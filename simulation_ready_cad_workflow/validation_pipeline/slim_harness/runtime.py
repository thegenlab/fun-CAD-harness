"""Native Chrono dynamics for a prepared scene; no author-generated result data."""
from __future__ import annotations

from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import sys
import time
from .runtime_progress import RuntimeProgress, worker_start

_worker_progress=worker_start(sys.argv[1]) if __name__=='__main__' and len(sys.argv)>1 else None

import numpy as np

from ..mechanism.pychrono_core import load_pychrono_core
from .contracts import segment_distance
from .measurements import evaluate_assertions
from .implementation import native_implementation_binding
from .trace_storage import TraceWriter
from .equivalent_elasticity import build_equivalent_elasticity
from .equivalent_collision_scope import build_equivalent_collision_scope
from .validation_scope import validate_validation_scope
from .measurement_registry import validate_runtime_registration, evaluate_native_trace
from .fixed_component_filter import build_fixed_component_filter
from .source_contact_filter import build_source_contact_filter
from .source_contact_normals import build_source_contact_normals
from .source_capsule_contacts import build_source_capsule_contacts
from .numerical import run_fixed_attempts, validate_solver_iterations
from .spur_gears import build_spur_gears

EXECUTION_IMPLEMENTATION = native_implementation_binding()

# Only main's exact supervised invocation executes native code in this process.
# An author importing simulate() gets the same process deadline as the CLI.
_managed_simulation = ContextVar("slim_managed_simulation", default=None)

CONTACT = {"friction": .3, "restitution": 0., "envelope_m": .0001,
           "margin_m": .0001, "mesh_swept_radius_m": 0.,
           "penetration_tolerance_m": .00005}


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def _pose(body):
    q = body.GetRot()
    return {"position_abs_m": _vec(body.GetPos()),
            "rotation_abs_quat_wxyz": [q.e0,q.e1,q.e2,q.e3]}


def _initialize_source_mesh_contacts(system, bodies):
    # Bullet's connected-triangle Populate replaces the model safe margin
    # with the swept radius. At r=0 this would reject every signed distance
    # <=0, including an initially touching support. Restore the margin after
    # Populate. Both the triangle AABB envelope and negative-distance window
    # must extend beyond the 50 um rejection limit; otherwise an overshot
    # contact disappears before it can trigger the fixed-step retry. These
    # search windows do not change that limit. Keep r=0 so source surfaces
    # are not inflated.
    system.GetCollisionSystem().Initialize()
    margins={}
    for key,body in bodies.items():
        if not body.IsCollisionEnabled(): continue
        model=body.GetCollisionModel()
        if not model.HasImplementation():
            raise RuntimeError("Source collision model was not initialized: "+key)
        before=float(model.GetSafeMargin())
        model.SetSafeMargin(CONTACT["margin_m"])
        effective=float(model.GetSafeMargin())
        if not math.isclose(effective,CONTACT["margin_m"],rel_tol=1e-6,abs_tol=1e-12):
            raise RuntimeError("Source collision safe margin was not applied: "+key)
        margins[key]={"after_populate_m":before,"effective_m":effective}
    return margins


def _joint_force_records(joints):
    records=[]
    for key,item in joints.items():
        link,spec=item["link"],item["spec"]
        first,second=link.GetFrame1Abs(),link.GetFrame2Abs()
        row={"joint_id":key,"body_a":spec["body_a"],"body_b":spec["body_b"],"active":bool(item["active"]),
             "point_a_abs_m":_vec(first.GetPos()),"point_b_abs_m":_vec(second.GetPos())}
        for side,frame,number in (("a",first,1),("b",second,2)):
            if item["active"]:
                wrench=getattr(link,f"GetReaction{number}")()
                row[f"force_{side}_abs_n"]=_vec(frame.TransformDirectionLocalToParent(wrench.force))
                row[f"torque_{side}_abs_nm"]=_vec(frame.TransformDirectionLocalToParent(wrench.torque))
            else:
                # Chrono retains the last solved wrench on a disabled link.
                # The disabled constraint exerts no current reaction.
                row[f"force_{side}_abs_n"]=[0.,0.,0.]
                row[f"torque_{side}_abs_nm"]=[0.,0.,0.]
        records.append(row)
    return records


def _axis_quat(c, axis):
    direction = np.asarray(axis,dtype=float)
    cosine = float(direction[2])
    if cosine < -1+1e-12:
        return c.ChQuaterniond(0,1,0,0)
    xyz = np.cross([0.,0.,1.],direction)
    q = np.r_[1+cosine,xyz]
    q /= np.linalg.norm(q)
    return c.ChQuaterniond(*q)


def _world_region(c, body, region):
    return [_vec(body.TransformPointLocalToParent(c.ChVector3d(*p))) for p in region["points"]]


def _active_at(item, t, duration):
    return float(item.get("start_s",0)) <= t < float(item.get("end_s",duration))


def _effective_inertia(c, bodies, records, spec, axis, *, fixed_components=()):
    """Controller estimate over currently rigid components; source bodies remain separate."""
    first, second = (spec["body_a"],), (spec["body_b"],)
    # The ordinary two-body path never enumerates unrelated bodies.
    if fixed_components:
        for component in fixed_components:
            if spec["body_a"] in component: first = tuple(component)
            if spec["body_b"] in component: second = tuple(component)
    if first == second:
        return 1e14  # An internal fixed connection has no relative response.
    b = bodies[spec["body_b"]]
    anchor = np.asarray(_vec(b.TransformPointLocalToParent(c.ChVector3d(
        *(np.asarray(spec["anchor"])-records[spec["body_b"]]["position_m"])))))
    inverse = 0.
    for members in (first, second):
        if any(bodies[key].IsFixed() for key in members): continue
        inertia = 0.
        for key in members:
            body, record = bodies[key], records[key]
            if spec["type"] == "revolute":
                local_axis = np.asarray(_vec(body.TransformDirectionParentToLocal(c.ChVector3d(*axis))))
                inertia += float(local_axis@np.asarray(record["inertia_kg_m2"])@local_axis)
                inertia += record["mass_kg"]*np.linalg.norm(np.cross(np.asarray(_vec(body.GetPos()))-anchor,axis))**2
            else:
                inertia += record["mass_kg"]
        inverse += 1/max(inertia,1e-14)
    return 1/max(inverse,1e-14)


_runtime_history_binding = ContextVar("runtime_history_binding", default=None)

def _simulate(prepared: dict, test: dict, output: Path, recorder: TraceWriter, *, max_step_s=.001, solver_iterations=160, progress=None) -> dict:
    # Windows Python 3.12 monotonic uses GetTickCount64 (15.625 ms).
    # Per-step deltas often quantize to zero; QPC resolves the actual cost.
    started = time.perf_counter()
    progress=progress or RuntimeProgress(output)
    progress.mark('loading_native',requested_duration_s=test['duration_s'])
    solver_iterations=validate_solver_iterations(solver_iterations)
    measurement = prepared.get("registered_measurements", {}).get(test["id"])
    if measurement is not None:
        validate_runtime_registration(measurement, test)
    c = load_pychrono_core()
    progress.mark('creating_system')
    system = c.ChSystemNSC()
    # The system constructor resets these defaults. Set them afterwards.
    c.ChCollisionModel.SetDefaultSuggestedEnvelope(CONTACT["envelope_m"])
    c.ChCollisionModel.SetDefaultSuggestedMargin(CONTACT["margin_m"])
    system.SetCollisionSystemType(c.ChCollisionSystem.Type_BULLET)
    system.SetNumThreads(3,3,1)
    system.SetGravitationalAcceleration(c.ChVector3d(*prepared.get("gravity_m_s2",[0,-9.81,0])))
    system.SetSolverType(c.ChSolver.Type_PSOR)
    solver = system.GetSolver().AsIterative()
    solver.SetMaxIterations(solver_iterations)
    # PSOR's stopping estimate samples residuals during a sweep. Later
    # coupled contact updates can invalidate a previously sampled zero.
    # Use the fixed sweep budget, as required by the first-step force probe.
    solver.SetTolerance(0.)
    material = c.ChContactMaterialNSC()
    material.SetFriction(CONTACT["friction"])
    material.SetRestitution(CONTACT["restitution"])
    bodies, meshes, accumulators = {}, [], {}
    analytic_shapes, analytic_material = [], []
    created_capsules_by_body = {}
    records = {b["id"]: b for b in prepared["bodies"]}
    scope = prepared.get("validation_scope")
    equivalent_value = prepared.get("equivalent_elasticity")
    has_native_clutch = any(row["kind"] == "friction_clutch" for row in (equivalent_value or {}).get("elements",[]))
    effective_solver_iterations=max(solver_iterations,240) if has_native_clutch else solver_iterations
    if has_native_clutch:
        # Chrono 10 PSOR can stop on an old sweep residual before coupled shaft
        # constraints converge. The native hold/slip probe requires full sweeps.
        solver.SetMaxIterations(effective_solver_iterations)
    from ..trajectory.configuration import save_simulation_configuration
    import sys
    save_simulation_configuration(Path(output).parent,interpreter=sys.executable,test=test,kind='effective',
        numerical={'max_step_s':min(float(test.get('step_s',.001)),max_step_s),
                   'solver_iterations':effective_solver_iterations,'solver':'PSOR','solver_tolerance':0.,
                   'collision_system':'Bullet','threads':[3,3,1],
                   'gravity_m_s2':prepared.get('gravity_m_s2',[0,-9.81,0]),'contact':dict(CONTACT)},history_binding=_runtime_history_binding.get())
    if equivalent_value is not None:
        validate_validation_scope(scope, prepared["bodies"])
        if scope["mode"] != "equivalent_elastic":
            raise RuntimeError("Equivalent dynamics require their explicit model scope")
    elastic_references = set(scope["elastic_reference_body_ids"]) if scope else set()
    for record in prepared["bodies"]:
        progress.mark('loading_body',force=True,body_id=record['id'])
        body = c.ChBody()
        body.SetName(record["id"])
        body.SetFixed(record["fixed"])
        body.SetPos(c.ChVector3d(*record["position_m"]))
        body.SetRot(c.ChQuaterniond(*record["rotation_wxyz"]))
        body.SetMass(record["mass_kg"])
        inertia = np.asarray(record["inertia_kg_m2"])
        body.SetInertiaXX(c.ChVector3d(*np.diag(inertia)))
        body.SetInertiaXY(c.ChVector3d(inertia[0,1],inertia[0,2],inertia[1,2]))
        mesh = c.ChTriangleMeshConnected()
        if not mesh.LoadWavefrontMesh(str(record["mesh_file"]),True,False):
            raise RuntimeError(f"Could not load original mesh for {record['id']}")
        body.AddCollisionShape(c.ChCollisionShapeTriangleMesh(
            material,mesh,record["fixed"],False,CONTACT["mesh_swept_radius_m"]))
        for supplement in record.get("source_analytic_material", []):
            if supplement["type"] != "capsule":
                raise RuntimeError("Unsupported source analytic material")
            shape = c.ChCollisionShapeCapsule(material, supplement["radius_m"], supplement["straight_length_m"])
            frame = c.ChFramed(c.ChVector3d(*supplement["center_local_m"]),
                              _axis_quat(c, supplement["axis_source"]))
            body.AddCollisionShape(shape, frame)
            created_capsules_by_body.setdefault(record["id"], []).append(supplement)
            analytic_shapes.append((shape, frame))
            analytic_material.append({"body_id": record["id"], **supplement})
        body.EnableCollision(record["id"] not in elastic_references)
        system.Add(body)
        bodies[record["id"]] = body
        accumulators[record["id"]] = body.AddAccumulator()
        meshes.append(mesh)
    progress.mark('creating_connections',body_id=None)
    joints = {}
    joint_types = {"revolute": c.ChLinkLockRevolute, "prismatic": c.ChLinkLockPrismatic,
                   "spherical": c.ChLinkLockSpherical, "fixed": c.ChLinkLockLock}
    for spec in prepared.get("joints",[]):
        a,b = bodies[spec["body_a"]],bodies[spec["body_b"]]
        link = joint_types[spec["type"]]()
        link.SetName(spec["id"])
        link.Initialize(a,b,c.ChFramed(c.ChVector3d(*spec["anchor"]),_axis_quat(c,spec["axis"])))
        system.Add(link)
        initial_offset = np.asarray(_vec(a.GetPos()))-np.asarray(_vec(b.GetPos()))
        joints[spec["id"]] = {"spec":spec,"link":link,"active":True,
                              "coordinate":0.,"initial_offset":initial_offset,
                              "work_j":0.,"effort":0.,"rate":0.,"damping_work_j":0.,"damping_effort":0.}
    fixed_component_filter = build_fixed_component_filter(bodies,joints,time_s=float(system.GetChTime()))
    equivalent = (build_equivalent_elasticity(c,system,bodies,joints,equivalent_value)
                  if equivalent_value is not None else None)
    support_records = dict(joints)
    if equivalent is not None:
        support_records.update(equivalent.support_records)
    contact_filter = (build_equivalent_collision_scope(c,system,bodies,records,scope,
                      support_records=equivalent.support_records)
                      if equivalent is not None else None)
    spur_gears = (build_spur_gears(c,system,bodies,support_records,prepared["spur_gears"],
                  axis_quat=_axis_quat,friction=CONTACT["friction"],previous_filter=contact_filter)
                  if prepared.get("spur_gears") else None)
    contacts = {"count":0,"max_penetration_m":0.,"pairs":[]}
    capture_forces=False
    detailed_contacts=[]

    class ContactReporter(c.ReportContactCallback):
        def OnReportContact(self,pa,pb,plane,distance,radius,force,torque,a,b,offset):
            depth = max(0.,-float(distance))
            contacts["max_penetration_m"] = max(contacts["max_penetration_m"],depth)
            if depth > CONTACT["penetration_tolerance_m"] or len(contacts["pairs"]) < 8:
                contacts["pairs"].append({"body_a":str(a.GetPhysicsItem().GetName()),
                    "body_b":str(b.GetPhysicsItem().GetName()),"distance_m":float(distance),
                    "point_a_m":_vec(pa),"point_b_m":_vec(pb),"force_n":_vec(force)})
            if capture_forces:
                world_force=plane*force
                world_torque=plane*torque
                detailed_contacts.append({"body_a":str(a.GetPhysicsItem().GetName()),"body_b":str(b.GetPhysicsItem().GetName()),
                    "point_a_abs_m":_vec(pa),"point_b_abs_m":_vec(pb),"distance_m":float(distance),
                    "force_a_abs_n":[-x for x in _vec(world_force)],"force_b_abs_n":_vec(world_force),
                    "torque_a_abs_nm":[-x for x in _vec(world_torque)],"torque_b_abs_nm":_vec(world_torque)})
            return True

    reporter = ContactReporter()
    signals, events = {}, []
    integrated_angles = {key:np.zeros(3) for key in bodies}
    duration = float(test["duration_s"])
    requested_step = min(float(test.get("step_s",.001)),max_step_s)
    steps = math.ceil(duration/requested_step)
    dt = duration/steps
    failure = None
    contact_collector, measurement_contact_schema, contact_step = None, None, None
    if measurement is not None and test["id"] == "automatic_alarm_and_stop":
        from .clock_contacts.calibration_binding import create_registered_collector
        contact_collector, measurement_contact_schema = create_registered_collector(
            c, system, bodies, [("alarm_hammer", "left_bell"), ("alarm_hammer", "right_bell")],
            measurement["contact_calibration_bundle"])
    progress.mark('initializing_collision_system')
    source_mesh_margins=_initialize_source_mesh_contacts(system,bodies)
    progress.mark('preparing_source_contact_filters')
    source_contact_filter=build_source_contact_filter(c,system,bodies,records,
        penetration_tolerance_m=CONTACT["penetration_tolerance_m"],previous_filter=spur_gears or contact_filter)
    source_contact_normals=build_source_contact_normals(c,system,source_contact_filter)
    source_capsule_contacts=build_source_capsule_contacts(c,system,source_contact_filter,records,
        source_contact_normals,created_capsules_by_body=created_capsules_by_body)
    force_indices=set(np.linspace(0,steps,min(720,steps+1),dtype=int).tolist())
    recorded_force_indices=[0]
    applied_loads=[]
    native_step_seconds = 0.
    trace_sampling_seconds = 0.
    load_ids = [str(load.get("id", f"load_{index:03d}")) for index,load in enumerate(test.get("loads",[]))]
    load_work = {key:0. for key in load_ids}
    load_dissipation = {key:0. for key in load_ids}

    def applied_record(body_id,point,force=None,torque=None,*,kind,identifier):
        force=[float(x) for x in (force if force is not None else [0.,0.,0.])]
        torque=[float(x) for x in (torque if torque is not None else [0.,0.,0.])]
        if not any(force) and not any(torque):return
        applied_loads.append({"id":identifier,"kind":kind,"body_id":body_id,"point_abs_m":_vec(point),
                              "force_abs_n":force,"torque_abs_nm":torque})

    def append_signal(key, value):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"Nonfinite physical state: {key}")
        signals[key] = float(value)

    def record(t,force_evidence=None):
        nonlocal trace_sampling_seconds
        elastic_state = None
        signals.clear()
        append_signal("time_s",t)
        poses = {}
        for key,body in bodies.items():
            poses[key] = _pose(body)
            dot = abs(float(np.dot(poses[key]["rotation_abs_quat_wxyz"],records[key]["rotation_wxyz"])))
            append_signal(f"{key}.orientation_error_rad",2*math.acos(min(1.,dot)))
            values = {"position":(_vec(body.GetPos()),"m"),
                      "angle":(integrated_angles[key],"rad"),
                      "linear_speed":(_vec(body.GetPosDt()),"m_s"),
                      "angular_speed":(_vec(body.GetAngVelParent()),"rad_s")}
            for kind,(xyz,unit) in values.items():
                for axis,value in zip("xyz",xyz): append_signal(f"{key}.{kind}.{axis}_{unit}",value)
        for key,item in joints.items():
            append_signal(f"{key}.coordinate",item["coordinate"])
            append_signal(f"{key}.active",item["active"])
            append_signal(f"input.{key}.effort",item["effort"])
            append_signal(f"input.{key}.work_j",item["work_j"])
            append_signal(f"damping.{key}.work_j",item["damping_work_j"])
        for key in load_work:
            append_signal(f"load.{key}.work_j",load_work[key])
            append_signal(f"load.{key}.dissipated_work_j",load_dissipation[key])
        append_signal("contact.count",contacts["count"])
        append_signal("contact.max_penetration_m",contacts["max_penetration_m"])
        sample = {"time_s":t,"body_poses":poses,"force_evidence":force_evidence or {"available":False,"reason":"not_sampled"}}
        if contact_step is not None:
            # One canonical solver-grid label for the same committed step.
            # The collector separately retains its actual native start time.
            sample["contact_step"] = {**contact_step,"time_s":t}
        if equivalent is not None:
            elastic_state = equivalent.sample()
            sample["equivalent_elasticity"] = elastic_state
            for name in ("potential_energy_j", "dissipated_energy_j"):
                append_signal("equivalent."+name, elastic_state[name])
            for key,row in elastic_state["elements"].items():
                for name,value in row.items():
                    # The first sample is unsolved; this last-step quantity
                    # is then None and remains available only in the raw state.
                    if name != "applied_step_generalized_force" and isinstance(value,(int,float)):
                        append_signal("equivalent."+key+"."+name,value)
            if contact_filter is not None:
                sample["equivalent_contact_exclusions"] = list(contact_filter.last_exclusions)
        if spur_gears is not None:
            sample["spur_gears"] = spur_gears.sample()
            sample["spur_gear_contact_exclusions"] = list(spur_gears.last_exclusions)
            for key, row in sample["spur_gears"].items():
                for name, value in row.items(): append_signal(f"gear.{key}.{name}", value)
            append_signal("gear.friction_dissipated_energy_j", spur_gears.friction_energy_j)
        sampling_started = time.perf_counter()
        recorder.append(sample,signals)
        trace_sampling_seconds += time.perf_counter()-sampling_started
        return elastic_state

    def domain_failure(elastic_state,t):
        if elastic_state is None:
            return None
        outside = [key for key,row in elastic_state["elements"].items() if not row["within_working_range"]]
        if outside:
            return {"time_s":t,"kind":"equivalent_model_outside_working_range","elements":outside}
        return None

    progress.mark('recording_initial_state')
    initial_elastic_state = record(0.,{"available":False,"reason":"unsolved","step_start_s":0.,"step_end_s":0.})
    failure = domain_failure(initial_elastic_state,0.)
    if failure is not None:
        events.append(failure)
    loop_started = time.perf_counter()
    for step in range(steps):
        if failure is not None:
            break
        t = step*dt
        progress.mark('stepping',step_index=step+1,recorded_samples=recorder.count,last_recorded_time_s=t)
        capture_forces=step+1 in force_indices
        applied_loads=[]
        applied_boundary_loads=[]
        detailed_contacts=[]
        for key,body in bodies.items(): body.EmptyAccumulator(accumulators[key])
        gravity=np.asarray(prepared.get("gravity_m_s2",[0,-9.81,0]),dtype=float)
        for key,body in bodies.items():
            if not body.IsFixed():applied_record(key,body.GetPos(),gravity*records[key]["mass_kg"],kind="gravity",identifier=f"gravity:{key}")
        for key,item in joints.items():
            spec = item["spec"]
            a,b = bodies[spec["body_a"]],bodies[spec["body_b"]]
            ra,rb = spec["region_a"],spec["region_b"]
            gap = segment_distance(*_world_region(c,a,ra),*_world_region(c,b,rb))-ra["radius"]-rb["radius"]
            if item["active"] and gap > .00005:
                item["active"] = False
                item["link"].SetDisabled(True)
                events.append({"time_s":t,"kind":"connection_released","joint":key,"gap_m":gap})
        # Resolve every released fixed link before estimating any controller inertia.
        fixed_component_filter.refresh(time_s=t)
        for key,item in joints.items():
            spec = item["spec"]
            a,b = bodies[spec["body_a"]],bodies[spec["body_b"]]
            axis = np.asarray(_vec(b.TransformDirectionLocalToParent(c.ChVector3d(*spec["axis"]))))
            if spec["type"] == "revolute":
                rate = float(np.dot(np.subtract(_vec(a.GetAngVelParent()),_vec(b.GetAngVelParent())),axis))
            else:
                offset = np.subtract(_vec(a.GetPos()),_vec(b.GetPos()))
                relative_velocity = np.subtract(_vec(a.GetPosDt()),_vec(b.GetPosDt()))-np.cross(_vec(b.GetAngVelParent()),offset)
                rate = float(np.dot(relative_velocity,axis))
            item["rate"] = rate
            item["effort"] = 0.
            effective = _effective_inertia(c,bodies,records,spec,axis)
            control_effective = None
            for drive in test.get("inputs",[]):
                if drive["joint"] != key or not item["active"] or not _active_at(drive,t,duration): continue
                target = float(drive.get("value",0))
                if drive.get("wave"):
                    wave = drive["wave"]
                    target = wave.get("offset",0)+wave["amplitude"]*math.sin(2*math.pi*wave["frequency_hz"]*t)
                mode = drive["mode"]
                if mode in ("speed","position") and control_effective is None:
                    control_effective = _effective_inertia(c,bodies,records,spec,axis,
                        fixed_components=fixed_component_filter.plan["components"])
                gain = float(drive.get("gain",1))
                damping = float(drive.get("damping",.1))
                if mode == "speed": effort = gain*(target-rate)/(1+gain*dt/control_effective)
                elif mode == "position": effort = (gain*(target-item["coordinate"])-damping*rate)/(1+damping*dt/control_effective+gain*dt*dt/control_effective)
                else: effort = target
                item["effort"] += float(np.clip(effort,-drive["max_effort"],drive["max_effort"]))
            effort = item["effort"]
            # Passive joint damping is part of the declared local behavior,
            # not an output actuator and not credited as input work.
            damping = float(spec.get("damping",0))
            applied = effort-damping*rate/(1+damping*dt/effective) if item["active"] else 0.
            item["damping_effort"] = applied-effort
            if spec["type"] == "revolute":
                a.AccumulateTorque(accumulators[spec["body_a"]],c.ChVector3d(*(axis*applied)),False)
                b.AccumulateTorque(accumulators[spec["body_b"]],c.ChVector3d(*(-axis*applied)),False)
                joint_frame = item["link"].GetFrame2Abs()
                point = joint_frame.GetPos()
                for body_id,sign in ((spec["body_a"],1),(spec["body_b"],-1)):
                    applied_record(body_id,point,torque=axis*effort*sign,kind="drive",identifier=f"drive:{key}:{body_id}")
                    applied_record(body_id,point,torque=axis*(applied-effort)*sign,kind="joint_damping",identifier=f"damping:{key}:{body_id}")
            elif spec["type"] == "prismatic":
                point = b.TransformPointLocalToParent(c.ChVector3d(*(np.asarray(spec["anchor"])-records[spec["body_b"]]["position_m"])))
                a.AccumulateForce(accumulators[spec["body_a"]],c.ChVector3d(*(axis*applied)),point,False)
                b.AccumulateForce(accumulators[spec["body_b"]],c.ChVector3d(*(-axis*applied)),point,False)
                for body_id,sign in ((spec["body_a"],1),(spec["body_b"],-1)):
                    applied_record(body_id,point,force=axis*effort*sign,kind="drive",identifier=f"drive:{key}:{body_id}")
                    applied_record(body_id,point,force=axis*(applied-effort)*sign,kind="joint_damping",identifier=f"damping:{key}:{body_id}")
        for load_index,load in enumerate(test.get("loads",[])):
            if not _active_at(load,t,duration): continue
            body = bodies[load["body"]]
            point = c.ChVector3d(*load["point"]) if "point" in load else body.GetPos()
            force = np.asarray(load.get("force_n",[0.,0.,0.]),dtype=float)
            torque = np.asarray(load.get("torque_nm",[0.,0.,0.]),dtype=float)
            damping_torque = np.zeros(3)
            omega = np.asarray(_vec(body.GetAngVelParent()))
            if load.get("rotational_damping_nm_s_rad",0):
                axis = np.asarray(load.get("axis_abs",[0.,0.,1.]),dtype=float); axis /= np.linalg.norm(axis)
                damping_torque = -float(load["rotational_damping_nm_s_rad"])*float(omega@axis)*axis
                torque = torque+damping_torque
            if np.any(force):body.AccumulateForce(accumulators[load["body"]],c.ChVector3d(*force),point,False)
            if np.any(torque):body.AccumulateTorque(accumulators[load["body"]],c.ChVector3d(*torque),False)
            net_torque = torque+np.cross(np.asarray(_vec(point))-np.asarray(_vec(body.GetPos())),force)
            applied_boundary_loads.append((load_ids[load_index],body,force,net_torque,damping_torque,
                                           np.asarray(_vec(body.GetPosDt())),omega))
            applied_record(load["body"],point,force,torque,kind="load",identifier=f"load:{load.get('id',load['body'])}")
        # Accumulator loads are not time functions; refresh Chrono's force
        # cache now so the current controller effort acts in this step.
        if equivalent is not None:
            released = [row["id"] for row in equivalent_value["elements"]
                        if row["kind"] == "friction_clutch" and not joints[row["joint_id"]]["active"]]
            if released:
                failure = {"time_s":t,"kind":"equivalent_model_connection_released","elements":released}
                events.append(failure)
                break
            equivalent.before_step(dt)
            applied_loads.extend(equivalent.applied_load_records())
        source_capsule_contacts.begin_step()
        if spur_gears is not None:
            spur_gears.apply_friction(dt)
            applied_loads.extend(spur_gears.friction_records)
        system.Update()
        if equivalent is not None:
            equivalent.capture_native_step_loads()
        if contact_collector is not None:
            contact_collector.begin_step(step+1,t,dt)
        native_started = time.perf_counter()
        solved=system.DoStepDynamics(dt)
        native_step_seconds += time.perf_counter()-native_started
        if solved is False:
            failure={"kind":"solver_nonconvergence","time_s":t}
            break
        if any(not np.isfinite([*_vec(body.GetPos()),*_vec(body.GetPosDt()),*_vec(body.GetAngVelParent()),
                               body.GetRot().e0,body.GetRot().e1,body.GetRot().e2,body.GetRot().e3]).all()
               for body in bodies.values()):
            failure={"kind":"nonfinite_solved_state","time_s":t}
            break
        if equivalent is not None:
            equivalent.after_step()
            applied_loads.extend(equivalent.solved_load_records())
        if spur_gears is not None:
            spur_gears.after_step()
        if contact_collector is not None:
            contact_step = contact_collector.finish_step()
        contacts.update(count=int(system.GetNumContacts()),max_penetration_m=0.,pairs=[])
        if contacts["count"]: system.GetContactContainer().ReportAllContacts(reporter)
        if contacts["max_penetration_m"]>CONTACT["penetration_tolerance_m"] and not capture_forces:
            # Preserve the actual failure frame even if it was between normal
            # display samples; this is a read of the same solved contact state.
            capture_forces=True
            contacts.update(max_penetration_m=0.,pairs=[])
            system.GetContactContainer().ReportAllContacts(reporter)
        for key,body in bodies.items(): integrated_angles[key] += np.asarray(_vec(body.GetAngVelParent()))*dt
        for load_id,body,force,torque,damping_torque,before_v,before_w in applied_boundary_loads:
            mean_v = .5*(before_v+np.asarray(_vec(body.GetPosDt())))
            mean_w = .5*(before_w+np.asarray(_vec(body.GetAngVelParent())))
            load_work[load_id] += float(force@mean_v+torque@mean_w)*dt
            load_dissipation[load_id] -= float(damping_torque@mean_w)*dt
        for item in joints.values():
            spec = item["spec"]
            a,b = bodies[spec["body_a"]],bodies[spec["body_b"]]
            axis = np.asarray(_vec(b.TransformDirectionLocalToParent(c.ChVector3d(*spec["axis"]))))
            if spec["type"] == "revolute":
                rate_after=float(np.dot(np.subtract(_vec(a.GetAngVelParent()),_vec(b.GetAngVelParent())),axis))
                item["coordinate"] += rate_after*dt
            else:
                offset=np.subtract(_vec(a.GetPos()),_vec(b.GetPos()))
                velocity=np.subtract(_vec(a.GetPosDt()),_vec(b.GetPosDt()))-np.cross(_vec(b.GetAngVelParent()),offset)
                rate_after=float(np.dot(velocity,axis))
                item["coordinate"] = float(np.dot(np.subtract(_vec(a.GetPos()),_vec(b.GetPos())),axis)-np.dot(item["initial_offset"],spec["axis"]))
            item["work_j"] += item["effort"]*.5*(item["rate"]+rate_after)*dt
            item["damping_work_j"] -= item["damping_effort"]*.5*(item["rate"]+rate_after)*dt
        force_evidence=None
        if capture_forces:
            recorded_force_indices.append(step+1)
            force_evidence={"available":True,"step_start_s":t,"step_end_s":(step+1)*dt,
                "joint_reactions":_joint_force_records(joints),"contacts":detailed_contacts,
                "external_loads":applied_loads,"external_loads_complete":True}
        elastic_state = record((step+1)*dt,force_evidence)
        if equivalent is not None:
            failure = domain_failure(elastic_state,(step+1)*dt)
            if failure is not None:
                events.append(failure)
                break
        if contacts["max_penetration_m"] > CONTACT["penetration_tolerance_m"]:
            failure = {"kind":"contact_penetration","time_s":(step+1)*dt,**contacts}
            break
    loop_seconds = time.perf_counter()-loop_started
    progress.mark('finalizing',recorded_samples=recorder.count,
                  last_recorded_time_s=(recorder.count-1)*dt if recorder.count else None)
    recorded = recorder.view({})
    samples, signals = recorded["samples"], recorded["signals"]
    assertions = evaluate_assertions(test,signals,pose_samples=samples)
    if len(samples) != steps+1:
        assertions.append({"id":"complete_duration","passed":False,"error":"Simulation stopped before the requested duration"})
    trace = {"schema_version":"slim_physical_trace_v1","test_id":test["id"],
             "execution_implementation":dict(EXECUTION_IMPLEMENTATION),
             "connections":prepared.get("joints",[]),"gravity_m_s2":prepared.get("gravity_m_s2",[0,-9.81,0]),
             "force_evidence_sampling":{"solver_indices":recorded_force_indices,"maximum_regular_frames":720,
                 "coordinates":"world","joint_wrenches":"GetReaction1/2 transformed by their respective absolute joint frames",
                 "contact_wrenches":"contact plane times callback force/torque acts on body_b; body_a receives the opposite",
                 "timing":"applied loads at their application time; joint and contact reactions after the completed solver step"},
             "engine":"Project Chrono / NSC / Bullet", "collision_representation":"original_triangle_mesh",
             "settings":CONTACT,"requested_duration_s":duration,"step_s":dt,
             "controller":"inertia-stabilized sampled force feedback; actual effort and work recorded",
             "events":events,"failure":failure,"assertions":assertions,
             "physical_pass":failure is None and all(x["passed"] for x in assertions),
             "elapsed_seconds":time.perf_counter()-started}
    trace["timings"] = {"setup_seconds":loop_started-started,"simulation_loop_seconds":loop_seconds,
                        "native_step_seconds":native_step_seconds,"trace_sampling_seconds":trace_sampling_seconds,
                        "clock":"perf_counter", "clock_resolution_s":time.get_clock_info("perf_counter").resolution,
                        "scope":"Native stepping includes Chrono callbacks and is a subset of the loop; trace commit and outer audit are measured by the caller."}
    trace["native_fixed_component_collision_filter"] = fixed_component_filter.metadata()
    trace["source_mesh_collision_margins"] = source_mesh_margins
    if analytic_material:
        trace["collision_representation"] = "original source meshes with contained analytic axial material"
        trace["source_analytic_material"] = analytic_material
    trace["native_source_contact_filter"] = source_contact_filter.metadata()
    trace["native_source_contact_normals"] = source_contact_normals.metadata()
    trace["native_source_capsule_contacts"] = source_capsule_contacts.metadata()
    if spur_gears is not None:
        trace["spur_gears"] = spur_gears.metadata()
        trace["collision_representation"] += " with finite source-bound spur tooth transmission"
    trace["native_threads"] = {"chrono":system.GetNumThreadsChrono(),"collision":system.GetNumThreadsCollision(),"eigen":system.GetNumThreadsEigen()}
    trace["native_solver"] = {"type":"PSOR","max_iterations":effective_solver_iterations,
                              "tolerance":0.,"termination":"fixed_sweep_budget"}
    timestepper_type = system.GetTimestepper().GetType()
    trace["native_timestepper"] = {
        "type": next((name.removeprefix("Type_") for name in dir(c.ChTimestepper)
                      if name.startswith("Type_") and getattr(c.ChTimestepper,name) == timestepper_type),
                     str(timestepper_type)),
        "native_enum": int(timestepper_type),
    }
    if scope is not None:
        trace["validation_scope"] = scope
    if equivalent is not None:
        trace["equivalent_elasticity"] = equivalent.metadata()
        trace["equivalent_solver"] = {"type":"PSOR","max_iterations":effective_solver_iterations,
                                      "tolerance":0.}
        trace["collision_representation"] += " with declared equivalent elastic references and finite pair-local regions"
        if contact_filter is not None:
            trace["equivalent_collision_scope"] = contact_filter.metadata()
    if measurement is not None:
        trace["measurement_registration_signature"] = measurement["signature"]
        if measurement_contact_schema is not None:
            trace["measurement_contact_schema"] = measurement_contact_schema
        try:
            trace.update(evaluate_native_trace(recorder.view(trace), measurement))
        except Exception as exc:
            # Retain the completed native record, but never publish its raw
            # result as acceptance when the required measurement cannot run.
            trace.update(native_assertions=list(trace["assertions"]),
                         native_physical_pass=trace["physical_pass"],
                         physical_pass=False, measurement_error=str(exc))
            recorder.commit(trace)
            raise
    result=recorder.commit(trace)
    progress.mark('finished',failure=failure,recorded_samples=recorder.count)
    return result


def simulate(prepared: dict, test: dict, output: Path, *, max_step_s=.001, solver_iterations=160) -> dict:
    construction_project = os.environ.get("SIMREADY_CONSTRUCTION_PROJECT")
    call = (id(prepared), id(test), str(Path(output).resolve()))
    if construction_project and _managed_simulation.get() != call:
        from .construction_limits import run_imported_simulation
        return run_imported_simulation(prepared, test, output, Path(construction_project))
    solver_iterations=validate_solver_iterations(solver_iterations)
    progress=_worker_progress or RuntimeProgress(output)
    def run_once(path,step):
        progress.mark('running_attempt',force=True,attempt_trace_file=str(path),
                      attempt_progress_file=str(Path(path).with_suffix('.progress.json')),
                      recorded_samples=0,last_recorded_time_s=None)
        expected_samples=math.ceil(float(test["duration_s"])/step)+1
        attempt_progress=RuntimeProgress(path)
        attempt_progress.mark('opening_trace')
        try:
            with TraceWriter(path,expected_samples) as recorder:
                result=_simulate(prepared,test,path,recorder,max_step_s=step,solver_iterations=solver_iterations,
                                 progress=attempt_progress)
                progress.mark('attempt_finished',recorded_samples=recorder.count,
                              last_recorded_time_s=attempt_progress.state['last_recorded_time_s'])
                return result
        except BaseException as exc:
            attempt_progress.mark('error',error=f'{type(exc).__name__}: {exc}')
            raise
    try:
        result=run_fixed_attempts(run_once,test,output,max_step_s=max_step_s,solver_iterations=solver_iterations)
        progress.mark('finished')
        return result
    except BaseException as exc:
        progress.mark('error',error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    request_path = Path(sys.argv[1]).resolve()
    construction_project = os.environ.get("SIMREADY_CONSTRUCTION_PROJECT")
    if construction_project and os.environ.get("SIMREADY_CONSTRUCTION_GUARDED_REQUEST") != str(request_path):
        from .construction_limits import run_runtime_entrypoint
        raise SystemExit(run_runtime_entrypoint(request_path, Path(construction_project)))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    binding_token=_runtime_history_binding.set(request.get("history_binding") or {})
    output = Path(request["output"])
    token = _managed_simulation.set((id(request["scene"]), id(request["test"]), str(output.resolve())))
    try:
        print(json.dumps(simulate(request["scene"],request["test"],output,
                                 max_step_s=request.get("numerical_settings",{}).get("max_step_s",.001),
                                 solver_iterations=request.get("numerical_settings",{}).get("solver_iterations",160))))
    finally:
        _managed_simulation.reset(token)
        _runtime_history_binding.reset(binding_token)


if __name__ == "__main__":
    main()
