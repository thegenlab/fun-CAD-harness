"""Observe registered rigid-body NSC pairs; never invent FEA point velocities.

The native CustomCollisionCallback runs after contact creation and before the
timestepper advances. Post-solve contacts are matched to those exact points.
Primary source: projectchrono/chrono, ChSystem.cpp ComputeCollisions/AdvanceDynamics;
ChContactContainer.h SumAllContactForces. Runtime unit/sign calibration is required.
"""
from collections import defaultdict, deque
import math


class ContactCollectionError(RuntimeError):pass


def xyz(v):return [float(v.x),float(v.y),float(v.z)]
def sub(a,b):return [a[i]-b[i] for i in range(3)]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def norm(v):return math.sqrt(dot(v,v))
def point_velocity(body,point):
    return [a+b for a,b in zip(xyz(body.GetPosDt()),cross(xyz(body.GetAngVelParent()),sub(point,xyz(body.GetPos()))))]


class RigidContactCollector:
    def __init__(self,c,system,bodies,body_pairs,units_review):
        if not isinstance(system,c.ChSystemNSC):raise ContactCollectionError('Collector requires the calibrated NSC system')
        if (units_review.get('raw_reaction_units')!='N'
                or units_review.get('positive_world_reaction_applies_to')!='body_b'
                or units_review.get('max_support_relative_error',math.inf)>1e-3
                or units_review.get('max_momentum_balance_error_Ns',math.inf)>1e-10):
            raise ContactCollectionError('A real unit/sign/momentum reference review is required')
        supported={(row['solver_type'],row['timestepper_type']) for row in units_review['cases']}
        if (str(system.GetSolver().GetType()),str(system.GetTimestepper().GetType())) not in supported:
            raise ContactCollectionError('Selected solver/timestepper lacks this unit calibration')
        self.c,self.system,self.bodies=c,system,bodies
        self.pairs={frozenset(pair) for pair in body_pairs}
        required=set().union(*self.pairs)
        if any(name not in bodies or not isinstance(bodies[name],c.ChBody) for name in required):
            raise ContactCollectionError('Only explicitly registered ChBody pairs have pre-impact velocity support')
        self.armed=False;self.error=None;self.before=defaultdict(deque);self.after=[];self.collision_calls=0
        owner=self
        class BeforeReport(c.ReportContactCallback):
            def OnReportContact(self,pa,pb,plane,distance,radius,force,torque,a,b,offset):
                try:owner._before(pa,pb,plane,a,b)
                except Exception as exc:owner.error='pre-contact: '+repr(exc)
                return True
        class CollisionEnd(c.CustomCollisionCallback):
            def OnCustomCollision(self,active_system):
                if not owner.armed:return
                try:
                    owner.collision_calls+=1
                    owner.capture_time=float(active_system.GetChTime())
                    if abs(owner.capture_time-owner.native_start_s)>1e-9:
                        raise ContactCollectionError('Collision callback is not at the requested step start')
                    active_system.GetContactContainer().ReportAllContacts(owner.before_report)
                except Exception as exc:owner.error='collision hook: '+repr(exc)
        class AfterReport(c.ReportContactCallback):
            def OnReportContact(self,pa,pb,plane,distance,radius,force,torque,a,b,offset):
                try:owner._after(pa,pb,plane,distance,force,a,b,offset)
                except Exception as exc:owner.error='post-contact: '+repr(exc)
                return True
        self.before_report=BeforeReport();self.after_report=AfterReport();self.collision_hook=CollisionEnd()
        # Register this after any custom callbacks that add or alter contacts.
        system.RegisterCustomCollisionCallback(self.collision_hook)

    def _identity(self,pa,pb,plane,a,b):
        if a is None or b is None:return None
        names=(str(a.GetPhysicsItem().GetName()),str(b.GetPhysicsItem().GetName()))
        if frozenset(names) not in self.pairs:return None
        points=(xyz(pa),xyz(pb));normal=xyz(plane*self.c.ChVector3d(1,0,0))
        if not all(math.isfinite(v) for values in (*points,normal) for v in values):
            raise ContactCollectionError('Nonfinite registered contact geometry')
        return names,points,normal,(names,*points[0],*points[1],*normal)

    def _before(self,pa,pb,plane,a,b):
        identity=self._identity(pa,pb,plane,a,b)
        if identity is None:return
        names,points,normal,key=identity
        velocities=[point_velocity(self.bodies[name],point) for name,point in zip(names,points)]
        locals_=[xyz(self.bodies[name].TransformPointParentToLocal(self.c.ChVector3d(*point)))
                 for name,point in zip(names,points)]
        native=[]
        for obj,point in ((a,pa),(b,pb)):
            native.append(xyz(obj.GetContactPointSpeed(point)) if hasattr(obj,'GetContactPointSpeed') else None)
        errors=[norm(sub(v,n)) for v,n in zip(velocities,native) if n is not None]
        com_relative=sub(xyz(self.bodies[names[1]].GetPosDt()),xyz(self.bodies[names[0]].GetPosDt()))
        self.before[key].append({'body_a':names[0],'body_b':names[1],
            'point_a_abs_m':points[0],'point_b_abs_m':points[1],'normal_abs':normal,
            'point_velocity_a_before_impulse_m_s':velocities[0],'point_velocity_b_before_impulse_m_s':velocities[1],
            'closing_speed_before_impulse_m_s':-dot(sub(velocities[1],velocities[0]),normal),
            'closing_speed_COM_only_before_impulse_m_s':-dot(com_relative,normal),
            'native_point_velocity_agreement_error_m_s':max(errors) if errors else None,
            '_point_a_local_m':locals_[0],'_point_b_local_m':locals_[1]})

    def _after(self,pa,pb,plane,distance,force,a,b,offset):
        identity=self._identity(pa,pb,plane,a,b)
        if identity is None:return
        names,points,normal,key=identity
        if not self.before[key]:raise ContactCollectionError('A solved contact lacks an exact pre-solve point match')
        row=self.before[key].popleft()
        after=[]
        for name,field in zip(names,('_point_a_local_m','_point_b_local_m')):
            point=xyz(self.bodies[name].TransformPointLocalToParent(self.c.ChVector3d(*row.pop(field))))
            after.append(point_velocity(self.bodies[name],point))
        raw=xyz(force)
        if not all(math.isfinite(v) for v in raw):raise ContactCollectionError('Nonfinite contact reaction')
        row.update(normal_impulse_n_s=raw[0]*self.dt,
                   raw_callback_force_contact_n=raw,raw_callback_force_on_b_abs_n=xyz(plane*force),
                   distance_m=float(distance),constraint_offset=int(offset),
                   closing_speed_after_impulse_m_s=-dot(sub(after[1],after[0]),normal))
        self.after.append(row)

    def begin_step(self,step,start_s,dt):
        if self.armed:raise ContactCollectionError('Previous contact step was not completed')
        self.step,self.start_s,self.dt=int(step),float(start_s),float(dt)
        self.native_start_s=float(self.system.GetChTime())
        self.before=defaultdict(deque);self.after=[];self.error=None;self.collision_calls=0
        self.armed=True

    def finish_step(self):
        if not self.armed:raise ContactCollectionError('No contact step was started')
        try:
            self.system.GetContactContainer().ReportAllContacts(self.after_report)
            if self.error:raise ContactCollectionError(self.error)
            if self.collision_calls!=1:raise ContactCollectionError('Expected exactly one native collision event per committed step')
            if any(self.before.values()):raise ContactCollectionError('Pre-solve contact rows were not all matched after solve')
            if abs(float(self.system.GetChTime())-(self.native_start_s+self.dt))>1e-9:
                raise ContactCollectionError('System time did not advance by the recorded solver step')
            return {'step':self.step,'time_s':self.start_s+self.dt,'contacts':self.after,
                    'capture_hook':'native_collision_end_before_timestepper','pre_capture_time_s':self.capture_time,
                    'native_collision_callbacks':self.collision_calls,'native_step_start_s':self.native_start_s}
        finally:self.armed=False

    def measurement_schema(self):
        return {'sampling':'every_solver_step','time_basis':'solver_step_end','normal_impulse_unit':'N*s',
                'closing_speed_unit':'m/s','closing_speed_timing':'before_same_contact_impulse',
                'normal_impulse_sign':'compression_positive','closing_speed_sign':'approach_positive',
                'covered_body_pairs':[sorted(pair) for pair in sorted(self.pairs,key=lambda pair:tuple(sorted(pair)))],
                'impulse_basis':'callback force in N times dt, established by current rigid unit/sign/momentum reference runs',
                'pre_velocity_basis':'native end-of-collision callback; v + omega cross material contact arm'}
