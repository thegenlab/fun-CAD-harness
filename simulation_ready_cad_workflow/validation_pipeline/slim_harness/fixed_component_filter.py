"""Native collision families for currently effective fixed-joint components.

Only family/mask fields are changed. Geometry, body properties, constraints
and the existing equivalent-region narrowphase callback remain untouched.
"""
from copy import deepcopy


ALL_FAMILIES_MASK = (1 << 15)-1
MAX_COMPONENT_FAMILY = 14  # Chrono 10 implementation asserts family < 15.


class FixedComponentFilterError(RuntimeError):
    pass


def component_plan(body_ids, active_fixed_edges):
    identifiers = tuple(sorted(body_ids))
    parent = {key: key for key in identifiers}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for joint_id, first, second in active_fixed_edges:
        if first not in parent or second not in parent:
            raise FixedComponentFilterError(f"Fixed joint {joint_id} names a missing body")
        a, b = root(first), root(second)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = {}
    for key in identifiers:
        groups.setdefault(root(key), []).append(key)
    components = sorted((tuple(values) for values in groups.values()))
    rigid = [values for values in components if len(values) > 1]
    capacity_available = len(rigid) <= MAX_COMPONENT_FAMILY
    families = {key: 0 for key in identifiers}
    for family, members in enumerate(rigid if capacity_available else [], start=1):
        for key in members:
            families[key] = family
    masks = {key: ALL_FAMILIES_MASK & ~(1 << family) if family else ALL_FAMILIES_MASK
             for key, family in families.items()}
    return {"components": [list(values) for values in rigid], "families": families, "masks": masks,
            "capacity_available": capacity_available,
            "omitted_internal_body_pairs": sum(len(values)*(len(values)-1)//2 for values in rigid) if capacity_available else 0}


class FixedComponentCollisionFilter:
    def __init__(self, bodies, joints, *, time_s=0.):
        self.bodies, self.joints = bodies, joints
        self.models = {key: body.GetCollisionModel() for key, body in bodies.items()}
        self.baseline = {}
        for key, model in self.models.items():
            if model is None:
                continue
            row = {"family": int(model.GetFamily()), "mask": int(model.GetFamilyMask())}
            self.baseline[key] = row
            if row != {"family": 0, "mask": ALL_FAMILIES_MASK}:
                raise FixedComponentFilterError(
                    f"Unexpected preexisting native collision family/mask on {key}; preserve it and stop before changing filters")
        self.fixed_records = [(key, row) for key, row in sorted(joints.items()) if row["spec"]["type"] == "fixed"]
        self._signature = None
        self.plan = None
        self.events = []
        self.native_model_updates = 0
        self.refresh(time_s=time_s)

    def _active_edges(self):
        # The runtime owns `active`; also observe a directly disabled native link.
        return tuple((key, row["spec"]["body_a"], row["spec"]["body_b"])
                     for key, row in self.fixed_records
                     if bool(row["active"]) and not row["link"].IsDisabled())

    def refresh(self, *, time_s):
        edges = self._active_edges()
        if edges == self._signature:
            return False
        next_plan = component_plan(self.bodies, edges)  # Above capacity restores ordinary all-open masks.
        changed = []
        for key, model in self.models.items():
            if model is None:
                continue
            family, mask = next_plan["families"][key], next_plan["masks"][key]
            before = (int(model.GetFamily()), int(model.GetFamilyMask()))
            if before[0] != family:
                model.SetFamily(family)
            if before[1] != mask:
                model.SetFamilyMask(mask)
            if int(model.GetFamily()) != family or int(model.GetFamilyMask()) != mask:
                raise FixedComponentFilterError(f"Native family update did not take effect on {key}; do not advance dynamics")
            if before != (family, mask):
                changed.append(key)
        self._signature, self.plan = edges, next_plan
        self.native_model_updates += len(changed)
        self.events.append({"time_s": float(time_s), "active_fixed_joint_ids": [edge[0] for edge in edges],
                            "components": deepcopy(next_plan["components"]), "updated_body_ids": changed,
                            "omitted_internal_body_pairs": next_plan["omitted_internal_body_pairs"]})
        return True

    def metadata(self):
        return {"schema_version": "native_fixed_component_collision_filter_v1",
                "rule": "Omit native collision only within a currently effective fixed-joint connected component",
                "singleton_family": 0, "component_families_available": [1, MAX_COMPONENT_FAMILY],
                "all_families_mask": ALL_FAMILIES_MASK, "initial_native_family_state": deepcopy(self.baseline),
                "current_plan": deepcopy(self.plan), "events": deepcopy(self.events),
                "native_model_updates": self.native_model_updates,
                "body_ids_mass_geometry_joints_changed": False,
                "cross_component_contacts": "All native family pairs remain enabled",
                "capacity_overflow": "Restore family 0 and the default all-open mask for every model",
                "existing_equivalent_region_narrowphase": "Retained unchanged",
                "native_mesh_representation": "Retained unchanged; no mesh or collider-type replacement"}


def build_fixed_component_filter(bodies, joints, *, time_s=0.):
    return FixedComponentCollisionFilter(bodies, joints, time_s=time_s)
