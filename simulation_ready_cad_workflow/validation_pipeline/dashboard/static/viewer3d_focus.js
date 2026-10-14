export function isGeneratedSplitComponentNodeName(name) {
  const parts = String(name || "").split("__");
  return parts.length >= 4 && /^component_[0-9]{4}_/.test(parts.at(-1));
}

// Match THREE.PropertyBinding.sanitizeNodeName exactly. Older replay scenes
// may contain dots in their manifest node names even though GLTFLoader removes
// them from the corresponding Object3D names.
export function physicsNodeBindingKey(name) {
  return String(name || "").replace(/\s/g, "_").replace(/[\[\]\.:/]/g, "");
}

export function buildPhysicsComponentIndex(bodies) {
  const index = new Map();
  for (const body of bodies || []) {
    for (const component of body.components || []) {
      const binding = {
        ...component,
        bodyId: body.body_id,
        fixed: Boolean(body.fixed),
        groupIds: body.group_ids || [],
      };
      index.set(component.node, binding);
      index.set(physicsNodeBindingKey(component.node), binding);
    }
  }
  return index;
}

export function indexedPhysicsComponent(index, name) {
  return index.get(name) || index.get(physicsNodeBindingKey(name)) || null;
}

export function physicsComponentSolidRef(component) {
  if (!component?.bodyId || !component?.solid_id) return "";
  return `${component.bodyId}/${component.solid_id}`;
}

export function physicsFocusRole({ meshName, component, focus }) {
  if (!focus) return "";
  if (meshName === focus.primary_node) return "primary";
  if (meshName === focus.secondary_node) return "secondary";
  const solidRef = physicsComponentSolidRef(component);
  const solidPair = Array.isArray(focus.solid_pair) ? focus.solid_pair.map(String) : [];
  if (solidRef && solidRef === solidPair[0]) return "primary";
  if (solidRef && solidRef === solidPair[1]) return "secondary";
  return "";
}
