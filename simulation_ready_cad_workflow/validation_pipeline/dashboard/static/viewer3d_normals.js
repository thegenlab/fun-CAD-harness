import { Box3, BufferAttribute } from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";

export function sourceCreasedNormals(geometry, creaseAngle = Math.PI / 6) {
  const result = geometry.index ? geometry.toNonIndexed() : geometry.clone();
  const position = result.getAttribute("position");
  const bounds = new Box3().setFromBufferAttribute(position);
  const extent = Math.max(
    bounds.max.x - bounds.min.x,
    bounds.max.y - bounds.min.y,
    bounds.max.z - bounds.min.z,
  );
  if (!(extent > 0)) {
    result.computeVertexNormals();
    return result;
  }

  // Three r160 groups normal candidates in 0.01-unit cells. In metre-scale
  // CAD this puts entire springs in a few cells and makes smoothing quadratic.
  // Normalize only the temporary lookup coordinates; retain exact source poses
  // and vertex positions, including their original precision and attributes.
  const scale = 1000 / extent;
  const lookup = new Float64Array(position.count * 3);
  for (let i = 0; i < position.count; i += 1) {
    lookup[3 * i] = (position.getX(i) - bounds.min.x) * scale;
    lookup[3 * i + 1] = (position.getY(i) - bounds.min.y) * scale;
    lookup[3 * i + 2] = (position.getZ(i) - bounds.min.z) * scale;
  }
  result.setAttribute("position", new BufferAttribute(lookup, 3));
  try {
    toCreasedNormals(result, creaseAngle);
  } finally {
    result.setAttribute("position", position);
  }
  return result;
}
