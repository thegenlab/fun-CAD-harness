import assert from "node:assert/strict";
import { BoxGeometry, TorusGeometry, BufferAttribute } from "three";
import { sourceCreasedNormals } from "../dashboard/static/viewer3d_normals.js";

// A dense millimetre-scale curved part previously placed nearly all vertices
// in one normal-lookup cell. Its normals must agree with the same part in mm.
const small = new TorusGeometry(0.003, 0.0005, 48, 256);
const originalPositions = small.getAttribute("position").array.slice();
const originalIndex = small.index.array.slice();
const colors = new Float32Array(small.getAttribute("position").count * 3).fill(0.25);
small.setAttribute("color", new BufferAttribute(colors, 3));
const start = performance.now();
const smoothed = sourceCreasedNormals(small);
const milliseconds = performance.now() - start;
const large = small.clone().scale(1000, 1000, 1000).translate(100, -200, 300);
const reference = sourceCreasedNormals(large);
assert.deepEqual(small.getAttribute("position").array, originalPositions);
assert.deepEqual(small.index.array, originalIndex);
const sourceExpanded = small.toNonIndexed();
assert.deepEqual(smoothed.getAttribute("position").array, sourceExpanded.getAttribute("position").array);
assert.deepEqual(smoothed.getAttribute("color").array, sourceExpanded.getAttribute("color").array);
const normals = smoothed.getAttribute("normal");
const expected = reference.getAttribute("normal");
let maxDifference = 0;
for (let i = 0; i < normals.array.length; i += 1) {
  assert.ok(Number.isFinite(normals.array[i]));
  maxDifference = Math.max(maxDifference, Math.abs(normals.array[i] - expected.array[i]));
}
assert.ok(maxDifference < 0.002, `Normals depend on source units: ${maxDifference}`);

// Preserve hard edges and do not modify a non-indexed input in place.
const box = new BoxGeometry(0.004, 0.006, 0.008).toNonIndexed();
const before = box.getAttribute("normal").array.slice();
const boxResult = sourceCreasedNormals(box);
assert.deepEqual(box.getAttribute("normal").array, before);
assert.deepEqual(boxResult.getAttribute("normal").array, before);
assert.notEqual(boxResult, box);
console.log(JSON.stringify({ triangles: normals.count / 3, milliseconds, maxDifference }));
