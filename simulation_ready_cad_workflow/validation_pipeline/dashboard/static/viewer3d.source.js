import { PartsController } from "./parts3d.js";
import "./workbench.js";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ViewHelper } from "three/examples/jsm/helpers/ViewHelper.js";
import { sourceCreasedNormals } from "./viewer3d_normals.js";
import { mechanicsForces, setupMechanics, updateMechanics, layoutMechanicsLabels } from "./viewer3d_mechanics.js";
import {
  buildPhysicsComponentIndex,
  indexedPhysicsComponent,
  isGeneratedSplitComponentNodeName,
  physicsComponentSolidRef as componentSolidRef,
  physicsFocusRole as focusRoleForComponent,
} from "./viewer3d_focus.js";

const mounted = new Map();
const FIXED_DISPLAY_MODES = new Set(["outline", "xray", "solid"]);
const DEFAULT_FIXED_DISPLAY_MODE = "outline";
const DEFAULT_PHYSICS_FIXED_DISPLAY_MODE = "solid";
const FIXED_XRAY_OPACITY = 0.16;
const MOVING_DISPLAY_MODES = new Set(["solid", "edges", "outline"]);
const DEFAULT_MOVING_DISPLAY_MODE = "solid";
const MAX_LOCAL_HEAT_SPOTS = 8;
const MAX_LOCAL_HEAT_SEGMENTS = 8;

const PALETTES = {
  design: {
    fixed: { color: 0x71849b, opacity: 1, roughness: 0.72, metalness: 0.08 },
    moving: { color: 0xd9a441, opacity: 1, roughness: 0.58, metalness: 0.12 },
    axis: 0xd9a441,
    background: 0xf4f7fa,
  },
  mechanism: {
    fixed: { color: 0x8a96a3, opacity: 0, roughness: 0.8, metalness: 0.02 },
    moving: { color: 0x22a86f, opacity: 1, roughness: 0.52, metalness: 0.1 },
    axis: 0x13885a,
    background: 0xf3f8f6,
  },
  physics: {
    fixed: { color: 0x7e8fa1, opacity: 0, roughness: 0.78, metalness: 0.04 },
    moving: { color: 0xd09d3d, opacity: 1, roughness: 0.54, metalness: 0.1 },
    axis: 0xe47222,
    background: 0xf1f4f7,
  },
};

const MOVING_WORDS = /(?:^|[\s_\-.])(moving|movable|dynamic|actuated|driven|drawer|slider|carriage|door|lid|lever|mobile|capsule)(?:$|[\s_\-.])/i;
const FIXED_WORDS = /(?:^|[\s_\-.])(fixed|static|base|frame|housing|cabinet|context|ground|body0)(?:$|[\s_\-.])/i;

function defaultFixedDisplayMode(mode) {
  return mode === "physics"
    ? DEFAULT_PHYSICS_FIXED_DISPLAY_MODE
    : DEFAULT_FIXED_DISPLAY_MODE;
}

function fixedDisplayModeFromUrl(mode) {
  const fallback = defaultFixedDisplayMode(mode);
  try {
    const requested = new URLSearchParams(window.location.search).get("fixed");
    return FIXED_DISPLAY_MODES.has(requested) ? requested : fallback;
  } catch (_) {
    return fallback;
  }
}

function saveFixedDisplayMode(mode, viewerMode) {
  try {
    const url = new URL(window.location.href);
    if (mode === defaultFixedDisplayMode(viewerMode)) url.searchParams.delete("fixed");
    else url.searchParams.set("fixed", mode);
    history.replaceState(null, "", url);
  } catch (_) {
    // Snapshot viewers and embedded documents may not expose a mutable URL.
  }
}

function movingDisplayModeFromUrl() {
  try {
    const requested = new URLSearchParams(window.location.search).get("moving");
    return MOVING_DISPLAY_MODES.has(requested) ? requested : DEFAULT_MOVING_DISPLAY_MODE;
  } catch (_) {
    return DEFAULT_MOVING_DISPLAY_MODE;
  }
}

function saveMovingDisplayMode(mode) {
  try {
    const url = new URL(window.location.href);
    if (mode === DEFAULT_MOVING_DISPLAY_MODE) url.searchParams.delete("moving");
    else url.searchParams.set("moving", mode);
    history.replaceState(null, "", url);
  } catch (_) {
    // Snapshot viewers and embedded documents may not expose a mutable URL.
  }
}

function hiddenBodyIdsFromUrl() {
  try {
    return new Set(
      String(new URLSearchParams(window.location.search).get("hidden") || "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
  } catch (_) {
    return new Set();
  }
}

function saveHiddenBodyIds(bodyIds) {
  try {
    const url = new URL(window.location.href);
    const values = [...bodyIds].sort();
    if (values.length) url.searchParams.set("hidden", values.join(","));
    else url.searchParams.delete("hidden");
    history.replaceState(null, "", url);
  } catch (_) {
    // Snapshot viewers and embedded documents may not expose a mutable URL.
  }
}

function motionBoundsUsefulForCamera(modelBox, recordedBounds) {
  if (!recordedBounds || !Array.isArray(recordedBounds.min) || !Array.isArray(recordedBounds.max)) return false;
  if (recordedBounds.min.length < 3 || recordedBounds.max.length < 3) return false;
  const modelSize = Math.max(modelBox.getSize(new THREE.Vector3()).length(), 1e-6);
  const recordedSize = new THREE.Vector3(
    recordedBounds.max[0] - recordedBounds.min[0],
    recordedBounds.max[1] - recordedBounds.min[1],
    recordedBounds.max[2] - recordedBounds.min[2],
  ).length();
  // Ignore runaway envelopes such as a 90 m flyaway around a centimetre-scale clip.
  return recordedSize <= modelSize * 8;
}

function vectorFrom(raw, fallback = null) {
  if (!raw) return fallback;
  const clean = String(raw).trim().toLowerCase();
  const named = {
    x: [1, 0, 0], "+x": [1, 0, 0], "-x": [-1, 0, 0],
    y: [0, 1, 0], "+y": [0, 1, 0], "-y": [0, -1, 0],
    z: [0, 0, 1], "+z": [0, 0, 1], "-z": [0, 0, -1],
  };
  if (named[clean]) return new THREE.Vector3(...named[clean]);

  let value = raw;
  try {
    value = JSON.parse(raw);
  } catch (_) {
    value = String(raw).replace(/[\[\](){}]/g, "").split(/[\s,;]+/).filter(Boolean);
  }
  if (!Array.isArray(value) && value && typeof value === "object") {
    value = [value.x, value.y, value.z];
  }
  if (!Array.isArray(value) || value.length < 3) return fallback;
  const numbers = value.slice(0, 3).map(Number);
  if (!numbers.every(Number.isFinite)) return fallback;
  return new THREE.Vector3(...numbers);
}

function semanticText(object) {
  const chunks = [];
  for (let node = object; node; node = node.parent) {
    if (node.name) chunks.push(node.name);
    const data = node.userData || {};
    for (const key of ["role", "motion", "state", "semantic", "questRole", "capsuleRole", "ownership"]) {
      if (data[key] != null) chunks.push(String(data[key]));
    }
    if (data.moving === true || data.dynamic === true) chunks.push("moving");
    if (data.fixed === true || data.static === true) chunks.push("fixed");
  }
  return chunks.join(" ");
}

function isMoving(object) {
  const semantic = semanticText(object);
  if (FIXED_WORDS.test(semantic)) return false;
  return MOVING_WORDS.test(semantic);
}

function applySurfaceOpacity(material, opacity) {
  const numeric = Number(opacity);
  const resolvedOpacity = Number.isFinite(numeric)
    ? Math.max(0, Math.min(1, numeric))
    : 1;
  const outlineOnly = resolvedOpacity <= 0;
  const smoothXray = !outlineOnly && resolvedOpacity < 1;
  const shaderModeChanged = material.transparent !== smoothXray || material.alphaHash;
  material.opacity = resolvedOpacity;
  material.visible = !outlineOnly;
  material.transparent = smoothXray;
  material.alphaHash = false;
  material.depthTest = true;
  material.depthWrite = !smoothXray;
  material.userData.questTransparencyMode = outlineOnly
    ? "outline-only"
    : smoothXray
      ? "smooth-uniform-xray"
      : "opaque-depth";
  const heatState = material.userData.questHeatState;
  if (heatState) {
    heatState.xrayEnabled = smoothXray ? 1 : 0;
    if (heatState.uniforms) heatState.uniforms.questXrayEnabled.value = heatState.xrayEnabled;
  }
  if (shaderModeChanged) material.needsUpdate = true;
}

function attachLocalHeatShader(material) {
  const heatState = {
    count: 0,
    radius: 0.01,
    enabled: 0,
    xrayEnabled: 0,
    colorRamp: 0,
    points: Array.from({ length: MAX_LOCAL_HEAT_SPOTS }, () => new THREE.Vector4()),
    segmentCount: 0,
    segmentStarts: Array.from({ length: MAX_LOCAL_HEAT_SEGMENTS }, () => new THREE.Vector4()),
    segmentEnds: Array.from({ length: MAX_LOCAL_HEAT_SEGMENTS }, () => new THREE.Vector4()),
    uniforms: null,
  };
  material.userData.questHeatState = heatState;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.questHeatCount = { value: heatState.count };
    shader.uniforms.questHeatRadius = { value: heatState.radius };
    shader.uniforms.questHeatEnabled = { value: heatState.enabled };
    shader.uniforms.questXrayEnabled = { value: heatState.xrayEnabled };
    shader.uniforms.questHeatColorRamp = { value: heatState.colorRamp };
    shader.uniforms.questHeatPoints = { value: heatState.points };
    shader.uniforms.questHeatSegmentCount = { value: heatState.segmentCount };
    shader.uniforms.questHeatSegmentStarts = { value: heatState.segmentStarts };
    shader.uniforms.questHeatSegmentEnds = { value: heatState.segmentEnds };
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec3 vQuestWorldPosition;",
      )
      .replace(
        "#include <project_vertex>",
        "#include <project_vertex>\nvQuestWorldPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;",
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vQuestWorldPosition;
uniform int questHeatCount;
uniform float questHeatRadius;
uniform float questHeatEnabled;
uniform float questXrayEnabled;
uniform float questHeatColorRamp;
uniform vec4 questHeatPoints[${MAX_LOCAL_HEAT_SPOTS}];
uniform int questHeatSegmentCount;
uniform vec4 questHeatSegmentStarts[${MAX_LOCAL_HEAT_SEGMENTS}];
uniform vec4 questHeatSegmentEnds[${MAX_LOCAL_HEAT_SEGMENTS}];
float questDistanceToSegment(vec3 point, vec3 startPoint, vec3 endPoint) {
  vec3 line = endPoint - startPoint;
  float lengthSquared = dot(line, line);
  if (lengthSquared < 1e-12) return distance(point, startPoint);
  float along = clamp(dot(point - startPoint, line) / lengthSquared, 0.0, 1.0);
  return distance(point, startPoint + along * line);
}`,
      )
      .replace(
        "#include <opaque_fragment>",
        `float questHeat = 0.0;
for (int questIndex = 0; questIndex < ${MAX_LOCAL_HEAT_SPOTS}; questIndex++) {
  if (questIndex < questHeatCount) {
    vec4 questSpot = questHeatPoints[questIndex];
    float questDistance = distance(vQuestWorldPosition, questSpot.xyz);
    float questFalloff = 1.0 - smoothstep(questHeatRadius * 0.18, questHeatRadius, questDistance);
    questHeat = max(questHeat, questFalloff * questSpot.w);
  }
}
for (int questIndex = 0; questIndex < ${MAX_LOCAL_HEAT_SEGMENTS}; questIndex++) {
  if (questIndex < questHeatSegmentCount) {
    vec4 questStart = questHeatSegmentStarts[questIndex];
    vec4 questEnd = questHeatSegmentEnds[questIndex];
    float questDistance = questDistanceToSegment(
      vQuestWorldPosition,
      questStart.xyz,
      questEnd.xyz
    );
    float questFalloff = 1.0 - smoothstep(
      questHeatRadius * 0.12,
      questHeatRadius * 0.72,
      questDistance
    );
    questHeat = max(questHeat, questFalloff * min(questStart.w, questEnd.w));
  }
}
vec3 questForceHighlightColor = vec3(0.68, 0.01, 0.55);
vec3 questHeatRampColor = mix(vec3(0.08, 0.40, 0.92), vec3(1.0, 0.80, 0.08), smoothstep(0.0, 0.5, questHeat));
questHeatRampColor = mix(questHeatRampColor, vec3(0.9, 0.06, 0.025), smoothstep(0.5, 1.0, questHeat));
questForceHighlightColor = mix(questForceHighlightColor, questHeatRampColor, questHeatColorRamp);
outgoingLight = mix(outgoingLight, questForceHighlightColor, clamp(questHeat * questHeatEnabled * 0.94, 0.0, 0.94));
vec3 questUniformXrayColor = vec3(0.49, 0.56, 0.64);
outgoingLight = mix(outgoingLight, questUniformXrayColor, questXrayEnabled);
#include <opaque_fragment>`,
      );
    heatState.uniforms = shader.uniforms;
  };
  material.customProgramCacheKey = () => "quest-local-force-flow-v3";
}

function setMaterialLocalHeat(material, hotspots, segments, radius, enabled, colorRamp = false) {
  const state = material?.userData?.questHeatState;
  if (!state) return;
  const values = enabled ? hotspots.slice(0, MAX_LOCAL_HEAT_SPOTS) : [];
  const flowSegments = enabled ? segments.slice(0, MAX_LOCAL_HEAT_SEGMENTS) : [];
  state.count = values.length;
  state.segmentCount = flowSegments.length;
  state.radius = Math.max(Number(radius) || 0.01, 1e-6);
  state.enabled = enabled && (values.length || flowSegments.length) ? 1 : 0;
  state.colorRamp = colorRamp ? 1 : 0;
  for (let index = 0; index < MAX_LOCAL_HEAT_SPOTS; index += 1) {
    const spot = values[index];
    state.points[index].set(
      Number(spot?.position?.x) || 0,
      Number(spot?.position?.y) || 0,
      Number(spot?.position?.z) || 0,
      clamp01(spot?.intensity),
    );
  }
  for (let index = 0; index < MAX_LOCAL_HEAT_SEGMENTS; index += 1) {
    const segment = flowSegments[index];
    state.segmentStarts[index].set(
      Number(segment?.start?.x) || 0,
      Number(segment?.start?.y) || 0,
      Number(segment?.start?.z) || 0,
      clamp01(segment?.intensity),
    );
    state.segmentEnds[index].set(
      Number(segment?.end?.x) || 0,
      Number(segment?.end?.y) || 0,
      Number(segment?.end?.z) || 0,
      clamp01(segment?.intensity),
    );
  }
  if (state.uniforms) {
    state.uniforms.questHeatColorRamp.value = state.colorRamp;
    state.uniforms.questHeatCount.value = state.count;
    state.uniforms.questHeatSegmentCount.value = state.segmentCount;
    state.uniforms.questHeatRadius.value = state.radius;
    state.uniforms.questHeatEnabled.value = state.enabled;
  }
}

function makeMaterial(spec) {
  const material = new THREE.MeshStandardMaterial({
    color: spec.color,
    roughness: spec.roughness,
    metalness: spec.metalness,
    opacity: spec.opacity,
    transparent: false,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
  });
  attachLocalHeatShader(material);
  applySurfaceOpacity(material, spec.opacity);
  return material;
}

function makeGhostEdges(geometry, spec, background) {
  const color = new THREE.Color(spec.color).lerp(new THREE.Color(background), 0.28);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry, 28),
    new THREE.LineBasicMaterial({
      color,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      toneMapped: false,
    }),
  );
  edges.name = "quest-ghost-edges";
  edges.renderOrder = 0;
  edges.raycast = () => {};
  return edges;
}

function disposeMaterial(material, seenTextures) {
  if (!material) return;
  for (const value of Object.values(material)) {
    if (value && value.isTexture && !seenTextures.has(value)) {
      seenTextures.add(value);
      value.dispose();
    }
  }
  material.dispose?.();
}

function disposeTree(root, detachedMaterials = []) {
  const geometries = new Set();
  const materials = new Set(detachedMaterials);
  const textures = new Set();
  root?.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    const list = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of list) if (material) materials.add(material);
  });
  for (const geometry of geometries) geometry.dispose?.();
  for (const material of materials) disposeMaterial(material, textures);
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function heatColor(value) {
  const t = clamp01(value);
  const low = new THREE.Color(0x9b7cff);
  const middle = new THREE.Color(0xe23fe0);
  const high = new THREE.Color(0xff165d);
  return t < 0.5 ? low.lerp(middle, t * 2) : middle.lerp(high, (t - 0.5) * 2);
}

function vectorMagnitude(values) {
  if (!Array.isArray(values) || values.length < 3) return 0;
  return Math.hypot(Number(values[0]) || 0, Number(values[1]) || 0, Number(values[2]) || 0);
}

function formatEvidenceNumber(value, unit, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "未记录";
  return `${number.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function physicsBodyLabel(value) {
  const labels = {
    ground: "固定机架",
    base: "底座",
    frame: "机架",
    housing: "外壳",
    handle: "手柄",
    blade: "雨刷片",
    linkage: "连杆组",
    bin: "桶身",
    tray: "托盘",
    pinion: "齿轮与手柄",
    rack: "齿条与闸门",
    pedal: "脚踏板",
    rod: "传动连杆",
    lid: "桶盖",
    crank: "驱动曲柄",
    coupler: "中间连杆",
    rocker: "雨刷摆臂",
  };
  return labels[value] || value || "未知刚体";
}

function physicsSolidLabel(value) {
  const match = String(value || "").match(/^solid_(\d+)$/);
  return match ? `STEP 实体 ${match[1]}` : value || "未选择";
}

function collisionRepresentationLabel(value) {
  const labels = {
    ChCollisionShapeBox: "准确长方体（来自 STEP）",
    ChCollisionShapeCylinder: "准确圆柱（来自 STEP）",
    ChCollisionShapeSphere: "准确球体（来自 STEP）",
    ChCollisionShapeTriangleMeshDynamicConvex: "原形 STEP 凸实体（不是自动凸包）",
    ChCollisionShapeTriangleMeshStaticNonconvex: "原形 STEP 非凸网格（固定件）",
  };
  return labels[value] || value || "未选择";
}

function physicsLandmarkLabel(value) {
  const labels = {
    pinion_bearing: "齿轮轴承",
    rack_guide: "齿条导向",
  };
  return labels[value] || String(value || "关节").replaceAll("_", " ");
}

function buildViewport(container, fixedDisplayMode = null, movingDisplayMode = null) {
  const viewport = document.createElement("div");
  viewport.className = "quest-3d-viewport";
  Object.assign(viewport.style, {
    position: "absolute",
    inset: "0",
    overflow: "hidden",
    borderRadius: "inherit",
    opacity: "0",
    pointerEvents: "none",
    transition: "opacity 160ms ease",
  });

  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "quest-3d-reset";
  reset.textContent = "重置视角";
  reset.setAttribute("aria-label", "重置三维视角");
  Object.assign(reset.style, {
    position: "absolute",
    zIndex: "2",
    top: "10px",
    right: "10px",
    border: "1px solid rgba(77, 98, 120, .28)",
    borderRadius: "7px",
    padding: "6px 9px",
    color: "#26384a",
    background: "rgba(255, 255, 255, .9)",
    boxShadow: "0 2px 8px rgba(28, 45, 62, .12)",
    font: "600 12px/1.2 system-ui, sans-serif",
    cursor: "pointer",
  });

  const fixedDisplay = fixedDisplayMode ? document.createElement("div") : null;
  const fixedButtons = [];
  if (fixedDisplay) {
    fixedDisplay.className = "quest-fixed-display";
    fixedDisplay.setAttribute("role", "group");
    fixedDisplay.setAttribute("aria-label", "\u56fa\u5b9a\u7ed3\u6784\u663e\u793a\u65b9\u5f0f");
    Object.assign(fixedDisplay.style, {
      position: "absolute",
      zIndex: "2",
      top: "48px",
      left: "10px",
      display: "flex",
      gap: "2px",
      padding: "3px",
      border: "1px solid rgba(77, 98, 120, .24)",
      borderRadius: "8px",
      background: "rgba(255, 255, 255, .9)",
      boxShadow: "0 2px 8px rgba(28, 45, 62, .1)",
    });
    for (const [mode, label] of [["outline", "\u7ebf\u6846"], ["xray", "\u900f\u89c6"], ["solid", "\u5b9e\u4f53"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.fixedDisplay = mode;
      button.textContent = label;
      button.setAttribute("aria-pressed", String(mode === fixedDisplayMode));
      Object.assign(button.style, {
        border: "0",
        borderRadius: "5px",
        padding: "5px 8px",
        color: "#344a5f",
        background: mode === fixedDisplayMode ? "#e2edf7" : "transparent",
        font: "600 11px/1.2 system-ui, sans-serif",
        cursor: "pointer",
      });
      fixedDisplay.append(button);
      fixedButtons.push(button);
    }
  }

  const movingDisplay = movingDisplayMode ? document.createElement("div") : null;
  const movingButtons = [];
  if (movingDisplay) {
    movingDisplay.className = "quest-moving-display";
    movingDisplay.setAttribute("role", "group");
    movingDisplay.setAttribute("aria-label", "运动零件显示方式");
    Object.assign(movingDisplay.style, {
      position: "absolute",
      zIndex: "2",
      top: "84px",
      left: "10px",
      display: "flex",
      gap: "2px",
      padding: "3px",
      border: "1px solid rgba(77, 98, 120, .24)",
      borderRadius: "8px",
      background: "rgba(255, 255, 255, .9)",
      boxShadow: "0 2px 8px rgba(28, 45, 62, .1)",
    });
    for (const [mode, label] of [["solid", "实体"], ["edges", "实体+线框"], ["outline", "线框"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.movingDisplay = mode;
      button.textContent = label;
      button.setAttribute("aria-pressed", String(mode === movingDisplayMode));
      Object.assign(button.style, {
        border: "0",
        borderRadius: "5px",
        padding: "5px 8px",
        color: "#5b3b12",
        background: mode === movingDisplayMode ? "#fff0cd" : "transparent",
        font: "600 11px/1.2 system-ui, sans-serif",
        cursor: "pointer",
      });
      movingDisplay.append(button);
      movingButtons.push(button);
    }
  }

  const hint = document.createElement("div");
  hint.className = "quest-3d-hint";
  hint.textContent = "\u753b\u5e03\u62d6\u52a8\u65cb\u8f6c \u00b7 \u6eda\u8f6e\u7f29\u653e \u00b7 \u53f3\u952e\u5e73\u79fb";
  Object.assign(hint.style, {
    position: "absolute",
    zIndex: "2",
    left: "10px",
    bottom: "9px",
    borderRadius: "6px",
    padding: "5px 7px",
    color: "#41566c",
    background: "rgba(255, 255, 255, .84)",
    font: "500 11px/1.2 system-ui, sans-serif",
    pointerEvents: "none",
  });

  const orientationPlate = document.createElement("div");
  orientationPlate.className = "quest-orientation-plate";
  orientationPlate.textContent = "\u62d6\u52a8\u65b9\u5411\u8f74 \u00b7 \u70b9\u51fb X/Y/Z \u5bf9\u9f50";
  Object.assign(orientationPlate.style, {
    position: "absolute",
    zIndex: "1",
    right: "5px",
    bottom: "5px",
    width: "118px",
    height: "118px",
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "center",
    paddingTop: "5px",
    boxSizing: "border-box",
    border: "1px solid rgba(77, 98, 120, .18)",
    borderRadius: "50%",
    color: "#51687d",
    background: "rgba(255, 255, 255, .64)",
    font: "600 9px/1.1 system-ui, sans-serif",
    pointerEvents: "none",
  });

  viewport.append(reset);
  if (fixedDisplay) viewport.append(fixedDisplay);
  if (movingDisplay) viewport.append(movingDisplay);
  viewport.append(orientationPlate);
  viewport.append(hint);
  container.append(viewport);
  return { viewport, reset, fixedButtons, movingButtons };
}

class QuestViewer {
  constructor(container) {
    this.container = container;
    this.modelSrc = container.dataset.modelSrc || "";
    const requestedMode = container.dataset.mode || "design";
    this.mode = ["mechanism", "physics"].includes(requestedMode) ? requestedMode : "design";
    this.palette = PALETTES[this.mode];
    this.supportsFixedDisplay = this.mode !== "design";
    this.supportsMovingDisplay = this.mode !== "design";
    this.fixedDisplayMode = this.supportsFixedDisplay
      ? fixedDisplayModeFromUrl(this.mode)
      : "solid";
    this.movingDisplayMode = this.supportsMovingDisplay ? movingDisplayModeFromUrl() : "solid";
    this.fixedDisplayListeners = [];
    this.movingDisplayListeners = [];
    this.hiddenBodyIds = hiddenBodyIdsFromUrl();
    this.gizmoListeners = [];
    this.replay = container.questReplay || null;
    this.sceneManifest = container.questSceneManifest || null;
    this.initialPhysicsState = container.questInitialState || {};
    this.axis = vectorFrom(container.dataset.axis);
    this.origin = vectorFrom(container.dataset.origin);
    this.disposed = false;
    this.detachedMaterials = [];
    this.historyModels = new Map();
    this.modelLoads = new Map();
    this.modelSelection = 0;
    this.restore = {
      position: container.style.position,
      minHeight: container.style.minHeight,
      fallback: [],
    };
    this.container.dataset.viewerStatus = "loading";
  }

  async init() {
    if (!this.modelSrc) throw new Error("missing model source");
    if (getComputedStyle(this.container).position === "static") {
      this.container.style.position = "relative";
    }
    if (this.container.getBoundingClientRect().height < 80) {
      this.container.style.minHeight = "280px";
    }
    const { viewport, reset, fixedButtons, movingButtons } = buildViewport(
      this.container,
      this.supportsFixedDisplay ? this.fixedDisplayMode : null,
      this.supportsMovingDisplay ? this.movingDisplayMode : null,
    );
    this.viewport = viewport;
    this.resetButton = reset;
    this.fixedDisplayButtons = fixedButtons;
    this.movingDisplayButtons = movingButtons;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.palette.background);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 10000);
    const up = vectorFrom(this.container.dataset.upAxis);
    if (up && up.lengthSq() > 0.5) this.camera.up.copy(up.normalize());
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    const minimumDpr = matchMedia("(max-width: 640px)").matches ? 1.5 : 2;
    const displayDpr = window.devicePixelRatio || 1;
    const stableTransparencyDpr = Math.min(Math.max(displayDpr, minimumDpr), 2);
    this.renderer.setPixelRatio(stableTransparencyDpr);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.renderer.domElement.style.display = "block";
    this.renderer.domElement.style.touchAction = "none";
    this.renderer.domElement.setAttribute("aria-label", "\u53ef\u62d6\u52a8\u65cb\u8f6c\u548c\u7f29\u653e\u7684\u4e09\u7ef4\u6a21\u578b\uff1b\u53f3\u4e0b\u89d2\u65b9\u5411\u8f74\u53ef\u62d6\u52a8\uff0c\u70b9\u51fb X/Y/Z \u53ef\u5bf9\u9f50\u89c6\u89d2");
    this.renderer.domElement.tabIndex = 0;
    this.viewport.prepend(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.001;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    this.controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    this.controls.touches.ONE = THREE.TOUCH.ROTATE;
    this.controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
    this.onControlsChange = () => this.renderOnce();
    this.controls.addEventListener("change", this.onControlsChange);
    this.setupOrientationGizmo();

    this.addLights();
    this.onPageVisibility=()=>{if(!document.hidden)this.renderOnce();};
    document.addEventListener("visibilitychange",this.onPageVisibility);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.onReset = (event) => {
      event.stopPropagation();
      this.resetView();
    };
    this.resetButton.addEventListener("click", this.onReset);
    for (const button of this.fixedDisplayButtons) {
      const handler = (event) => {
        event.stopPropagation();
        this.setFixedDisplayMode(button.dataset.fixedDisplay);
      };
      button.addEventListener("click", handler);
      this.fixedDisplayListeners.push(() => button.removeEventListener("click", handler));
    }

    for (const button of this.movingDisplayButtons) {
      const handler = (event) => {
        event.stopPropagation();
        this.setMovingDisplayMode(button.dataset.movingDisplay);
      };
      button.addEventListener("click", handler);
      this.movingDisplayListeners.push(() => button.removeEventListener("click", handler));
    }

    const gltf = await new GLTFLoader().loadAsync(this.modelSrc);
    if (this.disposed) {
      disposeTree(gltf.scene);
      return false;
    }
    this.model = gltf.scene;
    this.container.dataset.renderedModelSrc = this.modelSrc;
    this.styleModel();
    this.historyModels.set(this.modelSrc, {root:this.model, materials:this.detachedMaterials, styled:true});
    this.scene.add(this.model);
    this.fitView();
    this.addAxis();
    if (this.mode === "physics") this.setupPhysicsEvidence();
    this.parts = new PartsController(this);
    this.parts.attach(this.model);
    this.updateFixedDisplayUi();
    this.updateMovingDisplayUi();
    this.resize();
    this.showViewer();
    return true;
  }

  async historyModel(url) {
    if (this.historyModels.has(url)) return this.historyModels.get(url);
    if (this.modelLoads.has(url)) return this.modelLoads.get(url);
    const loading = new GLTFLoader().loadAsync(url).then(gltf => {
      if (this.disposed) { disposeTree(gltf.scene); return null; }
      const entry = {root:gltf.scene, materials:[], styled:false};
      this.historyModels.set(url, entry);
      return entry;
    }).finally(() => this.modelLoads.delete(url));
    this.modelLoads.set(url, loading);
    return loading;
  }

  trimHistory(keep = [], limit = 3) {
    const protectedUrls = new Set([this.modelSrc, ...keep]);
    const cap = Math.max(limit, protectedUrls.size);
    for (const [url, entry] of this.historyModels) {
      if (this.historyModels.size <= cap) break;
      if (protectedUrls.has(url)) continue;
      disposeTree(entry.root, entry.materials); this.historyModels.delete(url);
    }
    this.container.dataset.cachedModels = String(this.historyModels.size);
  }

  async replaceModel(url) {
    if (this.mode !== "design" || !url) return;
    const selection = ++this.modelSelection;
    if (url === this.modelSrc) {
      this.container.dataset.viewerStatus = "ready";
      this.container.dispatchEvent?.(new CustomEvent("quest-model-state", {bubbles:true,detail:{message:""}}));
      return;
    }
    const ready = this.historyModels?.get(url);
    if (!ready?.root) {
      this.container.dataset.viewerStatus = "loading";
      this.container.dispatchEvent?.(new CustomEvent("quest-model-state", {bubbles:true,detail:{message:"正在加载所选版本，暂时保留之前的模型"}}));
    }
    try {
      const entry = ready || await this.historyModel(url);
      if (!entry || this.disposed || selection !== this.modelSelection) { this.trimHistory(); return; }
      if (new THREE.Box3().setFromObject(entry.root).isEmpty()) throw new Error("empty model");
      this.parts?.detach();
      this.scene.remove(this.model);
      this.model = entry.root; this.modelSrc = url; this.detachedMaterials = entry.materials;
      this.container.dataset.renderedModelSrc = url;
      if (!entry.styled) { this.styleModel(); entry.materials = this.detachedMaterials; entry.styled = true; }
      this.scene.add(this.model);
      this.parts?.attach(this.model);
      this.fitView(false);
      this.historyModels.delete(url); this.historyModels.set(url, entry);
      this.trimHistory(); this.resize(); this.renderOnce();
      this.container.dataset.viewerStatus = "ready";
      this.container.dispatchEvent?.(new CustomEvent("quest-model-state", {bubbles:true,detail:{message:""}}));
    } catch {
      if (selection === this.modelSelection) {
        this.container.dataset.viewerStatus = "ready";
        this.container.dispatchEvent?.(new CustomEvent("quest-model-state", {bubbles:true,detail:{message:"所选版本加载失败；当前画面仍是之前保存的模型"}}));
      }
    }
  }

  async prefetchHistory(urls) {
    if (this.mode !== "design" || this.disposed) return;
    const chosen = urls.filter(url => url && url !== this.modelSrc).slice(0, 8);
    await Promise.allSettled(chosen.map(url => this.historyModel(url)));
    this.trimHistory(chosen, Math.max(3, chosen.length + 1));
  }

  setupOrientationGizmo() {
    this.viewHelper = new ViewHelper(this.camera, this.renderer.domElement);
    this.viewHelper.center.copy(this.controls.target);
    const canvas = this.renderer.domElement;
    const dimension = 128;
    let drag = null;
    const inGizmo = (event) => {
      const bounds = canvas.getBoundingClientRect();
      return event.clientX >= bounds.right - dimension
        && event.clientY >= bounds.bottom - dimension;
    };
    const listen = (type, handler, options = true) => {
      canvas.addEventListener(type, handler, options);
      this.gizmoListeners.push(() => canvas.removeEventListener(type, handler, options));
    };
    const stop = (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
    };
    const pointerDown = (event) => {
      if (event.button !== 0 || !inGizmo(event)) return;
      const offset = this.camera.position.clone().sub(this.controls.target);
      drag = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        spherical: new THREE.Spherical().setFromVector3(offset),
      };
      this.controls.enabled = false;
      try { canvas.setPointerCapture?.(event.pointerId); } catch (_) { /* Synthetic and legacy pointers may not be capturable. */ }
      stop(event);
    };
    const pointerMove = (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (Math.hypot(dx, dy) >= 4) drag.moved = true;
      const spherical = drag.spherical.clone();
      spherical.theta -= dx * 0.012;
      spherical.phi -= dy * 0.012;
      spherical.makeSafe();
      this.camera.position.copy(this.controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
      this.camera.lookAt(this.controls.target);
      this.viewHelper.center.copy(this.controls.target);
      this.controls.update();
      stop(event);
    };
    const finish = (event) => {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const click = !drag.moved;
      try { canvas.releasePointerCapture?.(event.pointerId); } catch (_) { /* Pointer may already be released. */ }
      drag = null;
      this.controls.enabled = true;
      this.viewHelper.center.copy(this.controls.target);
      if (click && this.viewHelper.handleClick(event)) this.animateOrientationGizmo();
      else this.renderOnce();
      stop(event);
    };
    listen("pointerdown", pointerDown, true);
    listen("pointermove", pointerMove, true);
    listen("pointerup", finish, true);
    listen("pointercancel", finish, true);
    const keydown = (event) => {
      const axis = String(event.key || "").toLowerCase();
      if (!["x", "y", "z"].includes(axis) || event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      const direction = {
        x: new THREE.Vector3(event.shiftKey ? -1 : 1, 0, 0),
        y: new THREE.Vector3(0, event.shiftKey ? -1 : 1, 0),
        z: new THREE.Vector3(0, 0, event.shiftKey ? -1 : 1),
      }[axis];
      const distance = this.camera.position.distanceTo(this.controls.target);
      this.camera.position.copy(this.controls.target).addScaledVector(direction, distance);
      this.camera.lookAt(this.controls.target);
      this.controls.update();
    };
    listen("keydown", keydown, false);
  }

  animateOrientationGizmo() {
    if (this.gizmoAnimationHandle) cancelAnimationFrame(this.gizmoAnimationHandle);
    let lastTime = performance.now();
    const tick = (time) => {
      const delta = Math.min(0.05, Math.max(0, (time - lastTime) / 1000));
      lastTime = time;
      this.viewHelper.update(delta);
      this.controls.update();
      this.renderOnce();
      if (this.viewHelper.animating && !this.disposed) this.gizmoAnimationHandle = requestAnimationFrame(tick);
      else this.gizmoAnimationHandle = 0;
    };
    this.gizmoAnimationHandle = requestAnimationFrame(tick);
  }

  addLights() {
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8190a0, 2.15));
    const key = new THREE.DirectionalLight(0xffffff, 2.8);
    key.position.set(4, 7, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xbdd7f0, 1.25);
    fill.position.set(-5, 2, -4);
    this.scene.add(fill);
  }

  styleModel() {
    const originalSurface = Boolean(this.container.closest(".studio-main"))
      && (this.mode !== "physics" || this.sceneManifest?.source_kind === "original_geometry");
    const meshes = [];
    this.model.traverse((object) => { if (object.isMesh) meshes.push(object); });
    this.physicsMeshes = [];
    for (const object of meshes) {
      object.userData.questPartSource = true;
      const sources = Array.isArray(object.material) ? object.material : [object.material];
      for (const source of sources) if (source) this.detachedMaterials.push(source);
      const component = this.physicsComponentForNode(object.name);
      const fixed = this.mode === "physics" ? Boolean(component?.fixed) : !isMoving(object);
      const spec = fixed ? this.palette.fixed : this.palette.moving;
      const materialSpec = {
        ...spec,
        opacity: fixed && this.supportsFixedDisplay
          ? this.fixedSurfaceOpacity()
          : !fixed && this.supportsMovingDisplay
            ? this.movingSurfaceOpacity()
            : spec.opacity,
      };
      if (originalSurface) {
        const sourceGeometry = object.geometry;
        object.geometry = sourceCreasedNormals(sourceGeometry);
        sourceGeometry.dispose();
      }
      const replacement = sources.map((source) => {
        const material = makeMaterial(originalSurface ? { ...materialSpec, color: source?.color || 0xffffff } : materialSpec);
        if (originalSurface) {
          material.vertexColors = Boolean(object.geometry.getAttribute("color"));
          material.map = source?.map || null;
          material.roughness = 0.68;
          material.metalness = 0.12;
        }
        return material;
      });
      object.material = Array.isArray(object.material) ? replacement : replacement[0];
      object.userData.questFixed = fixed;
      object.castShadow = false;
      object.receiveShadow = false;
      object.renderOrder = 0;
      if ((fixed && this.supportsFixedDisplay) || (!fixed && this.supportsMovingDisplay)) {
        const ghostEdges = makeGhostEdges(object.geometry, spec, this.palette.background);
        ghostEdges.visible = fixed
          ? this.fixedDisplayMode === "outline"
          : this.movingDisplayMode !== "solid";
        object.add(ghostEdges);
        object.userData.ghostEdges = ghostEdges;
      }
      if (this.mode === "physics") {
        object.userData.physicsComponent = component || this.physicsComponentFromName(object.name);
        object.userData.physicsBaseSpec = originalSurface ? { ...spec, color: replacement[0].color.getHex() } : spec;
        const overlay = new THREE.Mesh(
          object.geometry,
          new THREE.MeshBasicMaterial({
            color: 0x19344c,
            wireframe: true,
            transparent: true,
            opacity: 0.58,
            depthTest: true,
            depthWrite: false,
          }),
        );
        overlay.name = `${object.name}__collision_overlay`;
        overlay.visible = false;
        overlay.renderOrder = 5;
        overlay.raycast = () => {};
        object.add(overlay);
        object.userData.collisionOverlay = overlay;
        this.physicsMeshes.push(object);
      }
    }
  }

  fixedSurfaceOpacity() {
    if (this.fixedDisplayMode === "solid") return 1;
    if (this.fixedDisplayMode === "xray") return FIXED_XRAY_OPACITY;
    return 0;
  }

  movingSurfaceOpacity() {
    return this.movingDisplayMode === "outline" ? 0 : 1;
  }

  updateFixedDisplayUi() {
    for (const button of this.fixedDisplayButtons || []) {
      const active = button.dataset.fixedDisplay === this.fixedDisplayMode;
      button.setAttribute("aria-pressed", String(active));
      button.style.background = active ? "#e2edf7" : "transparent";
      button.style.color = active ? "#173f68" : "#344a5f";
    }
    if (this.container) this.container.dataset.fixedDisplay = this.fixedDisplayMode;
  }

  updateMovingDisplayUi() {
    for (const button of this.movingDisplayButtons || []) {
      const active = button.dataset.movingDisplay === this.movingDisplayMode;
      button.setAttribute("aria-pressed", String(active));
      button.style.background = active ? "#fff0cd" : "transparent";
      button.style.color = active ? "#6a3d00" : "#5b3b12";
    }
    if (this.container) this.container.dataset.movingDisplay = this.movingDisplayMode;
  }

  setFixedDisplayMode(mode) {
    if (!this.supportsFixedDisplay || !FIXED_DISPLAY_MODES.has(mode)) return;
    this.fixedDisplayMode = mode;
    saveFixedDisplayMode(mode, this.mode);
    this.updateFixedDisplayUi();
    if (this.mode === "physics" && this.physicsState) {
      this.applyPhysicsFrame();
      return;
    }
    this.applyFixedDisplay();
  }

  setMovingDisplayMode(mode) {
    if (!this.supportsMovingDisplay || !MOVING_DISPLAY_MODES.has(mode)) return;
    this.movingDisplayMode = mode;
    saveMovingDisplayMode(mode);
    this.updateMovingDisplayUi();
    if (this.mode === "physics" && this.physicsState) {
      this.applyPhysicsFrame();
      return;
    }
    this.applyMovingDisplay();
  }

  applyFixedDisplay() {
    if (!this.model) return;
    const opacity = this.fixedSurfaceOpacity();
    this.model.traverse((object) => {
      if (!object.isMesh || !object.userData.questFixed) return;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) applySurfaceOpacity(material, opacity);
      if (object.userData.ghostEdges) {
        object.userData.ghostEdges.visible = this.fixedDisplayMode === "outline";
      }
    });
    this.renderOnce();
  }

  applyMovingDisplay() {
    if (!this.model) return;
    const opacity = this.movingSurfaceOpacity();
    this.model.traverse((object) => {
      if (!object.isMesh || object.userData.questFixed) return;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) applySurfaceOpacity(material, opacity);
      if (object.userData.ghostEdges) {
        object.userData.ghostEdges.visible = this.movingDisplayMode !== "solid";
      }
    });
    this.renderOnce();
  }

  ensurePhysicsComponentIndex() {
    if (this.physicsComponentIndex) return this.physicsComponentIndex;
    this.physicsComponentIndex = buildPhysicsComponentIndex(
      this.sceneManifest?.bodies || [],
    );
    return this.physicsComponentIndex;
  }

  physicsComponentFromName(name) {
    const parts = String(name || "").split("__");
    if (parts.length < 3 || parts[0] !== "physics") return null;
    if (isGeneratedSplitComponentNodeName(name)) return null;
    return { node: name, bodyId: parts[1], solid_id: parts.slice(2).join("__"), fixed: false, representation: "未记录" };
  }

  physicsComponentForNode(name) {
    const index = this.ensurePhysicsComponentIndex();
    return indexedPhysicsComponent(index, name)
      || this.physicsComponentFromName(name);
  }

  physicsComponentSolidRef(component) {
    return componentSolidRef(component);
  }

  physicsFocusRole(mesh, focus = this.collisionFocus) {
    const component = mesh.userData.physicsComponent || this.physicsComponentForNode(mesh.name);
    return focusRoleForComponent({ meshName: mesh.name, component, focus });
  }

  physicsFocusNode(focus, role) {
    return this.physicsMeshes.find((mesh) => this.physicsFocusRole(mesh, focus) === role)?.name || "";
  }

  physicsBranchData() {
    return this.replay?.branches?.[this.physicsState?.branch || "physical"] || null;
  }

  listenPhysics(target, type, handler, options) {
    if (!target) return;
    target.addEventListener(type, handler, options);
    this.physicsListeners.push(() => target.removeEventListener(type, handler, options));
  }

  setupBodyVisibilityControls() {
    const bodies = this.sceneManifest?.bodies || [];
    if (!bodies.length) return;
    const available = new Set(bodies.map((body) => String(body.body_id)));
    this.hiddenBodyIds = new Set([...this.hiddenBodyIds].filter((bodyId) => available.has(bodyId)));
    const panel = document.createElement("details");
    panel.className = "quest-body-visibility";
    Object.assign(panel.style, {
      position: "absolute",
      zIndex: "4",
      top: "126px",
      left: "10px",
      width: "min(220px, calc(100% - 150px))",
      border: "1px solid rgba(77, 98, 120, .24)",
      borderRadius: "9px",
      color: "#26384a",
      background: "rgba(255, 255, 255, .94)",
      boxShadow: "0 3px 12px rgba(28, 45, 62, .12)",
      font: "600 11px/1.25 system-ui, sans-serif",
    });
    const summary = document.createElement("summary");
    summary.style.cursor = "pointer";
    summary.style.padding = "7px 9px";
    summary.style.listStyle = "none";
    panel.append(summary);
    const list = document.createElement("div");
    Object.assign(list.style, {
      display: "grid",
      gap: "5px",
      padding: "0 7px 7px",
      borderTop: "1px solid rgba(77, 98, 120, .13)",
    });
    this.bodyVisibilityButtons = [];
    for (const body of bodies) {
      const bodyId = String(body.body_id);
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.bodyId = bodyId;
      button.dataset.bodyFixed = String(Boolean(body.fixed));
      button.style.border = "1px solid rgba(77, 98, 120, .18)";
      button.style.borderRadius = "7px";
      button.style.padding = "6px 8px";
      button.style.textAlign = "left";
      button.style.cursor = "pointer";
      button.style.font = "600 11px/1.2 system-ui, sans-serif";
      this.listenPhysics(button, "click", (event) => {
        event.stopPropagation();
        this.toggleBodyVisibility(bodyId);
      });
      list.append(button);
      this.bodyVisibilityButtons.push(button);
    }
    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "\u5168\u90e8\u663e\u793a";
    Object.assign(reset.style, {
      border: "1px solid rgba(39, 126, 84, .28)",
      borderRadius: "7px",
      padding: "6px 8px",
      color: "#17623f",
      background: "#eef8f2",
      cursor: "pointer",
      font: "700 11px/1.2 system-ui, sans-serif",
    });
    this.listenPhysics(reset, "click", (event) => {
      event.stopPropagation();
      this.hiddenBodyIds.clear();
      this.parts?.hidden.clear();
      saveHiddenBodyIds(this.hiddenBodyIds);
      this.applyPhysicsFrame();
      this.parts?.publish();
    });
    list.append(reset);
    panel.append(list);
    this.viewport.append(panel);
    this.bodyVisibilityPanel = panel;
    this.bodyVisibilitySummary = summary;
    this.updateBodyVisibilityUi();
  }

  bodyIsVisible(bodyId) {
    const key=String(bodyId || "");
    if(this.hiddenBodyIds.has(key))return false;
    const entries=this.parts?.entries.filter(entry=>entry.body===key) || [];
    return !entries.length || entries.some(entry=>!this.parts.hidden.has(entry.id));
  }

  toggleBodyVisibility(bodyId) {
    const key = String(bodyId || "");
    if (!key) return;
    if (!this.bodyIsVisible(key)) {
      this.hiddenBodyIds.delete(key);
      for(const entry of this.parts?.entries || [])if(entry.body===key)this.parts.hidden.delete(entry.id);
    } else this.hiddenBodyIds.add(key);
    if (this.physicsState?.part) {
      const selected = this.physicsMeshes.find((mesh) => mesh.name === this.physicsState.part);
      if (selected?.userData?.physicsComponent?.bodyId === key && this.hiddenBodyIds.has(key)) {
        this.physicsState.part = "";
      }
    }
    saveHiddenBodyIds(this.hiddenBodyIds);
    this.applyPhysicsFrame();
    this.parts?.publish();
    this.emitPhysicsState();
  }

  updateBodyVisibilityUi() {
    const bodies = this.sceneManifest?.bodies || [];
    const visibleBodyCount = bodies.filter((body) => this.bodyIsVisible(body.body_id)).length;
    const componentCount = bodies.reduce(
      (count, body) => count + (body.components || []).length,
      0,
    );
    const visibleComponentCount = this.parts?.entries.length ? this.parts.entries.filter(entry=>entry.mesh.visible).length : bodies.reduce(
      (count, body) => count + (
        this.bodyIsVisible(body.body_id) ? (body.components || []).length : 0
      ),
      0,
    );
    if (this.bodyVisibilitySummary) {
      this.bodyVisibilitySummary.textContent = `\u521a\u4f53\u663e\u793a \u00b7 ${visibleBodyCount}/${bodies.length} \u00b7 \u7ec4\u4ef6 ${visibleComponentCount}/${componentCount}`;
    }
    for (const button of this.bodyVisibilityButtons || []) {
      const visible = this.bodyIsVisible(button.dataset.bodyId);
      button.setAttribute("aria-pressed", String(visible));
      button.textContent = `${visible ? "\u25cf" : "\u25cb"} ${physicsBodyLabel(button.dataset.bodyId)} \u00b7 ${button.dataset.bodyFixed === "true" ? "\u56fa\u5b9a" : "\u8fd0\u52a8"}`;
      button.style.color = visible ? "#253b50" : "#8795a3";
      button.style.background = visible ? "#f7fafc" : "#eef1f4";
      button.style.textDecoration = visible ? "none" : "line-through";
    }
  }

  setupPhysicsEvidence() {
    if (!this.replay?.branches?.physical || !this.sceneManifest?.bodies) {
      throw new Error("physics replay or scene manifest missing");
    }
    this.physicsRoot = this.container.closest(".physics-evidence-workspace") || this.container;
    this.physicsListeners = [];
    const requestedBranch = this.initialPhysicsState.branch;
    const branch = this.replay.branches[requestedBranch] ? requestedBranch : "physical";
    this.collisionFocus = this.initialPhysicsState.collisionFocus || null;
    this.physicsState = {
      branch: this.collisionFocus ? "physical" : branch,
      frame: Math.max(0, Number(this.initialPhysicsState.frame) || 0),
      layer: ["selected", "all", "off"].includes(this.initialPhysicsState.layer) ? this.initialPhysicsState.layer : "selected",
      part: String(this.initialPhysicsState.part || ""),
      heat: true,
      playing: false,
    };
    if (this.collisionFocus) {
      const sourceIndex = Number(this.collisionFocus.source_sample_index ?? this.collisionFocus.frame);
      const frames = this.replay.branches.physical?.frames || [];
      const replayIndex = frames.findIndex((frame) => Number(frame.source_sample_index) === sourceIndex);
      if (replayIndex >= 0) this.physicsState.frame = replayIndex;
      this.physicsState.part = String(
        this.physicsFocusNode(this.collisionFocus, "primary")
        || this.collisionFocus.primary_node
        || this.physicsState.part
        || "",
      );
    }
    if (!this.physicsMeshes.some((mesh) => mesh.name === this.physicsState.part)) this.physicsState.part = "";
    this.physicsUi = {
      frame: this.physicsRoot.querySelector("[data-physics-frame]"),
      frameLabel: this.physicsRoot.querySelector("[data-physics-frame-label]"),
      play: this.physicsRoot.querySelector("[data-physics-play]"),
      prev: this.physicsRoot.querySelector("[data-physics-prev]"),
      next: this.physicsRoot.querySelector("[data-physics-next]"),
      branchButtons: [...this.physicsRoot.querySelectorAll("[data-physics-branch]")],
      collision: this.physicsRoot.querySelector("[data-physics-collision]"),
      heat: this.physicsRoot.querySelector("[data-physics-heat]"),
      time: this.physicsRoot.querySelector("[data-physics-time]"),
      phase: this.physicsRoot.querySelector("[data-physics-phase]"),
      part: this.physicsRoot.querySelector("[data-physics-part]"),
      body: this.physicsRoot.querySelector("[data-physics-body]"),
      representation: this.physicsRoot.querySelector("[data-physics-representation]"),
      external: this.physicsRoot.querySelector("[data-physics-external]"),
      reaction: this.physicsRoot.querySelector("[data-physics-reaction]"),
      drift: this.physicsRoot.querySelector("[data-physics-drift]"),
      contact: this.physicsRoot.querySelector("[data-physics-contact]"),
      reactionBar: this.physicsRoot.querySelector("[data-physics-reaction-bar]"),
      availability: this.physicsRoot.querySelector("[data-physics-availability]"),
    };
    this.setupReactionLandmarks();
    this.setupCollisionFocus();
    this.setupBodyVisibilityControls();
    setupMechanics(this);
    this.setupMovingChainPathEvidence();

    this.listenPhysics(this.physicsUi.prev, "click", () => this.setPhysicsFrame(this.physicsState.frame - 1));
    this.listenPhysics(this.physicsUi.next, "click", () => this.setPhysicsFrame(this.physicsState.frame + 1));
    this.listenPhysics(this.physicsUi.play, "click", () => this.togglePhysicsPlayback());
    this.listenPhysics(this.physicsUi.frame, "input", (event) => this.setPhysicsFrame(Number(event.target.value)));
    for (const button of this.physicsUi.branchButtons) {
      this.listenPhysics(button, "click", () => this.setPhysicsBranch(button.dataset.physicsBranch));
    }
    this.listenPhysics(this.physicsUi.collision, "click", () => {
      const order = ["selected", "all", "off"];
      this.physicsState.layer = order[(order.indexOf(this.physicsState.layer) + 1) % order.length];
      this.applyPhysicsFrame();
      this.emitPhysicsState();
    });
    this.listenPhysics(this.physicsUi.heat, "click", () => {
      this.physicsState.heat = !this.physicsState.heat;
      this.applyPhysicsFrame();
    });

    this.listenPhysics(this.renderer.domElement, "keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        this.setPhysicsFrame(this.physicsState.frame + (event.key === "ArrowLeft" ? -1 : 1));
      } else if (event.key === " ") {
        event.preventDefault();
        this.togglePhysicsPlayback();
      } else if (event.key.toLowerCase() === "r") {
        this.resetView();
      } else if (event.key.toLowerCase() === "c") {
        this.physicsUi.collision?.click();
      }
    });
    this.onVisibilityChange = () => { if (document.hidden) this.pausePhysicsPlayback(); };
    this.listenPhysics(document, "visibilitychange", this.onVisibilityChange);
    this.intersectionObserver = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) this.pausePhysicsPlayback();
    }, { threshold: 0.05 });
    this.intersectionObserver.observe(this.container);

    const contextLost = (event) => {
      event.preventDefault();
      this.pausePhysicsPlayback();
      this.fail();
    };
    this.listenPhysics(this.renderer.domElement, "webglcontextlost", contextLost, false);
    this.applyPhysicsFrame();
    if (this.collisionFocus) this.focusCollisionView();
  }

  setupMovingChainPathEvidence() {
    const contract = this.replay?.branches?.physical?.moving_chain_path_contract;
    const points = contract?.centerline_points_abs_m;
    const regions = contract?.edge_region_ids;
    const cumulative = contract?.cumulative_lengths_m;
    if (!Array.isArray(points)
      || points.length < 3
      || !Array.isArray(regions)
      || regions.length !== points.length - 1
      || !Array.isArray(cumulative)
      || cumulative.length !== points.length) return;

    const regionColors = {
      "rigid.tangent_span": 0xd8a22e,
      "flex.input_sprocket": 0xef6f2c,
      "flex.output_sprocket": 0x2aa7c8,
    };
    const verticesByRegion = new Map();
    for (let index = 0; index < regions.length; index += 1) {
      const region = String(regions[index] || "rigid.tangent_span");
      const vertices = verticesByRegion.get(region) || [];
      vertices.push(...points[index], ...points[index + 1]);
      verticesByRegion.set(region, vertices);
    }
    const group = new THREE.Group();
    group.name = "physics-moving-chain-path";
    for (const [region, vertices] of verticesByRegion) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(vertices, 3),
      );
      const line = new THREE.LineSegments(
        geometry,
        new THREE.LineBasicMaterial({
          color: regionColors[region] || 0xd8a22e,
          transparent: true,
          opacity: 0.88,
          depthTest: false,
          depthWrite: false,
        }),
      );
      line.name = `physics-chain-region-${region}`;
      line.renderOrder = 8;
      line.raycast = () => {};
      group.add(line);
    }

    const markerCount = Math.max(
      1,
      Math.min(256, Number(contract.visual_marker_count) || 64),
    );
    const authoredRadius = Number(contract.chain_collision_radius_m) || 0;
    const markerRadius = Math.max(
      this.axisLength * 0.0025,
      Math.min(authoredRadius * 0.55, this.axisLength * 0.012),
    );
    const markers = new THREE.InstancedMesh(
      new THREE.SphereGeometry(markerRadius, 8, 6),
      new THREE.MeshStandardMaterial({
        color: 0xf5bc33,
        emissive: 0x503000,
        roughness: 0.45,
        metalness: 0.12,
        depthTest: false,
      }),
      markerCount,
    );
    markers.name = "physics-chain-phase-markers";
    markers.renderOrder = 9;
    markers.raycast = () => {};
    group.add(markers);
    group.visible = false;
    this.scene.add(group);
    this.movingChainPathGroup = group;
    this.movingChainPathMarkers = markers;
    this.movingChainPathContract = contract;
  }

  movingChainPointAt(distance) {
    const contract = this.movingChainPathContract;
    const points = contract?.centerline_points_abs_m || [];
    const cumulative = contract?.cumulative_lengths_m || [];
    const total = Number(contract?.total_length_m) || 0;
    if (points.length < 2 || cumulative.length !== points.length || total <= 0) {
      return null;
    }
    const wrapped = ((Number(distance) % total) + total) % total;
    let edge = 0;
    while (edge + 1 < cumulative.length - 1
      && Number(cumulative[edge + 1]) < wrapped) edge += 1;
    const leftDistance = Number(cumulative[edge]) || 0;
    const rightDistance = Number(cumulative[edge + 1]) || leftDistance;
    const fraction = rightDistance > leftDistance
      ? (wrapped - leftDistance) / (rightDistance - leftDistance)
      : 0;
    return new THREE.Vector3(...points[edge]).lerp(
      new THREE.Vector3(...points[edge + 1]),
      Math.max(0, Math.min(1, fraction)),
    );
  }

  updateMovingChainPathEvidence(frame, branch) {
    if (!this.movingChainPathGroup || !this.movingChainPathMarkers) return;
    const contract = branch?.moving_chain_path_contract;
    const state = frame?.moving_chain_path_state;
    const expectedHash = String(this.movingChainPathContract?.contract_sha256 || "");
    const visible = Boolean(
      contract
      && state
      && String(contract.contract_sha256 || "") === expectedHash
      && String(state.contract_sha256 || "") === expectedHash
      && Number(contract.material_motion_per_phase_sign) === -1,
    );
    this.movingChainPathGroup.visible = visible;
    if (!visible) return;
    const markerCount = this.movingChainPathMarkers.count;
    const total = Number(contract.total_length_m) || 0;
    const spacing = Number(contract.visual_marker_spacing_m)
      || total / Math.max(markerCount, 1);
    const phase = Number(state.phase_modulo_m) || 0;
    const materialDirection = Number(contract.material_motion_per_phase_sign);
    const matrix = new THREE.Matrix4();
    for (let index = 0; index < markerCount; index += 1) {
      const point = this.movingChainPointAt(
        materialDirection * phase + index * spacing,
      );
      matrix.identity();
      if (point) matrix.setPosition(point);
      this.movingChainPathMarkers.setMatrixAt(index, matrix);
    }
    this.movingChainPathMarkers.instanceMatrix.needsUpdate = true;
  }

  setPhysicsBranch(branch) {
    if (!this.replay?.branches?.[branch] || branch === this.physicsState.branch) return;
    const oldCount = this.physicsBranchData()?.frames?.length || 1;
    const fraction = oldCount > 1 ? this.physicsState.frame / (oldCount - 1) : 0;
    this.physicsState.branch = branch;
    const newCount = this.physicsBranchData()?.frames?.length || 1;
    this.physicsState.frame = Math.round(fraction * Math.max(0, newCount - 1));
    this.pausePhysicsPlayback();
    this.applyPhysicsFrame();
    this.emitPhysicsState();
  }

  setPhysicsFrame(frame) {
    const count = this.physicsBranchData()?.frames?.length || 1;
    this.physicsState.frame = Math.max(0, Math.min(count - 1, Math.round(Number(frame) || 0)));
    this.applyPhysicsFrame();
    this.emitPhysicsState();
  }

  togglePhysicsPlayback() {
    if (this.physicsState.playing) {
      this.pausePhysicsPlayback();
      return;
    }
    const count = this.physicsBranchData()?.frames?.length || 1;
    if (this.physicsState.frame >= count - 1) this.physicsState.frame = 0;
    this.physicsState.playing = true;
    this.playbackLastTime = 0;
    this.physicsPlaybackHandle = requestAnimationFrame((time) => this.physicsPlaybackTick(time));
    this.updatePhysicsUi();
  }

  physicsPlaybackTick(time) {
    if (!this.physicsState?.playing || this.disposed) return;
    if (!this.playbackLastTime || time - this.playbackLastTime >= 80) {
      this.playbackLastTime = time;
      const count = this.physicsBranchData()?.frames?.length || 1;
      if (this.physicsState.frame >= count - 1) {
        this.pausePhysicsPlayback();
        return;
      }
      this.physicsState.frame += 1;
      this.applyPhysicsFrame();
      this.emitPhysicsState();
    }
    this.physicsPlaybackHandle = requestAnimationFrame((next) => this.physicsPlaybackTick(next));
  }

  pausePhysicsPlayback() {
    if (!this.physicsState) return;
    this.physicsState.playing = false;
    if (this.physicsPlaybackHandle) cancelAnimationFrame(this.physicsPlaybackHandle);
    this.physicsPlaybackHandle = 0;
    this.updatePhysicsUi();
  }

  currentPhysicsFrame() {
    const frames = this.physicsBranchData()?.frames || [];
    if (!frames.length) return null;
    this.physicsState.frame = Math.max(0, Math.min(frames.length - 1, this.physicsState.frame));
    return frames[this.physicsState.frame];
  }

  usesTransmissionArrows() {
    return Array.isArray(this.replay?.rotation_annotations)
      && this.replay.rotation_annotations.length > 0;
  }

  inputLoadsForFrame(frame) {
    const explicit = (frame?.input_loads || []).filter(
      (item) => vectorMagnitude(item?.vector_abs) > 1e-12,
    );
    if (explicit.length) return explicit;
    const external = frame?.external_load || {};
    if (vectorMagnitude(external.vector_abs) <= 1e-12) return [];
    return [{
      id: "primary",
      body_id: frame?.active_input_body_id || this.replay?.input?.body_id,
      point_abs_m: frame?.active_input_point_abs_m,
      ...external,
    }];
  }

  worldInputPoint(frame, load = null) {
    if (Array.isArray(load?.point_abs_m)) {
      return new THREE.Vector3(...load.point_abs_m);
    }
    if (Array.isArray(frame?.active_input_point_abs_m)) {
      return new THREE.Vector3(...frame.active_input_point_abs_m);
    }
    const input = this.replay.input || {};
    const local = input.point_body_local_m;
    const bodyId = load?.body_id || frame?.active_input_body_id || input.body_id;
    const pose = frame?.body_poses?.[bodyId];
    if (Array.isArray(local) && pose) {
      const q = pose.rotation_abs_quat_wxyz || [1, 0, 0, 0];
      const point = new THREE.Vector3(...local);
      point.applyQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]));
      point.add(new THREE.Vector3(...(pose.position_abs_m || [0, 0, 0])));
      return point;
    }
    return new THREE.Vector3(...(input.point_abs_m_at_start || [0, 0, 0]));
  }

  relevantTransmissionBodies(frame) {
    const seeds = this.inputLoadsForFrame(frame)
      .map((item) => String(item.body_id || ""))
      .filter(Boolean);
    const relevant = new Set(seeds);
    if (!seeds.length) return relevant;
    const adjacency = new Map();
    for (const edge of this.replay?.transmission_edges || []) {
      if (edge.engaged_field && frame?.[edge.engaged_field] !== true) continue;
      const first = String(edge.first_body_id || "");
      const second = String(edge.second_body_id || "");
      if (!first || !second) continue;
      if (!adjacency.has(first)) adjacency.set(first, []);
      if (!adjacency.has(second)) adjacency.set(second, []);
      adjacency.get(first).push(second);
      adjacency.get(second).push(first);
    }
    const queue = [...seeds];
    while (queue.length) {
      const bodyId = queue.shift();
      for (const connected of adjacency.get(bodyId) || []) {
        if (relevant.has(connected)) continue;
        relevant.add(connected);
        queue.push(connected);
      }
    }
    return relevant;
  }

  worldRotationOrigin(frame, annotation) {
    const pose = frame?.body_poses?.[annotation?.body_id];
    const local = annotation?.origin_body_local_m;
    if (pose && Array.isArray(local)) {
      const q = pose.rotation_abs_quat_wxyz || [1, 0, 0, 0];
      return new THREE.Vector3(...local)
        .applyQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]))
        .add(new THREE.Vector3(...(pose.position_abs_m || [0, 0, 0])));
    }
    return new THREE.Vector3(...(annotation?.origin_abs_m || [0, 0, 0]));
  }

  rotationArrowRadius(annotation) {
    if (!this.rotationArrowRadii) this.rotationArrowRadii = new Map();
    const key = String(annotation?.body_id || "");
    if (this.rotationArrowRadii.has(key)) return this.rotationArrowRadii.get(key);
    const origin = new THREE.Vector3(...(annotation?.origin_body_local_m || [0, 0, 0]));
    const axis = new THREE.Vector3(...(annotation?.axis_abs || [0, 1, 0])).normalize();
    let radialExtent = 0;
    for (const mesh of this.physicsMeshes || []) {
      if (String(mesh.userData?.physicsComponent?.bodyId || "") !== key) continue;
      mesh.geometry.computeBoundingBox();
      const box = mesh.geometry.boundingBox;
      if (!box) continue;
      for (const x of [box.min.x, box.max.x]) {
        for (const y of [box.min.y, box.max.y]) {
          for (const z of [box.min.z, box.max.z]) {
            const offset = new THREE.Vector3(x, y, z).sub(origin);
            offset.addScaledVector(axis, -offset.dot(axis));
            radialExtent = Math.max(radialExtent, offset.length());
          }
        }
      }
    }
    const radius = Math.max(
      this.axisLength * 0.018,
      Math.min(
        radialExtent > 0 ? radialExtent * 0.44 : this.axisLength * 0.045,
        this.axisLength * 0.075,
      ),
    );
    this.rotationArrowRadii.set(key, radius);
    return radius;
  }

  makeRotationArrow(origin, axis, directionSign, radius, bodyId) {
    const normalizedAxis = axis.clone().normalize();
    let radial = new THREE.Vector3(1, 0, 0);
    if (Math.abs(radial.dot(normalizedAxis)) > 0.85) radial.set(0, 0, 1);
    radial.addScaledVector(normalizedAxis, -radial.dot(normalizedAxis)).normalize();
    const tangentBasis = new THREE.Vector3().crossVectors(normalizedAxis, radial).normalize();
    const center = origin.clone().addScaledVector(normalizedAxis, radius * 0.32);
    const startAngle = -Math.PI * 0.72;
    const span = Math.PI * 1.44;
    const points = [];
    for (let index = 0; index <= 28; index += 1) {
      const angle = startAngle + directionSign * span * (index / 28);
      points.push(
        center.clone()
          .addScaledVector(radial, Math.cos(angle) * radius)
          .addScaledVector(tangentBasis, Math.sin(angle) * radius),
      );
    }
    const group = new THREE.Group();
    group.name = `physics-gear-rotation__${bodyId}`;
    const curve = new THREE.CatmullRomCurve3(points);
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 36, Math.max(radius * 0.045, 0.00025), 7, false),
      new THREE.MeshBasicMaterial({
        color: 0x20a66a,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    tube.renderOrder = 14;
    tube.raycast = () => {};
    group.add(tube);
    const tangent = points.at(-1).clone().sub(points.at(-2)).normalize();
    const head = new THREE.Mesh(
      new THREE.ConeGeometry(radius * 0.14, radius * 0.38, 12),
      new THREE.MeshBasicMaterial({
        color: 0x0d7f4c,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    head.position.copy(points.at(-1)).addScaledVector(tangent, -radius * 0.07);
    head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
    head.renderOrder = 15;
    head.raycast = () => {};
    group.add(head);
    return group;
  }

  setupCollisionFocus() {
    this.physicsCollisionMarkers = [];
    const focus = this.collisionFocus;
    const position = focus?.contact_position_abs_m;
    if (!Array.isArray(position) || position.length < 3) return;
    const overlapMinimum = focus.overlap_bounds_min_abs_m;
    const overlapMaximum = focus.overlap_bounds_max_abs_m;
    let overlapSize = 0;
    if (Array.isArray(overlapMinimum) && Array.isArray(overlapMaximum)) {
      overlapSize = Math.max(
        ...overlapMaximum.slice(0, 3).map((value, index) => Math.abs(Number(value) - Number(overlapMinimum[index]))),
      );
    }
    const radius = Math.max(this.axisLength * 0.022, overlapSize * 0.65, 0.0015);
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 20, 14),
      new THREE.MeshStandardMaterial({
        color: 0xff2020,
        emissive: 0x5d0000,
        metalness: 0.0,
        roughness: 0.42,
        transparent: true,
        opacity: 0.96,
        depthWrite: false,
      }),
    );
    marker.name = `physics-penetration__${focus.event_id || "event"}`;
    marker.position.set(...position.slice(0, 3).map(Number));
    marker.visible = false;
    marker.renderOrder = 12;
    marker.raycast = () => {};
    this.scene.add(marker);
    this.physicsCollisionMarkers.push(marker);

    if (Array.isArray(overlapMinimum) && Array.isArray(overlapMaximum)) {
      const minimum = new THREE.Vector3(...overlapMinimum.slice(0, 3).map(Number));
      const maximum = new THREE.Vector3(...overlapMaximum.slice(0, 3).map(Number));
      const size = maximum.clone().sub(minimum);
      const center = minimum.clone().add(maximum).multiplyScalar(0.5);
      if (size.x > 0 && size.y > 0 && size.z > 0) {
        const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
        const volume = new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({
            color: 0xff2828,
            transparent: true,
            opacity: 0.34,
            depthWrite: false,
            side: THREE.DoubleSide,
          }),
        );
        volume.position.copy(center);
        volume.visible = false;
        volume.renderOrder = 10;
        volume.raycast = () => {};
        const outline = new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry),
          new THREE.LineBasicMaterial({ color: 0xc00000, transparent: true, opacity: 0.95 }),
        );
        outline.raycast = () => {};
        volume.add(outline);
        this.scene.add(volume);
        this.physicsCollisionMarkers.push(volume);
      }
    }
  }

  updateCollisionFocus(frame) {
    const active = Boolean(
      this.collisionFocus
      && this.physicsState.branch === "physical"
      && Number(frame?.source_sample_index) === Number(
        this.collisionFocus.source_sample_index ?? this.collisionFocus.frame
      )
    );
    for (const marker of this.physicsCollisionMarkers || []) marker.visible = active;
  }

  focusCollisionView() {
    const focus = this.collisionFocus;
    if (!focus) return;
    const minimumRaw = focus.overlap_bounds_min_abs_m;
    const maximumRaw = focus.overlap_bounds_max_abs_m;
    let box = null;
    if (Array.isArray(minimumRaw) && Array.isArray(maximumRaw)) {
      box = new THREE.Box3(
        new THREE.Vector3(...minimumRaw.slice(0, 3).map(Number)),
        new THREE.Vector3(...maximumRaw.slice(0, 3).map(Number)),
      );
      const rawSize = box.getSize(new THREE.Vector3());
      box.expandByScalar(Math.max(rawSize.x, rawSize.y, rawSize.z, 0.001) * 4.5);
    } else {
      box = new THREE.Box3();
      for (const mesh of this.physicsMeshes) {
        if (this.physicsFocusRole(mesh, focus)) box.expandByObject(mesh);
      }
    }
    if (!box || box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maximum = Math.max(size.x, size.y, size.z, 0.001);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const distance = (maximum / (2 * Math.tan(fov / 2))) * 1.7;
    let direction = new THREE.Vector3(...(focus.contact_normal_abs || [1.15, 0.78, 1.2])).normalize();
    if (direction.lengthSq() < 1e-12) direction = new THREE.Vector3(1.15, 0.78, 1.2).normalize();
    direction.add(new THREE.Vector3(0.25, 0.18, 0.28)).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.camera.near = Math.max(maximum / 1000, 0.00001);
    this.camera.far = Math.max(distance * 80, maximum * 100);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  worldLandmarkPoint(frame, landmark) {
    const parentPose = frame?.body_poses?.[landmark?.parent_body_id];
    const local = landmark?.origin_parent_local_m;
    if (parentPose && Array.isArray(local)) {
      const quaternion = parentPose.rotation_abs_quat_wxyz || [1, 0, 0, 0];
      return new THREE.Vector3(...local)
        .applyQuaternion(new THREE.Quaternion(quaternion[1], quaternion[2], quaternion[3], quaternion[0]))
        .add(new THREE.Vector3(...(parentPose.position_abs_m || [0, 0, 0])));
    }
    return new THREE.Vector3(...(landmark?.origin_abs_m || [0, 0, 0]));
  }

  physicsHeatEvidence(frame, branch) {
    const spots = new Map();
    const segments = new Map();
    const nodes = new Map();
    const addNode = (bodyId, position, intensity, kind, id) => {
      const key = String(bodyId || "");
      if (!key || !position) return;
      const node = {
        id: String(id || kind || "evidence"),
        position: position.clone(),
        intensity: clamp01(intensity),
        kind,
      };
      if (!spots.has(key)) spots.set(key, []);
      if (!nodes.has(key)) nodes.set(key, []);
      spots.get(key).push(node);
      nodes.get(key).push(node);
    };
    const enabled = this.physicsState.branch === "physical"
      && this.physicsState.heat
      && !this.collisionFocus
      && !this.usesTransmissionArrows();
    if (!enabled) return { spots, segments };

    const external = frame?.external_load || {};
    const externalMagnitude = Number(external.magnitude) || vectorMagnitude(external.vector_abs);
    const externalMaximum = Number(branch?.maxima?.external_load) || externalMagnitude;
    if (externalMagnitude > 1e-12) {
      addNode(
        this.replay?.input?.body_id,
        this.worldInputPoint(frame),
        Math.max(0.16, externalMagnitude / Math.max(externalMaximum, 1e-12)),
        "external-load",
        "external-load",
      );
    }

    if (branch?.field_availability?.per_joint_bushing_reaction) {
      const values = frame?.bushing_reaction_forces_n || {};
      const maximum = Number(branch?.maxima?.max_bushing_reaction_force_n) || 0;
      for (const landmark of this.replay?.landmarks || []) {
        const value = Number(values[landmark.id]);
        if (!Number.isFinite(value) || value <= 1e-12) continue;
        const point = this.worldLandmarkPoint(frame, landmark);
        const intensity = maximum > 0 ? Math.max(0.14, value / maximum) : 0.14;
        const nodeId = "joint:" + String(landmark.id || "unnamed");
        addNode(landmark.parent_body_id, point, intensity, "joint-reaction", nodeId);
        addNode(landmark.child_body_id, point, intensity, "joint-reaction", nodeId);
      }
    }

    for (const [bodyId, bodyNodes] of nodes.entries()) {
      const unique = [];
      const seen = new Set();
      for (const node of bodyNodes) {
        if (seen.has(node.id)) continue;
        seen.add(node.id);
        unique.push(node);
      }
      if (unique.length < 2) continue;
      const bodySegments = [];
      const connected = new Set([0]);
      while (connected.size < unique.length && bodySegments.length < MAX_LOCAL_HEAT_SEGMENTS) {
        let best = null;
        for (const fromIndex of connected) {
          for (let toIndex = 0; toIndex < unique.length; toIndex += 1) {
            if (connected.has(toIndex)) continue;
            const distanceSquared = unique[fromIndex].position.distanceToSquared(unique[toIndex].position);
            if (!best || distanceSquared < best.distanceSquared) {
              best = { fromIndex, toIndex, distanceSquared };
            }
          }
        }
        if (!best) break;
        const from = unique[best.fromIndex];
        const to = unique[best.toIndex];
        bodySegments.push({
          start: from.position.clone(),
          end: to.position.clone(),
          intensity: Math.min(from.intensity, to.intensity),
          kind: "rigid-body-force-flow",
        });
        connected.add(best.toIndex);
      }
      bodySegments.sort((left, right) => right.intensity - left.intensity);
      segments.set(bodyId, bodySegments);
    }
    for (const values of spots.values()) values.sort((left, right) => right.intensity - left.intensity);
    return { spots, segments };
  }
  setupReactionLandmarks() {
    this.physicsReactionMarkers = [];
    const radius = Math.max(this.axisLength * 0.018, 0.0015);
    for (const landmark of this.replay?.landmarks || []) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(radius, 14, 10),
        new THREE.MeshStandardMaterial({
          color: 0x8c9aa8,
          emissive: 0x000000,
          metalness: 0.05,
          roughness: 0.55,
          transparent: true,
          opacity: 0.92,
          depthWrite: false,
        }),
      );
      marker.name = `physics-reaction__${landmark.id || "joint"}`;
      marker.userData.physicsLandmark = landmark;
      marker.visible = false;
      marker.renderOrder = 8;
      marker.raycast = () => {};
      this.scene.add(marker);
      this.physicsReactionMarkers.push(marker);
    }
  }

  updateReactionLandmarks(frame, branch) {
    const values = frame?.bushing_reaction_forces_n || {};
    const maximum = Number(branch?.maxima?.max_bushing_reaction_force_n) || 0;
    const enabled = this.physicsState.branch === "physical"
      && this.physicsState.heat
      && !this.collisionFocus
      && !this.usesTransmissionArrows()
      && Boolean(branch?.field_availability?.per_joint_bushing_reaction);
    for (const marker of this.physicsReactionMarkers || []) {
      const landmark = marker.userData.physicsLandmark || {};
      const value = Number(values[landmark.id]);
      const connectedBodyVisible = this.bodyIsVisible(landmark.parent_body_id)
        || this.bodyIsVisible(landmark.child_body_id);
      marker.visible = enabled && connectedBodyVisible && Number.isFinite(value);
      if (!marker.visible) continue;
      marker.position.copy(this.worldLandmarkPoint(frame, landmark));
      const ratio = maximum > 0 ? clamp01(value / maximum) : 0;
      marker.material.color.copy(heatColor(ratio));
      marker.material.emissive.copy(heatColor(ratio)).multiplyScalar(0.18);
      marker.scale.setScalar(0.8 + ratio * 0.75);
    }
  }

  updateForceArrows(frame) {
    if (this.forceArrows) {
      this.scene.remove(this.forceArrows);
      disposeTree(this.forceArrows);
      this.forceArrows = null;
    }
    if (this.physicsState.branch !== "physical"
      || !this.physicsState.heat
      || this.collisionFocus) return;
    const loads = this.inputLoadsForFrame(frame);
    if (!loads.length) return;
    const group = new THREE.Group();
    group.name = "physics-input-loads";
    const recordedMaximum = Number(this.physicsBranchData()?.maxima?.external_load) || 0;
    for (const load of loads) {
      if (!this.bodyIsVisible(load.body_id)) continue;
      const vector = load.vector_abs || [0, 0, 0];
      const magnitude = vectorMagnitude(vector);
      if (magnitude <= 1e-12) continue;
      const maximum = Math.max(recordedMaximum, magnitude);
      const ratio = clamp01(magnitude / Math.max(maximum, 1e-12));
      const direction = new THREE.Vector3(...vector).normalize();
      const length = Math.max(this.axisLength * (0.14 + ratio * 0.25), this.axisLength * 0.12);
      const point = this.worldInputPoint(frame, load);
      const arrow = new THREE.ArrowHelper(
        direction,
        point,
        length,
        this.usesTransmissionArrows() ? 0xf06a24 : heatColor(ratio),
        length * 0.22,
        length * 0.09,
      );
      arrow.name = `physics-input-load__${load.id || load.body_id || "input"}`;
      arrow.line.material.depthTest = false;
      arrow.line.material.depthWrite = false;
      arrow.cone.material.depthTest = false;
      arrow.cone.material.depthWrite = false;
      arrow.renderOrder = 14;
      arrow.raycast = () => {};
      group.add(arrow);
      if (this.usesTransmissionArrows()) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(Math.max(this.axisLength * 0.009, 0.0008), 14, 10),
          new THREE.MeshBasicMaterial({
            color: 0xff8a45,
            depthTest: false,
            depthWrite: false,
            toneMapped: false,
          }),
        );
        marker.position.copy(point);
        marker.renderOrder = 15;
        marker.raycast = () => {};
        group.add(marker);
      }
    }
    if (!group.children.length) {
      disposeTree(group);
      return;
    }
    this.forceArrows = group;
    this.scene.add(group);
  }

  updateRotationArrows(frame) {
    if (this.rotationArrows) {
      this.scene.remove(this.rotationArrows);
      disposeTree(this.rotationArrows);
      this.rotationArrows = null;
    }
    this.currentRotationArrowCount = 0;
    if (!this.usesTransmissionArrows()
      || this.physicsState.branch !== "physical"
      || !this.physicsState.heat
      || this.collisionFocus) return;
    const relevant = this.relevantTransmissionBodies(frame);
    const velocities = frame?.body_angular_velocities_abs_rad_s || {};
    const group = new THREE.Group();
    group.name = "physics-gear-rotation-arrows";
    for (const annotation of this.replay.rotation_annotations) {
      const bodyId = String(annotation.body_id || "");
      if (!bodyId
        || !relevant.has(bodyId)
        || !this.bodyIsVisible(bodyId)) continue;
      const velocity = velocities[bodyId];
      const axis = new THREE.Vector3(...(annotation.axis_abs || [0, 1, 0]));
      const signedSpeed = Array.isArray(velocity)
        ? new THREE.Vector3(...velocity).dot(axis.clone().normalize())
        : 0;
      if (!Number.isFinite(signedSpeed) || Math.abs(signedSpeed) <= 1e-6) continue;
      const arrow = this.makeRotationArrow(
        this.worldRotationOrigin(frame, annotation),
        axis,
        signedSpeed >= 0 ? 1 : -1,
        this.rotationArrowRadius(annotation),
        bodyId,
      );
      group.add(arrow);
      this.currentRotationArrowCount += 1;
    }
    if (!group.children.length) {
      disposeTree(group);
      return;
    }
    this.rotationArrows = group;
    this.scene.add(group);
  }

  applyPhysicsFrame() {
    const frame = this.currentPhysicsFrame();
    if (!frame) return;
    const branch = this.physicsBranchData();
    const forceEvidence = this.mechanics ? mechanicsForces(frame, branch, this.mechanics.metric) : null;
    const heatEvidence = this.mechanics
      ? {spots:this.mechanics.heat ? forceEvidence.spots : new Map(),segments:this.mechanics.heat && this.mechanics.flow ? forceEvidence.segments : new Map()}
      : this.physicsHeatEvidence(frame, branch);
    const heatSpotsByBody = heatEvidence.spots;
    const heatSegmentsByBody = heatEvidence.segments;
    const heatRadius = Math.max(this.axisLength * 0.09, 0.0015);
    const selectedName = this.physicsState.part;
    const focus = this.physicsState.branch === "physical" ? this.collisionFocus : null;
    for (const mesh of this.physicsMeshes) {
      const component = mesh.userData.physicsComponent || {};
      const bodyVisible = this.bodyIsVisible(component.bodyId);
      mesh.visible = bodyVisible;
      const pose = frame.body_poses?.[component.bodyId];
      if (pose) {
        const q = pose.rotation_abs_quat_wxyz || [1, 0, 0, 0];
        mesh.position.set(...(pose.position_abs_m || [0, 0, 0]));
        mesh.quaternion.set(q[1], q[2], q[3], q[0]);
      }
      const selected = mesh.name === selectedName;
      const focusRole = this.physicsFocusRole(mesh, focus);
      const focusPrimary = focusRole === "primary";
      const focusSecondary = focusRole === "secondary";
      const focused = focusPrimary || focusSecondary;
      const base = mesh.userData.physicsBaseSpec || (component.fixed ? this.palette.fixed : this.palette.moving);
      const color = focusPrimary
        ? new THREE.Color(0x267fca)
        : focusSecondary
          ? new THREE.Color(0xef9b32)
          : focus
            ? new THREE.Color(0x9aa7b4)
            : selected
              ? new THREE.Color(0x27c7d4)
              : new THREE.Color(base.color);
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) {
        material.color.copy(color);
        material.emissive?.set(focused ? (focusPrimary ? 0x082a48 : 0x4a2600) : selected ? 0x063b44 : 0x000000);
        const baseOpacity = component.fixed ? this.fixedSurfaceOpacity() : this.movingSurfaceOpacity();
        applySurfaceOpacity(material, focused || selected ? 1 : focus ? 0.13 : baseOpacity);
        setMaterialLocalHeat(
          material,
          heatSpotsByBody.get(String(component.bodyId || "")) || [],
          heatSegmentsByBody.get(String(component.bodyId || "")) || [],
          heatRadius,
          bodyVisible && !focused && !selected && !focus
            && (component.fixed ? this.fixedDisplayMode === "solid" : this.movingDisplayMode !== "outline"),
          Boolean(this.mechanics),
        );
      }
      const ghostEdges = mesh.userData.ghostEdges;
      if (ghostEdges) {
        ghostEdges.visible = bodyVisible
          && (component.fixed ? this.fixedDisplayMode === "outline" : this.movingDisplayMode !== "solid")
          && !focused
          && !selected;
      }
      const overlay = mesh.userData.collisionOverlay;
      const showOverlay = this.physicsState.layer === "all"
        || (this.physicsState.layer === "selected" && (focused || selected));
      if (overlay) {
        overlay.visible = bodyVisible && showOverlay;
        overlay.material.color.set(focusPrimary ? 0x7ec8ff : focusSecondary ? 0xffcb7d : selected ? 0x00e6f2 : 0x17344d);
        overlay.material.opacity = focused || selected ? 0.95 : 0.38;
      }
    }
    this.parts?.apply();
    this.updateBodyVisibilityUi();
    this.updateReactionLandmarks(frame, branch);
    this.updateForceArrows(frame);
    this.updateRotationArrows(frame);
    this.updateMovingChainPathEvidence(frame, branch);
    this.updateCollisionFocus(frame);
    updateMechanics(this, frame, branch, forceEvidence);
    this.updatePhysicsUi();
    this.renderOnce();
  }

  updatePhysicsUi() {
    if (!this.physicsUi || !this.physicsState) return;
    const branch = this.physicsBranchData() || {};
    const frame = this.currentPhysicsFrame();
    const frames = branch.frames || [];
    if (this.physicsUi.frame) {
      this.physicsUi.frame.max = String(Math.max(0, frames.length - 1));
      this.physicsUi.frame.value = String(this.physicsState.frame);
    }
    if (this.physicsUi.frameLabel) this.physicsUi.frameLabel.textContent = `${this.physicsState.frame + 1} / ${Math.max(1, frames.length)}`;
    if (this.physicsUi.play) this.physicsUi.play.textContent = this.physicsState.playing ? "❚❚ 暂停" : "▶ 播放";
    for (const button of this.physicsUi.branchButtons) {
      button.setAttribute("aria-pressed", String(button.dataset.physicsBranch === this.physicsState.branch));
    }
    if (this.physicsUi.collision) {
      const words = { selected: "所选", all: "全部", off: "关闭" };
      const inactive = this.physicsState.branch === "reduced" ? " · 仅参考" : "";
      this.physicsUi.collision.textContent = `碰撞体：${words[this.physicsState.layer]}${inactive}`;
    }
    if (this.physicsUi.heat) {
      const label = this.usesTransmissionArrows() ? "传动箭头" : "力流证据";
      this.physicsUi.heat.textContent = `${label}：${this.physicsState.heat ? "开" : "关"}`;
      this.physicsUi.heat.setAttribute("aria-pressed", String(this.physicsState.heat));
    }
    if (!frame) return;
    if (this.physicsUi.time) this.physicsUi.time.textContent = formatEvidenceNumber(frame.time_s, "s", 3);
    if (this.physicsUi.phase) this.physicsUi.phase.textContent = frame.phase_id || "未命名阶段";
    const external = frame.external_load || {};
    if (this.physicsUi.external) this.physicsUi.external.textContent = formatEvidenceNumber(external.magnitude, external.unit, 2);
    const reaction = Number(frame.max_bushing_reaction_force_n);
    if (this.physicsUi.reaction) this.physicsUi.reaction.textContent = Number.isFinite(reaction) ? formatEvidenceNumber(reaction, "N", 2) : "未记录";
    const drifts = [...Object.values(frame.joint_drifts_m || {}), ...Object.values(frame.support_drifts_m || {})].map(Number).filter(Number.isFinite);
    const maxDrift = drifts.length ? Math.max(...drifts) : null;
    if (this.physicsUi.drift) this.physicsUi.drift.textContent = maxDrift == null ? "未记录" : formatEvidenceNumber(maxDrift * 1000, "mm", 3);
    if (this.physicsUi.contact) this.physicsUi.contact.textContent = `${Number(frame.contact_count) || 0} 个`;
    const reactionMax = Number(branch.maxima?.max_bushing_reaction_force_n) || 0;
    const reactionRatio = reactionMax > 0 && Number.isFinite(reaction) ? clamp01(reaction / reactionMax) : 0;
    if (this.physicsUi.reactionBar) {
      this.physicsUi.reactionBar.style.width = `${Math.round(reactionRatio * 100)}%`;
      this.physicsUi.reactionBar.style.backgroundColor = `#${heatColor(reactionRatio).getHexString()}`;
    }
    const selected = this.physicsMeshes.find((mesh) => mesh.name === this.physicsState.part);
    const component = selected?.userData?.physicsComponent;
    if (this.physicsUi.part) this.physicsUi.part.textContent = physicsSolidLabel(component?.solid_id || component?.solidId);
    if (this.physicsUi.body) this.physicsUi.body.textContent = component ? `刚体：${physicsBodyLabel(component.bodyId)} · ${component.fixed ? "固定" : "运动"}` : "可以查看它属于哪个刚体、对应哪个 STEP 实体。";
    if (this.physicsUi.representation) this.physicsUi.representation.textContent = collisionRepresentationLabel(component?.representation);
    if (this.physicsUi.availability) {
      const available = branch.field_availability || {};
      const reactionLandmarkIds = new Set(
        (this.replay?.landmarks || []).map((landmark) => String(landmark?.id || "")),
      );
      this.physicsUi.availability.textContent = this.physicsState.branch === "reduced"
        ? "简化分支使用理想约束；碰撞线框只用于几何对照，不参与该分支求解。"
        : this.usesTransmissionArrows()
          ? `橙色直箭头标出 ${this.inputLoadsForFrame(frame).length} 处当前输入；绿色弧箭头标出 ${this.currentRotationArrowCount || 0} 个与输入连通且正在转动的传动齿轮。未连通齿轮不画箭头。`
        : available.per_joint_bushing_reaction && reactionLandmarkIds.size > 0
          ? (() => {
              const reactions = Object.entries(frame.bushing_reaction_forces_n || {})
                .map(([id, value]) => [id, Number(value)])
                .filter(([id, value]) => reactionLandmarkIds.has(String(id)) && Number.isFinite(value))
                .sort((left, right) => right[1] - left[1]);
              return reactions.length
                ? `载荷点与连接反力已在对应表面生成局部深色热斑；当前最高为${physicsLandmarkLabel(reactions[0][0])} ${formatEvidenceNumber(reactions[0][1], "N", 2)}。`
                : "逐关节连接反力已记录，有数值时会在对应位置生成局部深色热斑。";
            })()
          : available.per_joint_bushing_reaction
            ? "逐关节反力数值已记录，但该机构没有可匹配的空间锚点，因此不生成空间热斑。"
            : "该轨迹没有保存逐关节的反力位置，因此不生成空间热斑，只显示已记录的数值。";
      const contactCoordinatesAvailable = Boolean(available.contact_positions || available.contact_points)
        || Array.isArray(frame.contact_points_abs_m);
      if (this.physicsState.branch === "physical" && Number(frame.contact_count) > 0 && !contactCoordinatesAvailable) {
        this.physicsUi.availability.textContent += " 接触次数已记录，但该轨迹没有保存接触点坐标，因此不伪造接触热斑。";
      }
    }
    if (this.collisionFocus) {
      const pair = this.collisionFocus.solid_pair || [];
      if (this.physicsUi.part) this.physicsUi.part.textContent = pair.join(" ↔ ");
      if (this.physicsUi.body) this.physicsUi.body.textContent = `穿模证据 · 原始轨迹帧 ${this.collisionFocus.source_sample_index ?? this.collisionFocus.frame}`;
      if (this.physicsUi.representation) this.physicsUi.representation.textContent = "蓝色与橙色为碰撞实体；红色为检测接触邻域";
      if (this.physicsUi.availability) {
        const exact = Number(frame.source_sample_index) === Number(this.collisionFocus.source_sample_index ?? this.collisionFocus.frame);
        this.physicsUi.availability.textContent = exact
          ? "当前正是审查失败时间点；红色标记由独立 STEP 实体复审生成。"
          : "已离开失败时间点；可拖动时间轴比较穿模出现前后的状态。";
      }
    }
  }

  emitPhysicsState() {
    if (!this.physicsState) return;
    this.container.dispatchEvent(new CustomEvent("quest-physics-state", {
      bubbles: true,
      detail: {
        branch: this.physicsState.branch,
        frame: this.physicsState.frame,
        layer: this.physicsState.layer,
        part: this.physicsState.part,
      },
    }));
  }

  fitView(reset = true) {
    const box = new THREE.Box3().setFromObject(this.model);
    const recordedBounds = this.sceneManifest?.motion_bounds_abs_m;
    if (this.mode === "physics" && motionBoundsUsefulForCamera(box, recordedBounds)) {
      box.expandByPoint(new THREE.Vector3(...recordedBounds.min));
      box.expandByPoint(new THREE.Vector3(...recordedBounds.max));
    }
    if (box.isEmpty()) throw new Error("empty model");
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const diagonal = Math.max(size.length(), 0.001);
    const maxDimension = Math.max(size.x, size.y, size.z, diagonal * 0.25);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const direction = new THREE.Vector3(1.15, 0.78, 1.2).normalize();
    let distance = (maxDimension / (2 * Math.tan(fov / 2))) * 1.55;
    if (this.container.closest(".studio-main")) {
      const right = new THREE.Vector3().crossVectors(this.camera.up, direction).normalize();
      const up = new THREE.Vector3().crossVectors(direction, right).normalize();
      const vertical = Math.tan(fov / 2);
      const horizontal = vertical * Math.max(this.camera.aspect, 0.1);
      let fit = 0;
      for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
        const point = new THREE.Vector3(x, y, z).sub(center);
        fit = Math.max(fit, point.dot(direction) + Math.max(Math.abs(point.dot(right)) / horizontal, Math.abs(point.dot(up)) / vertical));
      }
      distance = fit * 1.10;
    }

    this.homeTarget = center.clone();
    this.homePosition = center.clone().addScaledVector(direction, distance);
    this.camera.near = Math.max(diagonal / 1000, 0.0001);
    this.camera.far = Math.max(distance * 60, diagonal * 60);
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = diagonal * 0.08;
    this.controls.maxDistance = diagonal * 15;
    this.axisLength = diagonal * 0.32;
    if (reset) this.resetView(false);
  }

  addAxis() {
    if (this.mode !== "mechanism" || !this.axis || this.axis.lengthSq() < 1e-12) return;
    const direction = this.axis.clone().normalize();
    const origin = this.origin?.clone() || this.homeTarget.clone();
    const length = Math.max(this.axisLength, 0.001);
    const arrow = new THREE.ArrowHelper(direction, origin, length, this.palette.axis, length * 0.2, length * 0.08);
    arrow.name = "quest-motion-axis";
    this.axisHelper = arrow;
    this.scene.add(arrow);
  }

  resetView(animate = true) {
    if (!this.homePosition || !this.homeTarget) return;
    this.camera.position.copy(this.homePosition);
    this.controls.target.copy(this.homeTarget);
    this.controls.update();
    if (animate) this.renderOnce();
  }

  resize() {
    if (!this.renderer || this.disposed) return;
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.renderOnce();
  }

  renderOnce() {
    if (document.hidden || !this.renderer || !this.scene || !this.camera || this.disposed) return;
    layoutMechanicsLabels(this);
    this.renderer.autoClear = true;
    this.renderer.render(this.scene, this.camera);
    if (this.viewHelper) {
      this.renderer.autoClear = false;
      this.viewHelper.render(this.renderer);
      this.renderer.autoClear = true;
    }
  }

  showViewer() {
    for (const fallback of this.container.querySelectorAll("img, video, picture, .evidence-3d-fallback")) {
      if (fallback.closest(".quest-3d-viewport")) continue;
      this.restore.fallback.push({
        element: fallback,
        opacity: fallback.style.opacity,
        pointerEvents: fallback.style.pointerEvents,
        ariaHidden: fallback.getAttribute("aria-hidden"),
      });
      fallback.style.opacity = "0";
      fallback.style.pointerEvents = "none";
      fallback.setAttribute("aria-hidden", "true");
    }
    this.viewport.style.opacity = "1";
    this.viewport.style.pointerEvents = "auto";
    const status = this.container.querySelector("[data-3d-status]");
    if (status) status.textContent = "3D 已载入";
    this.container.dataset.viewerStatus = "ready";
  }

  fail() {
    if (this.disposed) return;
    this.container.dataset.viewerStatus = "error";
    const status = this.container.querySelector("[data-3d-status]");
    if (status) status.textContent = "3D 未载入，已显示静态备用图";
    this.container.title ||= "三维模型暂时无法加载，已保留静态预览";
    this.dispose(false);
  }

  dispose(restoreContainer = true) {
    this.mechanics?.toolbar?.remove();
    if (this.disposed) return;
    this.disposed = true;
    this.parts?.dispose();
    document.removeEventListener("visibilitychange",this.onPageVisibility);
    if (this.physicsState) {
      this.physicsState.playing = false;
      if (this.physicsPlaybackHandle) cancelAnimationFrame(this.physicsPlaybackHandle);
    }
    this.intersectionObserver?.disconnect();
    for (const remove of this.physicsListeners || []) remove();
    this.physicsListeners = [];
    this.resizeObserver?.disconnect();
    this.resetButton?.removeEventListener("click", this.onReset);
    for (const remove of this.fixedDisplayListeners || []) remove();
    this.fixedDisplayListeners = [];
    for (const remove of this.movingDisplayListeners || []) remove();
    this.movingDisplayListeners = [];
    this.controls?.removeEventListener("change", this.onControlsChange);
    for (const remove of this.gizmoListeners || []) remove();
    this.gizmoListeners = [];
    if (this.gizmoAnimationHandle) cancelAnimationFrame(this.gizmoAnimationHandle);
    this.viewHelper?.dispose();
    this.controls?.dispose();
    disposeTree(this.scene, this.detachedMaterials);
    for (const entry of this.historyModels.values()) {
      if (entry.root !== this.model) disposeTree(entry.root, entry.materials);
    }
    this.historyModels.clear();
    this.scene?.clear();
    this.renderer?.renderLists?.dispose();
    this.renderer?.dispose();
    this.renderer?.forceContextLoss();
    this.viewport?.remove();
    if (restoreContainer) {
      this.container.style.position = this.restore.position;
      this.container.style.minHeight = this.restore.minHeight;
      for (const item of this.restore.fallback) {
        item.element.style.opacity = item.opacity;
        item.element.style.pointerEvents = item.pointerEvents;
        if (item.ariaHidden == null) item.element.removeAttribute("aria-hidden");
        else item.element.setAttribute("aria-hidden", item.ariaHidden);
      }
      delete this.container.dataset.viewerStatus;
    }
  }
}

async function mount(container) {
  if (!(container instanceof Element)) return false;
  const current = mounted.get(container);
  if (current) {
    await current.promise;
    await current.replaceModel(container.dataset.modelSrc);
    return true;
  }
  const viewer = new QuestViewer(container);
  mounted.set(container, viewer);
  viewer.promise = viewer.init().catch(() => {
    viewer.fail();
    mounted.delete(container);
    return false;
  });
  return viewer.promise;
}

function selectTargets(root) {
  const targets = [];
  if (root instanceof Element && root.matches(".evidence-3d")) targets.push(root);
  if (root?.querySelectorAll) targets.push(...root.querySelectorAll(".evidence-3d"));
  return targets;
}

function mountAll(root = document) {
  for (const [container, viewer] of mounted) {
    if (!container.isConnected) {
      viewer.dispose();
      mounted.delete(container);
    }
  }
  return Promise.allSettled(selectTargets(root).map(mount));
}

function disposeAll(root = null) {
  for (const [container, viewer] of mounted) {
    const inScope = !root || root === container || root.contains?.(container);
    if (!inScope) continue;
    viewer.dispose();
    mounted.delete(container);
  }
}

function parts(root, action, id, value) {
  for (const [container, viewer] of mounted) {
    if ((root === container || root?.contains?.(container)) && container.isConnected && container.getAttribute("aria-hidden") !== "true") {
      if (action) viewer.parts?.action(action, id, value);
      return viewer.parts?.snapshot() || null;
    }
  }
  return null;
}

function prefetch(root, urls) {
  for (const [container, viewer] of mounted) {
    if (root === container || root.contains?.(container)) viewer.prefetchHistory(urls);
  }
}

window.Quest3D = Object.assign(window.Quest3D || {}, { mountAll, disposeAll, prefetch, parts });

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => mountAll(), { once: true });
} else {
  queueMicrotask(() => mountAll());
}
