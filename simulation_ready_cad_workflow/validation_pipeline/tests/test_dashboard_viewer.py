from pathlib import Path
import json
import shutil
import subprocess
import unittest


class DashboardViewerAssetTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_collision_focus_matches_all_components_of_each_solid(self) -> None:
        focus_module = (
            Path(__file__).resolve().parents[1]
            / "dashboard"
            / "static"
            / "viewer3d_focus.js"
        )
        script = f"""
          import {{ buildPhysicsComponentIndex, indexedPhysicsComponent, isGeneratedSplitComponentNodeName, physicsNodeBindingKey, physicsFocusRole }} from {json.dumps(focus_module.as_uri())};
          const focus = {{
            primary_node: "physics__frame__tube",
            secondary_node: "physics__wheel__rim",
            solid_pair: ["frame/tube", "wheel/rim"],
          }};
          const role = (meshName, bodyId, solid_id) => physicsFocusRole({{
            meshName,
            component: {{ bodyId, solid_id }},
            focus,
          }});
          const componentIndex = buildPhysicsComponentIndex([{{
            body_id: "input_crank",
            fixed: false,
            group_ids: ["crank_group"],
            components: [{{
              node: "physics__input_crank__solid_2__component_0001_input.chainring_backing",
              solid_id: "solid_2",
            }}],
          }}]);
          const resolvedLoadedNode = indexedPhysicsComponent(
            componentIndex,
            "physics__input_crank__solid_2__component_0001_inputchainring_backing",
          );
          console.log(JSON.stringify({{
            primaryParts: [
              role("physics__frame__tube__component_0001_a", "frame", "tube"),
              role("physics__frame__tube__component_0002_b", "frame", "tube"),
            ],
            secondary: role("physics__wheel__rim__component_0001_c", "wheel", "rim"),
            exactLegacy: role("physics__frame__tube", "wrong", "wrong"),
            unrelated: role("physics__fork__blade", "fork", "blade"),
            splitDetected: isGeneratedSplitComponentNodeName(
              "physics__frame__tube__component_0001_a"
            ),
            legacyDetected: isGeneratedSplitComponentNodeName("physics__frame__tube"),
            namedLikeComponentDetected: isGeneratedSplitComponentNodeName(
              "physics__frame__component_0001_base"
            ),
            oldManifestBinding: physicsNodeBindingKey(
              "physics__input_crank__solid_2__component_0001_input.chainring_backing"
            ),
            loadedNodeBinding: physicsNodeBindingKey(
              "physics__input_crank__solid_2__component_0001_inputchainring_backing"
            ),
            resolvedLoadedNode,
          }}));
        """
        completed = subprocess.run(
            ["node", "--input-type=module", "--eval", script],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["primaryParts"], ["primary", "primary"])
        self.assertEqual(result["secondary"], "secondary")
        self.assertEqual(result["exactLegacy"], "primary")
        self.assertEqual(result["unrelated"], "")
        self.assertTrue(result["splitDetected"])
        self.assertFalse(result["legacyDetected"])
        self.assertFalse(result["namedLikeComponentDetected"])
        self.assertEqual(result["oldManifestBinding"], result["loadedNodeBinding"])
        self.assertEqual(result["resolvedLoadedNode"]["bodyId"], "input_crank")
        self.assertEqual(result["resolvedLoadedNode"]["solid_id"], "solid_2")
        self.assertEqual(result["resolvedLoadedNode"]["groupIds"], ["crank_group"])

    def test_viewer_uses_smooth_xray_and_inspection_controls(self) -> None:
        dashboard_root = Path(__file__).resolve().parents[1] / "dashboard" / "static"
        source = (dashboard_root / "viewer3d.source.js").read_text(encoding="utf-8")
        bundle = (dashboard_root / "viewer3d.js").read_text(encoding="utf-8")
        legal = (dashboard_root / "viewer3d.js.LEGAL.txt").read_text(encoding="utf-8")
        app_source = (dashboard_root / "app.js").read_text(encoding="utf-8")
        app_css = (dashboard_root / "app.css").read_text(encoding="utf-8")

        self.assertIn('import { ViewHelper } from "three/examples/jsm/helpers/ViewHelper.js"', source)
        self.assertNotIn("three-wboit", source)
        self.assertNotIn("WboitPass", source)
        self.assertIn("material.transparent = smoothXray", source)
        self.assertIn("material.alphaHash = false", source)
        self.assertIn("material.depthWrite = !smoothXray", source)
        self.assertIn("questUniformXrayColor", source)
        self.assertIn('"smooth-uniform-xray"', source)
        self.assertIn("function makeGhostEdges", source)
        self.assertIn("new THREE.EdgesGeometry(geometry, 28)", source)
        self.assertIn("FIXED_DISPLAY_MODES", source)
        self.assertIn('DEFAULT_PHYSICS_FIXED_DISPLAY_MODE = "solid"', source)
        self.assertIn("fixedDisplayModeFromUrl(this.mode)", source)
        self.assertIn("defaultFixedDisplayMode(viewerMode)", source)
        self.assertIn("MOVING_DISPLAY_MODES", source)
        self.assertIn('[["outline",', source)
        self.assertIn('["xray",', source)
        self.assertIn('["solid",', source)
        self.assertIn("movingDisplayModeFromUrl()", source)
        self.assertIn('movingDisplay.className = "quest-moving-display"', source)
        self.assertIn('this.movingDisplayMode !== "solid"', source)
        self.assertIn('this.movingDisplayMode !== "outline"', source)

        self.assertIn("function motionBoundsUsefulForCamera", source)
        self.assertIn("recordedSize <= modelSize * 8", source)
        self.assertIn("motionBoundsUsefulForCamera(box, recordedBounds)", source)

        self.assertIn("function hiddenBodyIdsFromUrl", source)
        self.assertIn("function saveHiddenBodyIds", source)
        self.assertIn("setupBodyVisibilityControls()", source)
        self.assertIn('panel.className = "quest-body-visibility"', source)
        self.assertIn("toggleBodyVisibility(bodyId)", source)
        self.assertIn("this.bodyIsVisible(component.bodyId)", source)
        self.assertIn("buildPhysicsComponentIndex", source)
        self.assertIn("indexedPhysicsComponent", source)

        self.assertIn("setupOrientationGizmo()", source)
        self.assertIn("new ViewHelper(this.camera, this.renderer.domElement)", source)
        self.assertIn("new THREE.Spherical().setFromVector3(offset)", source)
        self.assertIn("this.viewHelper.handleClick(event)", source)
        self.assertIn('orientationPlate.className = "quest-orientation-plate"', source)

        self.assertIn("attachLocalHeatShader", source)
        self.assertIn("uniform vec4 questHeatPoints", source)
        self.assertIn("physicsHeatEvidence(frame, branch)", source)
        self.assertIn("uniform int questHeatSegmentCount", source)
        self.assertIn('kind: "rigid-body-force-flow"', source)
        self.assertIn("worldLandmarkPoint(frame, landmark)", source)
        self.assertIn("setMaterialLocalHeat(", source)
        self.assertNotIn("heatColor(heatRatio)", source)
        self.assertIn("quest-local-force-flow-v3", source)
        self.assertIn("questForceHighlightColor", source)
        self.assertIn("vec3(0.68, 0.01, 0.55)", source)

        for asset in (source, bundle):
            self.assertIn("smooth-uniform-xray", asset)
            self.assertIn("opaque-depth", asset)
            self.assertIn("outline-only", asset)
            self.assertIn("quest-ghost-edges", asset)
            self.assertIn("quest-fixed-display", asset)
            self.assertIn("quest-moving-display", asset)
            self.assertIn("quest-body-visibility", asset)
            self.assertIn("quest-orientation-plate", asset)
            self.assertIn("quest-local-force-flow-v3", asset)
            self.assertIn("physics-chain-phase-markers", asset)

        self.assertIn("Three.js Authors", legal)
        self.assertNotIn("Stephens Nunnally", legal)
        self.assertRegex(app_css, r"\.viewer-loading\s*\{[^}]*pointer-events:\s*none;")
        self.assertIn('role.startsWith("design-review-")', app_source)
        self.assertNotIn("latestReviewIso", app_source)
        self.assertIn('modelCard(visual("cad-scene"), visual("cad-overview")', app_source)
        self.assertIn('cleanUrl.searchParams.delete("hidden")', app_source)
        self.assertIn("力流证据：开", app_source)
        self.assertIn("不是有限元应力云图", app_source)
        self.assertIn("紫红＝反力点与传力路径", app_source)
        self.assertIn("传动箭头：开", app_source)
        self.assertIn("橙色直箭头＝输入位置", app_source)
        self.assertIn("绿色弧箭头＝相关齿轮旋向", app_source)
        self.assertIn("usesTransmissionArrows()", source)
        self.assertIn("relevantTransmissionBodies(frame)", source)
        self.assertIn("updateRotationArrows(frame)", source)
        self.assertIn("new THREE.TubeGeometry", source)
        self.assertIn("physics-gear-rotation-arrows", source)
        self.assertIn("setupMovingChainPathEvidence()", source)
        self.assertIn("updateMovingChainPathEvidence(frame, branch)", source)
        self.assertIn("physics-chain-phase-markers", source)
        self.assertIn("physics-moving-chain-path", source)
        self.assertIn("visibleComponentCount", source)
        self.assertIn("\\u521a\\u4f53\\u663e\\u793a", source)
        self.assertIn("\\u7ec4\\u4ef6", source)
        self.assertIn("physicsComponentSolidRef(component)", source)
        self.assertIn("physicsFocusRole(mesh, focus", source)
        focus_source = (dashboard_root / "viewer3d_focus.js").read_text(encoding="utf-8")
        self.assertIn("solidRef === solidPair[0]", focus_source)
        self.assertIn("solidRef === solidPair[1]", focus_source)
        self.assertIn("this.physicsFocusRole(mesh, focus)", source)
        self.assertNotIn("const focusPrimary = mesh.name === focus?.primary_node", source)
        self.assertIn("/^component_[0-9]{4}_/", focus_source)
        self.assertIn("background: #d51aa5", app_css)
        self.assertIn("background: #f06a24", app_css)
        self.assertIn("background: #20a66a", app_css)

    def test_live_refresh_distinguishes_follow_current_from_pinned_history(self) -> None:
        dashboard_root = Path(__file__).resolve().parents[1] / "dashboard" / "static"
        app_source = (dashboard_root / "app.js").read_text(encoding="utf-8")

        self.assertIn("let liveRefreshInFlight = false;", app_source)
        self.assertIn("let loadRequestSerial = 0;", app_source)
        self.assertIn("requestSerial !== loadRequestSerial", app_source)
        self.assertIn("state.runId !== selectedRunId", app_source)
        self.assertIn("item.id === selectedProjectId", app_source)
        self.assertIn('selectionMode: "follow_current"', app_source)
        self.assertIn('state.selectionMode === "pinned"', app_source)
        self.assertIn('state.selectionMode = "follow_current"', app_source)
        self.assertIn('state.selectionMode = "pinned"', app_source)
        self.assertIn("projectDefaultRunId", app_source)
        self.assertIn("desiredRunId", app_source)
        self.assertIn("error.status = response.status", app_source)
        self.assertIn("if (error?.status !== 409) throw error", app_source)
        self.assertIn('fetchJson(`${boot.apiBase}/runs?fresh=1`)', app_source)
        self.assertNotIn("await loadRun(rows[0].id", app_source)
        self.assertNotIn("查看全部 8 关", app_source)
        self.assertNotIn("尚无发布版本", app_source)
        self.assertIn("最近通过", app_source)
        self.assertIn("last_pass", app_source)
        self.assertIn("}[stageKind()]", app_source)

    def test_physics_stage_prefers_load_with_video_and_step_replay(self) -> None:
        dashboard_root = Path(__file__).resolve().parents[1] / "dashboard" / "static"
        app_source = (dashboard_root / "app.js").read_text(encoding="utf-8")

        self.assertIn("const physicsEvidenceForLoad = (loadItem) =>", app_source)
        self.assertIn("const selectBestPhysicsEvidenceLoad = () =>", app_source)
        self.assertIn('stageKind() !== "branch-simulation"', app_source)
        self.assertIn('"physical-diagnostic-video"', app_source)
        self.assertIn("selectBestPhysicsEvidenceLoad();", app_source)
        self.assertIn("Step 3D", app_source)

    def test_revision_replay_uses_project_history_and_run_scoped_previews(self) -> None:
        dashboard_root = Path(__file__).resolve().parents[1] / "dashboard" / "static"
        app_source = (dashboard_root / "app.js").read_text(encoding="utf-8")
        app_css = (dashboard_root / "app.css").read_text(encoding="utf-8")

        self.assertIn('revisionReplayOpen: false', app_source)
        self.assertIn("function revisionMilestoneIds", app_source)
        self.assertIn("function revisionReplayPanel", app_source)
        self.assertIn("function revisionReplayStep", app_source)
        self.assertIn("function revisionReplayFrames", app_source)
        self.assertIn("function revisionPreviewMatches", app_source)
        self.assertIn("function revisionReplayPlaybackFrames", app_source)
        self.assertIn("function bindRevisionReplayEvents", app_source)
        self.assertIn('q.set("replayRun", state.revisionReplayRunId)', app_source)
        self.assertIn("run=${encodeURIComponent(item.id)}", app_source)
        self.assertIn("item?.revision_preview_path", app_source)
        self.assertIn("item.revision_preview_sha256", app_source)
        self.assertIn("验证状态变化", app_source)
        self.assertNotIn("这一版发生了什么", app_source)
        self.assertIn(".app-shell.has-revision-replay", app_css)
        self.assertIn(".app-shell.has-revision-replay > .workspace", app_css)
        self.assertIn(".revision-timeline-track", app_css)
        self.assertIn(".revision-replay-visual img[hidden]", app_css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", app_css)
        self.assertIn("function revisionCompareVisual", app_source)
        self.assertIn("function renderRevisionImageDifference", app_source)
        self.assertIn("/revision-geometry?base=", app_source)
        self.assertIn('q.set("compareRun", state.revisionReplayCompareRunId)', app_source)
        self.assertIn("harness 派生碰撞网格记录，不是 STEP 布尔差", app_source)
        self.assertIn("function exportRevisionReplayVideo", app_source)
        self.assertIn("function cacheRevisionVideoFrames", app_source)
        self.assertIn('mediaRecorderStateChange(recorder, "pause", "pause")', app_source)
        self.assertIn("canvas.captureStream(30)", app_source)
        self.assertIn("画面未变，仅验证状态变化", app_source)
        self.assertIn("▶ 播放 ${playbackFrames.length} 个画面", app_source)
        self.assertIn("const frames = revisionReplayFrames(project);", app_source)
        self.assertIn("const frames = revisionReplayPlaybackFrames(project);", app_source)
        self.assertIn("function animateRevisionVideoWindow", app_source)
        self.assertIn(
            "recordingStartedAt + recordingPauseOffsetMs + (index + 1) * 1000",
            app_source,
        )
        self.assertNotIn("Promise.all(milestones.map(loadRevisionVideoImage))", app_source)
        self.assertIn('method: "POST"', app_source)
        self.assertIn("/revision-video", app_source)
        self.assertIn(".revision-compare-viewport", app_css)
        self.assertIn(".revision-geometry-counts", app_css)
        self.assertIn(".revision-replay-stack-image.is-previous", app_css)
        self.assertIn(".revision-frame-sequence", app_css)
        self.assertIn(".revision-tick.no-frame", app_css)
        self.assertIn(".revision-tick.same-frame", app_css)
        self.assertIn("selectedTick.offsetLeft", app_source)
        self.assertIn("function revisionMeshApiUrl", app_source)
        self.assertIn("/revision-mesh?run=", app_source)
        self.assertIn("/revision-frame?run=", app_source)
        self.assertIn("function revisionInteractiveMesh", app_source)
        self.assertIn("state.revisionReplayPlaying", app_source)
        self.assertIn("window.Quest3D?.mountAll(replacement)", app_source)


if __name__ == "__main__":
    unittest.main()
