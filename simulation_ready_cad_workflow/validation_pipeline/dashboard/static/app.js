(() => {
  "use strict";

  const boot = window.QUEST_BOOT || { mode: "snapshot", data: null };
  // Optional server surfaces: legacy maintenance, team snapshots, run launching.
  const features = boot.features || {};
  const app = document.getElementById("app");
  const wb = window.Workbench;
  // Language switch (static/i18n.js): English mode translates the rendered DOM;
  // tr() covers text that never reaches it (dialogs, notifications, canvas).
  const i18n = window.i18n || { lang: "zh", t: (text) => text, toggleMarkup: () => "" };
  const tr = (text) => i18n.t(text);
  let savedNotices = {}; try { savedNotices = JSON.parse(localStorage.getItem("workbench-notice-seen") || "{}"); } catch {}
  const noticeTracker = new wb.Notices(savedNotices);
  const humanDecisionCapability = new URLSearchParams(
    window.location.hash.slice(1),
  ).get("human") || "";
  const dashboardSurface = window.location.pathname.slice((boot.basePath || "").length).startsWith("/maintenance")
    ? "loop-loop"
    : "loop";
  const requestedStudio=new URLSearchParams(window.location.search).get("studio");
  const state = {
    runs: [],
    projects: [],
    teamLibrary: {items: [], local_url: '/'},
    data: boot.data || null,
    runId: boot.data?.run?.id || "",
    stageId: boot.data?.run?.default_stage_id || "",
    loadId: boot.data?.load_cases?.[0]?.id || "",
    mechanismId: boot.data?.mechanisms?.[0]?.id || "",
    evidenceId: "",
    evidenceSearch: "",
    evidenceScope: "stage",
    physicsBranch: "physical",
    physicsFrame: 0,
    physicsLayer: "selected",
    physicsPart: "",
    physicsFocusEvidenceId: "",
    stale: false,
    loading: false,
    liveSignature: "",
    catalogSignature: "",
    selectionMode: "follow_current",
    revisionReplayOpen: false,
    revisionReplayRunId: "",
    revisionReplayPreviousRunId: "",
    revisionReplayPlaying: false,
    revisionReplayCompare: false,
    revisionReplayCompareRunId: "",
    revisionReplayCompareMode: "wipe",
    revisionReplayWipe: 50,
    revisionReplayVideoExporting: false,
    resourceReviewContinuation: null,
    surface: dashboardSurface,
    roleSessions: [],
    roleSessionsLoading: false,
    roleSessionsError: "",
    roleSessionsUpdatedAt: "",
    roleSessionsSignature: "",
    roleSessionRole: "worker",
    eventKernel: { status: "NOT_STARTED" },
    studioPanel: "versions",
    studioDrawer: "activity",
    trajectory: null, trajectoryRun: "", trajectoryTurn: "", trajectoryBefore: null,
    trajectoryLoading: false, trajectoryError: "", trajectoryFollow: true, trajectoryViewKey: "",
    studioMode: ["design","motion","physics"].includes(requestedStudio) ? requestedStudio : new URLSearchParams(window.location.search).get("display")==="mechanics" ? "motion" : "design",
    studioMotionId: new URLSearchParams(window.location.search).get("motion") || "",
    studioAssetId: "",
    studioSidebarOpen: true,
    designFramePinned: "",
    trajectoryTimeWindow: null,
    historyScope: new URLSearchParams(window.location.search).get("history")==="project" ? "project" : "revision",
    history: null, historyRun: "", historyPosition: null, historySource: null, historySourceName: "",
    historySourceOffset: 0, historySourceLoading: false, historyError: "", historyWidth: 420, mobileHistoryPane: "model",
    projectSearch: "", projectStatusFilter: "all", projectRailCollapsed: false,
    summaries: {}, summaryReady: false, trajectorySearch: "", notificationsEnabled: false,
    thumbnails: {}, compareA: null, compareB: null, compareOpen: false, compareView: "iso", compareData: null, compareWipe: 50, markerNotice: null,
    config: null, configKey: "", configLoading: false,
    historyPlaying: false, historyInterval: 1000, activity: null,
    progressConnected: false,
    progressReceivedAt: 0,
    progressWorkflowReceivedAt: 0,
    launchStatus: null,
  };
  let liveWatchController = null;
  let launchStatusTimer = null;
  let launchStatusSerial = 0;
  const teamFolders = new Map();
  document.addEventListener('toggle', (event) => {
    if (event.target.matches?.('details[data-team-folder]')) {
      teamFolders.set(event.target.dataset.teamFolder, event.target.open);
    }
  }, true);
  function isImportedSnapshot(workflow = state.data?.live_workflow) {
    return workflow?.stop_reason === "imported_snapshot" || String(boot.basePath || "").includes("/imports/");
  }
  let liveWatchGeneration = 0;
  let liveRefreshInFlight = false;
  let loadRequestSerial = 0;
  let workspaceSwitchSerial = 0;
  let summaryGeneration = 0;
  let revisionReplayTimer = null;
  let revisionReplayRenderSerial = 0;
  const revisionGeometryCache = new Map();
  const revisionFrameObjectUrls = new Map();
  let revisionFrameCacheProjectId = "";
  let revisionFrameCacheSignature = "";
  let revisionFrameCachePromise = null;
  let revisionFrameCacheGeneration = 0;
  let roleSessionTimer = null;
  let roleSessionController = null;
  let roleSessionGeneration = 0;
  const roleSessionScrollPositions = new Map();
  let progressTimer = null;
  let progressController = null;
  let progressGeneration = 0;
  let timelineCommitTimer = null;
  let historyController = null;
  let historyRequestKey = "";
  let sourceRequestSerial = 0;
  let studioClockTimer = null;
  let summaryTimer = null;
  let summaryPending = false;
  let historyPlayTimer = null;
  let historySeekSerial = 0;
  let configController = null;
  let thumbnailObserver = null;
  const visibleProjects = new Set();
  const thumbnailPending = new Set();
  let compareTimer = null;
  let compareController = null;
  const trajectoryExpanded = new Set();

  const statusWords = {
    passed: "PASS", evidence_complete: "EVIDENCE_COMPLETE",
    evidence_incomplete: "EVIDENCE_INCOMPLETE",
    warning: "通过 · 有提醒", running: "进行中",
    awaiting_review: "等待审批",
    failed: "失败", blocked: "阻塞", stale: "已失效", not_started: "未闭环",
  };
  const shortStatus = {
    passed: "通过", evidence_complete: "证据齐全",
    evidence_incomplete: "证据不全",
    warning: "通过 · 有提醒", running: "进行中",
    awaiting_review: "待审批",
    failed: "失败", blocked: "阻塞", stale: "已失效", not_started: "未完成",
  };
  const evidenceHealthWords = {
    complete: "证据完整",
    incomplete: "证据未完成",
    stale: "证据失效",
    conflict: "证据冲突",
    legacy_unverified: "待复核",
  };
  const executionLaneWords = {
    QUEUED: "等待调度",
    WAITING_FOR_RESOURCE: "等待内存/资源",
    RUNNING: "运行中",
    PASSED: "通过",
    FAILED: "失败",
    CANCELLED: "已停止",
  };
  const revisionReasonWords = {
    TASK_CERTIFIED: "机构任务已认证",
    WHOLE_MACHINE_TASK_PASSED: "整机任务已通过",
    PHYSICAL_PREFLIGHT_FAILED: "物理预检未通过",
    PHYSICAL_DISPATCH_PREFLIGHT_FAILED: "物理任务派发预检未通过",
    PHYSICAL_SIMULATION_FAILED: "真实物理仿真未通过",
    PHYSICAL_BRANCH_FAILED: "真实物理分支未通过",
    INTAKE_REJECTED: "进入仿真前检查未通过",
    DESIGN_QUALITY_FAILED: "设计审查未通过",
    DESIGN_QUALITY_BLOCKED: "设计审查阻塞",
    RUNNING: "正在生成或验证",
    ITERATION_SCOPE_COMPLETE: "本轮定向检查完成，继续迭代",
    COMPLETED_WITH_TIMEBOX: "限时执行已结束，合同证据尚未完成",
    RESOURCE_BLOCKED: "资源暂时不足，等待自动续跑",
    REPEATED_ERROR_PAUSED: "同一错误连续出现 3 次，Worker 已暂停",
    SUPERSEDED: "已被后续版本替代",
    FAILED: "本版本未通过",
  };
  const loadPurpose = {
    nominal_light: "轻质量推拉，确认基本开合",
    nominal_loaded: "较重质量推拉，确认带载开合",
    zero_input_gravity: "不施加外力，确认不会自己滑动",
    transverse_probe: "从侧面施力，确认机构不会歪斜",
    held_out_mid: "用未参与调参的中等条件检查泛化",
    task_envelope_roundtrip: "完整打开再关闭，检查全行程任务",
  };
  const rackLoadPurpose = {
    nominal_light: "轻载手摇升降，确认齿轮齿条能顺畅带动闸门",
    nominal_loaded: "提高负载手摇升降，确认限位与啮合仍可靠",
    zero_input_gravity: "不摇动手柄，确认闸门不会在重力下自己下滑",
    held_out_mid: "用中等扭矩升降，检查不同负载下仍可靠",
    task_rack_roundtrip: "完整升起再降下，检查整段行程和回原位",
    transverse_rack_probe: "从非工作方向施加扰动，确认闸门不会误升降",
  };
  const angularLoadPurpose = {
    nominal_light: "轻质量开合，确认铰链能正常转动",
    nominal_loaded: "较重质量开合，确认带载仍能转动",
    zero_input_gravity: "不施加扭矩，确认机构不会自行转开",
    transverse_torque_probe: "从非目标方向施加扭矩，确认转轴不会歪斜",
    held_out_mid: "用未参与调参的中等条件检查泛化",
    task_angle_roundtrip: "完整打开再关闭，检查全角度任务",
  };
  const familyWords = { prismatic: "直线滑动", revolute: "绕轴转动", compound: "复合机构", fixed: "固定连接" };
  const clockLoadPurpose = {
    nominal_light: "轻载驱动整套齿轮系，检查三针和铃锤传动",
    nominal_loaded: "提高轴系负载，检查带载走时是否仍稳定",
    zero_input_gravity: "不施加扭矩，检查表针不会自行明显走动",
    held_out_mid: "用未参与调参的中等条件检查泛化",
    shaft_transverse_probe: "从非工作方向扰动轴系，检查支撑是否稳固",
    task_clock_cycle: "正常走时，并同时核对时针、分针、秒针比例",
    task_manual_time_setting: "只转后冠，检查表针能被离合与齿轮系带动",
    task_alarm_cycle: "只驱动发条轴，检查铃锤能经传动自行摆动",
  };
  const shield = '<svg viewBox="0 0 28 32" fill="none" aria-hidden="true"><path d="M14 2 25 6v9c0 7.2-4.6 12.8-11 16-6.4-3.2-11-8.8-11-16V6l11-4Z" stroke="currentColor" stroke-width="2"/><path d="m8.5 16 3.2 3.2 7.2-7.3" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const fileIcon = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 3h8l4 4v14H6V3Z" stroke="currentColor" stroke-width="1.7"/><path d="M14 3v5h5M9 13h6m-6 3h6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';

  const esc = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const statusLabel = (status, compact = false) => (compact ? shortStatus : statusWords)[status] || String(status || "未知");
  const mark = (status) => ({ passed: "✓", evidence_complete: "◆", evidence_incomplete: "□", warning: "!", awaiting_review: "!", failed: "×", blocked: "■", running: "↻", stale: "⌛", not_started: "○" }[status] || "○");
  const stage = () => state.data?.stages?.find((item) => item.id === state.stageId) || state.data?.stages?.[0] || null;
  const stageKind = () => stage()?.kind || stage()?.id || "";
  const scopedLoads = () => {
    const all = state.data?.load_cases || [];
    const ids = stage()?.scope?.load_case_ids || [];
    const capsuleId = stage()?.scope?.capsule_id || "";
    return all.filter((item) =>
      (!capsuleId || item.capsule_id === capsuleId)
      && (!ids.length || ids.includes(item.id))
    );
  };
  const load = () => scopedLoads().find((item) => item.id === state.loadId) || scopedLoads()[0] || null;
  const mechanism = () => state.data?.mechanisms?.find((item) => item.id === state.mechanismId) || state.data?.mechanisms?.[0] || null;
  const ensureScopedSelection = () => {
    const current = stage();
    const loadIds = current?.scope?.load_case_ids || [];
    if (loadIds.length && !loadIds.includes(state.loadId)) state.loadId = loadIds[0];
    const capsuleId = current?.scope?.capsule_id || "";
    if (capsuleId && state.data?.mechanisms?.some((item) => item.id === capsuleId)) {
      state.mechanismId = capsuleId;
    }
  };
  const physicsEvidenceForLoad = (loadItem) => {
    const loadId = loadItem?.id || "";
    const capsuleId = loadItem?.capsule_id || stage()?.scope?.capsule_id || "";
    const evidence = state.data?.evidence || [];
    const replayArtifact = evidence.find((item) => {
      const detail = item.details || {};
      return detail.visual_role === "physics-evidence-replay"
        && detail.load_case_id === loadId
        && (!capsuleId || detail.capsule_id === capsuleId);
    }) || null;
    const physical = replayArtifact?.details?.replay?.branches?.physical || null;
    const frameCount = Array.isArray(physical?.frames)
      ? physical.frames.length
      : Number(physical?.retained_frame_count || 0);
    const video = evidence.find((item) => {
      const detail = item.details || {};
      return detail.load_case_id === loadId
        && (!capsuleId || detail.capsule_id === capsuleId)
        && item.preview_type === "video"
        && [
          "physical-diagnostic-video",
          "physical-video",
          "load-comparison-video",
        ].includes(detail.visual_role);
    }) || null;
    return { replayArtifact, frameCount, video };
  };
  const selectBestPhysicsEvidenceLoad = () => {
    if (stageKind() !== "branch-simulation") return false;
    const available = scopedLoads().map((item) => {
      const evidence = physicsEvidenceForLoad(item);
      return {
        item,
        evidence,
        score: (evidence.frameCount > 0 ? 2 : 0) + (evidence.video ? 1 : 0),
      };
    });
    const current = available.find(({ item }) => item.id === state.loadId);
    if ((current?.score || 0) > 0) return false;
    const preferred = available.sort((left, right) => right.score - left.score)[0];
    if (!preferred?.score || preferred.item.id === state.loadId) return false;
    state.loadId = preferred.item.id;
    state.evidenceId = "";
    state.physicsBranch = "physical";
    state.physicsFrame = 0;
    state.physicsPart = "";
    state.physicsFocusEvidenceId = "";
    return true;
  };
  const projectForRun = (runId) => state.projects.find((item) => (item.runs || []).some((run) => run.id === runId)) || null;
  const buildProjectCatalog = (runs) => {
    const grouped = new Map();
    (runs || []).forEach((run) => {
      const projectId = run.project_id || run.package_id || run.id;
      if (!grouped.has(projectId)) grouped.set(projectId, []);
      grouped.get(projectId).push(run);
    });
    return [...grouped.entries()].map(([id, versions]) => {
      const current = versions[0] || {};
      return {
        id,
        label: current.package_id || current.project_id || id,
        current_run_id: current.id || "",
        default_run_id: current.id || "",
        latest_run_id: current.id || "",
        published_run_id: current.status === "passed" ? current.id || "" : "",
        last_pass_run_id: current.status === "passed" ? current.id || "" : "",
        latest: current,
        published: current.status === "passed" ? current : null,
        last_pass: current.status === "passed" ? current : null,
        status: current.status || "not_started",
        version_count: versions.length,
        runs: versions,
      };
    });
  };
  const applyRunCatalog = (payload) => {
    state.teamLibrary = payload.team_library || state.teamLibrary;
    const signature = JSON.stringify({
      projects: (payload?.projects || []).map((item) => [
        item.id,
        item.default_run_id || item.current_run_id,
        item.latest_run_id,
        item.last_pass_run_id || item.published_run_id,
        item.latest_status,
        item.status,
        item.version_count,
      ]),
      runs: (payload?.runs || []).map((item) => [
        item.id,
        item.status,
        item.updated_at,
        item.revision_preview_path,
        item.revision_preview_sha256,
      ]),
    });
    const changed = signature !== state.catalogSignature;
    state.runs = payload?.runs || [];
    state.projects = payload?.projects?.length ? payload.projects : buildProjectCatalog(state.runs);
    state.catalogSignature = signature;
    return changed;
  };
  const loadDescription = (item) => {
    const clock = item?.mechanism_kind === "clock" || String(item?.program || "").includes("clock") || String(item?.id || "").includes("clock") || String(item?.id || "").includes("alarm");
    if (clock) return clockLoadPurpose[item?.id] || item?.purpose || "未说明";
    const rackPinion = String(item?.program || "").includes("rack_pinion") || String(item?.id || "").includes("rack_");
    if (rackPinion) return rackLoadPurpose[item?.id] || item?.purpose || "未说明";
    return ((item?.coordinate_kind === "angular" ? angularLoadPurpose : loadPurpose)[item?.id] || item?.purpose || "未说明");
  };

  function fmt(value, unit = "") {
    if (value == null || value === "") return "—";
    const shown = typeof value === "number" ? (Math.abs(value) >= 100 ? value.toFixed(1) : Number(value.toFixed(4)).toString()) : String(value);
    return shown + (unit ? ` ${unit}` : "");
  }

  function fmtTime(value) {
    if (!value) return "未知";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
  }
  function formatProjectDuration(value) {
    const seconds = Math.max(0, Math.round(Number(value) || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    if (hours) return `${hours}时${minutes}分${remainder ? `${remainder}秒` : ""}`;
    if (minutes) return `${minutes}分${remainder ? `${remainder}秒` : ""}`;
    return `${remainder}秒`;
  }

  function formatProjectTokens(value, compact = false) {
    const tokens = Math.max(0, Math.round(Number(value) || 0));
    if (!compact) return tokens.toLocaleString("zh-CN");
    if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(2)}M`;
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
    return String(tokens);
  }

  function projectUsagePanel(usage) {
    const record = usage || { status: "unrecorded" };
    const status = record.status || "unrecorded";
    if (status !== "recorded") {
      const unavailable = status === "unavailable";
      const label = unavailable ? "记录不可用" : "尚未记录";
      const explanation = unavailable
        ? "项目投入文件存在，但未通过一致性校验，因此不显示可能错误的数字。"
        : "这个机械还没有项目级耗时与 Token 记录；看板不会把缺失记录显示成 0。";
      const issueRows = (record.issues || []).map((item) => `<li>${esc(item)}</li>`).join("");
      return `<details class="project-usage" data-status="${esc(status)}">
        <summary title="查看项目累计投入说明"><span class="project-usage-label">累计投入</span><strong>${label}</strong><span class="usage-chevron">⌄</span></summary>
        <div class="project-usage-popover"><div class="usage-popover-head"><div><span>整个机械项目</span><h3>${label}</h3></div><b>${unavailable ? "校验失败" : "无记录"}</b></div><p>${explanation}</p>${issueRows ? `<ul class="usage-issues">${issueRows}</ul>` : ""}<small>预期记录：${esc(record.source_path || ".runtime_runs/project_usage/projects/<package_id>.json")}</small></div>
      </details>`;
    }

    const completed = Number(record.completed_session_count || 0);
    const running = Number(record.running_session_count || 0);
    const pendingTokens = Number(record.pending_token_session_count || 0);
    const currentRevision = record.current_revision || "—";
    const revisionCount = Number(record.revision_count || 0);
    const recordedRevisionCount = Number(record.recorded_revision_count || 0);
    const missingRevisionCount = Number(record.missing_revision_count || 0);
    const runningWallTime = (record.running_sessions || []).reduce((total, item) => {
      const started = Date.parse(item.started_at || "");
      return total + (Number.isFinite(started) ? Math.max(0, (Date.now() - started) / 1000) : 0);
    }, 0);
    const totalTime = formatProjectDuration(Number(record.total_wall_time_seconds || 0) + runningWallTime);
    const benchmarkTokens = record.benchmark_total_tokens;
    const totalTokens = benchmarkTokens == null ? "不可比" : formatProjectTokens(benchmarkTokens);
    const tokenSource = record.benchmark_token_counter_source || "无统一来源";
    const mixedTokenSources = Boolean(record.mixed_token_counter_sources);
    const unallocatedBackfill = Number(record.unallocated_backfill_tokens || 0);
    const completedRows = (record.completed_sessions || []).map((item) => `<tr>
      <td><span>${esc(item.run_id || "未绑定版本")}</span><small>${esc(item.session_id)}</small></td>
      <td>${esc(formatProjectDuration(item.wall_time_seconds))}</td>
      <td title="${esc(formatProjectTokens(item.total_tokens))} Token"><span>${esc(formatProjectTokens(item.total_tokens, true))}</span><small>${esc(item.token_counter_source || "未知口径")}</small></td>
      <td>${esc(fmtTime(item.completed_at))}</td>
    </tr>`).join("");
    const runningRows = (record.running_sessions || []).map((item) => `<tr class="usage-running-row"><td><span>记录中</span><small>${esc(item.run_id || item.session_id)}</small></td><td>${esc(formatProjectDuration(Math.max(0, (Date.now() - Date.parse(item.started_at || "")) / 1000)))}</td><td title="${esc(formatProjectTokens(item.total_tokens))} Token">${esc(formatProjectTokens(item.total_tokens, true))}</td><td>${esc(fmtTime(item.started_at))}</td></tr>`).join("");
    const revisionRows = (record.revision_records || []).map((item) => {
      const usageStatus = item.usage_status || "missing";
      const detail = item.run_status && item.run_status !== "UNFINISHED" ? item.run_status : "运行未结束";
      if (usageStatus === "recorded") return `<tr>
        <td><span>${esc(item.revision)}</span><small>${esc(detail)}</small></td>
        <td>${esc(formatProjectDuration(item.wall_time_seconds))}</td>
        <td title="${esc(formatProjectTokens(item.total_tokens))} Token"><span>${esc(formatProjectTokens(item.total_tokens, true))}</span><small>${esc((item.token_counter_sources || []).join(", ") || "未知口径")}</small></td>
        <td>${esc(fmtTime(item.completed_at))}</td>
      </tr>`;
      if (usageStatus === "running") return `<tr class="usage-running-row">
        <td><span>${esc(item.revision)}</span><small>${esc(detail)}</small></td>
        <td>成本记录中</td><td>${esc(formatProjectTokens(item.total_tokens, true))}</td><td>${esc(fmtTime(item.started_at))}</td>
      </tr>`;
      if (usageStatus === "awaiting_token") return `<tr class="usage-running-row">
        <td><span>${esc(item.revision)}</span><small>${esc(detail)}</small></td>
        <td>${esc(formatProjectDuration(item.wall_time_seconds))}</td>
        <td>Token 待结算</td><td>${esc(fmtTime(item.completed_at))}</td>
      </tr>`;
      return `<tr class="usage-missing-row">
        <td><span>${esc(item.revision)}</span><small>${esc(detail)}</small></td>
        <td colspan="3">该版本未记录耗时和 Token</td>
      </tr>`;
    }).join("");
    return `<details class="project-usage" data-status="recorded">
      <summary title="点击查看每轮耗时与 Token">
        <span class="project-usage-label">累计投入${running ? `<i>记录中</i>` : ""}</span>
        <span class="usage-mini"><small>耗时</small><strong>${esc(totalTime)}</strong></span>
        <span class="usage-mini" title="${esc(totalTokens)} Token；${esc(tokenSource)}"><small>Token</small><strong>${benchmarkTokens == null ? "不可比" : esc(formatProjectTokens(benchmarkTokens, true))}</strong></span>
        <span class="usage-mini"><small>版本</small><strong>${esc(currentRevision)}</strong></span>
        <span class="usage-chevron">⌄</span>
      </summary>
      <div class="project-usage-popover">
        <div class="usage-popover-head"><div><span>整个机械项目累计</span><h3>${esc(record.package_id || record.project_id || "当前机械")}</h3></div><b>${pendingTokens ? `${pendingTokens} 个版本 Token 待结算` : (missingRevisionCount ? `${missingRevisionCount} 个版本缺记录` : (mixedTokenSources ? "存在旧 Token 口径" : (running ? `${running} 次记录中` : "记录完整")))}</b></div>
        <div class="usage-total-grid"><div><span>累计耗时</span><strong>${esc(totalTime)}</strong></div><div><span>可比工作 Token</span><strong>${esc(totalTokens)}</strong><small>${esc(tokenSource)} · 非计费 Token</small></div><div><span>版本成本记录</span><strong>${recordedRevisionCount}/${revisionCount || completed}</strong></div></div>
        ${unallocatedBackfill ? `<p class="usage-backfill-note">其中 ${esc(formatProjectTokens(unallocatedBackfill))} Token 来自启用逐版本记录前的项目历史，只计入项目总量，不归属任何 rxx。</p>` : ""}
        <div class="usage-session-scroll"><table><thead><tr><th>运行版本</th><th>耗时</th><th>Token</th><th>完成时间</th></tr></thead><tbody>${revisionRows || runningRows + completedRows || `<tr><td colspan="4">尚无运行版本</td></tr>`}</tbody></table></div>
        <small>Token 只汇总同一计数来源；旧口径保留但不混加。每条记录绑定具体 rxx · 来源：${esc(record.source_path || "项目投入记录")}</small>
      </div>
    </details>`;
  }

  function revisionOrdinal(item) {
    const revision = String(item?.revision || item?.name || "");
    const match = /^r(\d+)$/i.exec(revision);
    if (match) return Number(match[1]);
    const updated = new Date(item?.updated_at || "").getTime();
    return Number.isFinite(updated) ? updated : 0;
  }

  function revisionReplayProject() {
    return projectForRun(state.revisionReplayRunId)
      || projectForRun(state.runId)
      || state.projects[0]
      || null;
  }

  function revisionReplayVersions(project = revisionReplayProject()) {
    return [...(project?.runs || [])].sort((left, right) => (
      revisionOrdinal(left) - revisionOrdinal(right)
      || String(left.updated_at || "").localeCompare(String(right.updated_at || ""))
    ));
  }

  function revisionReplayFrames(project = revisionReplayProject()) {
    return revisionReplayVersions(project).filter(
      (item) => Boolean(item?.revision_preview_path),
    );
  }

  function revisionPreviewMatches(left, right) {
    const leftSha = String(left?.revision_preview_sha256 || "");
    const rightSha = String(right?.revision_preview_sha256 || "");
    return Boolean(leftSha && rightSha && leftSha === rightSha);
  }

  function revisionReplayPlaybackFrames(project = revisionReplayProject()) {
    const frames = revisionReplayFrames(project);
    return frames.filter(
      (item, index) => index === 0 || !revisionPreviewMatches(item, frames[index - 1]),
    );
  }

  function revisionReplayStepNumbers(frames) {
    const numbers = new Map();
    let step = 0;
    frames.forEach((item, index) => {
      if (index === 0 || !revisionPreviewMatches(item, frames[index - 1])) step += 1;
      numbers.set(item.id, step);
    });
    return numbers;
  }

  function revisionMilestoneIds(project = revisionReplayProject()) {
    const versions = revisionReplayVersions(project);
    if (versions.length <= 12) return new Set(versions.map((item) => item.id));
    const target = 12;
    const chosen = new Set([0, versions.length - 1]);
    const importantRunIds = new Set([
      state.runId,
      state.revisionReplayRunId,
      project?.published_run_id,
      project?.last_pass_run_id,
      project?.current_run_id,
      project?.latest_run_id,
    ].filter(Boolean));
    versions.forEach((item, index) => {
      if (importantRunIds.has(item.id)) chosen.add(index);
    });
    const transitionIndices = new Set();
    versions.forEach((item, index) => {
      const previous = versions[index - 1];
      if (previous && previous.status !== item.status) transitionIndices.add(index);
    });
    while (chosen.size < target) {
      let bestIndex = -1;
      let bestScore = -1;
      versions.forEach((item, index) => {
        if (chosen.has(index)) return;
        const nearest = Math.min(...[...chosen].map((selected) => Math.abs(selected - index)));
        const transitionBonus = transitionIndices.has(index) ? versions.length * 0.18 : 0;
        const score = nearest + transitionBonus;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = index;
        }
      });
      if (bestIndex < 0) break;
      chosen.add(bestIndex);
    }
    return new Set([...chosen].map((index) => versions[index].id));
  }

  function revisionReason(item) {
    const raw = String(item?.status_raw || item?.verdict?.reason_code || "").trim();
    return revisionReasonWords[raw] || statusLabel(item?.status, true);
  }

  function revisionPreviewApiUrl(item) {
    if (boot.mode !== "live" || !item?.id || !item?.revision_preview_path) return "";
    return `${boot.apiBase}/revision-frame?run=${encodeURIComponent(item.id)}`;
  }

  function revisionMeshApiUrl(item) {
    if (boot.mode !== "live" || !item?.id) return "";
    return `${boot.apiBase}/revision-mesh?run=${encodeURIComponent(item.id)}`;
  }

  function revisionPreviewUrl(item) {
    return revisionFrameObjectUrls.get(item?.id) || revisionPreviewApiUrl(item);
  }

  async function ensureRevisionFrameCache(project, onProgress = null) {
    const projectId = String(project?.id || "");
    const frames = revisionReplayFrames(project);
    const signature = frames.map(
      (item) => `${item.id}:${item.revision_preview_path}:${item.revision_preview_sha256 || ""}`,
    ).join("|");
    if (!projectId || !frames.length) return [];
    if (state.revisionReplayVideoExporting && revisionFrameCachePromise) {
      await revisionFrameCachePromise;
      return frames.map(
        (item) => revisionFrameObjectUrls.get(item.id) || null,
      );
    }
    if (
      revisionFrameCacheProjectId === projectId
      && revisionFrameCacheSignature === signature
      && revisionFrameCachePromise
    ) {
      const cached = await revisionFrameCachePromise;
      onProgress?.(cached.filter(Boolean).length, frames.length);
      return cached;
    }
    revisionFrameCacheGeneration += 1;
    const generation = revisionFrameCacheGeneration;
    revisionFrameObjectUrls.forEach((url) => URL.revokeObjectURL(url));
    revisionFrameObjectUrls.clear();
    revisionFrameCacheProjectId = projectId;
    revisionFrameCacheSignature = signature;
    const selectedIndex = Math.max(
      0,
      frames.findIndex((item) => item.id === state.revisionReplayRunId),
    );
    const orderedFrames = [...frames].sort((left, right) => (
      Math.abs(frames.indexOf(left) - selectedIndex)
      - Math.abs(frames.indexOf(right) - selectedIndex)
    ));
    revisionFrameCachePromise = (async () => {
      let cursor = 0;
      let completed = 0;
      const worker = async () => {
        while (cursor < orderedFrames.length) {
          const item = orderedFrames[cursor];
          cursor += 1;
          let objectUrl = "";
          try {
            const response = await fetch(revisionPreviewApiUrl(item), {
              cache: "no-store",
            });
            if (!response.ok) throw new Error("frame unavailable");
            objectUrl = URL.createObjectURL(await response.blob());
          } catch (_error) {
            objectUrl = "";
          }
          if (generation === revisionFrameCacheGeneration) {
            if (objectUrl) revisionFrameObjectUrls.set(item.id, objectUrl);
          } else if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
          }
          completed += 1;
          onProgress?.(completed, frames.length);
        }
      };
      await Promise.all(
        Array.from(
          { length: Math.min(8, orderedFrames.length) },
          () => worker(),
        ),
      );
      return frames.map((item) => revisionFrameObjectUrls.get(item.id) || null);
    })();
    return revisionFrameCachePromise;
  }

  function revisionCompareBase(versions, selected) {
    const selectedIndex = versions.findIndex((item) => item.id === selected?.id);
    const explicitIndex = versions.findIndex(
      (item) => item.id === state.revisionReplayCompareRunId,
    );
    if (explicitIndex >= 0 && explicitIndex < selectedIndex) return versions[explicitIndex];
    return versions[selectedIndex - 1] || null;
  }

  function revisionGeometryCacheKey(baseId, targetId) {
    return `${baseId || ""}::${targetId || ""}`;
  }

  function revisionGeometryItemNames(items, limit = 3) {
    const names = (items || []).slice(0, limit).map((item) => item.key).filter(Boolean);
    const remaining = Math.max(0, (items || []).length - names.length);
    return `${names.map((name) => `<code>${esc(name)}</code>`).join("")}${remaining ? `<small>另 ${remaining} 项</small>` : ""}`;
  }

  function revisionGeometryProofMarkup(base, target) {
    if (!base || !target) return "";
    const cached = revisionGeometryCache.get(revisionGeometryCacheKey(base.id, target.id));
    if (!cached || cached.status === "loading") {
      return `<section class="revision-geometry-proof is-loading" aria-live="polite"><div><span>碰撞几何证据</span><strong>正在核对组件哈希…</strong></div></section>`;
    }
    if (cached.status === "error" || !cached.data?.available) {
      return `<section class="revision-geometry-proof is-unavailable" aria-live="polite"><div><span>碰撞几何证据</span><strong>这两个版本没有可比清单</strong><small>图片仍可划屏查看，但不会推断实体增删。</small></div></section>`;
    }
    const data = cached.data;
    const summary = data.summary || {};
    return `<section class="revision-geometry-proof" aria-live="polite">
      <header><div><span>碰撞网格记录变化</span><strong>${esc(base.revision || base.name)} → ${esc(target.revision || target.name)}</strong></div><small>组件引用 + SHA-256 + 装配位姿</small></header>
      <div class="revision-geometry-counts">
        <div data-change="added"><b>+${Number(summary.added || 0)}</b><span>新增</span></div>
        <div data-change="removed"><b>−${Number(summary.removed || 0)}</b><span>删除</span></div>
        <div data-change="modified"><b>△${Number(summary.modified || 0)}</b><span>修改</span></div>
        <div data-change="unchanged"><b>${Number(summary.unchanged || 0)}</b><span>未变</span></div>
      </div>
      <details>
        <summary>查看组件引用</summary>
        <div class="revision-geometry-items">
          ${summary.added ? `<div data-change="added"><b>新增</b>${revisionGeometryItemNames(data.added)}</div>` : ""}
          ${summary.removed ? `<div data-change="removed"><b>删除</b>${revisionGeometryItemNames(data.removed)}</div>` : ""}
          ${summary.modified ? `<div data-change="modified"><b>修改</b>${revisionGeometryItemNames(data.modified)}</div>` : ""}
          ${!summary.added && !summary.removed && !summary.modified ? "<p>碰撞实体及其哈希完全一致。</p>" : ""}
        </div>
      </details>
      <p>这是 harness 派生碰撞网格记录，不是 STEP 布尔差；组件改名或重排会显示为删除和新增。</p>
    </section>`;
  }

  function revisionCompareVisual(selected, base, targetUrl) {
    const baseUrl = revisionPreviewUrl(base);
    const hasPair = Boolean(baseUrl && targetUrl);
    const mode = state.revisionReplayCompareMode === "difference" ? "difference" : "wipe";
    return `<figure class="revision-replay-visual is-comparing" data-status="${esc(selected.status)}">
      <div class="revision-compare-viewport" data-compare-mode="${mode}" style="--revision-wipe:${Number(state.revisionReplayWipe || 50)}%">
        ${hasPair ? `
          <img id="revision-compare-base-image" class="revision-preview-image revision-compare-base" src="${esc(baseUrl)}" alt="${esc(`${base.revision || base.name} 固定机位设计视图`)}">
          <div class="revision-compare-target-layer"><img id="revision-compare-target-image" class="revision-preview-image" src="${esc(targetUrl)}" alt="${esc(`${selected.revision || selected.name} 固定机位设计视图`)}"></div>
          <canvas id="revision-difference-canvas" aria-label="固定机位图片变化辅助高亮" hidden></canvas>
          <span class="revision-compare-tag is-base">A · ${esc(base.revision || base.name)}</span>
          <span class="revision-compare-tag is-target">B · ${esc(selected.revision || selected.name)}</span>
          <span class="revision-compare-divider" aria-hidden="true"></span>
        ` : ""}
        <div class="revision-replay-placeholder" ${hasPair ? "hidden" : ""}><strong>需要两个固定机位画面</strong><span>组件哈希仍会照常核对，不用其他版本图片代替。</span></div>
      </div>
      <div class="revision-compare-controls">
        <div class="revision-compare-modes" aria-label="对比方式">
          <button type="button" data-compare-mode="wipe" aria-pressed="${mode === "wipe" ? "true" : "false"}">划屏</button>
          <button type="button" data-compare-mode="difference" aria-pressed="${mode === "difference" ? "true" : "false"}">变化热区</button>
        </div>
        <label class="revision-wipe-control" ${mode === "difference" ? "hidden" : ""}><span>A</span><input id="revision-wipe-range" type="range" min="5" max="95" value="${Number(state.revisionReplayWipe || 50)}" aria-label="A/B 划屏位置"><span>B</span></label>
        <small>${mode === "difference" ? "彩色区域是固定机位图片差异，仅用于定位；实体结论以右侧哈希统计为准。" : "拖动分界线查看同一固定机位的两个真实版本。"}</small>
      </div>
      <figcaption><strong>${esc(base.revision || base.name)} / ${esc(selected.revision || selected.name)}</strong><span>A/B 对比</span></figcaption>
    </figure>`;
  }

  function revisionInteractiveMesh(selected, fallbackUrl) {
    const meshUrl = revisionMeshApiUrl(selected);
    return `<figure class="revision-replay-visual revision-replay-interactive" data-status="${esc(selected.status)}">
      <div class="revision-replay-mesh evidence-3d" data-model-src="${esc(meshUrl)}" data-mode="design" role="img" aria-label="${esc(`${selected.revision || selected.name} 可交互 3D Mesh`)}，可拖动旋转、滚轮缩放、右键平移">
        ${fallbackUrl ? `<img class="evidence-3d-fallback" src="${esc(fallbackUrl)}" alt="${esc(`${selected.revision || selected.name} 固定机位备用图`)}">` : ""}
        <span class="viewer-loading" data-3d-status>正在载入该版本 Mesh…</span>
        <div class="viewer-legend"><span><i class="legend-fixed design"></i>固定结构</span><span><i class="legend-moving"></i>运动组件</span></div>
      </div>
      <figcaption><strong>${esc(selected.revision || selected.name)}</strong><span>${esc(statusLabel(selected.status))} · 可交互 Mesh</span></figcaption>
      <div class="revision-frame-sequence" data-frame-change="mesh"><strong>3D MESH</strong><span>拖拽旋转 · 滚轮缩放 · 右键平移</span></div>
    </figure>`;
  }

  function revisionReplayPanel() {
    if (!state.revisionReplayOpen || boot.mode !== "live") return "";
    const project = revisionReplayProject();
    const versions = revisionReplayVersions(project);
    const frames = revisionReplayFrames(project);
    const playbackFrames = revisionReplayPlaybackFrames(project);
    if (!project || !versions.length) return "";
    const selected = versions.find((item) => item.id === state.revisionReplayRunId)
      || versions.find((item) => item.id === state.runId)
      || versions[versions.length - 1];
    const selectedIndex = versions.findIndex((item) => item.id === selected.id);
    const frameIndex = frames.findIndex((item) => item.id === selected.id);
    const frameNumberById = revisionReplayStepNumbers(frames);
    const unchangedFrameIds = new Set(
      frames
        .filter((item, index) => index > 0 && revisionPreviewMatches(item, frames[index - 1]))
        .map((item) => item.id),
    );
    const selectedStep = frameNumberById.get(selected.id) || 0;
    const selectedFrameUnchanged = unchangedFrameIds.has(selected.id);
    const selectedOrdinal = revisionOrdinal(selected);
    const hasPreviousVersion = versions.some(
      (item) => revisionOrdinal(item) < selectedOrdinal,
    );
    const hasNextVersion = versions.some(
      (item) => revisionOrdinal(item) > selectedOrdinal,
    );
    const previousFrame = frames.find(
      (item) => item.id === state.revisionReplayPreviousRunId,
    );
    const previous = versions[selectedIndex - 1] || null;
    const milestoneIds = revisionMilestoneIds(project);
    const rawStatus = String(selected.status_raw || selected.verdict?.reason_code || "NOT_AVAILABLE");
    const previousRaw = String(previous?.status_raw || previous?.verdict?.reason_code || "");
    const versionChange = !previous
      ? "项目起始版本"
      : previousRaw === rawStatus
        ? `继续处理：${revisionReason(selected)}`
        : `${revisionReason(previous)} → ${revisionReason(selected)}`;
    const selectedRun = state.data?.run?.id === selected.id ? state.data.run : null;
    const progress = selectedRun?.progress
      ? `${selectedRun.progress.completed}/${selectedRun.progress.total}`
      : "打开详情查看";
    const comparisons = selectedRun?.comparisons
      ? `${selectedRun.comparisons.passed}/${selectedRun.comparisons.total}`
      : "—";
    const evidenceHealth = selected.evidence_health?.status_label
      || evidenceHealthWords[selected.evidence_health?.status]
      || evidenceHealthWords[selected.evidence_health]
      || "按正式终态";
    const previewUrl = revisionPreviewUrl(selected);
    const previousPreviewUrl = previousFrame?.id !== selected.id
      ? revisionPreviewUrl(previousFrame)
      : "";
    const compareBase = revisionCompareBase(versions, selected);
    const canCompare = Boolean(compareBase);
    const compareEnabled = Boolean(state.revisionReplayCompare && compareBase);
    const compareOptions = versions
      .slice(0, Math.max(0, selectedIndex))
      .map((item) => `<option value="${esc(item.id)}" ${item.id === compareBase?.id ? "selected" : ""}>A · ${esc(item.revision || item.name)} · ${esc(statusLabel(item.status, true))}</option>`)
      .join("");
    const ticks = versions.map((item) => {
      const milestone = milestoneIds.has(item.id);
      const active = item.id === selected.id;
      const frameNumber = frameNumberById.get(item.id) || 0;
      const hasFrame = frameNumber > 0;
      const unchangedFrame = unchangedFrameIds.has(item.id);
      const revision = item.revision || item.name || "未编号";
      const frameLabel = !hasFrame
        ? "无 3D 帧，"
        : unchangedFrame
          ? `画面未变，沿用 Step ${frameNumber}，`
          : `变化 Step ${frameNumber}，`;
      return `<button type="button" class="revision-tick ${milestone ? "milestone" : ""} ${hasFrame ? "has-frame" : "no-frame"} ${unchangedFrame ? "same-frame" : ""} ${active ? "selected" : ""}" data-revision-run-id="${esc(item.id)}" ${hasFrame ? `data-revision-step="${frameNumber}"` : ""} data-status="${esc(item.status)}" aria-current="${active ? "true" : "false"}" aria-label="${esc(`${frameLabel}${revision}，${statusLabel(item.status)}，${revisionReason(item)}`)}">
        <span class="revision-dot">${active ? mark(item.status) : ""}</span>
        ${milestone ? `<span class="revision-label"><strong>${esc(revision)}</strong><small>${esc(unchangedFrame ? "画面未变" : statusLabel(item.status, true))}</small></span>` : ""}
      </button>`;
    }).join("");
    return `<section class="revision-replay-shell" id="revision-replay" aria-label="机械版本回放">
      <header class="revision-replay-head">
        <div><span>单版本交互 Mesh · 播放使用固定机位</span><h2>${esc(project.label || project.id)} · ${versions.length} 个 Mesh 版本 / ${playbackFrames.length} 个播放画面</h2></div>
        <div class="revision-replay-actions">
          <button type="button" id="revision-replay-prev" aria-label="上一个版本" ${hasPreviousVersion ? "" : "disabled"}>←</button>
          <button type="button" id="revision-replay-play" aria-pressed="${state.revisionReplayPlaying ? "true" : "false"}" ${playbackFrames.length > 1 ? "" : "disabled"}>${state.revisionReplayPlaying ? "❚❚ 暂停" : `▶ 播放 ${playbackFrames.length} 个画面`}</button>
          <button type="button" id="revision-replay-next" aria-label="下一个版本" ${hasNextVersion ? "" : "disabled"}>→</button>
          <button type="button" id="revision-replay-compare" aria-pressed="${compareEnabled ? "true" : "false"}" ${canCompare ? "" : "disabled"}>${compareEnabled ? "退出对比" : "A/B 对比"}</button>
          <button type="button" id="revision-replay-export-video" title="预计 ${playbackFrames.length} 秒；连续重复图片不制造假动画" ${state.revisionReplayVideoExporting ? "disabled" : ""}>${state.revisionReplayVideoExporting ? "生成中…" : "导出去重视频"}</button>
          <button type="button" id="revision-replay-close" aria-label="关闭版本回放">×</button>
        </div>
      </header>
      <div class="revision-replay-stage">
        ${compareEnabled ? revisionCompareVisual(selected, compareBase, previewUrl) : state.revisionReplayPlaying ? `<figure class="revision-replay-visual ${previousPreviewUrl && previewUrl ? "is-stacked" : ""}" data-status="${esc(selected.status)}" data-frame-change="${selectedFrameUnchanged ? "unchanged" : previewUrl ? "changed" : "missing"}">
          ${previousPreviewUrl && previewUrl ? `<img class="revision-replay-stack-image is-previous" src="${esc(previousPreviewUrl)}" alt="" aria-hidden="true">` : ""}
          ${previewUrl ? `<img id="revision-replay-image" class="revision-replay-stack-image is-current" src="${esc(previewUrl)}" alt="${esc(`${selected.revision || selected.name} 固定机位设计视图`)}">` : ""}
          <div class="revision-replay-placeholder" ${previewUrl ? "hidden" : ""}><strong>此版本没有固定机位画面</strong><span>仍保留正式状态和版本位置，不以其他版本图片代替。</span></div>
          <figcaption><strong>${esc(selected.revision || selected.name)}</strong><span>${esc(statusLabel(selected.status))}${frameIndex >= 0 ? ` · Step ${selectedStep}/${playbackFrames.length}${selectedFrameUnchanged ? " · 画面未变" : ""}` : " · 无 3D 帧"}</span></figcaption>
          <div class="revision-frame-sequence" role="status" aria-live="polite" data-frame-change="${selectedFrameUnchanged ? "unchanged" : previewUrl ? "changed" : "missing"}"><strong>${frameIndex >= 0 ? `STEP ${selectedStep}` : "NO FRAME"}</strong><span>${esc(selected.revision || selected.name)} · ${selectedFrameUnchanged ? "画面未变，仅验证状态变化" : previewUrl ? "同视角原图 · 每个不同画面 1 秒" : "没有固定视角图片"}</span></div>
        </figure>` : revisionInteractiveMesh(selected, previewUrl)}
        <aside class="revision-replay-story">
          <span>验证状态变化</span>
          <h3>${esc(versionChange)}</h3>
          <p><code>${esc(rawStatus)}</code></p>
          ${compareEnabled ? `<label class="revision-compare-base-select"><span>对照版本 A</span><select id="revision-compare-base">${compareOptions}</select></label>` : ""}
          ${compareEnabled ? "" : `<dl>
            <div><dt>正式结论</dt><dd>${esc(revisionReason(selected))}</dd></div>
            <div><dt>阶段完成</dt><dd>${esc(progress)}</dd></div>
            <div><dt>工况通过</dt><dd>${esc(comparisons)}</dd></div>
            <div><dt>证据状态</dt><dd>${esc(evidenceHealth)}</dd></div>
          </dl>`}
          ${compareEnabled ? revisionGeometryProofMarkup(compareBase, selected) : ""}
          <button type="button" id="revision-replay-open-run">进入此版本详情</button>
        </aside>
      </div>
      <div class="revision-timeline">
        <div class="revision-timeline-copy"><strong>3D Step 轨道</strong><span>${playbackFrames.length} 个不同画面 · ${frames.length - playbackFrames.length} 个重复画面 · ${versions.length - frames.length} 个缺帧版本</span></div>
        <div class="revision-timeline-scroll"><div class="revision-timeline-track">${ticks}</div></div>
        <div class="revision-timeline-legend"><span data-frame-change="unchanged">◌ 画面未变</span><span data-status="passed">✓ 通过</span><span data-status="running">↻ 进行中</span><span data-status="failed">× 失败</span></div>
      </div>
    </section>`;
  }

  function updateUrl() {
    if (boot.mode !== "live") return;
    const q = new URLSearchParams(location.search);
    if (state.runId) q.set("run", state.runId);
    q.set("view", state.selectionMode === "pinned" ? "pinned" : "current");
    if (state.stageId) q.set("stage", state.stageId);
    if (state.loadId) q.set("load", state.loadId);
    if(state.historyScope==="project")q.set("history","project");else q.delete("history");
    if (state.designFramePinned) q.set("meshframe", state.designFramePinned);
    else q.delete("meshframe");
    q.set("studio",state.studioMode);
    if(state.studioMode==="motion" && studioMotion()?.id)q.set("motion",studioMotion().id);else q.delete("motion");
    if (stageKind() === "branch-simulation" || ["motion","physics"].includes(state.studioMode)) {
      q.set("branch", state.physicsBranch);
      q.set("frame", String(state.physicsFrame));
      q.set("layer", state.physicsLayer);
      if (state.physicsPart) q.set("part", state.physicsPart);
      else q.delete("part");
      if (state.physicsFocusEvidenceId) q.set("collision", state.physicsFocusEvidenceId);
      else q.delete("collision");
    } else {
      for (const key of ["branch", "frame", "layer", "part", "collision"]) q.delete(key);
    }
    if (state.revisionReplayOpen) {
      q.set("replay", "1");
      if (state.revisionReplayRunId) q.set("replayRun", state.revisionReplayRunId);
      if (state.revisionReplayCompare && state.revisionReplayCompareRunId) {
        q.set("replayCompare", "1");
        q.set("compareRun", state.revisionReplayCompareRunId);
        q.set("compareMode", state.revisionReplayCompareMode);
      } else {
        q.delete("replayCompare");
        q.delete("compareRun");
        q.delete("compareMode");
      }
    } else {
      q.delete("replay");
      q.delete("replayRun");
      q.delete("replayCompare");
      q.delete("compareRun");
      q.delete("compareMode");
    }
    history.replaceState(
      null,
      "",
      `${location.pathname}?${q}${location.hash}`,
    );
  }

  function topbar(data) {
    const run = data.run;
    const warningCount = Number(data.overview?.warning?.count || data.stages?.filter((item) => item.status === "warning").length || 0);
    const health = run.evidence_health || {};
    const verdictText = run.status === "passed" && health.status === "conflict"
      ? "PASS · 证据冲突"
      : run.status === "passed" && health.status === "legacy_unverified"
        ? "PASS · 待复核"
        : run.status === "passed" && warningCount
          ? "PASS · 含提醒"
          : statusLabel(run.status);
    const verdictTitle = health.status === "conflict"
      ? `正式终态保持 ${statusLabel(run.status)}；${health.status_label || "证据存在冲突"}`
      : warningCount
        ? `闭环通过，但有 ${warningCount} 条非阻塞提醒`
        : "整机结论只读取 harness 写入的正式运行终态";
    const selectedProject = projectForRun(state.runId);
    const projectOptions = state.projects.length
      ? state.projects.map((item) => {
        const lastPass = item.last_pass || item.published;
        const latest = item.latest;
        const lastPassHealth = lastPass?.evidence_health
          && lastPass.evidence_health !== "complete"
          ? ` / ${evidenceHealthWords[lastPass.evidence_health] || lastPass.evidence_health}`
          : "";
        const latestHealth = latest?.evidence_health
          && latest.evidence_health !== "complete"
          ? ` / ${evidenceHealthWords[latest.evidence_health] || latest.evidence_health}`
          : "";
        const versionStatus = latest?.id === lastPass?.id
          ? `当前 ${latest.revision || "未编号"} · ${statusLabel(latest.status)}${latestHealth}`
          : lastPass
            ? `最近通过 ${lastPass.revision || "未编号"} · ${statusLabel(lastPass.status)}${lastPassHealth}${latest ? ` · 最新 ${latest.revision || "未编号"} · ${statusLabel(latest.status)}${latestHealth}` : ""}`
            : latest
              ? `最新 ${latest.revision || "未编号"} · ${statusLabel(latest.status)}${latestHealth}`
              : "暂无可查看版本";
        return `<option value="${esc(item.id)}" ${item.id === selectedProject?.id ? "selected" : ""}>${esc(item.label || item.id)} · ${esc(versionStatus)} · ${item.version_count || item.runs?.length || 1} 版</option>`;
      }).join("")
      : `<option>${esc(run.project_id || run.name)}</option>`;
    const versions = selectedProject?.runs || state.runs.filter((item) => item.id === state.runId);
    const versionOptions = versions.length
      ? versions.map((item) => {
        const revision = item.revision || item.name || "未编号";
        const packageVersion = item.package_version ? ` · v${item.package_version}` : "";
        return `<option value="${esc(item.id)}" ${item.id === state.runId ? "selected" : ""}>${esc(revision + packageVersion)} · ${esc(statusLabel(item.status))}</option>`;
      }).join("")
      : `<option>${esc(run.revision || run.name)}</option>`;
    const picker = boot.mode === "live"
      ? `<div class="run-picker"><label class="picker-field project-field"><span>机械项目</span><select id="project-select" aria-label="切换机械项目" ${state.loading ? "disabled" : ""}>${projectOptions}</select></label><label class="picker-field version-field"><span>版本</span><select id="run-select" aria-label="切换项目版本" title="${esc(run.display_name || run.name)}" ${state.loading ? "disabled" : ""}>${versionOptions}</select></label></div><button id="refresh-button" class="icon-button" type="button" aria-label="重新载入看板" title="重新载入页面与最新程序">↻</button>`
      : `<div class="snapshot-run"><span>离线快照 · 仅此运行</span><strong>${esc(run.display_name || run.name)}</strong></div>`;
    const replayButton = boot.mode === "live" && state.surface === "loop"
      ? `<button id="revision-replay-button" class="revision-replay-button" type="button" aria-pressed="${state.revisionReplayOpen ? "true" : "false"}" title="沿项目时间轴查看机械版本变化"><span>↔</span>版本回放</button>`
      : "";
    return `<header class="topbar">
      <div class="brand"><span class="brand-mark">${shield}</span><div><h1>Loop 项目看板</h1><p>项目执行、资源审批与双角色会话</p></div></div>
      <div class="run-controls">${picker}${replayButton}<button id="export-button" class="export-button" type="button"><span>⇩</span>导出快照</button><span class="mobile-run-name">当前实验：${esc(run.display_name || run.name)}</span></div>
      <div class="run-summary">
        <div class="verdict" data-status="${esc(run.status)}" title="${esc(verdictTitle)}"><span class="verdict-mark">${mark(run.status)}</span><span>闭环判定${health.status_label ? ` · ${esc(health.status_label)}` : ""}</span><strong>${esc(verdictText)}</strong></div>
        ${projectUsagePanel(data.project_usage)}
        <div class="summary-count"><span>阶段完成</span><strong>${run.progress.completed}/${run.progress.total}</strong></div>
        <div class="summary-count"><span>仿真完成</span><strong>${run.jobs.completed}/${run.jobs.total}</strong></div>
        <div class="summary-count"><span>工况通过</span><strong>${run.comparisons.passed}/${run.comparisons.total}</strong></div>
      </div>
    </header>`;
  }

  function questPanel(data) {
    const items = data.stages.map((item) => `<li class="quest-item" data-status="${esc(item.status)}">
      <span class="quest-node">${item.order}</span>
      <button class="quest-button ${item.id === state.stageId ? "selected" : ""}" type="button" data-stage-id="${esc(item.id)}" aria-current="${item.id === state.stageId ? "step" : "false"}">
        <span class="quest-copy"><strong>${esc(item.short_title)}</strong><small>${esc(item.goal)}</small></span>
        <span class="quest-status"><i>${mark(item.status)}</i>${esc(statusLabel(item.status, true))}</span>
      </button>
    </li>`).join("");
    return `<aside class="quest-panel">
      <div class="panel-title"><div><span>主线任务</span><h2>任务战役路线</h2></div><strong>${data.run.progress.completed}/${data.run.progress.total}</strong></div>
      <ol class="quest-list">${items}</ol>
      <span class="quest-swipe-hint">← 左右滑动查看本项目 ${data.stages.length} 个阶段 →</span>
      <div class="quest-legend"><span data-status="passed">✓ 通过</span><span data-status="evidence_complete">◆ 证据齐全</span><span data-status="warning">! 提醒</span><span data-status="failed">× 失败</span><span data-status="not_started">○ 未完成</span></div>
    </aside>`;
  }

  function flowBoard(item) {
    const list = (title, values, cls) => `<section class="flow-column ${cls}"><h3>${title}</h3><ul>${(values || []).map((value) => `<li>${esc(value)}</li>`).join("")}</ul></section>`;
    const steps = (item.process || []).map((value, index) => `<li><span>${index + 1}</span>${esc(value)}</li>`).join("");
    const counts = `${item.inputs?.length || 0} 项输入 · ${item.process?.length || 0} 步 · ${item.outputs?.length || 0} 项产出`;
    return `<details class="flow-details" id="stage-flow"><summary><span>本阶段输入 → 过程 → 产出</span><small>${counts}，点击展开</small></summary><div class="flow-board">${list("输入", item.inputs, "inputs")}<section class="flow-column process"><h3>这一阶段做什么</h3><ol>${steps}</ol></section>${list("生成", item.outputs, "outputs")}</div></details>`;
  }

  function mediaSource(item) {
    if (!item) return "";
    if (item.preview_data_uri) return item.preview_data_uri;
    return boot.mode === "live" ? `${boot.apiBase}/file?run=${encodeURIComponent(state.runId)}&path=${encodeURIComponent(item.path)}` : "";
  }

  function visual(role, filters = {}) {
    const matches = (state.data?.evidence || []).filter((item) => {
      const detail = item.details || {};
      return detail.visual_role === role
        && (!filters.loadId || detail.load_case_id === filters.loadId)
        && (!filters.capsuleId || !detail.capsule_id || detail.capsule_id === filters.capsuleId);
    });
    if (role.startsWith("design-review-")) {
      matches.sort((left, right) => Number(right.details?.review_round || 0) - Number(left.details?.review_round || 0));
    }
    if (filters.capsuleId) {
      const exact = matches.find(
        (item) => item.details?.capsule_id === filters.capsuleId,
      );
      if (exact) return exact;
      const scopedCapsules = new Set([
        ...(state.data?.mechanisms || []).map((item) => item.id),
        ...(state.data?.whole_machines || []).map((item) => item.capsule_id),
      ].filter(Boolean));
      if (scopedCapsules.size > 1) return null;
    }
    return matches[0] || null;
  }

  function mediaCard(item, title, note, cls = "") {
    if (!item) return `<article class="media-card media-missing ${cls}"><div class="media-card-head"><strong>${esc(title)}</strong></div><div class="media-placeholder"><span>暂无画面</span><small>${esc(note || "本次实验没有登记这项可视化")}</small></div></article>`;
    const src = mediaSource(item);
    const role = item.details?.visual_role || "";
    const availability = item.preview_type === "video" ? "● 可查看 · 可全屏" : "● 可查看";
    const media = !src
      ? `<div class="media-placeholder"><span>画面未嵌入</span><small>请在在线看板查看</small></div>`
      : item.preview_type === "video"
        ? `<video data-visual-role="${esc(role)}" controls loop muted playsinline preload="metadata" src="${esc(src)}"></video>`
        : `<a class="image-link" href="${esc(src)}" target="_blank" rel="noopener" aria-label="放大查看：${esc(title)}"><img loading="lazy" src="${esc(src)}" alt="${esc(title)}"><span>点击放大</span></a>`;
    return `<article class="media-card ${cls}"><div class="media-card-head"><strong>${esc(title)}</strong><span class="media-availability">${availability}</span></div>${media}<p>${esc(note || item.summary || "")}</p></article>`;
  }

  function modelCard(scene, fallback, title, note, mode = "design", options = {}) {
    if (!scene) return mediaCard(fallback, title, note, "hero-media model-fallback-card");
    const src = mediaSource(scene);
    if (!src) return mediaCard(fallback, title, `${note} 当前快照没有嵌入交互模型。`, "hero-media model-fallback-card");
    const fallbackSrc = mediaSource(fallback);
    const axis = JSON.stringify(options.axis || []);
    const origin = JSON.stringify(options.origin || []);
    const legend = mode === "mechanism"
      ? `<span><i class="legend-fixed"></i>整机背景</span><span><i class="legend-moving mechanism"></i>当前机构</span>`
      : `<span><i class="legend-fixed design"></i>固定结构</span><span><i class="legend-moving"></i>运动组件</span>`;
    return `<article class="media-card hero-media model-media"><div class="media-card-head"><strong>${esc(title)}</strong><span class="media-availability">● 可查看</span></div>
      <div class="evidence-3d" data-model-src="${esc(src)}" data-mode="${esc(mode)}" data-axis="${esc(axis)}" data-origin="${esc(origin)}" role="img" aria-label="${esc(title)}，可拖动旋转、滚轮缩放、右键平移">
        ${fallbackSrc ? `<img class="evidence-3d-fallback" src="${esc(fallbackSrc)}" alt="${esc(title)} 静态备用图">` : ""}
        <span class="viewer-loading" data-3d-status>正在载入 3D…</span>
        <div class="viewer-legend">${legend}</div>
      </div><p>${esc(note)}</p></article>`;
  }

  function facts(rows) {
    return `<div class="fact-row">${rows.map(([label, value]) => `<div><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("")}</div>`;
  }

  function stageTaskPlanning(data) {
    const item = stage();
    const metrics = item?.metrics || [];
    const cards = metrics.map((metric) => `<article class="gate-key" data-status="${esc(item.status)}"><span>${mark(item.status)}</span><div><strong>${esc(metric.label)}</strong><p>${esc(metric.value)}</p></div></article>`).join("");
    const authored = metrics.find((metric) => metric.label === "新增测试任务")?.value || "0";
    const coverage = data.task_planning?.execution_coverage || [];
    const taskCards = coverage.map((task) => {
      const binding = [task.capsule_id, task.load_case_id, task.metric_id].filter(Boolean).join(" · ");
      const branches = (task.required_branches || []).map((branch) => `${branch}: ${statusLabel(task.branch_status?.[branch] || "not_started", true)}`).join(" / ");
      return `<article class="gate-key" data-status="${esc(task.status)}"><span>${mark(task.status)}</span><div><strong>${esc(task.test_id)}</strong><small>${esc(binding || "尚未绑定执行对象")}</small><p>${esc(task.goal)}</p><em>${esc(branches || "尚未声明执行分支")}</em></div></article>`;
    }).join("");
    const executionSection = coverage.length
      ? `<div class="section-heading"><div><h3>${coverage.length} 项测试规划与执行对应关系</h3><p>每项都显示自己使用的机构 capsule、工况、指标和分支；一个任务的 PASS 不会代替另一个任务。</p></div></div><div class="gate-key-grid">${taskCards}</div>`
      : "";
    return `<section class="stage-canvas"><article class="selected-load-summary" data-status="${esc(item.status)}"><div><span>接收任务</span><h3>${esc(item.title)}</h3><p>${esc(item.why)}</p></div><div class="branch-facts"><span>任务来源<strong>冻结的人类要求</strong></span><span>按需测试任务<strong>${esc(authored)}</strong></span></div></article><div class="gate-key-grid">${cards}</div>${executionSection}</section>`;
  }

  function pauseBanner(data) {
    const pause = data.run?.pause;
    if (!pause || data.run?.status_raw !== "REPEATED_ERROR_PAUSED") return "";
    const where = String(pause.previous_stage || pause.stage || "");
    const code = String(pause.error_code || "");
    const message = String(pause.message || "");
    const resume = String(pause.resume_status || "");
    return `<section class="resource-continuation-banner" data-status="paused" role="alert" aria-live="assertive"><span>Loop · 已暂停</span><div><strong>同一错误连续出现 ${esc(String(pause.occurrences || 3))} 次，Worker 已暂停</strong><p>${esc([where, code].filter(Boolean).join(" · "))}</p>${message ? `<p>${esc(message)}</p>` : ""}${resume ? `<p>恢复后回到：${esc(resume)}（使用 resume-paused-loop 命令）</p>` : ""}</div></section>`;
  }

  function resourceReviewPanel(data) {
    const review = data.resource_review;
    const continuationStatuses = new Set(["IDLE", "QUEUED", "RUNNING", "DELIVERED", "FAILED"]);
    const continuationWindowStatuses = new Set(["TASK_PLAN_REQUIRED", "PLAN_REVIEW_REQUIRED", "ASSET_DRAFT_REQUIRED", "CERTIFICATION_SUCCESSOR_REQUIRED", "DESIGN_REQUIRED"]);
    const projectedContinuation = review?.continuation;
    const hasProjectedContinuation = projectedContinuation
      && continuationStatuses.has(String(projectedContinuation.status || ""));
    const localContinuation = state.resourceReviewContinuation?.runId === data.run?.id
      ? state.resourceReviewContinuation
      : null;
    const continuation = hasProjectedContinuation
      ? projectedContinuation
      : localContinuation;
    const continuationStatus = String(continuation?.status || "");
    const continuationMatchesRequest = !review?.pending
      || continuation?.request_id === review.request?.request_id;
    const inContinuationWindow = !review?.pending
      && continuationWindowStatuses.has(String(review?.loop_status || ""));
    if (
      continuation
      && inContinuationWindow
      && continuationStatus !== "IDLE"
      && continuationStatuses.has(continuationStatus)
      && continuationMatchesRequest
    ) {
      const decision = String(continuation.decision || "");
      const delivered = continuationStatus === "DELIVERED";
      const failed = continuationStatus === "FAILED";
      let message = "审批已保存，正在继续同一个 Worker";
      if (decision === "REVISE") {
        message = delivered
          ? "同一个 Worker 已收到修改意见，正在重做提案"
          : "审核意见已发送，正在让同一个 Worker 重做提案";
      } else if (decision === "REJECT") {
        message = delivered
          ? "同一个 Worker 已收到拒绝结果，正在重写提案"
          : "拒绝决定已发送，正在让同一个 Worker 重写提案";
      } else if (delivered) {
        message = "同一个 Worker 已收到批准结果，正在继续任务";
      }
      const heading = failed ? "审批已保存，但 Worker 没有继续" : message;
      const detail = failed && continuation.message
        ? `<p>${esc(continuation.message)}</p>`
        : "";
      const retry = failed && boot.mode === "live" && continuation.request_id
        ? `<p><button type="button" data-resource-continuation-retry data-request-id="${esc(continuation.request_id)}">重试继续同一个 Worker</button> <span id="resource-continuation-message" role="status" aria-live="polite"></span></p>`
        : "";
      const role = failed ? "alert" : "status";
      const live = failed ? "assertive" : "polite";
      return `<section class="resource-continuation-banner" data-status="${esc(continuationStatus.toLowerCase())}" data-decision="${esc(decision)}" role="${role}" aria-live="${live}"><span>Loop · 审批已完成</span><div><strong>${esc(heading)}</strong>${detail}${retry}</div></section>`;
    }
    if (!review?.pending) return "";
    if (review.error || !review.request) {
      return `<section class="resource-review-panel resource-review-error" data-status="failed"><div><span>项目资源审批</span><h2>申请无法安全读取</h2><p>${esc(review.error || "资源申请缺失或已变化，审批按钮已关闭。")}</p></div></section>`;
    }
    const request = review.request;
    const kindWords = { authored_test: "测试程序", mechanism_template: "关节执行适配器" };
    const resources = (request.resources || []).map((item) => `<details class="resource-review-item"><summary><span>${esc(kindWords[item.kind] || item.kind)}</span><strong>${esc(item.logical_path)}</strong><small>${esc(item.size_bytes)} bytes · ${esc(String(item.sha256 || "").slice(0, 12))}…</small></summary><div class="resource-review-meta"><code>${esc(item.path)}</code><code>SHA-256 ${esc(item.sha256)}</code></div><pre>${esc(item.preview || "文件内容不可预览")}${item.preview_truncated ? "\n…预览已截断" : ""}</pre></details>`).join("");
    const dependencies = (request.dependencies || []).map((item) => `<li><strong>${esc(item.name)}</strong><span>v${esc(item.version)} · ${esc(item.sha256 || "未绑定哈希")}</span></li>`).join("");
    const jointPlan = (request.joint_plan || []).map((item) => `<li><strong>${esc(item.joint_id)}</strong><span>${esc(item.goal)}</span></li>`).join("");
    const testPlan = (request.test_plan || []).map((item) => `<li><strong>${esc(item.test_id)}</strong><span>${esc(item.goal)}</span></li>`).join("");
    const standardTopology = review.standard_role_topology === true;
    const reviewMode = review.resource_review_mode === "AI" ? "AI" : "HUMAN";
    const modeControl = standardTopology && boot.mode === "live"
      ? `<div class="resource-review-mode"><div><strong>项目资源审核方式</strong><span>${reviewMode === "AI" ? "AI 资源审核已启用；只批准本项目中的精确文件字节" : "人类审核项目资源；AI 资源审核休眠，但物理前外形审核仍自动运行"}</span></div>${humanDecisionCapability ? `<button type="button" data-resource-review-mode="${reviewMode === "AI" ? "HUMAN" : "AI"}" aria-pressed="${reviewMode === "AI" ? "true" : "false"}">${reviewMode === "AI" ? "关闭 AI 资源审核" : "启用 AI 资源审核"}</button>` : `<span class="resource-review-control-locked">请从服务启动日志中的“Human Loop dashboard”链接打开，才能切换审核方式。</span>`}<span id="resource-review-mode-message" role="status" aria-live="polite"></span></div>`
      : "";
    let controls = `<p class="resource-review-snapshot">离线快照只展示申请，审批请打开在线 Loop 看板。</p>`;
    if (boot.mode === "live" && standardTopology && reviewMode === "AI") {
      const reviewerActivation = review.reviewer_activation || {};
      const reviewerFailed = reviewerActivation.status === "FAILED";
      controls = reviewerFailed
        ? `<div class="resource-review-ai-wait" data-status="failed" role="alert"><strong>AI 审核员未能唤醒</strong><p>${esc(reviewerActivation.error_code || "ROLE_WAKE_FAILED")} · 人类可切回人类审核，或等人处理编排问题。</p></div>`
        : `<div class="resource-review-ai-wait"><strong>等待已登记的 AI 审核员</strong><p>Harness 会把这份哈希绑定申请发送到它自己的可见会话。浏览器不能代替它填写身份或决定。</p><span id="resource-review-message" role="status" aria-live="polite"></span></div>`;
    } else if (boot.mode === "live" && standardTopology && humanDecisionCapability) {
      controls = `<form class="resource-review-form" id="resource-review-form" aria-busy="false"><div class="resource-review-fields"><label><span>人类审核人</span><input id="resource-review-approver" maxlength="256" required placeholder="姓名或可审计身份"></label></div><p class="resource-review-authority-note">人类 APPROVE 只授权这些字节进入正式入库流程；本按钮不会直接复制或发布到全局目录。</p><label><span>审核意见</span><textarea id="resource-review-note" maxlength="8192" required placeholder="说明为什么批准，或具体要 Worker 修改什么"></textarea></label><div class="resource-review-actions"><button type="button" data-resource-decision="APPROVE">批准并授权入库流程</button><button type="button" data-resource-decision="REVISE">退回修改</button><button type="button" data-resource-decision="REJECT">拒绝</button><span id="resource-review-message" role="status" aria-live="polite" data-tone="idle"></span></div></form>`;
    } else if (boot.mode === "live" && standardTopology) {
      controls = `<div class="resource-review-ai-wait"><strong>人类审核控制已锁定</strong><p>请从服务启动日志中的“Human Loop dashboard”链接打开本页；普通旁观链接只能查看申请。</p></div>`;
    } else if (boot.mode === "live") {
      controls = `<form class="resource-review-form" id="resource-review-form" aria-busy="false"><div class="resource-review-fields"><label><span>审核人</span><input id="resource-review-approver" maxlength="256" required placeholder="姓名或审核 Agent ID"></label><label><span>审核身份</span><select id="resource-review-approver-type"><option value="reviewer_ai">AI 审核员 · 仅本项目</option><option value="human">人类审核 · 可授权复用</option></select></label></div><label><span>审核意见</span><textarea id="resource-review-note" maxlength="8192" required placeholder="说明为什么批准，或具体要 Worker 修改什么"></textarea></label><div class="resource-review-actions"><button type="button" data-resource-decision="APPROVE">批准</button><button type="button" data-resource-decision="REVISE">退回修改</button><button type="button" data-resource-decision="REJECT">拒绝</button><span id="resource-review-message" role="status" aria-live="polite" data-tone="idle"></span></div></form>`;
    }
    return `<section class="resource-review-panel" data-status="awaiting_review"><header><div><span>Loop · 当前项目资源审批</span><h2>Worker 新增 ${request.resources?.length || 0} 个资源，等待独立审核</h2><p>这里只审批关节执行适配器和测试程序。文件完整性已通过，但不代表设计内容正确。</p></div><div class="resource-review-request"><strong>${esc(request.request_id)}</strong><small>bundle ${esc(String(request.bundle_sha256 || "").slice(0, 16))}…</small></div></header>${modeControl}<div class="resource-review-grid"><div><h3>待审文件</h3>${resources}</div><div class="resource-review-context"><details><summary>计划使用这些关节与测试</summary><h4>关节</h4><ul>${jointPlan || "<li>无新增关节计划</li>"}</ul><h4>测试</h4><ul>${testPlan}</ul></details><details><summary>依赖 ${request.dependencies?.length || 0} 项</summary><ul>${dependencies || "<li>无额外依赖</li>"}</ul></details><details><summary>完整性检查</summary><ul>${(request.integrity?.checks || []).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></details></div></div>${controls}</section>`;
  }

  function setResourceReviewMessage(node, message, tone) {
    if (!node) return;
    node.textContent = message;
    node.dataset.tone = tone;
    node.setAttribute("role", tone === "error" ? "alert" : "status");
    node.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
  }

  function staticVisualReviewPanel(data) {
    const review = data.static_visual_review || {};
    if (!review.recorded) return "";
    const status = String(review.status || "UNKNOWN");
    const words = {
      PENDING: "等待独立外形审核",
      APPROVED: "独立外形审核已通过",
      CHANGES_REQUESTED: "独立审核要求修改外形",
      BUDGET_EXHAUSTED_CONTINUE: "外形问题未解决，已继续物理验证",
      REJECTED: "独立外形审核已拒绝",
      EVIDENCE_REQUIRED: "需要补齐六视图",
      INVALID: "独立审核记录损坏",
    };
    const tone = status === "APPROVED" ? "passed"
      : status === "PENDING" ? "awaiting_review"
      : ["EVIDENCE_REQUIRED", "INVALID"].includes(status) ? "blocked" : "failed";
    const callIndex = Number(review.call_index || review.calls_used || 0);
    const maxCalls = Number(review.max_calls || 0);
    const callProgress = maxCalls > 0
      ? `第 ${callIndex || "—"}/${maxCalls} 次`
      : `第 ${callIndex || "—"} 次（无固定轮次上限）`;
    const decision = review.decision || {};
    const findings = (decision.findings || []).map((item) => `<li><strong>${esc(item.view || "assembly")}</strong><span>${esc(item.issue || "")}</span><small>${esc(item.requested_change || "")}</small></li>`).join("");
    const rendererVersion = String(review.renderer_version || "").trim();
    const harnessViews = String(review.render_authority || "") === "harness";
    const viewNote = harnessViews
      ? `Harness 从冻结 STEP 统一生成的权威审核视图${rendererVersion ? `（renderer ${rendererVersion}）` : ""}`
      : "历史/来源未验证";
    const roleLabels = {
      front: "前视", rear: "后视", left: "左视", right: "右视",
      top: "俯视", bottom: "仰视", iso: "等轴总览",
    };
    const views = (review.views || []).map((item) => {
      const role = String(item.role || "");
      const kind = String(item.kind || "view");
      const title = roleLabels[role] || `${role} · ${kind}`;
      const caption = harnessViews
        ? viewNote
        : `独立 Reviewer 在物理仿真前实际查看的哈希绑定 ${kind} 图片。`;
      return mediaCard(
        visual(`static-review-${role}`),
        title,
        caption,
        "design-review-media",
      );
    }).join("");
    const feedback = decision.feedback
      ? `<div class="plain-warning"><strong>Reviewer 意见</strong><p>${esc(decision.feedback)}</p>${findings ? `<ul>${findings}</ul>` : ""}</div>`
      : "";
    const qualification = review.appearance_qualification || "PENDING";
    return `<section class="static-visual-review-panel" data-status="${tone}"><header><div><span>物理前独立外形门</span><h3>${esc(words[status] || status)}</h3><p>${callProgress}；外观资格 ${esc(qualification)}；${viewNote}。这里只判断外形与关键构件，绝不代表物理通过。</p></div><strong>${esc(status)}</strong></header><div class="design-review-gallery">${views}</div>${feedback}</section>`;
  }

  function stageOne(data) {
    const item = stage();
    const quality = data.design_quality || {};
    const solidCount = (item.metrics || []).find((metric) => metric.label === "CAD 零件")?.value || "—";
    const goal = quality.plain_language_goal || "这次历史运行没有保存造物目标，无法从文件名反推。";
    const prompt = quality.original_prompt || "历史运行未保存原始 Prompt；这不是通过状态，新生成任务必须记录。";
    const brief = quality.engineering_brief || {};
    const briefRows = Object.entries(brief).map(([key, value]) => {
      const shown = Array.isArray(value) ? value.join("、") : value && typeof value === "object" ? Object.entries(value).map(([name, amount]) => `${name}: ${amount}`).join("；") : String(value || "—");
      const labels = { overall_dimensions_mm: "整机尺寸", mechanism: "运动机构", major_components: "主要组件", materials: "材料", manufacturing_assumptions: "制造方式", coordinate_convention: "坐标约定" };
      return `<div><span>${esc(labels[key] || key)}</span><strong translate="no">${esc(shown)}</strong></div>`;
    }).join("");
    const reported = quality.reported_findings || quality.open_findings || {};
    const reportedCount = Number(reported.P0 || 0) + Number(reported.P1 || 0) + Number(reported.P2 || 0);
    const supplemental = quality.supplemental_issues || [];
    const reviewStatus = quality.recorded ? statusLabel(quality.status, true) : "历史记录缺失";
    const reviewViews = [
      ["design-review-iso", "等轴总览"], ["design-review-front", "正面结构"],
      ["design-review-side", "侧面结构"], ["design-review-top", "顶部结构"],
      ["design-review-section", "内部/剖面"],
    ].map(([role, title]) => {
      const view = visual(role);
      return view ? mediaCard(view, title, "Worker 补充图（worker_supplemental）；review_authority=false；不是 Harness 冻结审核视图。", "design-review-media") : "";
    }).join("");
    const reviewGallery = reviewViews ? `<div class="design-review-gallery">${reviewViews}</div>` : `<div class="design-review-missing"><strong>没有 Worker 补充图（worker_supplemental）</strong><span>这是可选项；Harness 冻结 STEP 后生成权威六视图。</span></div>`;
    const completenessNote = quality.recorded
      ? `${quality.review_round_count || 0} 轮 · 提交方备注 ${reportedCount} 个${supplemental.length ? ` · 补充问题 ${supplemental.length} 个不阻塞` : ""} · 不签发通过`
      : "Prompt 和包装文件都必须补齐";
    return `<section class="stage-canvas design-stage">
      <div class="design-intent-grid">
        <article class="design-goal-card"><span>我们在造什么</span><strong translate="no">${esc(goal)}</strong><small>这句话先说明对象和用途，再看下面的 CAD。</small></article>
        <details class="design-prompt-card" open><summary>原始生成 Prompt</summary><pre>${esc(prompt)}</pre></details>
        <article class="design-review-status" data-status="${esc(quality.status || "warning")}"><span>设计输入与包装完整性</span><strong>${esc(reviewStatus)}</strong><small>${completenessNote}</small></article>
      </div>
      ${briefRows ? `<article class="engineering-brief"><div><span>工程样机说明</span><strong>同一套 CAD 同时服务查看、装配检查和仿真</strong></div><div class="engineering-brief-grid">${briefRows}</div></article>` : ""}
      ${modelCard(visual("cad-scene"), visual("cad-overview"), "本次设计包的权威整机 CAD", `直接在画面中拖动查看 ${solidCount} 个 CAD 实体；不存在另一套独立外观模型。`, "design")}
      ${staticVisualReviewPanel(data)}
      ${reviewGallery}
      ${facts([...(item.metrics || []).map((m) => [m.label, m.value]), ["当前版本", data.overview?.package?.version || "—"]])}
    </section>`;
  }

  function stageTwo(data) {
    const item = stage();
    const values = Object.fromEntries((item.metrics || []).map((m) => [m.label, m.value]));
    const warning = data.overview?.warning || {};
    return `<section class="stage-canvas"><div class="preflight-gates">
      <article data-status="${values["阻塞问题"] === "0" ? "passed" : "failed"}"><span>${values["阻塞问题"] === "0" ? "✓" : "×"}</span><div><strong>资料齐全</strong><small>${esc(values["阻塞问题"] || "—")} 个阻塞问题</small></div></article>
      <article data-status="${item.status === "blocked" ? "failed" : "passed"}"><span>${item.status === "blocked" ? "×" : "✓"}</span><div><strong>CAD 能打开</strong><small>整机几何已读取</small></div></article>
      <article data-status="${values["可继续"] === "是" ? "passed" : "blocked"}"><span>${values["可继续"] === "是" ? "✓" : "■"}</span><div><strong>可以继续</strong><small>交给后续机构验证</small></div></article>
    </div>${warning.count ? `<div class="plain-warning"><strong>提醒，不阻塞：</strong><span>${esc(warning.title || warning.summary)}</span><p>${esc(warning.impact || "")}</p></div>` : ""}</section>`;
  }

  function mechanismButtons(data) {
    return (data.mechanisms || []).map((item, index) => `<button type="button" class="capsule-button ${item.id === state.mechanismId ? "selected" : ""}" data-mechanism-id="${esc(item.id)}"><span>${index + 1}</span><strong>${esc(item.id)}</strong><small>${esc(familyWords[item.family] || item.family || "未知运动")}</small></button>`).join("");
  }

  function stageThree(data) {
    const item = mechanism();
    const axis = Array.isArray(item?.axis) ? item.axis.join(", ") : "—";
    const scene = visual("mechanism-scene", { capsuleId: item?.id });
    const fallback = visual("mechanism-map", { capsuleId: item?.id });
    return `<section class="stage-canvas">
      <div class="mechanism-toolbar"><div><strong>选择一个机构查看范围</strong><span>拖动整机；灰色是背景，彩色是当前机构</span></div><div class="capsule-buttons">${mechanismButtons(data)}</div></div>
      ${modelCard(scene, fallback, item ? `机构：${item.id}` : "机构范围", "点击机构按钮会切换高亮范围；蓝色箭头表示允许的运动方向。", "mechanism", { axis: item?.axis || [], origin: item?.origin_m || [] })}
      ${item ? facts([["运动方式", familyWords[item.family] || item.family], ["固定组", String(item.fixed_groups?.length || 0)], ["运动组", String(item.moving_groups?.length || 0)], ["运动方向", axis]]) : ""}
    </section>`;
  }

  function stageFour() {
    const side = visual("side-clearance");
    const rail = visual("rail-contact");
    const closeups = side || rail ? `<div class="closeup-grid">${side ? mediaCard(side, "侧向间隙：0.10 mm", "缝隙已放大，导向壁和运动件没有穿模。") : ""}${rail ? mediaCard(rail, "导轨面：设计接触", "接触面贴合，但两件零件没有正体积重叠。") : ""}</div>` : "";
    return `<section class="stage-canvas geometry-canvas">${mediaCard(visual("collision-overview"), "碰撞预检总览", "固定结构与运动组件来自本次 CAD 派生的碰撞形状。", "geometry-overview")}${closeups}</section>`;
  }

  function branchRunStatus(item) {
    if (item?.branch_status) return item.branch_status;
    const statuses = [item?.physical?.status, item?.reduced?.status].filter(Boolean);
    return statuses.length === 2 && statuses.every((value) => value === "passed" || value === "warning") ? "passed" : "failed";
  }

  function applicableChecks(item) {
    return (item?.checks || []).filter((check) => check.physical_pass != null || check.reduced_pass != null);
  }

  function checkStats(item) {
    const rows = applicableChecks(item);
    const passed = rows.filter((check) => [check.physical_pass, check.reduced_pass].filter((value) => value != null).every(Boolean)).length;
    return { rows, passed, total: rows.length };
  }

  function branchVideos() {
    const item = load();
    const simplifiedConstraint = item?.coordinate_kind === "angular" ? "转轴约束" : "直线约束";
    const idealLabel = item?.ideal_label || "简化机构";
    const idealCopy = item?.ideal_kind === "ideal_reference"
      ? "由机构方程直接计算刚体姿态的简化动力学分支；不运行 Chrono 或碰撞，也不声称接触、受力或真实动态响应。"
      : `相同外部条件下，用${simplifiedConstraint}替代内部物理连接。`;
    return `<div class="branch-compare"><div class="branch-toolbar"><div><strong>真实物理与理想目标</strong><span>两者证据职责不同，不能互相替代</span></div><button type="button" data-sync-branches>▶ 同步播放</button></div><div class="media-pair">${mediaCard(visual("physical-video"), "左：真实物理模型", "使用逐零件实体碰撞和物理连接求解，内部理想关节为 0。")}${mediaCard(visual("reduced-video"), `右：${idealLabel}`, idealCopy)}</div></div>`;
  }

  function loadReplay(item) {
    const filters = { loadId: item?.id, capsuleId: item?.capsule_id };
    const video = visual("load-comparison-video", filters);
    const diagnosticVideo = visual("physical-diagnostic-video", filters);
    const referenceVideo = visual("reference-comparison-video", { capsuleId: item?.capsule_id });
    const idealLabel = item?.ideal_label || "简化机构";
    if (!video && diagnosticVideo) {
      const replay = visual("physics-evidence-replay", filters)?.details?.replay;
      const complete = replay?.branches?.physical?.status === "COMPLETED";
      return mediaCard(
        diagnosticVideo,
        item ? `${item.label}：真实物理${complete ? "" : "部分"}轨迹` : "真实物理保存轨迹",
        complete
          ? "视频只回放已保存的真实物理状态，没有重新运行仿真。"
          : "视频只覆盖超时前已原子保存的物理状态，用于诊断，不具备证书权威。",
        "hero-media load-replay",
      );
    }
    if (!video && referenceVideo) {
      return mediaCard(
        referenceVideo,
        "真实物理与理想参考",
        "左侧是真实物理保存轨迹，右侧是由机构方程计算出的各零件理想姿态；两边同步回放，没有重跑仿真。",
        "hero-media reference-comparison-replay",
      );
    }
    const missingNote = item && branchRunStatus(item) !== "passed"
      ? "本工况没有完整、可核对的物理与理想证据，因此不会生成假视频。"
      : item?.ideal_kind === "ideal_reference"
        ? "固定镜头对比视频尚未生成；可在上方 3D 回放中切换到方程计算的简化动力学分支。"
        : "如果没有画面，表示本次实验没有登记这项保存轨迹。";
    return mediaCard(video, item ? `${item.label}：真实物理与${idealLabel}` : "本工况运行回放", video ? `左边是真实物理，右边是${idealLabel}；视频只回放已保存证据，没有重新运行仿真。` : missingNote, "hero-media load-replay");
  }

  function shortHash(value) {
    const text = String(value || "");
    return text.length > 18 ? `${text.slice(0, 9)}…${text.slice(-7)}` : text || "未记录";
  }

  function physicsEvidenceReplay(item) {
    const filters = { loadId: item?.id, capsuleId: item?.capsule_id };
    const scene = visual("physics-evidence-scene", { capsuleId: item?.capsule_id });
    const manifest = visual("physics-evidence-scene-manifest", { capsuleId: item?.capsule_id });
    const replayArtifact = visual("physics-evidence-replay", filters);
    const replay = replayArtifact?.details?.replay;
    if (!scene || !replay?.branches?.physical) {
      return `<div class="physics-evidence-missing">${loadReplay(item)}<p>本工况没有登记可交互轨迹索引，因此只显示固定镜头回放。</p></div>`;
    }
    const src = mediaSource(scene);
    const fallback = mediaSource(visual("collision-overview"));
    const collision = replay.collision || {};
    const physical = replay.branches.physical || {};
    const traceComplete = physical.status === "COMPLETED";
    const checkpoint = physical.checkpoint || {};
    const completionPercent = Number.isFinite(Number(checkpoint.completion_fraction))
      ? Math.round(Number(checkpoint.completion_fraction) * 100)
      : null;
    const completenessLabel = traceComplete
      ? "完整物理轨迹"
      : `部分物理轨迹${completionPercent == null ? "" : ` · ${completionPercent}%`}`;
    const frameTotal = Math.max(1, Number(physical.retained_frame_count || 1));
    const hashes = replay.provenance || {};
    const sourceManifest = manifest?.details?.scene_manifest || {};
    const transmissionArrowMode = Array.isArray(replay.rotation_annotations)
      && replay.rotation_annotations.length > 0;
    const legacyReaction = !physical.field_availability?.per_joint_bushing_reaction;
    const branchNote = transmissionArrowMode
      ? "直箭头标出当前输入位置；弧箭头只画在与当前输入连通、且保存轨迹记录了旋向的齿轮上。"
      : legacyReaction
      ? "旧轨迹只保存全机构最大连接反力；逐关节位置不推断。"
      : "本轨迹保存逐关节连接反力，可按关节位置检查。";
    const forceLegend = transmissionArrowMode
      ? `<span><i class="legend-input-arrow"></i>橙色直箭头＝输入位置</span><span><i class="legend-rotation-arrow"></i>绿色弧箭头＝相关齿轮旋向；无箭头＝不参与当前传动</span>`
      : `<span><i class="legend-force-flow"></i>紫红＝反力点与传力路径（非应力）</span>`;
    const forceToggle = transmissionArrowMode
      ? `<button type="button" data-physics-heat aria-pressed="true" title="输入位置和齿轮旋向均来自保存轨迹；只显示与当前输入连通的已声明传动轮">传动箭头：开</button>`
      : `<button type="button" data-physics-heat aria-pressed="true" title="反力点来自物理轨迹；路径由同一刚体上的反力点连接推导，不是有限元应力云图">力流证据：开</button>`;
    const responseIndicator = transmissionArrowMode
      ? `<div class="transmission-arrow-scale"><span>当前显示</span><strong>输入位置 + 相关齿轮旋向</strong><small>不再按受力大小绘制热力图</small></div>`
      : `<div class="reaction-scale"><span>当前帧反力强度</span><i><b data-physics-reaction-bar></b></i><small>低　→　高</small></div>`;
    return `<article class="physics-evidence-workspace" data-load-case="${esc(item?.id || "")}">
      <header class="physics-evidence-head"><div><span>不是视频：可逐帧核对</span><h3>${traceComplete ? "保存轨迹" : "部分保存轨迹"} 3D 回放</h3><p>${traceComplete ? "默认显示真实 physical 轨迹；切换到理想参考时只用于运动对照，不能承担物理判定。" : "回放只覆盖超时前已原子保存的刚体位姿、载荷和接触；可用于诊断，不能用于签发证书。"}</p></div><strong data-status="${traceComplete ? "passed" : "warning"}">${traceComplete ? "✓ 来源已锁定" : "! 未完整跑完"}</strong></header>
      <div class="physics-proof-chips">
        <span class="${traceComplete ? "proof-good" : ""}">完整性 <b>${esc(completenessLabel)}</b></span>
        <span>轨迹 SHA <b title="${esc(physical.source_trace_sha256 || "")}">${esc(shortHash(physical.source_trace_sha256))}</b></span>
        <span>CAD SHA <b title="${esc(hashes.assembly_sha256 || "")}">${esc(shortHash(hashes.assembly_sha256))}</b></span>
        <span class="proof-good">自动整件凸包：${collision.convex_hull_used || collision.whole_body_convex_hull_used ? "使用" : "未使用"}</span>
        <span class="proof-good">可视化重跑仿真：否</span>
      </div>
      <div class="physics-evidence-layout">
        <div class="physics-view-column">
          <div class="evidence-3d physics-evidence-3d" data-model-src="${esc(src)}" data-mode="physics" role="application" aria-label="可旋转、逐帧检查并选择碰撞实体的物理轨迹">
            ${fallback ? `<img class="evidence-3d-fallback" src="${esc(fallback)}" alt="物理证据静态备用图">` : ""}
            <span class="viewer-loading" data-3d-status>正在载入物理证据 3D…</span>
            <div class="physics-frame-badge"><strong data-physics-phase>—</strong><span data-physics-time>0.000 s</span></div>
            <div class="viewer-legend physics-legend"><span><i class="legend-fixed design"></i>固定结构</span><span><i class="legend-moving"></i>运动实体</span><span><i class="legend-selected"></i>青色＝选中，不是受力</span>${forceLegend}</div>
          </div>
          <div class="physics-playback" aria-label="物理轨迹播放控制">
            <div class="physics-step-buttons"><button type="button" data-physics-prev aria-label="上一帧">|◀</button><button type="button" data-physics-play aria-label="播放或暂停">▶ 播放</button><button type="button" data-physics-next aria-label="下一帧">▶|</button><strong data-physics-frame-label>1 / ${frameTotal}</strong></div>
            <input type="range" min="0" max="${frameTotal - 1}" value="0" step="1" data-physics-frame aria-label="选择轨迹帧">
            ${replay.branches?.reduced ? `<div class="physics-branch-switch" role="group" aria-label="选择轨迹分支"><button type="button" data-physics-branch="physical" aria-pressed="true">真实物理</button><button type="button" data-physics-branch="reduced" aria-pressed="false">${esc(item?.ideal_label || "简化关节")}</button></div>` : `<div class="physics-branch-switch"><span>本工况没有可生成姿态的简化分支</span></div>`}
            <div class="physics-layer-buttons"><button type="button" data-physics-collision>碰撞体：所选</button>${forceToggle}</div>
          </div>
        </div>
        <aside class="physics-inspector" aria-live="polite">
          <section><span>当前部件</span><h4 data-physics-part>点击 3D 中的零件</h4><p data-physics-body>可以查看它属于哪个刚体、对应哪个 STEP 实体。</p></section>
          <section class="physics-collider-detail"><span>碰撞表示</span><strong data-physics-representation>未选择</strong><p>${esc(collision.dynamic_rule || "逐 STEP 实体碰撞")}</p><small>场景实体 ${esc(collision.component_count || sourceManifest.component_count || "—")} 个</small></section>
          <section class="physics-live-values"><div><span>外加载荷</span><strong data-physics-external>—</strong></div><div><span>最大连接反力</span><strong data-physics-reaction>—</strong></div><div><span>最大关节偏移</span><strong data-physics-drift>—</strong></div><div><span>接触点</span><strong data-physics-contact>—</strong></div></section>
          ${responseIndicator}
          <p class="physics-availability-note" data-physics-availability>${esc(branchNote)}</p>
          <details class="physics-provenance"><summary>来源锁定与证明边界</summary><dl><div><dt>原始轨迹</dt><dd>${esc(physical.source_trace || "未记录")}</dd></div><div><dt>载荷 SHA</dt><dd>${esc(hashes.load_case_sha256 || "未记录")}</dd></div><div><dt>计划 SHA</dt><dd>${esc(hashes.plan_sha256 || "未记录")}</dd></div><div><dt>验证器 SHA</dt><dd>${esc(hashes.validator_source_sha256 || "未记录")}</dd></div></dl><p>这能证明画面与工作区保存的轨迹/CAD 绑定；它不是外部签名，也不单独证明仿真器数学实现绝对正确。</p></details>
          ${replay.held_out ? `<div class="held-out-proof"><strong>保留工况</strong><span>这组条件未参与调参，用来检查模型是否只会复现标定动画。</span></div>` : ""}
        </aside>
      </div>
    </article>`;
  }

  function loadCards(data, mode) {
    return `<div class="load-grid">${scopedLoads().map((item, index) => {
      const simulation = mode === "simulation";
      const lane = item.execution_lane || {};
      const cardStatus = simulation ? (lane.status || branchRunStatus(item)) : item.status;
      const stats = checkStats(item);
      const angular = item.coordinate_kind === "angular";
      const values = simulation
        ? `<span>质量 ${fmt(item.moving_mass_kg, "kg")}</span><span>最大${angular ? "扭矩" : "力"} ${fmt(angular ? item.max_torque_nm : item.max_force_n, angular ? "N·m" : "N")}</span>`
        : `<span>真实 ${fmt(angular ? item.physical?.max_angle_deg : item.physical?.max_progress_mm, angular ? "°" : "mm")}</span><span>${esc(item.ideal_label || "简化")} ${fmt(angular ? item.reduced?.max_angle_deg : item.reduced?.max_progress_mm, angular ? "°" : "mm")}</span>`;
      const result = simulation
        ? (lane.state
          ? `${mark(lane.status)} ${executionLaneWords[lane.state] || lane.state}`
          : (cardStatus === "passed" ? "✓ 两边记录齐全" : `${mark(cardStatus)} 记录不完整`))
        : `${mark(item.status)} ${stats.passed}/${stats.total} 项门槛通过`;
      const recorded = physicsEvidenceForLoad(item.id);
      const replayLabel = simulation && (recorded.frameCount || recorded.video)
        ? ` · ${recorded.video ? "▶ 视频" : ""}${recorded.video && recorded.frameCount ? " · " : ""}${recorded.frameCount ? `${recorded.frameCount} Step 3D` : ""}`
        : "";
      return `<button type="button" class="load-card ${item.id === state.loadId ? "selected" : ""}" data-load-id="${esc(item.id)}" data-status="${esc(cardStatus)}">
        <span class="load-number">${index + 1}</span><div><strong>${esc(item.label)}</strong><small>${esc(loadDescription(item))}</small></div>
        <p>${values}</p><i>${result}${replayLabel}</i>
      </button>`;
    }).join("")}</div>`;
  }

  function selectedLoadSummary() {
    const item = load();
    if (!item) return "";
    const branchStatus = branchRunStatus(item);
    const idealLabel = item.ideal_label || "简化记录";
    const idealNature = item.ideal_kind === "ideal_reference" ? "轻量方程参考" : "动态轨迹";
    const lane = item.execution_lane || {};
    const laneFact = lane.state
      ? `<span>执行线路<strong>${esc(executionLaneWords[lane.state] || lane.state)}</strong></span>`
      : "";
    return `<article class="selected-load-summary" data-status="${esc(lane.status || branchStatus)}"><div><span>当前查看</span><h3>${esc(item.label)}</h3><p>${esc(lane.reason || loadDescription(item))}</p></div><div class="branch-facts">${laneFact}<span>真实物理<strong>${fmt(item.physical?.sample_count, "点")}</strong></span><span>${esc(idealLabel)}<strong>${fmt(item.reduced?.sample_count, "点")}</strong></span><span>理想侧性质<strong>${idealNature}</strong></span></div></article>`;
  }

  function retainedFailureComparison(item) {
    const filters = { loadId: item?.id, capsuleId: item?.capsule_id };
    const failure = visual("runtime-failure-snapshot", filters);
    if (!failure) return "";
    const completed = visual("load-comparison-video", filters);
    const missing = failure.details?.missing_requirement || "失败时缺少完整、合法的物理结果";
    return `<section class="retained-failure-compare"><div class="section-heading"><div><h3>失败前后对照</h3><p>只保留代表性失败，不堆积重复快照；右侧在本工况通过后自动出现完成证据。</p></div></div><div class="media-pair">${mediaCard(failure, "失败时：缺少什么", missing)}${mediaCard(completed, "通过后：补齐后的实体", completed ? "同一工况的真实物理与理想证据已经完整保存。" : "当前还没有合法完成的同工况证据。")}</div></section>`;
  }

  function stageFive(data) {
    const item = load();
    const loads = scopedLoads();
    const certificateStage = data.stages.find((candidate) => (candidate.kind || candidate.id) === "certificate");
    const audit = data.physical_audit || {};
    const contact = audit.component_contact || {};
    const auditStatus = audit.status || "not_started";
    const auditSummary = audit.recorded
      ? `<article class="selected-load-summary physical-audit-summary" data-status="${esc(auditStatus)}"><div><span>真实实体复审</span><h3>${auditStatus === "passed" ? "已通过，可生成理想参考" : "未通过，不得沿用为新 PASS"}</h3><p>${audit.source === "legacy_saved_trace_migration" ? "旧运行追加复审；历史证书没有被改写。" : "真实物理全部完成后，用保存姿态独立复查 STEP 实体。"}</p></div><div class="branch-facts"><span>工况通过<strong>${audit.passed_trace_count || 0}/${audit.trace_count || 0}</strong></span><span>已审 3D 时间点<strong>${audit.selected_pose_count || 0}</strong></span><span>最大检出穿入<strong>${fmt(audit.max_penetration_mm, "mm")}</strong></span><span>对照侧<strong>${esc(item?.ideal_label || "未记录理想参考")}</strong></span></div></article>`
      : "";
    const contactSummary = contact.recorded
      ? `<article class="selected-load-summary contact-intent-summary" data-status="${contact.forbidden_contact_failure_count || contact.uncertain_attribution_event_count ? "failed" : "passed"}"><div><span>零件级接触意图</span><h3>接触证据已绑定到具体 STEP 实体</h3><p>柔性支撑和解析传动不会被冒充成真实工作接触；只有正确阶段的实体对接触才能计入。</p></div><div class="branch-facts"><span>应发生接触<strong>${contact.expected_observed_count || 0}/${contact.expected_count || 0}</strong></span><span>柔性支撑<strong>${contact.support_count || 0}</strong></span><span>解析传动<strong>${contact.analytical_transmission_count || 0}</strong></span><span>禁碰/归因问题<strong>${(contact.forbidden_contact_failure_count || 0) + (contact.uncertain_attribution_event_count || 0)}</strong></span></div></article>`
      : "";
    const separationTitle = audit.recorded
      ? (auditStatus === "passed" ? "实体审查通过，再看理想目标" : "实体审查失败：旧对照记录只作为历史证据")
      : "这一关先验收真实物理，再生成方程计算的简化动力学分支";
    const separationNote = audit.recorded && auditStatus !== "passed"
      ? "修好几何并按物理优先流程重跑前，不能签发新证书。"
      : "理想参考不运行碰撞；是否满足运动要求，由后续认证阶段按明确门槛判断。";
    const certificateLink = certificateStage
      ? `<button type="button" data-stage-id="${esc(certificateStage.id)}">去第 ${certificateStage.order} 阶段看判定 →</button>`
      : "";
    return `<section class="stage-canvas">${auditSummary}${contactSummary}<div class="section-heading"><div><h3>${loads.length} 种必要测试</h3><p>每一种都是独立任务；点击任务卡会切换真实 3D 证据、理想参考和已有回放。</p></div><strong>${data.run.jobs.completed}/${data.run.jobs.total} 物理仿真完成</strong></div><div class="physics-load-workspace">${loadCards(data, "simulation")}${physicsEvidenceReplay(item)}</div>${selectedLoadSummary()}<details class="fixed-camera-replay"><summary>查看固定镜头或已有对照</summary>${loadReplay(item)}</details>${retainedFailureComparison(item)}<article class="stage-separation"><div><strong>${separationTitle}</strong><p>${separationNote}</p></div>${certificateLink}</article></section>`;
  }

  function gateValue(value, passed, margin, unit) {
    if (passed == null) return esc(fmt(value, unit));
    return `<span class="${passed ? "gate-pass" : "gate-fail"}">${esc(fmt(value, unit))} ${passed ? "✓" : "×"}${margin == null ? "" : `<small>余量 ${margin >= 0 ? "+" : ""}${esc(fmt(margin, unit))}</small>`}</span>`;
  }

  function gateTable(item) {
    const checks = item?.checks || [];
    if (!checks.length) return `<p class="empty-note">没有可读取的门槛明细。</p>`;
    const idealHeader = item?.ideal_label || "简化模型";
    const rows = checks.map((check) => `<tr><th>${esc(check.label)}${check.note ? `<small>${esc(check.note)}</small>` : ""}</th><td>${esc(`${check.rule} ${fmt(check.limit, check.unit)}`)}</td><td>${gateValue(check.physical, check.physical_pass, check.physical_margin, check.unit)}</td><td>${gateValue(check.reduced, check.reduced_pass, check.reduced_margin, check.unit)}</td></tr>`).join("");
    return `<div class="gate-table-wrap"><table class="gate-table"><thead><tr><th>检查项</th><th>门槛</th><th>真实物理</th><th>${esc(idealHeader)}</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function gateSummary(item) {
    if (!item) return "";
    const stats = checkStats(item);
    const idealLabel = item.ideal_label || "简化";
    const rows = stats.rows.map((check) => {
      const branchResults = [check.physical_pass, check.reduced_pass].filter((value) => value != null);
      const passed = branchResults.length > 0 && branchResults.every(Boolean);
      const limit = typeof check.limit === "string" ? `${check.rule} ${check.limit}` : `${check.rule} ${fmt(check.limit, check.unit)}`;
      const physical = check.physical == null ? "不适用" : fmt(check.physical, check.unit);
      const reduced = check.reduced == null ? "不适用" : fmt(check.reduced, check.unit);
      return `<article class="gate-key" data-status="${passed ? "passed" : "failed"}"><span>${passed ? "✓" : "×"}</span><div><strong>${esc(check.label)}</strong><small>门槛 ${esc(limit)}</small><p>真实 ${esc(physical)} · ${esc(idealLabel)} ${esc(reduced)}</p>${check.note ? `<em>${esc(check.note)}</em>` : ""}</div></article>`;
    }).join("");
    const amplitudeNote = item.ideal_kind === "ideal_reference"
      ? "理想侧是方程目标，不是第二次非凸包物理仿真。"
      : item.coordinate_kind === "angular"
        ? "两条曲线不要求逐时刻角度相等。"
        : "真实与理想动力学的最大位移不要求数值相等。";
    return `<article class="gate-summary" data-status="${esc(item.status)}"><header><div><span>本工况为什么通过</span><strong>${stats.passed}/${stats.total} 项门槛满足</strong></div><b>${mark(item.status)} ${esc(statusLabel(item.status, true))}</b></header><p>${esc(item.gate_policy || "按运动方向、约束、接触与任务边界判断。")} ${amplitudeNote}</p><div class="gate-key-grid">${rows}</div></article>`;
  }

  function stageSix(data) {
    const item = load();
    const loads = scopedLoads();
    const comparisonCaption = item?.coordinate_kind === "angular"
      ? "横轴是时间，纵轴是绕目标转轴的角度；最终判定以上方门槛卡为准。"
      : "横轴是时间，纵轴是沿机构方向的位移；最终判定以上方门槛卡为准。";
    return `<section class="stage-canvas"><div class="section-heading"><div><h3>${loads.length} 个任务逐项认证</h3><p>选中一项，先拖转核对 3D 物理轨迹，再看“门槛—实测—结论”。</p></div><strong>${data.run.comparisons.passed}/${data.run.comparisons.total} 通过</strong></div>
      <div class="physics-load-workspace">${loadCards(data, "comparison")}${physicsEvidenceReplay(item)}</div>
      <details class="fixed-camera-replay"><summary>查看固定镜头或已有对照</summary>${loadReplay(item)}</details>
      ${gateSummary(item)}
      <div class="comparison-focus">${mediaCard(visual("comparison-chart", { loadId: item?.id, capsuleId: item?.capsule_id }), item ? `${item.label}：真实物理与${item.ideal_label || "理想目标"}` : "工况对比", comparisonCaption, "comparison-chart")}<article class="comparison-verdict" data-status="${esc(item?.status || "not_started")}"><span>${mark(item?.status)}</span><div><strong>${esc(item ? statusLabel(item.status, true) : "未选择")}</strong><p>${esc(item?.gate_policy || "检查运动方向、横向约束、接触和任务边界。")}</p></div></article></div>
      <details class="audit-details"><summary>查看完整表格：门槛、观测值与余量</summary>${gateTable(item)}</details>
    </section>`;
  }

  function stageSeven(data) {
    const item = stage();
    const referenceOnly = Boolean(data.overview?.replacement?.reference_only);
    const referenceVideo = visual("reference-comparison-video");
    const historical = Boolean(data.overview?.replacement?.historical_terminal_authority);
    const headline = referenceOnly
      ? "仅保留理想参考 · 未启用替代"
      : historical
      ? "历史运行已启用 · 文件审核提示"
      : item.status === "passed"
        ? "简化机构已启用"
        : "简化机构未启用";
    const caption = referenceOnly
      ? "左边是真实 physical 证据；右边仅是理想运动参考，不得用于补救失败或签发物理结论。"
      : "左边保留真实物理结果作为回退依据；右边是被允许用于后续整机任务的简化动画。";
    const replay = referenceOnly && referenceVideo
      ? mediaCard(
          referenceVideo,
          "真实物理与理想参考",
          "左侧是真实物理保存轨迹，右侧是由机构方程计算出的各零件理想姿态；两边同步回放。",
          "hero-media reference-comparison-replay",
        )
      : branchVideos();
    return `<section class="stage-canvas"><article class="activation-result" data-status="${esc(item.status)}"><span>${mark(item.status)}</span><div><small>${referenceOnly ? "参考策略" : "替换结果"}</small><h3>${headline}</h3><p>${esc(item.why)}</p></div></article>${replay}<p class="pair-caption">${caption}</p></section>`;
  }

  function stageEightClock(data, selectedWhole = null) {
    const whole = selectedWhole || data.whole_machine || {};
    const radToDeg = (value) => value == null ? null : Number(value) * 180 / Math.PI;
    const taskDetails = (item) => {
      const evidence = item.evidence || {};
      if (item.id === "run_three_hand_clock") {
        return `<span>分针行程 <strong>${fmt(radToDeg(evidence.minute_span_rad), "°")}</strong></span><span>由发条轴输入，经整套齿轮系传到表针</span>`;
      }
      if (item.id === "verify_three_hand_ratio") {
        return `<span>时针/分针 <strong>${fmt(evidence.hour_to_minute_ratio)}</strong>（目标 1/12）</span><span>秒针/分针 <strong>${fmt(evidence.second_to_minute_ratio)}</strong>（目标 60）</span><span>方向 <strong>${evidence.same_display_direction ? "一致" : "不一致"}</strong></span>`;
      }
      if (item.id === "set_time_manually") {
        return `<span>后冠转动 <strong>${fmt(radToDeg(evidence.setting_crown_span_rad), "°")}</strong></span><span>分针被带动 <strong>${fmt(radToDeg(evidence.minute_span_rad), "°")}</strong></span><span>外力只加在后冠</span>`;
      }
      if (item.id === "ring_twin_bell_alarm") {
        return `<span>铃锤摆幅 <strong>${fmt(radToDeg(evidence.alarm_hammer_swing_rad), "°")}</strong></span><span>外力只加在发条轴，没有直接推铃锤</span>`;
      }
      if (item.id === "zero_time_and_resume") {
        return `<span>回零残差 <strong>${fmt(radToDeg(evidence.zero_residual_rad), "°")}</strong></span><span>松手漂移 <strong>${fmt(radToDeg(evidence.release_drift_rad), "°")}</strong></span><span>松手后继续走 <strong>${fmt(radToDeg(evidence.post_release_time_travel_rad), "°")}</strong></span>`;
      }
      if (item.id === "alarm_after_one_minute") {
        return `<span>秒针触发行程 <strong>${fmt(radToDeg(evidence.trigger_second_hand_travel_rad), "°")}</strong></span><span>触发前铃锤摆幅 <strong>${fmt(radToDeg(evidence.pretrigger_alarm_hammer_span_rad), "°")}</strong></span><span>响铃时仍在走时 <strong>${fmt(radToDeg(evidence.ring_timekeeping_travel_rad), "°")}</strong></span>`;
      }
      return "";
    };
    const tasks = (whole.stages || []).map((item, index) => `<article class="whole-task clock-task" data-status="${esc(item.status)}"><span>${mark(item.status)}</span><div><small>功能 ${index + 1}</small><strong>${esc(item.label)}</strong><p>${esc(item.summary || "")}</p></div><div class="task-checks clock-checks">${taskDetails(item)}</div></article>`).join("");
    const missionStrip = whole.task_total
      ? `<div class="whole-phase-strip clock-missions">${(whole.stages || []).map((item, index) => `${index ? "<b>→</b>" : ""}<span><i>${index + 1}</i>${esc(item.label)}</span>`).join("")}</div>`
      : "";
    const proofNote = `<div class="whole-nonclaim"><strong>如何避免“固定动画”误解：</strong>本阶段视频只回放证书后的整机任务轨迹；物理与认证阶段保留真实实体、理想参考、逐帧 3D、输入、支撑漂移和碰撞体来源。结论来自保存的数值轨迹，不由视频本身判定。</div>`;
    return `<section class="stage-canvas">${mediaCard(visual("whole-machine-video", { capsuleId: whole.capsule_id }), `闹钟 ${whole.task_total || 0} 项整机功能回放`, "按本项目登记的任务顺序查看；每项都有独立结果卡。", "hero-media whole-video")}${missionStrip}<article class="whole-standard"><strong>本阶段验收逻辑</strong><span>${whole.task_total || 0} 项功能必须分别通过</span><span>每项只允许其声明的输入</span><span>各项独立门槛必须满足</span></article><div class="whole-task-grid">${tasks || `<div class="empty-note">当前没有登记闹钟整机任务结果。</div>`}</div>${facts([["任务通过", `${whole.task_passed || 0}/${whole.task_total || 0}`], ["保存轨迹", fmt(whole.sample_count, "点")], ["结果", whole.status === "passed" ? "完整闭环通过" : "未通过"]])}${proofNote}</section>`;
  }

  function stageEightSingle(data, selectedWhole = null) {
    const whole = selectedWhole || data.whole_machine || {};
    if (whole.required === false) {
      return `<section class="stage-canvas"><article class="activation-result" data-status="passed"><span>✓</span><div><small>无需重复替代</small><h3>单机构整机直接采用 physical 任务证据</h3><p>该机械本身就是一个完整机构，未生成 reduced 整机回放。任务是否通过只看第 5、6 关的真实物理轨迹和任务门槛。</p></div></article></section>`;
    }
    if (whole.coordinate?.kind === "clock_multifunction") return stageEightClock(data, whole);
    const coordinate = whole.coordinate || {
      kind: "linear", name: "位置", unit: "mm", initial: 0,
      tolerance_label: "位置误差", tolerance: whole.position_tolerance_mm,
      effort_label: "控制力", effort_unit: "N", effort_limit: whole.max_control_force_n,
      constraint_label: "最大横向偏移（观测）", constraint_unit: "mm", constraint_value: whole.max_lateral_shift_mm,
      secondary_constraint_label: "最大转动（观测）", secondary_constraint_unit: "°", secondary_constraint_value: whole.max_rotation_deg,
    };
    const coordinateValue = (item, key, legacy) => item?.[key] ?? item?.[legacy];
    const tasks = (whole.stages || []).map((item) => {
      const target = coordinateValue(item, "target", "target_mm");
      const observed = coordinateValue(item, "observed", "observed_mm");
      const error = coordinateValue(item, "error", "error_mm");
      const tolerance = coordinateValue(item, "tolerance", "position_tolerance_mm") ?? coordinate.tolerance;
      const effort = coordinateValue(item, "max_effort", "max_force_n");
      const effortLimit = coordinateValue(item, "effort_limit", "force_limit_n") ?? coordinate.effort_limit;
      const effortPass = item.effort_pass ?? item.force_pass;
      return `<article class="whole-task" data-status="${esc(item.status)}"><span>${mark(item.status)}</span><div><strong>${esc(item.label)}任务</strong><small>目标 ${fmt(target, coordinate.unit)} · 实际 ${fmt(observed, coordinate.unit)}</small></div><div class="task-checks"><span data-status="${item.error_pass === false ? "failed" : "passed"}">${esc(coordinate.tolerance_label)} ${fmt(error, coordinate.unit)} ≤ ${fmt(tolerance, coordinate.unit)}</span><span data-status="${item.duration_pass === false ? "failed" : "passed"}">用时 ${fmt(item.duration_s, "s")} ≤ ${fmt(item.duration_limit_s, "s")}</span><span data-status="${effortPass === false ? "failed" : "passed"}">最大${esc(coordinate.effort_label)} ${fmt(effort, coordinate.effort_unit)} ≤ ${fmt(effortLimit, coordinate.effort_unit)}</span></div></article>`;
    }).join("");
    const timeline = whole.task_total
      ? `<div class="whole-phase-strip"><span><i>0</i>初始 <strong>${fmt(coordinate.initial, coordinate.unit)}</strong></span>${(whole.stages || []).map((item, index) => `<b>→</b><span><i>${index + 1}</i>${esc(item.label)} <strong>${fmt(coordinateValue(item, "observed", "observed_mm"), coordinate.unit)}</strong></span>`).join("")}</div>`
      : "";
    const standard = whole.task_total
      ? `<article class="whole-standard"><strong>本关验收标准</strong><span>${esc(coordinate.tolerance_label)} ≤ ${fmt(coordinate.tolerance, coordinate.unit)}</span><span>每段用时 ≤ ${fmt(whole.max_stage_duration_s, "s")}</span><span>${esc(coordinate.effort_label)} ≤ ${fmt(coordinate.effort_limit, coordinate.effort_unit)}</span></article>`
      : "";
    const constraintFacts = [
      [coordinate.constraint_label, fmt(coordinate.constraint_value, coordinate.constraint_unit)],
      [coordinate.secondary_constraint_label, fmt(coordinate.secondary_constraint_value, coordinate.secondary_constraint_unit)],
    ].filter(([label]) => label);
    const nonclaim = whole.task_total
      ? `<div class="whole-nonclaim"><strong>碰撞/穿透：</strong>本阶段使用已认证的简化机构，不运行高保真内部接触，因此不把“无穿透”作为本阶段结论；几何穿模检查请看“碰撞预检”阶段。</div>`
      : `<div class="whole-nonclaim">本次运行还没有登记整机任务目标、结果或回放；空目录不会被解释成已完成。</div>`;
    return `<section class="stage-canvas">${mediaCard(visual("whole-machine-video", { capsuleId: whole.capsule_id }), `整机 ${whole.task_total || 0} 项任务回放`, "点击播放或全屏，按本项目登记的任务顺序核对；下方任务卡给出目标、实测和允许范围。", "hero-media whole-video")}${timeline}${standard}<div class="whole-task-grid">${tasks || `<div class="empty-note">当前实验没有登记整机任务结果，因此这里不会显示成通过。</div>`}</div>${facts([["任务通过", `${whole.task_passed || 0}/${whole.task_total || 0}`], ...constraintFacts, ["运动记录", fmt(whole.sample_count, "点")]])}${nonclaim}</section>`;
  }

  function stageEight(data) {
    const wholeMachines = (data.whole_machines || []).filter(Boolean);
    if (wholeMachines.length <= 1) {
      return stageEightSingle(data, wholeMachines[0] || data.whole_machine || {});
    }
    const sections = wholeMachines.map((whole, index) => `
      <article class="whole-machine-scope" data-status="${esc(whole.status)}">
        <header class="whole-machine-scope-heading">
          <span>整机任务组 ${index + 1}/${wholeMachines.length}</span>
          <h3>${esc(whole.capsule_id || `机构 ${index + 1}`)}</h3>
          <small>${whole.task_passed || 0}/${whole.task_total || 0} 项通过</small>
        </header>
        ${stageEightSingle(data, whole)}
      </article>`).join("");
    return `<div class="whole-machine-multi">${sections}</div>`;
  }

  function stageCanvas(data) {
    return ({
      "task-planning": stageTaskPlanning,
      "design-package": stageOne, intake: stageTwo, "mechanism-plan": stageThree,
      "geometry-preflight": stageFour, "branch-simulation": stageFive,
      certificate: stageSix, replacement: stageSeven, "whole-machine": stageEight,
    }[stageKind()] || (() => ""))(data);
  }

  function missionPanel(data) {
    const item = stage();
    if (!item) return `<main class="mission-panel" id="mission-workspace"><p>没有可展示的阶段。</p></main>`;
    return `<main class="mission-panel" id="mission-workspace" tabindex="-1"><header class="mission-header"><div><span>当前查看 ${item.order}/${data.stages.length}</span><h2>${item.order}. ${esc(item.title)}</h2><p>${esc(item.goal)}</p></div><div class="stage-badge" data-status="${esc(item.status)}"><i>${mark(item.status)}</i><strong>${esc(statusLabel(item.status, true))}</strong></div></header><div class="stage-verdict" data-status="${esc(item.status)}"><strong>本关结论</strong><span>${esc(item.why)}</span></div><nav class="mobile-jumpbar" aria-label="本关快速跳转"><button type="button" data-jump=".stage-canvas">看画面与结果</button><button type="button" data-jump="#stage-flow">看输入输出</button><button type="button" data-jump="#evidence-panel">看证据</button></nav>${stageCanvas(data)}${flowBoard(item)}</main>`;
  }

  function evidenceForCurrent(data) {
    let items = data.evidence || [];
    const current = stage();
    const currentLoad = load();
    const currentCapsule = current?.scope?.capsule_id || currentLoad?.capsule_id || "";
    const matchesCurrentCapsule = (item) => {
      const capsuleId = item?.details?.capsule_id || "";
      return !currentCapsule || !capsuleId || capsuleId === currentCapsule;
    };
    if (state.evidenceScope === "stage" && current) {
      let ids = new Set(current.evidence_ids || []);
      if ((current.kind || current.id) === "branch-simulation" && currentLoad) {
        ids = new Set([...currentLoad.evidence_ids.filter((id) => data.evidence.find((x) => x.id === id)?.kind !== "comparison"), ...data.evidence.filter((x) => matchesCurrentCapsule(x) && (["physical-video", "reduced-video"].includes((x.details || {}).visual_role) || (["load-comparison-video", "physical-diagnostic-video", "runtime-failure-snapshot", "penetration-failure-closeup"].includes((x.details || {}).visual_role) && (x.details || {}).load_case_id === currentLoad.id) || ["queue", "physical-preflight", "legacy-physical-audit", "penetration-visual"].includes(x.kind))).map((x) => x.id)]);
      }
      if ((current.kind || current.id) === "certificate" && currentLoad) {
        ids = new Set([...currentLoad.evidence_ids.filter((id) => data.evidence.find((x) => x.id === id)?.kind === "comparison"), ...data.evidence.filter((x) => matchesCurrentCapsule(x) && (x.kind === "certificate" || ["reference-comparison-video", "physics-evidence-scene", "physics-evidence-scene-manifest"].includes((x.details || {}).visual_role) || (((x.details || {}).visual_role === "physics-evidence-replay" || ["comparison-chart", "load-comparison-video"].includes((x.details || {}).visual_role)) && (x.details || {}).load_case_id === currentLoad.id))).map((x) => x.id)]);
      }
      items = items.filter((item) => ids.has(item.id));
    }
    const q = state.evidenceSearch.trim().toLocaleLowerCase();
    return q ? items.filter((item) => [item.label, item.kind, item.path, item.summary].join(" ").toLocaleLowerCase().includes(q)) : items;
  }

  function ensureEvidence(data) {
    const items = evidenceForCurrent(data);
    if (items.some((item) => item.id === state.evidenceId)) return;
    const priority = {
      "task-planning": ["task-planning", "published-evidence"],
      "design-package": ["interactive-model", "step", "package", "visual-image"], intake: ["intake"],
      "mechanism-plan": ["interactive-model", "plan", "visual-image"], "geometry-preflight": ["geometry", "visual-image"],
      "branch-simulation": ["penetration-visual", "physical-preflight", "legacy-physical-audit", "trace-physical", "queue", "visual-video"],
      certificate: ["interactive-model", "visual-video", "certificate", "comparison"], replacement: ["visual-video", "replacement", "certificate"],
      "whole-machine": ["whole-machine", "visual-video", "replacement"],
    }[stageKind()] || [];
    state.evidenceId = (priority.map((kind) => items.find((x) => x.kind === kind && x.exists)).find(Boolean) || items.find((x) => x.exists) || items[0])?.id || "";
  }

  function evidencePreview(item) {
    if (!item.preview_type) return "";
    if (item.preview_type === "model") return `<p class="model-evidence-note">这个 3D 证据已显示在本关主画面，可直接拖动检查。</p>`;
    const src = mediaSource(item);
    if (!src) return `<p class="empty-note">快照没有嵌入这项画面。</p>`;
    return item.preview_type === "video" ? `<video class="evidence-media" controls loop muted playsinline preload="metadata" src="${esc(src)}"></video>` : `<a href="${esc(src)}" target="_blank" rel="noopener"><img class="evidence-media" loading="lazy" src="${esc(src)}" alt="${esc(item.label)}"></a>`;
  }

  function evidenceAudit(item) {
    const d = item.details || {};
    if (d.type === "trace") {
      const angular = d.coordinate_kind === "angular";
      return facts(angular
        ? [["分支", d.branch], ["记录点", String(d.sample_count || "—")], ["最大转角", fmt(d.max_angle_deg, "°")], ["转轴漂移", fmt(d.max_pivot_drift_mm, "mm")], ["轴外转动", fmt(d.max_off_axis_rotation_deg, "°")]]
        : [["分支", d.branch], ["记录点", String(d.sample_count || "—")], ["最大行程", fmt(d.max_progress_mm, "mm")], ["横向偏移", fmt(d.max_lateral_shift_mm, "mm")], ["最大转动", fmt(d.max_rotation_deg, "°")]]);
    }
    if (d.type === "comparison") return `<div class="audit-summary"><strong>${item.status === "passed" ? "本工况为什么通过" : "本工况为什么失败"}</strong><p>${esc(d.policy || "")}</p></div><details class="audit-details"><summary>查看门槛明细</summary>${gateTable(d)}</details>`;
    if (d.type === "certificate") {
      const angular = Math.abs(d.certifiable_angle_limit_deg || 0) > 0 || Math.abs(d.angle_deg?.max || 0) > 0;
      const range = angular ? `${fmt(d.angle_deg?.min, "°")} – ${fmt(d.angle_deg?.max, "°")}` : `${fmt(d.progress_mm?.min, "mm")} – ${fmt(d.progress_mm?.max, "mm")}`;
      return `${facts([["证书", d.status === "passed" ? "已签发" : "未签发"], ["设计包", `${d.package_id} v${d.design_version}`], ["质量范围", `${fmt(d.mass_kg?.min, "kg")} – ${fmt(d.mass_kg?.max, "kg")}`], [angular ? "已测角度" : "已测行程", range]])}<p class="evidence-boundary">${esc(d.boundary)}</p>`;
    }
    if (d.type === "penetration-event") {
      const pair = (d.solid_pair || []).join(" ↔ ") || "未记录";
      return `${facts([["时间", fmt(d.time_s, "s", 4)], ["原始轨迹帧", String(d.source_sample_index ?? "—")], ["最大检出穿入", fmt(d.penetration_depth_mm, "mm", 4)], ["碰撞实体", pair]])}<p class="evidence-boundary">红色是检测到的接触位置和局部重叠邻域，不声称是精确布尔交集体积。</p><button type="button" class="penetration-focus-button" data-penetration-focus="${esc(item.id)}">在 3D 中定位这个时间点</button>`;
    }
    if (d.type === "replacement") return facts([["状态", d.normalized_status === "passed" ? "已启用" : statusLabel(d.normalized_status, true)], ["运动方式", familyWords[d.family] || d.family], ["证书匹配", d.certificate_hash_matches ? "是" : "否"], ["设计包匹配", d.package_hash_matches ? "是" : "否"]]);
    if (d.type === "whole-machine") {
      const coordinate = d.coordinate || { constraint_label: "最大横向偏移", constraint_unit: "mm", constraint_value: d.max_lateral_shift_mm, secondary_constraint_label: "最大转动", secondary_constraint_unit: "°", secondary_constraint_value: d.max_rotation_deg };
      return `${facts([["任务通过", `${d.task_passed}/${d.task_total}`], [coordinate.constraint_label, fmt(coordinate.constraint_value, coordinate.constraint_unit)], [coordinate.secondary_constraint_label, fmt(coordinate.secondary_constraint_value, coordinate.secondary_constraint_unit)]])}<p class="evidence-boundary">${esc(d.boundary)}</p>`;
    }
    return "";
  }

  function evidenceDetail(item) {
    return `<div class="evidence-detail">${evidencePreview(item)}<p>${esc(item.summary || "没有额外摘要。")}</p>${evidenceAudit(item)}<details class="file-source"><summary>文件来源</summary><code>${esc(item.path || "未解析")}</code>${item.sha256 ? `<code>SHA-256 ${esc(item.sha256)}</code>` : ""}<button type="button" class="copy-path" data-path="${esc(item.path)}">复制路径</button></details></div>`;
  }

  function evidenceList(data) {
    const words = { "task-planning": "任务与测试规划", step: "整机 CAD", package: "文件清单", contract: "装配说明", intake: "检查结果", plan: "机构说明", geometry: "碰撞预检", queue: "仿真队列", load: "测试条件", "trace-physical": "真实运动记录", "trace-reduced": "历史简化运动记录", "trace-ideal-reference": "简化动力学分支（方程计算）", "trace-ideal-dynamics": "按需理想动力学", comparison: "工况对比", certificate: "运动证书", replacement: "简化机构", "whole-machine": "整机结果", "whole-machine-plan": "任务目标与容差", "visual-image": "图片", "visual-video": "视频", "interactive-model": "交互 3D", "run-status": "运行状态", "physical-preflight": "实体预检", "legacy-physical-audit": "旧运行复审", "penetration-visual": "穿模位置特写" };
    const items = evidenceForCurrent(data);
    const verdictKinds = new Set(["intake", "comparison", "certificate", "replacement", "whole-machine", "run-status", "physical-preflight", "legacy-physical-audit", "penetration-visual"]);
    if (!items.length) return `<p class="empty-note">当前阶段没有符合条件的证据。</p>`;
    return items.map((item) => {
      const open = item.id === state.evidenceId;
      const carriesVerdict = verdictKinds.has(item.kind);
      const itemStatus = !item.exists ? "not_started" : carriesVerdict ? item.status : "running";
      const stateCopy = !item.exists ? " · 文件缺失" : carriesVerdict ? ` · 结论${statusLabel(item.status, true)}` : " · 已找到";
      const itemMark = !item.exists ? "○" : carriesVerdict ? mark(item.status) : "●";
      return `<article class="evidence-item ${open ? "expanded" : ""}" data-status="${esc(itemStatus)}"><button type="button" class="evidence-button" data-evidence-id="${esc(item.id)}" aria-expanded="${open}"><span class="evidence-file">${fileIcon}</span><span><strong>${esc(item.label)}</strong><small>${esc(words[item.kind] || item.kind)}${esc(stateCopy)}</small></span><i>${itemMark}</i></button>${open ? evidenceDetail(item) : ""}</article>`;
    }).join("");
  }

  function evidencePanel(data) {
    const current = stage();
    const count = evidenceForCurrent(data).length;
    const boundary = current?.caveats?.[0] || "只依据当前实验中已登记的证据判定。";
    const search = count > 8 || state.evidenceSearch ? `<input id="evidence-search" class="evidence-search" type="search" value="${esc(state.evidenceSearch)}" placeholder="搜索证书、图片或结果">` : "";
    return `<aside class="evidence-panel" id="evidence-panel"><div class="panel-title evidence-title"><div><span>可追溯文件</span><h2>证据检查器</h2></div><button id="scope-toggle" type="button">${state.evidenceScope === "stage" ? `本阶段 ${count} 项` : `全部 ${count} 项`}</button></div><div class="evidence-boundary-top"><strong>这项结论的边界</strong><p>${esc(boundary)}</p></div>${search}<div id="evidence-list" class="evidence-list">${evidenceList(data)}</div><div class="evidence-integrity" data-status="${data.diagnostics?.length ? "warning" : "running"}">${data.diagnostics?.length ? `! ${data.diagnostics.length} 条文件诊断` : "● 已登记文件均可追溯"}</div></aside>`;
  }

  function statusbar(data) {
    const project = projectForRun(state.runId);
    const defaultRun = project?.runs?.find((item) => item.id === (project.default_run_id || project.current_run_id));
    const latestRun = project?.runs?.find((item) => item.id === project.latest_run_id);
    const selectedRevision = data.run.revision || data.run.name;
    const defaultRevision = defaultRun?.revision || defaultRun?.name || "";
    const selectionText = state.selectionMode === "pinned" && defaultRun?.id !== state.runId
      ? `固定查看 ${selectedRevision} · 项目当前 ${defaultRevision} ${statusLabel(defaultRun?.status, true)}`
      : latestRun && defaultRun && latestRun.id !== defaultRun.id
        ? `项目当前 ${defaultRevision} ${statusLabel(defaultRun.status, true)} · 最新尝试 ${latestRun.revision || latestRun.name} ${statusLabel(latestRun.status, true)}`
        : `跟随项目当前 ${selectedRevision}`;
    return `<footer class="statusbar"><span><i class="${state.stale ? "stale" : ""}"></i><strong>${boot.mode === "live" ? (state.stale ? "数据已过期" : "在线") : "离线快照"}</strong></span><span><strong>版本视图</strong>${esc(selectionText)}</span><span><strong>更新时间</strong>${esc(fmtTime(data.run.updated_at))}</span><span class="status-boundary"><strong>可信边界</strong>整机结论来自 harness 终态；证据问题单独显示</span></footer>`;
  }

  function restorePhysicsState(query) {
    const branch = query.get("branch");
    const layer = query.get("layer");
    state.physicsBranch = ["physical", "reduced"].includes(branch) ? branch : "physical";
    state.physicsLayer = ["selected", "all", "off"].includes(layer) ? layer : "selected";
    state.physicsFrame = Math.max(0, Number.parseInt(query.get("frame") || "0", 10) || 0);
    state.physicsPart = query.get("part") || "";
    state.physicsFocusEvidenceId = query.get("collision") || "";
  }

  function preparePhysicsViewer() {
    const node = document.querySelector(".physics-evidence-3d");
    if (!node) return;
    const currentLoad = load();
    const capsuleId = currentLoad?.capsule_id || stage()?.scope?.capsule_id || "";
    const replayArtifact = visual("physics-evidence-replay", { loadId: state.loadId, capsuleId });
    const manifestArtifact = visual("physics-evidence-scene-manifest", { capsuleId });
    const focusArtifact = (state.data?.evidence || []).find((item) => item.id === state.physicsFocusEvidenceId);
    node.questReplay = replayArtifact?.details?.replay || null;
    node.questSceneManifest = manifestArtifact?.details?.scene_manifest || null;
    node.questInitialState = {
      branch: state.physicsBranch,
      frame: state.physicsFrame,
      layer: state.physicsLayer,
      part: state.physicsPart,
      collisionFocus: focusArtifact?.details?.dashboard_focus || null,
    };
    bindPhysicsState(node);
  }

  function bindPhysicsState(node) {
    if(node.dataset.physicsStateBound)return;
    node.dataset.physicsStateBound="true";
    node.addEventListener("quest-physics-state", (event) => {
      const detail = event.detail || {};
      if (["physical", "reduced"].includes(detail.branch)) state.physicsBranch = detail.branch;
      if (["selected", "all", "off"].includes(detail.layer)) state.physicsLayer = detail.layer;
      if (Number.isInteger(detail.frame)) state.physicsFrame = detail.frame;
      state.physicsPart = detail.part || "";
      updateUrl();
    });
  }

  const roleDefinitions = {
    worker: { label: "Worker", short: "W", boundary: "执行项目任务" },
    project_resource_reviewer: { label: "项目审核员", short: "审核", boundary: "项目资源 + 物理前外形" },
    harness_operator: { label: "Harness 操作员", short: "HO", boundary: "休眠等待 Harness interrupt" },
    harness_change_reviewer: { label: "Harness 改动审核员", short: "HR", boundary: "只审 Harness 修改提案" },
  };

  const activationEventLabels = {
    task_start: "启动 Worker 任务",
    loop_continue: "继续 Worker 任务",
    resource_request: "项目资源审批请求",
    resource_decision: "项目资源审批结果",
    static_visual_review: "物理前独立外形审核",
    harness_defect: "Harness 缺陷已记录",
    orchestration_stall: "编排卡住",
    harness_change_proposal: "Harness 修改提案审核",
    harness_change_decision: "Harness 修改人类决定",
    supervision_start: "旧版监督启动",
  };

  const harnessPlainLanguage = {
    role_bootstrap: "这个角色会话已经登记，现在休眠；只有 Harness 发来与本角色匹配的 interrupt 时才会工作。",
    task_start: "开始处理当前机械任务。先核对这次激活绑定的 Observation 和冻结要求，再从普通 Loop 暴露的 Action 开始工作。",
    loop_continue: "继续当前机械任务。读取这次绑定的最新 Observation，只执行普通 Loop 当前允许的下一步 Action。",
    resource_decision: "项目资源审核结果已经记录。核对绑定的决定和最新 Observation，再按审核意见继续；不要自行批准资源。",
    resource_request: "独立审核当前绑定的项目机构或测试资源及其真实文件，只对这一份申请给出 APPROVE、REVISE 或 REJECT。",
    static_visual_review: "实际查看当前 STEP 绑定的前、后、左、右、上、下六张图片，独立决定通过、退回修改或拒绝；这不会授予物理 PASS。",
    harness_defect: "Harness 缺陷已记录。不要修改设计，不要唤醒 AI，不要重试；等人修复 Harness 后再用 recover-internal-harness-error 重开这次提交。",
    orchestration_stall: "编排已卡住并已记录。不要唤醒 AI，不要重试；等人处理后从当前任务继续。",
    supervision_start: "读取这次绑定的旧版监督事件，只在其原有权限范围内处理。",
  };

  const rolesForSurface = () => state.surface === "loop-loop"
    ? ["worker", "project_resource_reviewer", "harness_operator", "harness_change_reviewer"]
    : ["worker", "project_resource_reviewer"];

  function surfaceHref(pathname) {
    const target = new URL(window.location.href);
    target.pathname = `${boot.basePath || ""}${pathname}`;
    if (state.runId) target.searchParams.set("run", state.runId);
    return `${target.pathname}${target.search}${target.hash}`;
  }

  function globalSidebar() {
    const loopCurrent = state.surface === "loop";
    return `<nav class="global-sidebar" aria-label="统一看板视图">
      <div class="global-sidebar-brand" aria-label="机械闭环统一看板"><span>${shield}</span><strong>闭环</strong></div>
      <div class="global-sidebar-links">
        <a href="${esc(surfaceHref("/"))}" class="${loopCurrent ? "is-current" : ""}" aria-current="${loopCurrent ? "page" : "false"}" title="查看 Worker、项目审核员、项目资源与物理前外形审核"><span>Loop</span><small>项目</small></a>
        ${features.maintenance ? `<a href="${esc(surfaceHref("/maintenance"))}" class="${loopCurrent ? "" : "is-current"}" aria-current="${loopCurrent ? "false" : "page"}" title="查看四角色会话和 Harness 提案审批"><span>Loop<br>Loop</span><small>维护</small></a>` : ""}
      </div>
      <div class="global-sidebar-human" data-enabled="${humanDecisionCapability ? "true" : "false"}" title="${humanDecisionCapability ? "人类审批权限已启用" : "当前为只读旁观链接"}"><i></i><span>${humanDecisionCapability ? "可审批" : "只读"}</span></div>
    </nav>`;
  }

  function roleSessionTone(status) {
    const value = String(status || "").toLowerCase().replaceAll("_", "");
    if (["active", "running", "inprogress", "working", "handlinginterrupt"].includes(value)) return "running";
    if (["completed", "complete", "passed", "delivered"].includes(value)) return "passed";
    if (["failed", "interrupted", "blocked", "error", "invalid", "unavailable", "unsupported"].includes(value)) return "failed";
    if (["waiting", "queued", "awaitingreview", "interruptqueued"].includes(value)) return "awaiting_review";
    return "not_started";
  }

  function roleSessionStatus(status) {
    const value = String(status || "DORMANT").toUpperCase();
    return {
      ACTIVE: "正在执行",
      RUNNING: "正在执行",
      IN_PROGRESS: "正在执行",
      READY: "休眠 · 等待 interrupt",
      DORMANT: "休眠 · 等待 interrupt",
      WAITING_FOR_INTERRUPT: "休眠 · 等待 interrupt",
      INTERRUPT_QUEUED: "interrupt 已排队",
      HANDLING_INTERRUPT: "正在执行",
      NOT_LOADED: "未载入",
      NOTLOADED: "未载入",
      WAITING: "休眠 · 等待 interrupt",
      QUEUED: "interrupt 已排队",
      COMPLETED: "已完成",
      DELIVERED: "已交付",
      INTERRUPTED: "处理被中止",
      FAILED: "失败",
      BLOCKED: "阻塞",
      UNAVAILABLE: "会话不可用",
      UNSUPPORTED: "会话不可用",
    }[value] || value;
  }

  function eventKernelPresentation() {
    const status = String(state.eventKernel?.status || "NOT_STARTED").toUpperCase();
    const lane = eventKernelBoundary().lane;
    const label = {
      RUNNING: "运行中",
      QUEUED: "启动中",
      WAITING_HUMAN_RESOURCE_REVIEW: "等待人类审批",
      WAITING_EXTERNAL_AUTHORITY: "等待外部决定",
      WAITING_HARNESS_RESULT: "等待机械测试",
      MAINTENANCE_ROLE_COMPLETED: "维护角色已完成",
      COMPLETE: "任务完成",
      FAILED: "故障",
      INVALID: "状态无效",
      NOT_STARTED: "未启动",
    }[status] || "休眠";
    return {
      status,
      label: lane === "WORKER_RECOVERY_EXHAUSTED"
        ? "Worker 等待你的说明"
        : lane === "HARNESS_DEFECT"
          ? "Harness 缺陷 · 等人"
          : label,
      tone: roleSessionTone(status),
    };
  }

  function eventKernelBoundary() {
    const projectedLane = String(state.eventKernel?.lane || "");
    const projectedError = String(state.eventKernel?.error_code || "");
    if (projectedLane) return { lane: projectedLane, errorCode: projectedError };
    try {
      const detail = JSON.parse(String(state.eventKernel?.detail || ""));
      return {
        lane: String(detail?.lane || ""),
        errorCode: String(detail?.error_code || projectedError),
      };
    } catch (_error) {
      return { lane: "", errorCode: projectedError };
    }
  }

  function roleSessionTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(date);
  }

  function roleSessionFor(roleName) {
    const definition = roleDefinitions[roleName];
    const session = state.roleSessions.find((item) => item.role === roleName) || {
      role: roleName,
      status: "DORMANT",
      displayStatus: "WAITING_FOR_INTERRUPT",
      providerStatus: "",
      available: null,
      errorCode: "",
      messages: [],
      updatedAt: "",
    };
    const providerStatus = String(session.providerStatus || "")
      .toLowerCase()
      .replaceAll("_", "");
    const status = session.available === false
      ? "UNAVAILABLE"
      : session.displayStatus
      || (["active", "running", "inprogress", "working"].includes(providerStatus)
        ? "HANDLING_INTERRUPT"
        : "WAITING_FOR_INTERRUPT");
    return { definition, session, status };
  }

  function activeRoleName() {
    const roles = rolesForSurface();
    if (!roles.includes(state.roleSessionRole)) state.roleSessionRole = roles[0];
    return state.roleSessionRole;
  }

  function roleSessionSelector(roleName) {
    const { definition, session, status } = roleSessionFor(roleName);
    const selected = roleName === activeRoleName();
    const updated = roleSessionTime(session.updatedAt);
    return `<button type="button" class="role-chat-role" id="role-tab-${esc(roleName)}" role="tab" aria-selected="${selected ? "true" : "false"}" aria-controls="role-chat-panel" tabindex="${selected ? "0" : "-1"}" data-role-select="${esc(roleName)}" data-status="${esc(roleSessionTone(status))}">
      <span class="role-avatar" aria-hidden="true">${esc(definition.short)}</span>
      <span class="role-chat-role-copy"><strong>${esc(definition.label)}</strong><small>${esc(definition.boundary)}</small></span>
      <span class="role-status"><i></i>${esc(roleSessionStatus(status))}</span>
      ${updated ? `<time datetime="${esc(session.updatedAt)}">${esc(updated)}</time>` : ""}
    </button>`;
  }

  function roleChatMessage(message, definition) {
    const time = roleSessionTime(message.createdAt);
    if (message.kind === "harness_event" && message.direction === "input") {
      const interruptPrefix = /^Harness interrupt\s*·\s*/i;
      const legacyEvent = message.text.match(/^Harness\s*·\s*([a-z_]+)\s*·\s*(.*)$/i);
      const legacyKind = legacyEvent?.[1] || "";
      const isInterrupt = interruptPrefix.test(message.text)
        || Boolean(legacyKind && legacyKind !== "role_bootstrap")
        || Boolean(message.eventKind && message.eventKind !== "role_bootstrap");
      const text = interruptPrefix.test(message.text)
        ? message.text.replace(interruptPrefix, "")
        : legacyEvent
          ? `${legacyKind === "role_bootstrap" ? "角色会话已就位" : activationEventLabels[legacyKind] || legacyKind} · ${legacyEvent[2]}`
          : message.text.replace(/^Harness\s*·\s*/i, "");
      const bindingDetails = message.technicalText
        ? `<details class="role-chat-binding-details"><summary>查看技术绑定</summary><code>${esc(message.technicalText)}</code></details>`
        : "";
      return `<li class="role-chat-message role-chat-input"${message.id ? ` data-message-id="${esc(message.id)}"` : ""}>
        <article><header>${time ? `<time>${esc(time)}</time>` : ""}<span>${isInterrupt ? "Interrupt 输入" : "会话输入"}</span><strong>Harness</strong></header><div class="role-chat-bubble"><p>${esc(text)}</p>${bindingDetails}</div></article>
        <span class="role-chat-message-avatar" aria-hidden="true">H</span>
      </li>`;
    }
    return `<li class="role-chat-message"${message.id ? ` data-message-id="${esc(message.id)}"` : ""}>
      <span class="role-chat-message-avatar" aria-hidden="true">${esc(definition.short)}</span>
      <article><header><strong>${esc(definition.label)}</strong><span>Agent 消息</span>${time ? `<time>${esc(time)}</time>` : ""}</header><div class="role-chat-bubble"><p translate="no">${esc(message.text)}</p></div></article>
    </li>`;
  }

  function roleSessionTranscript(roleName) {
    const { definition, session, status } = roleSessionFor(roleName);
    const messages = (session.messages || []).map(
      (message) => roleChatMessage(message, definition),
    ).join("");
    const unavailable = session.available === false
      ? `<li class="role-session-notice" data-tone="error" role="alert"><strong>会话不可用</strong>${session.errorCode ? ` · ${esc(session.errorCode)}` : ""}</li>`
      : "";
    const emptyMessage = status === "HANDLING_INTERRUPT"
      ? "已收到 Harness 指令，正在执行。"
      : status === "INTERRUPT_QUEUED"
        ? "Harness interrupt 已排队，等待该会话取件。"
        : "当前休眠；Harness 只在该角色需要处理事件时发送 interrupt。";
    return `<section class="role-chat-conversation" id="role-chat-panel" role="tabpanel" aria-labelledby="role-tab-${esc(roleName)}" data-status="${esc(roleSessionTone(status))}">
      <header class="role-chat-active-head"><span class="role-avatar" aria-hidden="true">${esc(definition.short)}</span><div><h3>${esc(definition.label)}</h3><p>${esc(definition.boundary)}</p></div><strong class="role-status"><i></i>${esc(roleSessionStatus(status))}</strong></header>
      <ol class="role-chat-stream" data-role-chat-stream data-role="${esc(roleName)}" aria-label="${esc(definition.label)} 的连续会话记录">${unavailable}${messages || (unavailable ? "" : `<li class="role-chat-empty"><span>${esc(definition.short)}</span><strong>会话等待中</strong><p>${esc(emptyMessage)}</p></li>`)}</ol>
    </section>`;
  }

  function roleSessionsContent() {
    const roles = rolesForSurface();
    const selectedRole = activeRoleName();
    const selectors = roles.map(roleSessionSelector).join("");
    const note = state.roleSessionsError
      ? `<div class="role-session-notice" data-tone="error" role="status">${esc(state.roleSessionsError)}</div>`
      : state.roleSessionsLoading && !state.roleSessions.length
        ? `<div class="role-session-notice" role="status">正在连接可见角色会话…</div>`
        : "";
    return `${note}<div class="role-chat-layout"><aside class="role-chat-roster" aria-label="角色会话列表"><div class="role-chat-roster-head"><strong>会话</strong><span>${roles.length} 个角色</span></div><div class="role-chat-role-list" role="tablist" aria-label="切换角色会话" aria-orientation="vertical">${selectors}</div></aside>${roleSessionTranscript(selectedRole)}</div>`;
  }

  function roleSessionsPanel(context = "dock") {
    const updated = roleSessionTime(state.roleSessionsUpdatedAt);
    const count = rolesForSurface().length;
    const tag = context === "dock" ? "aside" : "section";
    const kernel = eventKernelPresentation();
    return `<${tag} class="role-sessions-panel role-sessions-${esc(context)}" aria-labelledby="role-sessions-title">
      <header class="role-sessions-head"><div><span>实时会话 · ${count} 个独立角色</span><h2 id="role-sessions-title">Loop 角色</h2><p>优先显示 Harness 指令的人话摘要与 Agent 回复；技术绑定可展开，完整提示、工具参数和权限凭据不会进入页面。</p></div><div><span class="event-kernel-status" data-event-kernel-status data-status="${esc(kernel.tone)}"><i></i><span data-event-kernel-label>Harness 内核 · ${esc(kernel.label)}</span></span><small>${updated ? `更新于 ${esc(updated)}` : "等待首次刷新"}</small><button type="button" data-role-session-refresh ${state.roleSessionsLoading ? "disabled" : ""}>刷新</button></div></header>
      <div class="role-sessions-content" data-role-sessions-content aria-live="polite">${roleSessionsContent()}</div>
    </${tag}>`;
  }

  function loopApprovalZone(data) {
    const review = resourceReviewPanel(data);
    const pending = Boolean(data.resource_review?.pending);
    const reviewMode = data.resource_review?.resource_review_mode === "AI" ? "AI" : "HUMAN";
    const reviewerActivation = data.resource_review?.reviewer_activation || {};
    const reviewerState = reviewMode === "AI"
      ? roleSessionStatus(reviewerActivation.status || "ACTIVE")
      : "资源审核休眠 · 外形审核独立";
    const modeControl = review
      ? ""
      : `<div class="human-approval-mode"><div><span>当前资源审核方式</span><strong>${reviewMode === "AI" ? "AI 审核" : "人类审核"}</strong></div><div><span>AI 资源审核员</span><strong>${esc(reviewerState)}</strong></div>${humanDecisionCapability
        ? `<button type="button" data-resource-review-mode="${reviewMode === "AI" ? "HUMAN" : "AI"}" aria-pressed="${reviewMode === "AI" ? "true" : "false"}">${reviewMode === "AI" ? "切换为人类资源审核" : "启用 AI 资源审核"}</button>`
        : `<small>使用 Human Loop 链接可切换</small>`}<span id="resource-review-mode-message" role="status" aria-live="polite"></span></div>`;
    return `<section class="human-approval-zone ${pending ? "has-pending" : "is-clear"}" aria-labelledby="human-approval-title">
      <header class="human-approval-head"><div><span>人类审批</span><h2 id="human-approval-title">项目资源审批</h2><p>关节模板与任务模板在这里决定。</p></div></header>
      ${modeControl}
      ${review || `<div class="human-approval-clear"><strong>当前没有待审批的项目资源</strong><span>${humanDecisionCapability ? "审批权限已就绪；新申请到达时会在这里展开。" : "当前为只读旁观链接；请使用 Human Loop 链接进行决定。"}</span></div>`}
    </section>`;
  }

  function workerRecoveryZone() {
    const { lane } = eventKernelBoundary();
    const loopStatus = String(state.data?.run?.status_raw || "");
    if (loopStatus === "REPEATED_ERROR_PAUSED") return "";
    if (lane === "HARNESS_DEFECT" || loopStatus === "EXTERNAL_VALIDATION_REVIEW_REQUIRED") return "";
    if (lane !== "WORKER_RECOVERY_EXHAUSTED" && lane !== "ORCHESTRATION_STALL") return "";
    return `<section class="human-approval-zone worker-recovery-zone" aria-labelledby="worker-recovery-title">
      <header class="human-approval-head"><div><span>需要人类说明</span><h2 id="worker-recovery-title">Worker 需要你的说明</h2><p lang="en">Worker needs clarification</p><p>自动恢复已经用完。这里补充工作流程或权限说明后，Harness 会唤醒同一个 Worker 从当前任务继续；不会批准资源、修改 Harness 或创建新 Worker。</p></div></header>
      ${humanDecisionCapability
        ? `<form class="worker-recovery-form" id="worker-recovery-form" aria-busy="false">
            <label><span>你的身份</span><input id="worker-recovery-actor" name="actor" maxlength="160" value="人类审批员" autocomplete="name"></label>
            <label><span>给 Worker 的说明</span><textarea id="worker-recovery-note" name="note" maxlength="2000" rows="5" placeholder="请说明它误解了什么、应从哪条合法流程继续；不要在这里审批资源。"></textarea></label>
            <button type="submit" data-worker-resume>发送说明并继续</button>
            <span id="worker-recovery-message" role="status" aria-live="polite"></span>
          </form>`
        : `<div class="human-approval-locked"><strong>当前是只读旁观链接</strong><span>请使用服务启动日志中的 Human Loop dashboard 链接打开此页，才能发送说明并恢复 Worker。</span></div>`}
    </section>`;
  }

  function harnessDefectBanner(data) {
    const defect = data.run?.harness_defect;
    const { lane } = eventKernelBoundary();
    const statusRaw = String(data.run?.status_raw || "");
    if (!defect && lane !== "HARNESS_DEFECT" && statusRaw !== "EXTERNAL_VALIDATION_REVIEW_REQUIRED") {
      return "";
    }
    const code = String(defect?.code || "");
    const owner = String(defect?.owner || "harness_code");
    const message = String(defect?.message || "Harness 缺陷已记录。不要重试，不要唤醒 AI。等人修复后用 recover-internal-harness-error 重开这次提交。");
    const envelope = defect?.envelope;
    const envelopeBlock = envelope == null || envelope === ""
      ? ""
      : `<pre class="harness-defect-envelope">${esc(typeof envelope === "string" ? envelope : JSON.stringify(envelope, null, 2))}</pre>`;
    return `<section class="resource-continuation-banner harness-defect-banner" data-status="failed" role="alert" aria-live="assertive"><span>Loop · Harness 缺陷</span><div><strong>Harness 缺陷已记录，正在等人处理</strong><p>${esc([code, owner].filter(Boolean).join(" · "))}</p><p>${esc(message)}</p>${envelopeBlock}</div></section>`;
  }

  const studioIcons = {
    cube: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z M3 8l9 5 9-5 M12 13v8 M7.5 5.5l9 5" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>',
    panel: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="M9 4v16" stroke="currentColor" stroke-width="1.5"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 11v6 M12 7v1" stroke="currentColor" stroke-width="1.8"/></svg>',
  };

  function studioProjectName(project) {
    const current = project?.id === projectForRun(state.runId)?.id;
    const records = [project, current ? state.data?.live_workflow : null, project?.latest, current ? state.data?.run : null].filter(Boolean);
    for (const record of records) {
      const candidate = record.human_title || record.machine_name || record.project_title || record.title || (typeof record.brief === "string" ? record.brief : record.brief?.title || record.brief?.machine_name);
      if (typeof candidate === "string" && candidate.trim() && candidate.trim().length <= 100) return candidate.trim().split(/[\r\n]/)[0];
    }
    const raw = String(project?.label || project?.id || state.data?.run?.project_id || state.data?.run?.name || "机械项目");
    const readable = raw
      .replace(/^(?:(?:package|project|formal|unattended|speed|probe|cold_start)[_-]+)+/i, "")
      .replace(/(?:^|[_-])20\d{6}(?=[_-]|$)/g, "")
      .replace(/(?:^|[_-])w\d+[a-z]?(?=[_-])/gi, "")
      .replace(/(?:[_-]?\d{5,})$/, "")
      .replace(/[_-]+/g, " ").trim();
    return readable || raw.replace(/^package[_-]/i, "");
  }

  function studioProjectShortName(name) {
    const value = String(name || "项目");
    if (/[\u3400-\u9fff]/.test(value)) return value.replace(/\s+/g, "").slice(0, 6);
    return value.split(/\s+/).slice(0, 2).join(" ");
  }

  function studioStatus() {
    const workflow = state.data?.live_workflow;
    if (!workflow) return statusLabel(state.data?.run?.status, true);
    const selectedRevision = String(state.data?.run?.revision || "").replace(/^r0*/, "");
    if (selectedRevision && selectedRevision !== String(Number(workflow.revision))) return statusLabel(state.data?.run?.status, true);
    const words = {
      running: "进行中", active: "进行中", completed: "已完成", complete: "已完成", passed: "已通过",
      activity_unknown: "运行状态待确认",
      failed: "未完成", blocked: "等待恢复", paused: "已暂停", stopped: "已停止",
      interrupted: "已中断", queued: "等待执行", authoring: "正在制作",
      validating: "正在验证", reviewing: "正在审核", repairing: "正在修改",
      criteria: "整理验收标准", author: "正在制作", author_complete: "作者已交付", geometry: "准备原始模型",
      preflight_review: "检查外观与测试", physics: "物理测试中", geometry_audit: "复查运动几何",
      final_review: "独立复核中", evidence_collection: "补充证据", harness_repair: "修复运行问题",
      revision_needed: "需要修正", author_followup: "继续制作", first_pass: "准备首轮测试", first_pass_blocked: "场景尚不可测试",
    };
    // How a finished run ended outranks the phase it ended in.
    const endings = {
      final_evaluation_failed: "最终评估未通过", partial_delivery: "部分交付（未全部通过）", budget_exhausted: "预算用完",
      provider_error: "Agent 服务异常", harness_error: "Harness 错误", scene_untestable: "场景无法测试", author_unfinished: "作者未完成交付",
    };
    const raw = String(workflow.phase || workflow.status || "").toLowerCase();
    const label = (!workflow.is_running && endings[workflow.outcome]) || words[raw] || (workflow.is_running ? "进行中" : words[String(workflow.status || "").toLowerCase()]) || statusLabel(state.data?.run?.status, true);
    return label;
  }

  function studioElapsed(now = Date.now()) {
    const workflow = state.data?.live_workflow;
    if (workflow) {
      let seconds = workflow.elapsed_seconds == null ? null : Number(workflow.elapsed_seconds);
      if (!Number.isFinite(seconds)) {
        const start = Date.parse(workflow.started_at || "");
        const end = Date.parse(workflow.ended_at || workflow.updated_at || "");
        seconds = Number.isFinite(start) && Number.isFinite(end) ? Math.max(0, (end - start) / 1000) : null;
      }
      if (seconds == null) return null;
      if (workflow.is_running && state.progressWorkflowReceivedAt) seconds += Math.max(0, (now - state.progressWorkflowReceivedAt) / 1000);
      return seconds;
    }
    const usage = state.data?.project_usage;
    return usage?.status === "recorded" ? Number(usage.total_wall_time_seconds) : null;
  }

  function studioTokenValue() {
    const usage = state.data?.live_usage;
    if (usage) return usage.total_tokens == null ? null : Number(usage.total_tokens);
    const legacy = state.data?.project_usage;
    return legacy?.status === "recorded" && legacy.benchmark_total_tokens != null ? Number(legacy.benchmark_total_tokens) : null;
  }

  function studioStats() {
    const seconds = studioElapsed();
    const tokens = studioTokenValue();
    const pending = Boolean(state.data?.live_usage?.active_tokens_pending);
    return `<div class="studio-stat"><span>项目耗时</span><strong data-studio-elapsed>${seconds == null ? "待记录" : esc(formatProjectDuration(seconds))}</strong></div><div class="studio-stat" title="${esc(state.data?.live_usage?.source || state.data?.project_usage?.benchmark_token_counter_source || "使用实际返回的 Token 记录")}"><span>${state.data?.live_usage?.external_author?.session_id ? "作者 Token（含缓存）" : "Token"} <i data-studio-token-pending>${pending ? "待更新" : ""}</i></span><strong data-studio-tokens>${tokens == null || !Number.isFinite(tokens) ? "待更新" : esc(formatProjectTokens(tokens, true))}</strong></div>`;
  }

  function designTimeline() {
    return state.historyRun === state.runId && state.history?.total ? state.history.items : state.data?.studio_preview?.timeline || [];
  }

  function designFrame() {
    const frames = designTimeline();
    if (!frames.length) return state.historyScope==="project" ? state.history?.selected || null : null;
    if (state.designFramePinned) {
      const pinned = frames.find((item) => item.id === state.designFramePinned) || (state.history?.selected?.id === state.designFramePinned ? state.history.selected : null);
      if (pinned) return pinned;
    }
    return frames[frames.length - 1];
  }

  function displayDesignFrame() {
    const selected = designFrame();
    if (selected?.model_url) return selected;
    if (selected && state.history?.selected?.id === selected.id && state.history?.display_fallback?.model_url) return state.history.display_fallback;
    const frames = designTimeline(), index = frames.findIndex(frame => frame.id === selected?.id);
    if (!selected?.created_at || index < 0) return null;
    return frames.slice(0,index).reverse().find(frame => frame.model_url && frame.created_at && Date.parse(frame.created_at) <= Date.parse(selected.created_at)) || null;
  }

  function designFrameIndex() {
    const frame = designFrame();
    const index=designTimeline().findIndex((item) => item.id === frame?.id);
    return index>=0 ? index : frame?.ordinal != null ? frame.ordinal-(state.history?.offset || 0) : -1;
  }

  function designFrameTime(value) {
    const ms = Date.parse(value || "");
    return Number.isFinite(ms) ? new Date(ms).toLocaleTimeString("zh-CN", { hour12: false }) : "";
  }

  function studioPreview() {
    const selected = studioAssets().find((item) => item.id === state.studioAssetId);
    if (selected) return {
      type: selected.details?.visual_role === "cad-scene" ? "model" : selected.preview_type === "video" ? "video" : selected.preview_type === "image" ? "image" : "file",
      url: mediaSource(selected), title: selected.title || selected.label || selected.path?.split(/[\\/]/).pop() || "产物",
      version: selected.sha256 || selected.path, status: "ready", source: "产物预览", artifact: selected,
    };
    const preview = state.data?.studio_preview;
    if (preview) {
      const original = preview.source_kind === "original_geometry";
      const frame = original ? designFrame() : null;
      const frameUrl = displayDesignFrame()?.model_url || "";
      return { type: original && (frame ? frameUrl : preview.model_url) ? "model" : original && !frame && preview.preview_url ? "image" : "empty",
        url: original ? (frame ? frameUrl : preview.model_url || preview.preview_url || "") : "",
        image: original ? preview.preview_url || "" : "", title: state.designFramePinned ? "历史模型" : "当前模型",
        version: frame?.id || preview.input_version || preview.source_revision || preview.model_url || "",
        status: preview.status || "pending", source: "原始几何", error: preview.error || "",
        revision: preview.source_revision || state.data?.live_workflow?.revision || state.data?.run?.revision,
      };
    }
    const scene = visual("cad-scene");
    const overview = visual("cad-overview");
    const originalScene = scene && !String(scene.details?.source_kind || "").includes("collider") && !String(scene.details?.source_kind || "").includes("proxy");
    return { type: originalScene && mediaSource(scene) ? "model" : overview && mediaSource(overview) ? "image" : "empty",
      url: originalScene ? mediaSource(scene) : mediaSource(overview), image: mediaSource(overview), title: "当前模型",
      version: scene?.sha256 || scene?.path || overview?.path || "", status: originalScene || overview ? "ready" : "pending", source: "原始几何",
      revision: state.data?.run?.revision,
    };
  }

  // The chosen motion, else the one with most frames, else the newest earlier revision's motion.
  function studioMotion() {
    const preview = state.data?.studio_preview;
    const items = preview?.motions || [];
    const fallback = preview?.fallback_motion;
    // A link may name a test that only has one frame in this revision; a choice in the list is kept as made.
    const chosen = items.find((item) => item.id === state.studioMotionId && (motionPlayable(item) || state.studioMotionPicked))
      || (fallback && fallback.id === state.studioMotionId ? fallback : null);
    return chosen || items.find((item) => item.id === preview?.best_motion_id)
      || fallback || items[0] || (preview?.motion_model_url ? preview : null);
  }

  function motionPlayable(motion) {
    return Number(motion?.motion_frame_count || 0) >= 2;
  }

  function studioMotionChoices() {
    const preview = state.data?.studio_preview;
    const fallback = preview?.fallback_motion;
    return [...(preview?.motions || []), ...(fallback && !(preview?.motions || []).some((item) => item.id === fallback.id) ? [fallback] : [])];
  }

  function watchMotionButton(motion) {
    const where = [motion.revision, motion.label || motion.load_case_id].filter(Boolean).join(" · ");
    return `<button type="button" class="studio-watch-motion" data-studio-mode="motion">▶ 观看运动回放<small><span translate="no">${esc(where)}</span> · ${esc(motion.motion_frame_count)} 帧</small></button>`;
  }

  function motionNotes(motion) {
    const notes = [];
    if (motion.revision && motion.revision !== state.data?.run?.revision) notes.push(`当前版本还没有可播放的运动，显示的是之前版本的运动：<span translate="no">${esc(motion.revision)}</span>`);
    if (motion.preview_kind === "gravity_settle") notes.push("运动预览：没有任何操作，只在重力下静置一小段时间，不是功能测试。");
    return notes.map((text) => `<small class="studio-motion-note">${text}</small>`).join("");
  }

  function studioViewKey() {
    const preview = studioPreview();
    if (state.studioMode === "motion") {
      const motion = studioMotion();
      return JSON.stringify([state.runId, "motion", motion?.id, motion?.motion_model_url, motion?.replay_url, motion?.replay_version]);
    }
    if (state.studioMode === "physics") return JSON.stringify([state.runId, "physics", state.loadId, mediaSource(visual("physics-evidence-scene", { capsuleId: load()?.capsule_id })), preview.version]);
    return JSON.stringify([state.runId, state.studioMode, state.compareOpen, state.compareData?.id, preview.type, preview.url, preview.version, preview.type === "empty" ? [preview.status, preview.error] : null]);
  }

  function projectHistoryChapters() {
    if(state.historyScope!=="project" || !state.history?.revisions)return "";
    return `<div class="history-chapters" aria-label="项目版本分界">${state.history.revisions.map(segment=>`<button type="button" data-history-chapter="${segment.offset}" ${segment.count ? "" : "disabled"} aria-current="${segment.run_id===state.runId ? "step" : "false"}">${esc(segment.revision)} · ${segment.count ? segment.count+" 条" : "无记录"}</button>`).join("")}</div>`;
  }

  async function toggleHistoryScope() {
    historySeekSerial++;
    stopHistoryPlay();state.compareOpen=false;compareController?.abort();
    const current=designFrame();
    state.historyScope=state.historyScope==="project" ? "revision" : "project";
    state.designFramePinned=current?.id || "";
    if(current?.capture==="legacy")state.designFramePinned=state.historyScope==="project" ? current.id : "";
    else if(state.historyScope==="project" && current && !current.capture)state.designFramePinned=`legacy:${state.data.run.revision}:${current.id}`;
    setDesignSelection(state.designFramePinned);
    state.selectionMode="pinned";state.historyPosition=null;state.historySource=null;
    state.history=null;state.historyRun="";
    updateUrl();render();await refreshHistory();refreshTrajectory();refreshHistorySource();
  }

  function studioTimelineBar() {
    const frames = designTimeline();
    const frame = designFrame();
    const index = Math.max(0, designFrameIndex()) + (state.history?.offset || 0);
    const total = state.history?.total || frames.length;
    const pinned = Boolean(state.designFramePinned);
    const time = designFrameTime(frame?.created_at);
    const building = ["building", "queued"].includes(frame?.preview_status || state.data?.studio_preview?.status);
    const failed = /FAIL|ERROR|incomplete|failed/.test(String(frame?.status || ""));
    return `<div class="studio-timeline" data-studio-timeline><span class="studio-timeline-title">修改历程</span>${boot.mode==="live" ? `<button type="button" data-history-scope aria-pressed="${state.historyScope==='project'}" title="${state.historyScope==='project' ? '回到所选版本的历史' : '串起这个项目所有版本的已保存变化'}">${state.historyScope==='project' ? '整个项目 · 返回当前版本' : '查看整个项目'}</button>` : ""}${projectHistoryChapters()}<button type="button" data-history-play ${total < 2 ? "disabled" : ""} aria-label="播放或暂停修改历程">${state.historyPlaying ? "暂停" : "▶"}</button><select data-history-speed aria-label="播放节奏">${[100,250,500,1000,2000].map(ms=>`<option value="${ms}" ${ms===state.historyInterval ? "selected" : ""}>${ms/1000} 秒</option>`).join("")}</select><div class="history-range-shell">${markerLayer(total)}<input type="range" min="0" max="${Math.max(0,total - 1)}" step="1" value="${index}" ${total<2 ? "disabled" : ""} data-timeline-frame aria-label="模型修改历程，拖动查看 Agent 本轮中的历次修改"></div><button type="button" data-history-pin="A" ${!frame ? "disabled" : ""}>${state.compareA ? "重设 A" : "设为 A"}</button><button type="button" data-history-pin="B" ${!frame ? "disabled" : ""}>${state.compareB ? "重设 B" : "设为 B"}</button><span class="studio-timeline-label" data-timeline-label>${frame ? esc(frame.revision || state.data?.run?.revision)+" · "+(frame.turn ? esc(frame.turn)+" · " : "") : ""}记录 ${total ? index + 1 : 0}/${total}${time ? ` · ${esc(time)}` : " · 时间未知"} · ${esc(({build:"工具构建",observed:"文件观测",recovered:"旧记录",legacy:"旧预览"})[frame?.capture] || "历史预览")}</span>${failed ? `<span class="studio-timeline-badge history-failed">尝试失败／未就绪</span>` : ""}${pinned ? `<span class="studio-timeline-badge">历史帧</span><button type="button" class="studio-timeline-action" data-timeline-trajectory>${frame?.capture ? "关联轨迹" : "项目轨迹"}</button><button type="button" class="studio-timeline-action" data-timeline-follow>跟随最新</button>` : `<span class="studio-timeline-live" data-timeline-live>${building ? "正在构建最新模型…" : "跟随最新"}</span>`}</div>`;
  }

  function setDesignSelection(frameId) {
    state.markerNotice=null;
    state.designFramePinned = frameId || "";
    state.trajectoryBefore = null; state.trajectoryTimeWindow = null; state.trajectoryFollow = !frameId;
    state.historySource = null; state.historySourceName = ""; state.historySourceOffset = 0; state.trajectoryFocusSnapshot = null; state.historyViewerNotice = "";
    state.trajectoryTurn="";
  }

  async function commitDesignFrame(frameId,seekSerial=historySeekSerial,fromPlayback=false) {
    const frame=designTimeline().find(item=>item.id===frameId) || (state.history?.selected?.id===frameId ? state.history.selected : null);
    if(state.historyScope==="project" && frame?.run_id && frame.run_id!==state.runId) {
      state.selectionMode="pinned";
      const result=await loadRun(frame.run_id,{historySelection:frame,historySeek:seekSerial});
      if(!result.ok && seekSerial===historySeekSerial)stopHistoryPlay();
      return;
    }
    setDesignSelection(frameId);
    updateUrl();
    if (fromPlayback && app.querySelector(".studio-model")) {
      applyHistoryPlaybackFrame();
      const frames = designTimeline();
      const index = designFrameIndex();
      if (index >= 0 && (index < 8 || index > frames.length - 8)) refreshHistory();
      return;
    }
    render();
    refreshHistory(); refreshTrajectory(); refreshHistorySource();
  }

  function openDesignFrameTrajectory() {
    state.studioDrawer = "activity"; state.mobileHistoryPane = "activity";
    state.trajectoryBefore = null; state.trajectoryTimeWindow = null;
    state.trajectoryFollow = !state.designFramePinned;
    render(); refreshTrajectory();
  }

  function historySelectionKey() {
    return JSON.stringify([state.runId, state.historyScope, state.designFramePinned, state.historyPosition]);
  }

  async function refreshHistory() {
    if (boot.mode !== "live" || !state.runId) return;
    const key = historySelectionKey();
    if (historyController && historyRequestKey === key) return;
    historyController?.abort();
    const controller = new AbortController(); historyController = controller; historyRequestKey = key;
    const run = state.runId;
    const query = new URLSearchParams({run,scope:state.historyScope});
    if (state.designFramePinned) query.set("selected", state.designFramePinned);
    if (state.historyPosition != null) query.set("around", String(state.historyPosition));
    try {
      const response = await fetch(`${boot.apiBase}/design-history?${query}`, {cache:"no-store", signal:controller.signal});
      if (!response.ok) throw new Error("历史记录暂不可用");
      const data = await response.json();
      if (key !== historySelectionKey()) return;
      const historyContents = value => JSON.stringify(value ? {...value, capture:{...value.capture, updated_ms:undefined, cpu_seconds:undefined}} : null);
      const previous = historyContents(state.history);
      state.history = data; state.historyRun = run; state.historyError = "";
      if(state.historyScope==="project" && !state.designFramePinned && state.historyPosition==null && !state.loading && !state.compareOpen && data.selected?.run_id && data.selected.run_id!==run) {
        await loadRun(data.selected.run_id,{historySelection:data.selected,historyFollow:true,historySeek:historySeekSerial});return;
      }
      if (previous !== historyContents(data)) { render(); refreshHistorySource(); }
    } catch (error) {
      if (!controller.signal.aborted && key === historySelectionKey()) state.historyError = error.message;
    } finally { if (historyController === controller) historyController = null; }
  }

  async function refreshHistorySource() {
    const selected = designFrame();
    if (state.studioDrawer !== "source" || !selected || selected.capture==="legacy" || !state.history?.total) return;
    const run = state.runId;
    const key = JSON.stringify([run, state.historyScope, selected.id, state.historySourceName, state.historySourceOffset, selected.source_file_count, selected.status]);
    if (state.historySource?._key === key) return;
    const serial = ++sourceRequestSerial;
    state.historySourceLoading = true;
    const query = new URLSearchParams({run, snapshot:selected.id, scope:state.historyScope, offset:String(state.historySourceOffset)});
    if (state.historySourceName) query.set("file",state.historySourceName);
    try {
      const response = await fetch(`${boot.apiBase}/design-source?${query}`,{cache:"no-store"});
      if (!response.ok) throw new Error("对应源码暂不可用");
      const data = await response.json();
      if (serial !== sourceRequestSerial || run !== state.runId || selected.id !== designFrame()?.id) return;
      state.historySource = {...data, _key:key}; state.historyError = "";
    } catch (error) { if (serial === sourceRequestSerial) state.historyError = error.message; }
    finally { if (serial === sourceRequestSerial) {state.historySourceLoading = false; render();} }
  }

  function historySourcePanel() {
    if(designFrame()?.capture==="legacy")return `<section class="history-source"><p>此旧预览没有可关联的源码快照。</p></section>`;
    const source = state.historySource;
    const valid = Boolean(source?.snapshot && source.snapshot === designFrame()?.id);
    return `<section class="history-source" data-source-key="${esc(source?._key || "")}"><p>${esc(state.historyError || (state.historySourceLoading ? "正在读取对应源码…" : "对应源码 · 已保存源码快照之间的差异"))}</p>
      ${valid ? `<label>文件<select data-history-source-file>${source.files.map(name=>`<option ${name===source.file ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></label>
      <p>${source.source_complete ? "已保存构建输入" : "源码关联可能不完整"}${source.partial ? " · 当前显示文件片段" : ""}</p>
      <details open><summary>源码差异${source.comparison_revision ? " · 对比 "+esc(source.comparison_revision) : ""}${source.comparison ? "" : " · 无可确认的前一版本"}</summary><pre>${esc(source.diff || "此片段没有变化")}</pre></details>
      <details><summary>源码内容</summary><pre>${esc(source.content || "此记录没有保存文本源码")}</pre></details>
      ${source.offset ? `<button type="button" data-history-source-first>回到文件开头</button>` : ""}
      ${source.next_offset != null ? `<button type="button" data-history-source-next>下一段</button>` : ""}` : `<p>选择一个已保存的历史记录查看源码。</p>`}</section>`;
  }

  function historyStatus() {
    const h = state.history || state.data?.studio_preview?.history;
    if (!h) return "";
    const capture = h.capture || {};
    const frame = designFrame();
    const fallback = displayDesignFrame();
    const pending = frame && !frame.model_url;
    return `<div class="history-status" role="status"><span data-history-viewer-notice>${esc(state.historyViewerNotice || "")}</span><span>${esc(capture.error ? "采集异常：" + capture.error : ({recording:"正在采集",idle:"采集已保存",interrupted:"采集中断",historical:"已有历史"})[capture.status] || "正在整理历史")}</span>
      <span>原始记录 ${(((h.storage?.original_asset_bytes || 0) + (h.storage?.indexed_log_bytes || 0))/1048576).toFixed(1)} MB · 预览缓存 ${((h.storage?.preview_cache_bytes || 0)/1048576).toFixed(1)} MB / 2 GB</span>
      ${pending ? `<span>${esc(frame.preview_error || (fallback ? `此记录尚无可用模型，显示 ${fallback.revision || "此前版本"} 保存的模型` : "预览尚未就绪"))}</span>` : ""}
      ${(h.markers || []).some(marker=>marker.ordinal==null && (!marker.run_id || marker.run_id===state.runId)) ? `<button type="button" data-revision-events title="这些记录未定位到具体模型">版本级事件 ${(h.markers || []).filter(marker=>marker.ordinal==null && (!marker.run_id || marker.run_id===state.runId)).length}</button>` : ""}
      ${frame?.capture === "observed" ? `<span>文件观测：不保证捕获瞬间反复覆盖的每个状态。</span>` : ""}</div>`;
  }

  function studioViewport() {
    const preview = studioPreview();
    let content;
    if (state.compareOpen) { content=comparisonViewport();
    } else if (state.studioMode === "motion" && studioMotion()) {
      const motion = studioMotion();
      content = !motionPlayable(motion) && motion.motion_frame_count != null
        ? `<div class="physics-evidence-workspace studio-motion-workspace studio-motion-empty"><p><strong>只记录到 1 帧，没有可播放的运动</strong>这次仿真开始后很快停止（通常是数值发散），原因见“验证与证据”。</p></div>`
        : `<div class="physics-evidence-workspace studio-motion-workspace">${motionNotes(motion)}<div class="studio-motion-model" data-model-src="${esc(motion.motion_model_url)}" data-scene-src="${esc(motion.scene_url)}" data-replay-src="${esc(motion.replay_url)}" data-mode="physics" role="application" aria-label="原始模型运动回放"><span class="viewer-loading" data-3d-status>正在载入原形运动记录…</span></div><div class="studio-motion-controls">${motion.status === "UNVERIFIED_PREFIX" ? `<small class="studio-partial-motion">未完成、未验证 · 仅 ${esc(motion.recorded_duration_s)} 秒 / ${esc(motion.motion_frame_count)} 帧 · 逐帧慢放</small>` : ""}<button type="button" data-physics-prev aria-label="上一帧">‹</button><button type="button" data-physics-play aria-label="播放或暂停真实运动记录">播放</button><button type="button" data-physics-next aria-label="下一帧">›</button><input type="range" min="0" max="1" value="0" data-physics-frame aria-label="运动时间轴"><span data-physics-time>0 s</span><small data-physics-frame-label></small></div></div>`;
    } else if (state.studioMode === "physics") {
      content = physicsEvidenceReplay(load());
    } else if (preview.type === "model") {
      content = `<div class="evidence-3d studio-model" data-model-src="${esc(preview.url)}" data-mode="design" role="application" aria-label="当前原始模型，拖动旋转、滚轮缩放、右键平移">${preview.image ? `<img class="evidence-3d-fallback" src="${esc(preview.image)}" alt="当前模型预览">` : ""}<span class="viewer-loading" data-3d-status>正在载入模型…</span></div>`;
    } else if (preview.type === "video") {
      content = `<video class="studio-media" controls playsinline loop src="${esc(preview.url)}"></video>`;
    } else if (preview.type === "image") {
      content = `<img class="studio-media" src="${esc(preview.url)}" alt="${esc(preview.title)}">`;
    } else if (preview.type === "file") {
      content = `<div class="studio-empty">${studioIcons.cube}<h2>${esc(preview.title)}</h2><p>此产物可下载后查看。</p><a class="studio-primary" href="${esc(preview.url)}" target="_blank" rel="noopener">打开产物 ↗</a></div>`;
    } else {
      const failed = /error|failed|unavailable/.test(String(preview.status).toLowerCase());
      content = `<div class="studio-empty" data-preview-status="${esc(preview.status)}">${studioIcons.cube}<h2>${failed ? "模型预览暂未就绪" : "等待模型生成"}</h2><p>${failed ? "原始产物已保留，预览服务将显示可用结果。" : "产物保存后会自动出现在这里。"}</p>${preview.error ? `<details><summary>查看原因</summary><p>${esc(typeof preview.error === "string" ? preview.error : JSON.stringify(preview.error))}</p></details>` : ""}<span class="studio-empty-hint">自动同步 · 无需手动发布</span></div>`;
    }
    const timelineBar = !state.compareOpen && state.studioMode === "design" && (designTimeline().length > 0 || boot.mode==="live") ? studioTimelineBar() : "";
    return `<div class="studio-view-surface" data-studio-view-key="${esc(studioViewKey())}" data-studio-mode="${esc(state.studioMode)}" data-studio-run="${esc(state.runId)}" data-history-project="${esc(projectForRun(state.runId)?.id || "")}">${content}${timelineBar}</div>`;
  }

  function launchAvailable() {
    return boot.mode === "live" && Boolean(features.launch) && !isImportedSnapshot();
  }

  function launchHref() {
    return `${boot.basePath || ""}/launch${location.hash}`;
  }

  function runsHref() {
    return `${boot.basePath || ""}/runs${location.hash}`;
  }

  function launchButtons() {
    const workflow = state.data?.live_workflow;
    const status = state.launchStatus;
    if (!launchAvailable() || workflow?.workflow_kind !== "slim_harness" || status?.project !== workflow.project_id) return "";
    const locked = humanDecisionCapability ? "" : ' disabled title="请用终端打印的带 #human= 的链接打开看板后再操作"';
    if (status.stopping) return `<button type="button" class="studio-launch-action" disabled>正在停止…</button>`;
    if (status.can_stop) return `<button type="button" class="studio-launch-action" data-launch-action="stop"${locked}>停止运行</button>`;
    if (status.can_resume) return `<button type="button" class="studio-launch-action" data-launch-action="resume"${locked}>继续运行</button>`;
    return "";
  }

  async function refreshLaunchStatus() {
    const serial = ++launchStatusSerial;
    clearTimeout(launchStatusTimer);
    const workflow = state.data?.live_workflow;
    const project = workflow?.workflow_kind === "slim_harness" ? workflow.project_id : "";
    if (!launchAvailable() || !project) { state.launchStatus = null; return; }
    try {
      const response = await fetch(`${boot.apiBase}/launch/status?project=${encodeURIComponent(project)}`, { cache: "no-store" });
      const payload = response.ok ? await response.json() : null;
      if (serial !== launchStatusSerial) return;
      if (payload) state.launchStatus = payload;
    } catch (_error) {
      if (serial !== launchStatusSerial) return;
    }
    const slot = app.querySelector("[data-launch-controls]");
    if (slot) slot.innerHTML = launchButtons();
    if (!document.hidden) launchStatusTimer = setTimeout(refreshLaunchStatus, 5000);
  }

  async function launchAction(action, button) {
    const project = state.data?.live_workflow?.project_id;
    if (!project || !humanDecisionCapability) return;
    if (action === "stop" && !window.confirm(tr("停止这个运行？Harness 会保存已有进度并写出停止摘要，之后可以继续运行。"))) return;
    button.disabled = true;
    try {
      const response = await fetch(`${boot.apiBase}/launch/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Human-Decision-Capability": humanDecisionCapability },
        body: JSON.stringify({ project }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || payload.error || `${response.status} ${response.statusText}`);
      state.launchStatus = payload;
      toast(action === "stop" ? "已请求停止，Harness 正在保存进度" : "已继续运行");
    } catch (error) {
      toast(`${action === "stop" ? "停止" : "继续运行"}失败：${error.message}`);
    }
    refreshLaunchStatus();
  }

  function studioHeader() {
    const data = state.data;
    const project = projectForRun(state.runId);
    const projects = state.projects.map((item) => `<option value="${esc(item.id)}" title="${esc(item.id)}" translate="no" ${item.id === project?.id ? "selected" : ""}>${esc(studioProjectName(item))}</option>`).join("");
    const versions = (project?.runs || state.runs).map((item) => `<option value="${esc(item.id)}" ${item.id === state.runId ? "selected" : ""}>${esc(item.revision || item.name)} · ${esc(statusLabel(item.status, true))}</option>`).join("");
    const picker = boot.mode === "live" ? `<div class="studio-project-picker"><select id="project-select" aria-label="切换机械项目">${projects}</select><span>/</span><select id="run-select" aria-label="切换项目版本">${versions}</select></div>` : `<strong class="studio-snapshot-title" translate="no">${esc(data.run.display_name || data.run.name)}</strong>`;
    return `<header class="studio-header"><a class="studio-brand" href="${esc(surfaceHref("/"))}" aria-label="机械工作台">${studioIcons.cube}<span>机械工作台</span></a>${picker}<button type="button" class="studio-follow ${state.selectionMode === "follow_current" ? "is-active" : ""}" data-studio-follow aria-pressed="${state.selectionMode === "follow_current"}">${state.selectionMode === "follow_current" ? "跟随当前" : "固定版本"}</button>${launchAvailable() ? `<span class="studio-launch-controls" data-launch-controls>${launchButtons()}</span>` : ""}<div class="studio-header-spacer"><small class="studio-current-activity" data-current-activity>暂无公开活动记录</small></div><button type="button" class="notice-enable" data-enable-notifications>${state.notificationsEnabled ? "通知已开启" : "开启通知"}</button><span class="notice-count" data-notice-count></span><div class="studio-live-stats" data-studio-stats>${studioStats()}</div><button type="button" class="studio-icon-button" data-studio-drawer="overview" aria-label="查看项目详情" title="项目详情">${studioIcons.info}</button>${boot.mode === "live" && !isImportedSnapshot() ? `<a class="studio-new-task" href="${esc(runsHref())}" title="所有实验的运行状态、已运行时间和排队情况">运行监控</a>` : ""}${launchAvailable() ? `<a class="studio-new-task" href="${esc(launchHref())}" title="定义并启动一个新的 Harness 运行">新任务</a>` : ""}<button id="export-button" type="button" class="studio-export" title="导出当前运行的离线快照">导出 ↗</button>${i18n.toggleMarkup()}</header>`;
  }

  async function pollProjectSummary() {
    if (boot.mode!=="live" || summaryPending) return;
    const generation=summaryGeneration;
    summaryPending=true;
    try {
      const response=await fetch(`${boot.apiBase}/project-summary`,{cache:"no-store"});
      if(!response.ok)throw new Error("项目状态暂不可用");
      const summary=await response.json();
      if(generation!==summaryGeneration)return;
      const previous=JSON.stringify(state.summaries);
      const fresh=noticeTracker.update(summary);
      state.summaries=Object.fromEntries(summary.projects.map(project=>[project.id,project]));
      state.summaryReady=summary.ready;
      try {localStorage.setItem("workbench-notice-seen",JSON.stringify(noticeTracker.seen));} catch {}
      for(const item of fresh) {
        if(state.notificationsEnabled && "Notification" in window && Notification.permission==="granted") {
          try {
            const notice=new Notification(`${item.status==="passed" ? "✓" : "!"} ${item.title} · ${item.revision}`,{body:tr(item.reason || statusLabel(item.status,true)),tag:item.id});
            notice.onclick=()=>{window.focus();noticeTracker.acknowledge(item.id);state.selectionMode="pinned";loadRun(item.run_id,{announce:true});notice.close();};
          } catch {state.notificationsEnabled=false;}
        }
      }
      if(previous!==JSON.stringify(state.summaries))updateProjectFilter();
      updateActivity(); refreshProjectCovers();
    } catch { /* The last known project verdict stays visible during disconnection. */ }
    finally {
      if(generation!==summaryGeneration)return;
      summaryPending=false;
      if(summaryTimer)clearTimeout(summaryTimer);
      summaryTimer=setTimeout(pollProjectSummary,document.hidden ? 15000 : 5000);
    }
  }

  async function enableNotifications() {
    if(!("Notification" in window)) {alert(tr("此浏览器不支持系统通知，标题与站内提醒仍可使用。"));return;}
    if(state.notificationsEnabled)state.notificationsEnabled=false;
    else {
      const permission=Notification.permission==="default" ? await Notification.requestPermission() : Notification.permission;
      state.notificationsEnabled=permission==="granted";
      if(permission==="denied")alert(tr("浏览器已禁止系统通知，可在浏览器的网站设置中修改。"));
    }
    try {localStorage.setItem("workbench-notifications",String(state.notificationsEnabled));} catch {}
    render();
  }

  function updateActivity() {
    const project=projectForRun(state.runId);
    const summary=state.summaries[project?.id] || {title:studioProjectName(project),status:state.data?.run?.status};
    document.title=tr(wb.title(summary,noticeTracker.unread.size));
    const count=app.querySelector("[data-notice-count]");
    if(count) {
      count.textContent=noticeTracker.unread.size ? `${noticeTracker.unread.size} 条新结果` : "";
      count.title=[...noticeTracker.unread.values()].map(item=>`${item.title}: ${item.reason || item.status}`).join("\n");
    }
    const node=app.querySelector("[data-current-activity]");
    const activity=state.activity;
    if(!node)return;
    if(!activity?.available) {node.textContent="暂无公开活动记录";return;}
    const prefix=activity.running ? "正在" : wb.age(activity.at_ms)+"：";
    if(activity.detail==null) {node.textContent=`${prefix}${activity.summary}`;return;}
    // The detail is agent text (message, command, error); only the label is UI text.
    const detail=document.createElement("span");detail.setAttribute("translate","no");detail.textContent=activity.detail;
    node.replaceChildren(`${prefix}${activity.label}：`,detail);
  }

  async function refreshConfig(force=false) {
    if(state.studioDrawer!=="config" || boot.mode!=="live" || !state.runId)return;
    const snapshot=state.studioMode==="design" ? state.designFramePinned || (state.history?.total ? designFrame()?.id : "") : "";
    const testId=state.studioMode==="motion" ? studioMotion()?.load_case_id || "" : state.loadId;
    const key=JSON.stringify([state.runId,snapshot,testId,state.data?.live_workflow?.active_turn?.id]);
    if(state.configKey===key && (configController || state.config && !force))return;
    if(state.studioMode==="design" && designFrame()?.capture==="legacy") {
      configController?.abort();configController=null;state.configLoading=false;
      if(state.configKey!==key){state.configKey=key;state.config={revision:designFrame().revision,note:"此旧预览没有可关联的配置快照；未记录的字段不补当前默认值。"};render();}
      return;
    }
    const changedContext=state.configKey!==key;
    configController?.abort();const controller=new AbortController();configController=controller;
    state.configKey=key;state.configLoading=true;
    if(changedContext){state.config=null;render();}
    const query=new URLSearchParams({run:state.runId});
    if(snapshot)query.set("snapshot",snapshot);
    if(testId)query.set("test",testId);
    try {
      const response=await fetch(`${boot.apiBase}/run-config?${query}`,{cache:"no-store",signal:controller.signal});
      if(!response.ok)throw new Error("此版本没有可读取的配置");
      const data=await response.json();if(key!==state.configKey)return;
      const changed=JSON.stringify(state.config)!==JSON.stringify(data);state.config=data;
      if(changed)render();
    } catch(error) {if(!controller.signal.aborted){state.config={error:error.message};render();}}
    finally {if(configController===controller){configController=null;state.configLoading=false;}}
  }

  function configurationPanel() {
    const data=state.config;
    if(!data)return `<p>${state.configLoading ? "正在读取当时的配置…" : "配置未记录"}</p>`;
    const generation=data.generation || {}, worker=generation.worker || {};
    const values=Object.entries({"后端":worker.provider,"模型":worker.model,"推理强度":worker.reasoning_effort,"轮次时限（秒）":worker.turn_timeout_seconds,"执行程序":worker.executable});
    return `<section class="run-configuration"><p>${esc(data.error || `配置 · ${data.revision}${data.test_id ? " / "+data.test_id : ""}`)}</p><p>${esc(data.note || "")}</p><h3>运行配置</h3><dl>${values.map(([key,value])=>`<div><dt>${key}</dt><dd>${esc(value==null ? "未记录" : value)}</dd></div>`).join("")}</dl>
      <details><summary>预算配置</summary><pre>${esc(Object.keys(generation.budget || {}).length ? JSON.stringify(generation.budget,null,2) : "未记录")}</pre></details>
      <details open><summary>机械参数</summary><p>独立几何参数：${data.mechanical?.parameters ? "已记录" : "未记录，可查看源码"}</p><pre>${esc(data.mechanical ? JSON.stringify(data.mechanical,null,2) : "未记录")}</pre></details>
      <details><summary>仿真设置 · ${data.simulations?.length || 0} 份记录</summary>${(data.simulations || []).map(item=>`<article><strong>${esc(item.test_id || "工况")} · ${item.kind==="effective" ? "实际应用值" : "运行请求"}</strong><pre>${esc(JSON.stringify(item,null,2))}</pre></article>`).join("") || "未记录"}</details>
      <details><summary>配置来源</summary><ul>${(data.sources || []).map(item=>`<li>${esc(item)}</li>`).join("")}</ul></details><button type="button" data-copy-config>复制配置 JSON</button></section>`;
  }

  let playGeneration=0;
  function historyPrefetchCount() {
    if (!state.historyPlaying) return 3;
    if (state.historyInterval <= 100) return 8;
    if (state.historyInterval <= 250) return 6;
    return 3;
  }
  function prefetchNearbyHistory() {
    const frames = designTimeline();
    const index = Math.max(0, designFrameIndex());
    window.Quest3D?.prefetch(app, frames.slice(index, index + historyPrefetchCount() + 1).map((item) => item.model_url).filter(Boolean));
  }
  function applyHistoryPlaybackFrame() {
    const bar = app.querySelector("[data-studio-timeline]");
    const frames = designTimeline();
    const frame = designFrame();
    const index = Math.max(0, designFrameIndex()) + (state.history?.offset || 0);
    const total = state.history?.total || frames.length;
    const range = bar?.querySelector("[data-timeline-frame]");
    if (range) {
      range.max = String(Math.max(0, total - 1));
      range.value = String(index);
    }
    const label = bar?.querySelector("[data-timeline-label]");
    if (label && frame) {
      const time = designFrameTime(frame.created_at);
      label.textContent = `${frame.revision || state.data?.run?.revision || ""}${frame.turn ? " · " + frame.turn : ""} · 记录 ${total ? index + 1 : 0}/${total}${time ? ` · ${time}` : " · 时间未知"} · ${({build:"工具构建",observed:"文件观测",recovered:"旧记录",legacy:"旧预览"})[frame.capture] || "历史预览"}`;
    }
    const model = app.querySelector(".studio-model");
    const url = displayDesignFrame()?.model_url || "";
    if (model && url) model.dataset.modelSrc = url;
    prefetchNearbyHistory();
    window.Quest3D?.mountAll(app);
  }
  function stopHistoryPlay() {
    state.historyPlaying=false;playGeneration++;
    if(historyPlayTimer)clearTimeout(historyPlayTimer);historyPlayTimer=null;
    const button=app.querySelector("[data-history-play]");if(button)button.textContent="▶";
    if (state.studioDrawer === "source") refreshHistorySource();
    if (state.studioDrawer === "activity") refreshTrajectory();
  }
  async function startHistoryPlay() {
    const total=state.history?.total || designTimeline().length;if(!total)return;
    state.historyPlaying=true;const generation=++playGeneration;
    const position=()=>Math.max(0,designFrameIndex())+(state.history?.offset || 0);
    if(position()>=total-1)await selectHistoryPosition(0,(state.history?.offset || 0)===0 ? designTimeline()[0] : null,true);
    render();
    prefetchNearbyHistory();
    const tick=async()=>{
      if(!state.historyPlaying || generation!==playGeneration || document.hidden)return;
      const frame=designFrame();
      const displayed=displayDesignFrame();
      const viewer=app.querySelector(".studio-model");
      const waitingBrowser=displayed?.model_url && viewer?.dataset.viewerStatus==="loading";
      if(state.loading || waitingBrowser || ["queued","building","deferred"].includes(frame?.preview_status)) {
        if(!historyController) refreshHistory();
        prefetchNearbyHistory();
        historyPlayTimer=setTimeout(tick, Math.min(80, state.historyInterval));
        return;
      }
      const next=position()+1;
      if(next>=(state.history?.total || designTimeline().length)){stopHistoryPlay();return;}
      await selectHistoryPosition(next,designTimeline()[next-(state.history?.offset || 0)],true);
      if(generation===playGeneration)historyPlayTimer=setTimeout(tick,state.historyInterval);
    };
    historyPlayTimer=setTimeout(tick,state.historyInterval);
  }

  function filteredProjects() { return wb.filterProjects(state.projects,state.projectSearch,state.projectStatusFilter,state.summaries); }

  function projectListKey(filtered = filteredProjects()) {
    return JSON.stringify([
      filtered.items.map(project => [project.id, studioProjectName(project),
        state.summaries[project.id]?.status || project.status]),
      filtered.counts,
      state.projectStatusFilter,
      state.projects.length,
      teamLibraryTree(),
    ]);
  }

  function syncProjectSelection(root = app) {
    const selected = projectForRun(state.runId)?.id;
    root.querySelectorAll("[data-studio-project]").forEach(button => {
      const active = button.dataset.studioProject === selected;
      if (button.classList.contains("is-active") !== active) button.classList.toggle("is-active", active);
      if (active && button.getAttribute("aria-current") !== "true") button.setAttribute("aria-current", "true");
      if (!active && button.hasAttribute("aria-current")) button.removeAttribute("aria-current");
    });
  }

  function projectButtons(projects) {
    const selected=projectForRun(state.runId);
    return projects.map(project => {
      const item=project.latest || project.runs?.[0], label=studioProjectName(project);
      const summary=state.summaries[project.id];
      const imageUrl=state.thumbnails[project.id]?.url || summary?.thumbnail_url || (item?.revision_preview_path ? revisionPreviewUrl(item) : "");
      return `<button type="button" class="studio-project-thumb ${project.id===selected?.id ? "is-active" : ""}" data-studio-project="${esc(project.id)}" title="${esc(label)}"><span class="project-cover">${imageUrl ? `<img src="${esc(imageUrl)}" loading="lazy" alt="" data-studio-thumbnail>` : ""}<strong class="studio-project-initials" ${imageUrl ? "hidden" : ""}>${esc(label.slice(0,2))}</strong></span><span class="project-list-copy"><strong translate="no">${esc(label)}</strong><small>${esc(project.id)}</small><small>${esc(statusLabel(summary?.status || project.status,true))}</small></span><i class="studio-project-state" data-status="${esc(summary?.status || project.status)}"></i></button>`;
    }).join("") || '<p class="project-empty">没有匹配的项目</p>';
  }

  function observeProjectCovers() {
    thumbnailObserver?.disconnect(); visibleProjects.clear();
    if(!("IntersectionObserver" in window))return;
    thumbnailObserver=new IntersectionObserver(entries=>{
      for(const entry of entries) {const id=entry.target.dataset.studioProject; entry.isIntersecting ? visibleProjects.add(id) : visibleProjects.delete(id);}
      refreshProjectCovers();
    },{root:app.querySelector(".studio-project-scroll"),rootMargin:"120px"});
    app.querySelectorAll("[data-studio-project]").forEach(node=>thumbnailObserver.observe(node));
  }

  async function refreshProjectCovers() {
    if(document.hidden || boot.mode!=="live")return;
    for(const id of visibleProjects) {
      if(thumbnailPending.has(id) || Date.now()-(state.thumbnails[id]?.checkedAt || 0)<5000)continue;
      const project=state.projects.find(p=>p.id===id);
      const run=project?.default_run_id || project?.current_run_id || project?.runs?.[0]?.id;if(!run)continue;
      thumbnailPending.add(id);
      fetch(`${boot.apiBase}/project-thumbnail?run=${encodeURIComponent(run)}`,{cache:"no-store"}).then(response=>response.ok ? response.json() : null).then(data=>{
        const previous=state.thumbnails[id] || {};
        const url=data?.status==="ready" ? data.urls?.[0] : previous.url;
        state.thumbnails[id]={url,checkedAt:Date.now()};
        if(url && url!==previous.url) {
          const button=[...app.querySelectorAll("[data-studio-project]")].find(node=>node.dataset.studioProject===id);
          const cover=button?.querySelector(".project-cover");
          if(cover) {cover.innerHTML=`<img src="${esc(url)}" loading="lazy" alt="" data-studio-thumbnail><strong class="studio-project-initials" hidden>${esc(project.title?.slice(0,2) || "模型")}</strong>`;bindStudioMediaEvents();}
        }
      }).catch(()=>{state.thumbnails[id]={...state.thumbnails[id],checkedAt:Date.now()};}).finally(()=>thumbnailPending.delete(id));
    }
  }

  function markerLayer(total) {
    const markers=state.history?.markers || state.data?.studio_preview?.history?.markers || [];
    const grouped=new Map();
    for(const marker of markers) {
      if(marker.ordinal==null)continue;
      const group=grouped.get(marker.ordinal) || [];group.push(marker);grouped.set(marker.ordinal,group);
    }
    return `<div class="history-event-markers">${[...grouped].map(([ordinal,items])=>`<button type="button" data-history-marker="${ordinal}" style="left:${100*ordinal/Math.max(1,total-1)}%" class="${items.some(item=>/failed/.test(item.kind)) ? "is-failure" : ""}" aria-label="${esc(items.map(item=>item.label).join('；'))}" title="${esc(items.map(item=>item.label).join('；'))}">${items.length>1 ? items.length : "•"}</button>`).join("")}</div>`;
  }

  function pickComparison(which) {
    const frame=designFrame();if(!frame)return;
    stopHistoryPlay();
    const pin={project:projectForRun(state.runId)?.id,run:state.runId,snapshot:frame.legacy_id || frame.id,label:`${state.data.run.revision} · ${designFrameTime(frame.created_at) || frame.id}`};
    state[which==="A" ? "compareA" : "compareB"]=pin;state.compareData=null;
    if(state.compareA && state.compareB) {state.compareOpen=true;refreshComparison();}
    render();
  }

  async function refreshComparison() {
    if(document.hidden || !state.compareOpen || !state.compareA || !state.compareB)return;
    if(compareTimer)clearTimeout(compareTimer);
    compareController?.abort();const controller=new AbortController();compareController=controller;
    const key=JSON.stringify([state.compareA,state.compareB,state.compareView]);
    const query=new URLSearchParams({run:state.compareA.run,a:state.compareA.snapshot,other_run:state.compareB.run,b:state.compareB.snapshot,view:state.compareView});
    try {
      const response=await fetch(`${boot.apiBase}/history-compare?${query}`,{cache:"no-store",signal:controller.signal});
      const data=await response.json();if(!response.ok)throw new Error(data.detail || "无法对比所选模型");
      if(key!==JSON.stringify([state.compareA,state.compareB,state.compareView]) || !state.compareOpen)return;
      const changed=JSON.stringify(state.compareData)!==JSON.stringify(data);state.compareData=data;if(changed)render();
      if(["queued","building","deferred","ready"].includes(data.status))compareTimer=setTimeout(refreshComparison,data.status==="ready" ? 5000 : 1000);
    } catch(error) {if(!controller.signal.aborted){state.compareData={status:"failed",error:error.message};render();}}
    finally {if(compareController===controller)compareController=null;}
  }

  function comparisonViewport() {
    const data=state.compareData;
    const ready=data?.status==="ready" && data.urls?.length===2;
    return `<div class="history-comparison"><header><strong>A/B 原始模型对比</strong><select data-history-compare-view>${Object.entries({iso:"等轴测",front:"正视",side:"侧视",top:"俯视"}).map(([value,label])=>`<option value="${value}" ${value===state.compareView ? "selected" : ""}>${label}</option>`).join("")}</select><button type="button" data-history-compare-close>退出对比</button></header>
      <div class="revision-compare-viewport" style="--revision-wipe:${state.compareWipe}%">${ready ? `<img class="revision-preview-image revision-compare-base" src="${esc(data.urls[0])}" alt="A 模型"><div class="revision-compare-target-layer"><img class="revision-preview-image" src="${esc(data.urls[1])}" alt="B 模型"></div><span class="revision-compare-divider"></span>` : `<p class="compare-pending">${esc(data?.error || "正在准备同视角图片…")}</p>`}</div>
      <label>A <input data-history-wipe type="range" min="0" max="100" value="${state.compareWipe}"> B</label><small>A：${esc(state.compareA?.label || "")} · B：${esc(state.compareB?.label || "")} · 共用尺度，仅用于视觉定位变化</small></div>`;
  }

  function projectStatusChips(counts) {
    return Object.entries({all:"全部",running:"运行中",passed:"通过",failed:"失败",blocked:"阻塞",preparing:"准备中"}).map(([key,label])=>`<button type="button" data-project-status="${key}" aria-pressed="${state.projectStatusFilter===key}">${label} ${counts[key] || 0}</button>`).join("");
  }

  function projectRailKey() {
    return JSON.stringify([
      projectListKey(),
      state.projectSearch,
      state.projectStatusFilter,
      state.projectRailCollapsed,
      state.teamLibrary.local_url,
      (state.teamLibrary.items || []).map((item) => [item.id, item.status, item.url]),
      location.pathname,
    ]);
  }

  function studioProjectRail() {
    const filtered=filteredProjects();
    return `<nav class="studio-project-rail ${state.projectRailCollapsed ? "is-collapsed" : ""}" data-rail-key="${esc(projectRailKey())}" data-project-list-key="${esc(projectListKey(filtered))}" aria-label="机械项目"><header><strong>机械项目</strong><button type="button" data-project-collapse aria-label="展开或折叠项目列表">${state.projectRailCollapsed ? "›" : "‹"}</button></header><div class="project-filter"><input id="project-search" data-project-search type="search" value="${esc(state.projectSearch)}" placeholder="搜索名称或 ID" aria-label="搜索项目"><div class="project-status-chips">${projectStatusChips(filtered.counts)}</div></div><div class="studio-project-scroll">${teamLibraryTree()}${projectButtons(filtered.items)}</div><span class="studio-rail-count">${filtered.items.length} / ${state.projects.length} 项目</span></nav>`;
  }

  function teamFolderOpen(key, current) {
    if (state.projectSearch) return true;
    return teamFolders.has(key) ? teamFolders.get(key) : key === "team" || current;
  }

  function teamLibraryTree() {
    const library = state.teamLibrary;
    if (!features.team_library || !library.items.length) return '';
    const groups = new Map();
    const search = state.projectSearch.toLowerCase();
    const currentPath = location.pathname;
    const containsCurrent = (items) => items.some((item) => currentPath.startsWith(item.url));
    for (const item of library.items) {
      if (search && !`${item.owner} ${item.project} ${item.id}`.toLowerCase().includes(search)) continue;
      if (!groups.has(item.owner)) groups.set(item.owner, new Map());
      const projects = groups.get(item.owner);
      if (!projects.has(item.project)) projects.set(item.project, []);
      projects.get(item.project).push(item);
    }
    const folder = (key, label, body, current=false, href="") => {
      const title = href
        ? `<a class="team-project" href="${esc(href)}"${current ? ' aria-current="page"' : ""}>${esc(label)}</a>`
        : esc(label);
      return `<details data-team-folder="${esc(key)}" ${teamFolderOpen(key, current) ? "open" : ""}><summary>${title}</summary>${body}</details>`;
    };
    const members = [...groups].map(([owner, projects]) => {
      const ownerItems = [...projects.values()].flat();
      return folder(owner, owner,
        [...projects].map(([project, snapshots]) => folder(`${owner}/${project}`, project,
          snapshots.slice().reverse().map(item => `<a class="team-snapshot" href="${esc(item.url)}" title="${esc(item.id)}" ${currentPath.startsWith(item.url) ? 'aria-current="page"' : ''}><time>${esc(new Date(item.created_at).toLocaleString())}</time><span>${esc(statusLabel(item.status, true))}</span></a>`).join(''),
          containsCurrent(snapshots),
          snapshots[snapshots.length - 1].url)).join(''),
        containsCurrent(ownerItems));
    }).join('');
    const onLocal = !currentPath.includes("/imports/");
    return `<section class="team-library"><a class="team-library-home" href="${esc(library.local_url)}"${onLocal ? ' aria-current="page"' : ""}>本地项目</a>${folder('team', '团队快照', members, true)}</section>`;
  }

  function bindProjectButtons(root=app) {
    root.querySelectorAll("[data-studio-project]").forEach(button=>{
      if (button.dataset.projectBound) return; button.dataset.projectBound="true";
      button.addEventListener("click",()=>{
        const project=state.projects.find(item=>item.id===button.dataset.studioProject);
        const run=project?.default_run_id || project?.current_run_id || project?.runs?.[0]?.id;
        state.selectionMode="follow_current";state.studioAssetId="";if(run) loadRun(run,{announce:true});
      });
    });
  }

  function updateProjectFilter() {
    const rail=app.querySelector(".studio-project-rail");if(!rail)return;
    const filtered=filteredProjects();
    const key=projectListKey(filtered);
    if (rail.dataset.projectListKey === key) { rail.dataset.railKey=projectRailKey(); syncProjectSelection(rail); return; }
    const scroll=rail.querySelector(".studio-project-scroll"), scrollTop=scroll.scrollTop;
    rail.querySelector(".studio-project-scroll").innerHTML=teamLibraryTree()+projectButtons(filtered.items);
    rail.querySelector(".project-status-chips").innerHTML=projectStatusChips(filtered.counts);
    rail.querySelector(".studio-rail-count").textContent=`${filtered.items.length} / ${state.projects.length} 项目`;
    rail.dataset.projectListKey=key;rail.dataset.railKey=projectRailKey();
    scroll.scrollTop=scrollTop;
    syncProjectSelection(rail);
    bindProjectButtons(rail); observeProjectCovers();
  }

  function studioAssets() {
    const stored = (state.data?.evidence || []).filter((item) => item.exists && (item.path || item.preview_data_uri) && ["interactive-model", "step", "visual-image", "visual-video", "package"].includes(item.kind));
    const preview = state.data?.studio_preview;
    const generated = [];
    if (preview?.model_url) generated.push({ id: "studio-original-model", kind: "interactive-model", title: "当前原形 · 3D", exists: true,
      preview_data_uri: preview.model_url, details: { visual_role: "cad-scene", source_kind: "original_geometry" } });
    if (preview?.source_url) generated.push({ id: "studio-original-file", kind: "step", title: "模型源文件", exists: true,
      preview_data_uri: preview.source_url, path: preview.source_path });
    return [...generated, ...stored];
  }

  function versionReason(item) { const project=state.summaries[item.project_id]; return project?.versions?.find(row=>row.id===item.id)?.reason || item.failure_summary || "原因未记录"; }

  function studioNavigator() {
    const project = projectForRun(state.runId);
    const versions = project?.runs || state.runs;
    const assets = studioAssets();
    const rows = state.studioPanel === "assets"
      ? `<button type="button" class="studio-asset-row ${!state.studioAssetId ? "is-active" : ""}" data-studio-asset=""><span class="studio-asset-icon">${studioIcons.cube}</span><span><strong>当前原始模型</strong><small>自动同步的制作预览</small></span></button>${assets.map((item) => `<button type="button" class="studio-asset-row ${state.studioAssetId === item.id ? "is-active" : ""}" data-studio-asset="${esc(item.id)}"><span class="studio-asset-icon">${fileIcon}</span><span><strong>${esc(item.title || item.label || item.path?.split(/[\\/]/).pop() || item.kind)}</strong><small>${esc(item.preview_type === "video" ? "回放" : item.preview_type === "image" ? "图片" : item.kind === "step" ? "STEP 文件" : "产物")}</small></span></button>`).join("")}`
      : versions.map((item) => `<button type="button" class="studio-version-row ${item.id === state.runId ? "is-active" : ""}" data-studio-version="${esc(item.id)}"><span class="studio-version-node" data-status="${esc(item.status)}"></span><span><strong>${esc(item.revision || item.name)}${item.id === (project?.default_run_id || project?.current_run_id) ? '<i>当前</i>' : ""}</strong><small>${esc(statusLabel(item.status, true))}</small>${["failed","blocked"].includes(item.status) ? `<small class="version-reason" title="${esc(versionReason(item))}">${esc(versionReason(item))}</small>` : ""}</span><time>${esc(fmtTime(item.updated_at).split(" ").pop())}</time></button>`).join("");
    return `<aside class="studio-navigator" aria-label="版本与产物" ${!state.studioSidebarOpen ? "hidden" : ""}><div class="studio-nav-heading"><span>项目空间</span><strong title="${esc(`${studioProjectName(project)}\n${project?.id || state.data.run.project_id || state.data.run.name}`)}">${esc(studioProjectName(project))}</strong></div><div class="studio-nav-tabs" role="tablist" aria-label="项目内容"><button type="button" role="tab" data-studio-panel="versions" aria-selected="${state.studioPanel === "versions"}">版本 <span>${versions.length}</span></button><button type="button" role="tab" data-studio-panel="assets" aria-selected="${state.studioPanel === "assets"}">产物 <span>${assets.length}</span></button></div><div class="studio-nav-list">${rows || '<p class="studio-list-empty">产物生成后自动更新</p>'}</div><button id="revision-replay-button" type="button" class="studio-history-button" aria-pressed="${state.revisionReplayOpen}">↔ 查看版本演变</button><div class="studio-nav-footer"><i></i>产物自动发现与同步</div></aside>`;
  }

  function trajectoryPanel() {
    if (boot.mode !== "live") return `<p>Agent 轨迹在实时工作台中查看；离线模型快照不附带作者会话。</p>`;
    const data = state.trajectoryRun === state.runId ? state.trajectory : null;
    const visibleItems = wb.searchEvents(data?.items || [], state.trajectorySearch);
    const status = { in_progress: "执行中", running: "执行中", started: "执行中", completed: "已结束", failed: "失败", error: "失败" };
    const card = (item) => {
      const id = `${state.runId}:${data.turn}:${item.id}`;
      const stamp = item.at || item.at_ms ? `<time>${esc(fmtTime(item.at || new Date(item.at_ms).toISOString()))}</time>` : "";
      if (item.kind === "tool") return `<details class="trajectory-event trajectory-tool" data-trajectory-id="${esc(id)}" data-event-seq="${esc(item.event_seq || "")}" ${trajectoryExpanded.has(id) ? "open" : ""}><summary><span class="trajectory-kind">工具</span><strong>${esc(item.title)}</strong><span>${esc(status[item.status] || item.status || "已记录")}</span></summary><div><small>调用</small><pre>${esc(item.text)}</pre>${item.output ? `<small>执行结果</small><pre>${esc(item.output)}</pre>` : ""}${item.exit_code != null ? `<small>退出码 ${esc(item.exit_code)}</small>` : ""}${stamp}</div></details>`;
      return `<article class="trajectory-event trajectory-${esc(item.kind)}"><header><span class="trajectory-kind">${esc(item.title)}</span>${stamp}</header><pre>${esc(item.text || "已记录")}</pre></article>`;
    };
    return `<div class="agent-trajectory" data-agent-trajectory>
      <p class="trajectory-scope">项目级 Agent 轨迹 · 跨版本保留 · 只读</p>
      <div class="trajectory-toolbar"><label>作者轮次<select data-trajectory-turn aria-label="选择作者轮次"><option value="">跟随最新轮次</option>${(data?.turns || []).map((turn) => `<option value="${esc(turn.id)}" ${state.trajectoryTurn === turn.id ? "selected" : ""}>${esc(turn.label)}</option>`).join("")}</select></label><button type="button" data-trajectory-refresh ${state.trajectoryLoading ? "disabled" : ""}>刷新</button></div>
      <button type="button" data-history-live>回到最新</button><label class="trajectory-follow"><input type="checkbox" data-trajectory-follow ${state.trajectoryFollow ? "checked" : ""}> 跟随最新消息</label>
      ${state.trajectoryTimeWindow ? `<p class="trajectory-window" role="status">时段过滤 ${esc(designFrameTime(new Date(state.trajectoryTimeWindow.from_ms).toISOString()))} – ${state.trajectoryTimeWindow.to_ms != null ? esc(designFrameTime(new Date(state.trajectoryTimeWindow.to_ms).toISOString())) : "现在"} <button type="button" data-trajectory-clear-window>清除</button></p>` : ""}
      ${state.trajectoryError ? `<p class="trajectory-error" role="status">${esc(state.trajectoryError)}</p>` : ""}
      ${!data ? `<p role="status">${state.trajectoryLoading ? "正在读取会话记录…" : "尚未读取会话记录"}</p>` : `
        <details class="trajectory-task"><summary>任务要求</summary><pre>${esc(data.task)}</pre></details>
        <div class="trajectory-page"><span>${esc(data.turn || "暂无作者轮次")}${state.trajectoryBefore != null ? " · 较早记录" : ""}</span>${data.next_before != null ? `<button type="button" data-trajectory-earlier>更早记录</button>` : ""}${state.trajectoryBefore != null || state.trajectoryTurn ? `<button type="button" data-trajectory-latest>回到最新</button>` : ""}</div>
        <label class="trajectory-search-label">搜索已加载记录<input id="trajectory-search" data-trajectory-search type="search" value="${esc(state.trajectorySearch)}" placeholder="消息、工具或输出"></label><small data-trajectory-match-count>${visibleItems.length} / ${data.items.length} 条已加载记录</small>
        <p class="history-association">${esc(state.markerNotice || "")}</p><p class="history-association">${esc(data.message || (data.unknown_time_count ? `${data.unknown_time_count} 条旧事件缺少可靠时间戳` : data.association === "turn_only" ? "已定位作者轮次，具体调用关联未知" : ""))}</p><div class="trajectory-stream">${visibleItems.map(card).join("") || `<p>${data.events_recorded ? "本页没有可展示的消息；正在执行的记录会自动更新。" : "此轮未保存工具事件；下面仅展示实际保存的交付结果。"}</p>`}</div>
        ${data.response?.summary ? `<article class="trajectory-event trajectory-response"><header>作者本轮交付 · ${esc(data.response.status || "已保存")}</header><pre>${esc(data.response.summary)}</pre>${data.response.issues ? `<pre>${esc(data.response.issues)}</pre>` : ""}</article>` : ""}
        ${Object.keys(data.feedback || {}).length ? `<details class="trajectory-feedback"><summary>Harness ${esc(data.selected_revision)} 反馈 · ${esc(data.feedback.status || "已记录")}</summary>${data.feedback.issues ? `<h4>检查反馈</h4><pre>${esc(data.feedback.issues)}</pre>` : ""}${data.feedback.pending_tests ? `<h4>未完成测试</h4><pre>${esc(data.feedback.pending_tests)}</pre>` : ""}</details>` : ""}
        <small class="trajectory-note">按记录顺序显示公开消息、工具和返回结果。长输出会截断显示；未保存的事件不补写。</small>`}
      </div>`;
  }

  function bindTrajectoryEvents() {
    app.querySelector("[data-trajectory-search]")?.addEventListener("input", event => {
      state.trajectorySearch=event.target.value;
      const old=app.querySelector("[data-agent-trajectory]");
      const template=document.createElement("template");template.innerHTML=trajectoryPanel();
      const next=template.content.querySelector(".trajectory-stream");
      old?.querySelector(".trajectory-stream")?.replaceWith(next);
      bindTrajectoryCards();
      const count=app.querySelector("[data-trajectory-match-count]");
      if(count)count.textContent=`${wb.searchEvents(state.trajectory?.items || [],state.trajectorySearch).length} / ${state.trajectory?.items?.length || 0} 条已加载记录`;
    });
    app.querySelector("[data-history-live]")?.addEventListener("click", () => followLatestHistory());
    app.querySelector("[data-trajectory-turn]")?.addEventListener("change", (event) => {
      state.trajectoryTurn = event.target.value; state.trajectoryBefore = null; state.designFramePinned = ""; updateUrl(); refreshTrajectory();
    });
    app.querySelector("[data-trajectory-refresh]")?.addEventListener("click", () => refreshTrajectory());
    app.querySelector("[data-trajectory-earlier]")?.addEventListener("click", () => {
      state.trajectoryTurn = state.trajectory.turn; state.trajectoryBefore = state.trajectory.next_before;
      state.trajectoryFollow = false; refreshTrajectory();
    });
    app.querySelector("[data-trajectory-latest]")?.addEventListener("click", () => {
      followLatestHistory();
    });
    app.querySelector("[data-trajectory-clear-window]")?.addEventListener("click", () => {
      state.trajectoryTimeWindow = null; refreshTrajectory();
    });
    app.querySelector("[data-trajectory-follow]")?.addEventListener("change", (event) => { state.trajectoryFollow = event.target.checked; });
    bindTrajectoryCards();
  }

  function bindTrajectoryCards() {
    app.querySelectorAll("[data-trajectory-id]").forEach((node) => node.addEventListener("toggle", () => {
      if (!node.isConnected) return;
      if (node.open) trajectoryExpanded.add(node.dataset.trajectoryId); else trajectoryExpanded.delete(node.dataset.trajectoryId);
    }));
  }

  async function refreshTrajectory() {
    if (boot.mode !== "live" || state.studioDrawer !== "activity" || !state.data?.live_workflow || !state.runId || state.trajectoryLoading) return;
    const runId = state.runId;
    if (state.trajectoryRun !== runId) {
      state.trajectory = null; state.trajectoryTurn = ""; state.trajectoryBefore = null; trajectoryExpanded.clear();
    }
    state.trajectoryRun = runId; state.trajectoryLoading = true; state.trajectoryError = "";
    const requestedTurn = state.trajectoryTurn;
    const requestedBefore = state.trajectoryBefore;
    if(state.designFramePinned && designFrame()?.capture==="legacy") {
      state.trajectory={items:[],message:"此旧预览没有可确认的作者轮次或工具调用关联。",association:"unknown"};
      state.trajectoryRun=state.runId;state.trajectoryLoading=false;render();return;
    }
    const requestedSnapshot = state.designFramePinned;
    const requestedWindow = JSON.stringify(state.trajectoryTimeWindow);
    const currentSelection = () => runId === state.runId && requestedTurn === state.trajectoryTurn && requestedBefore === state.trajectoryBefore && requestedSnapshot === state.designFramePinned && requestedWindow === JSON.stringify(state.trajectoryTimeWindow);
    const query = new URLSearchParams({ run: runId });
    if (state.trajectoryTurn) query.set("turn", state.trajectoryTurn);
    if (state.designFramePinned && state.history?.total) query.set("snapshot", state.designFramePinned);
    const incremental = !state.designFramePinned && !state.trajectoryTimeWindow && state.trajectoryBefore == null && state.trajectoryFollow && state.trajectory?.cursor_kind === "sequence" && state.trajectory?.next_after != null;
    if (incremental) query.set("after", String(state.trajectory.next_after));
    if (state.trajectoryBefore != null) query.set("before", state.trajectoryBefore);
    if (state.trajectoryTimeWindow) {
      query.set("from_ms", String(state.trajectoryTimeWindow.from_ms));
      if (state.trajectoryTimeWindow.to_ms != null) query.set("to_ms", String(state.trajectoryTimeWindow.to_ms));
    }
    try {
      const response = await fetch(`${boot.apiBase}/agent-trajectory?${query}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`读取轨迹失败（${response.status}）`);
      const data = await response.json();
      if (!currentSelection()) return;
      if (incremental && data.turn === state.trajectory?.turn) {
        const combined = [...state.trajectory.items];
        for (const item of data.items) {
          const at = combined.findIndex(old => item.event_key ? old.event_key === item.event_key : old.id === item.id);
          if (at >= 0) combined[at] = {...item, id:combined[at].id}; else combined.push(item);
        }
        const retained = combined.slice(-100);
        data.items = retained;
        data.next_before = retained.length && (combined.length > 100 || state.trajectory.next_before != null) ? "seq:" + Math.min(...retained.map(item => Number(item.id))) : null;
      }
      state.trajectory = data;
    } catch (error) {
      if (currentSelection()) state.trajectoryError = error.message;
    } finally {
      state.trajectoryLoading = false;
      if (!currentSelection()) { refreshTrajectory(); return; }
      if (runId === state.runId && state.studioDrawer === "activity") {
        const slot = app.querySelector("[data-agent-trajectory]");
        const scroll = slot?.closest(".studio-drawer-body");
        const position = scroll?.scrollTop || 0;
        const nearEnd = !scroll || scroll.scrollHeight - position - scroll.clientHeight < 80;
        const viewKey = JSON.stringify([state.trajectory, state.trajectoryTurn, state.trajectoryBefore, state.trajectoryError, state.trajectoryFollow, state.trajectorySearch]);
        if (slot && state.trajectoryViewKey !== viewKey) {
          state.trajectoryViewKey = viewKey;
          slot.outerHTML = trajectoryPanel(); bindTrajectoryEvents();
          if (scroll) scroll.scrollTop = state.trajectoryFollow && nearEnd ? scroll.scrollHeight : position;
          if (scroll && state.trajectory?.focus_event && state.trajectoryFocusSnapshot !== state.designFramePinned) {
            const event = app.querySelector(`[data-event-seq="${state.trajectory.focus_event}"]`);
            if (event) scroll.scrollTop += event.getBoundingClientRect().top - scroll.getBoundingClientRect().top - 30;
            state.trajectoryFocusSnapshot = state.designFramePinned;
          }
        }
      }
    }
  }

  function studioTestOutcome(test) {
    if (test.execution_status === "running") return "正在运行";
    if (test.execution_status === "pending_result") return "结果未完成";
    if (test.execution_started === false || test.execution_status === "not_run") return "未运行";
    if (test.execution_status === "SIMULATION_TIMEOUT" || test.failure_status === "SIMULATION_TIMEOUT") return "超时";
    if (test.failure_status === "HARNESS_ERROR") return "运行错误";
    if (test.physical_pass && test.geometry_audit?.passed) return "通过";
    if (test.physical_pass && ["pending_audit", "not_completed"].includes(test.geometry_audit?.status)) return "物理已跑，检查未完成";
    return "失败";
  }

  function studioLiveTests() {
    const workflow = state.data?.live_workflow;
    if (Number(String(state.data?.run?.revision || "").replace(/^r/, "")) !== Number(workflow?.revision)) return "";
    const progress = workflow?.test_progress;
    if (!progress?.total) return "";
    return `<strong>本轮测试 · 已返回 ${esc(progress.returned)} / ${esc(progress.total)} 项结果</strong><div class="studio-test-items">${progress.tests.map(test => `<span data-test-status="${esc(test.execution_status)}">${esc(test.test_id)}：${esc(studioTestOutcome(test))}${test.execution_status === "SIMULATION_TIMEOUT" ? "（未完成）" : ""}</span>`).join("")}</div>`;
  }

  function partsPanel() {
    const data=state.compareOpen ? null : window.Quest3D?.parts(app.querySelector('.studio-view-surface'));
    if(!data?.total)return `<section class="parts-panel"><p>${state.compareOpen ? "退出 A/B 后查看零件。" : "等待原始模型或运动模型载入。"}</p></section>`;
    const groups=new Map();for(const entry of data.entries){const group=entry.body || entry.path.split('/').slice(0,-1).join('/') || "GLB 节点";if(!groups.has(group))groups.set(group,[]);groups.get(group).push(entry);}
    return `<section class="parts-panel"><header><strong>可见 ${data.visible} / ${data.total}</strong><button data-part-all>显示全部</button><button data-part-isolate ${data.selected ? "" : "disabled"}>隔离所选</button></header><p class="parts-note">${data.entries.some(entry=>entry.body) ? "按已记录刚体与实体分组" : "按 GLB 节点树展示"} · 点击模型或名称选择</p>${data.mode==="design" ? `<label class="parts-explosion">爆炸示意 <input type="range" data-part-explode min="0" max="100" value="${data.explosion}" aria-label="爆炸示意比例"><output>${data.explosion}%</output></label>` : `<p class="parts-note">运动回放使用已保存姿态；爆炸示意仅用于静态模型。</p>`}<div class="parts-tree">${[...groups].map(([group,entries])=>`<details open data-part-group="${esc(group)}"><summary>${esc(group)} <small>${entries.length}</small></summary>${entries.map(entry=>`<div class="parts-row ${entry.selected ? "is-selected" : ""}"><input type="checkbox" data-part-visible="${esc(entry.id)}" ${entry.visible ? "checked" : ""} aria-label="显示 ${esc(entry.label)}"><button data-part-select="${esc(entry.id)}" title="${esc(entry.path)}" aria-pressed="${entry.selected}">${esc(entry.label)}</button></div>`).join("")}</details>`).join("")}</div></section>`;
  }
  function refreshPartsPanel() {
    if(state.studioDrawer!=="parts")return;
    const panel=app.querySelector('.parts-panel');if(!panel)return;
    // Keyed by group name, not summary text: English mode rewrites the text.
    const open=new Map([...panel.querySelectorAll('details[data-part-group]')].map(node=>[node.dataset.partGroup,node.open]));
    const template=document.createElement('template');template.innerHTML=partsPanel();panel.replaceWith(template.content);
    for(const node of app.querySelectorAll('.parts-tree details[data-part-group]'))if(open.has(node.dataset.partGroup))node.open=open.get(node.dataset.partGroup);
    bindPartsEvents();
  }
  function bindPartsEvents() {
    const control=(action,id,value)=>window.Quest3D?.parts(app.querySelector('.studio-view-surface'),action,id,value);
    app.querySelectorAll('[data-part-select]').forEach(button=>button.addEventListener('click',()=>control('select',button.dataset.partSelect)));
    app.querySelectorAll('[data-part-visible]').forEach(input=>input.addEventListener('change',()=>control('visible',input.dataset.partVisible,input.checked)));
    app.querySelector('[data-part-all]')?.addEventListener('click',()=>control('all'));
    app.querySelector('[data-part-isolate]')?.addEventListener('click',()=>control('isolate',window.Quest3D?.parts(app.querySelector('.studio-view-surface'))?.selected));
    app.querySelector('[data-part-explode]')?.addEventListener('input',event=>{
      // Keep this input mounted throughout a drag.
      const input=event.target;partsDragging=true;control('explode',null,Number(input.value));partsDragging=false;
      input.nextElementSibling.textContent=input.value+'%';
    });
  }
  let partsDragging=false;
  app.addEventListener('quest-parts',()=>{if(!partsDragging)refreshPartsPanel();});

  function studioDrawer() {
    if (!state.studioDrawer) return "";
    const titles = { activity: "轨迹", source: "源码", config: "配置", parts: "零件", overview: "项目详情", evidence: "验证与证据", usage: "项目投入", controls: "流程控制" };
    const data = state.data;
    let content = "";
    if (state.studioDrawer === "overview") {
      const quality = data.design_quality || {};
      const prompt = data.live_workflow?.brief || quality.original_prompt || quality.prompt || data.overview?.package?.description || "";
      content = `<section class="studio-detail-summary"><span>当前版本</span><h2>${esc(data.run.revision || data.run.name)}</h2><p>${esc(data.run.status_raw || statusLabel(data.run.status))}</p>${prompt ? `<details open><summary>任务要求</summary><pre>${esc(prompt)}</pre></details>` : ""}<dl><div><dt>阶段</dt><dd>${esc(stage()?.title || studioStatus())}</dd></div><div><dt>验证</dt><dd>${esc(statusLabel(data.run.status, true))}</dd></div><div><dt>更新时间</dt><dd>${esc(fmtTime(data.run.updated_at))}</dd></div></dl></section>${staticVisualReviewPanel(data)}`;
    } else if (state.studioDrawer === "evidence") {
      if (data.live_workflow?.workflow_kind === "slim_harness") {
        const result = data.slim_result || {};
        const labels = { PASS: "本次任务已通过", NEEDS_REVISION: "检查问题已记录，需修正", HARNESS_ERROR: "本轮记录了运行错误", SIMULATION_TIMEOUT: "本轮存在超时，请查看逐项结果", ACTIVE_BUDGET_EXHAUSTED: "预算已用完，已有证据保留", INCOMPLETE_EVIDENCE: "证据尚不完整", EVIDENCE_REQUIRED: "需要补充证据", STALE_EVIDENCE: "产物已变化，等待重新验证" };
        const tests = result.tests || (result.issues || []).filter((item) => item?.test_id);
        const finalFailure = data.live_workflow.outcome === "final_evaluation_failed" && result.status === "EVIDENCE_REQUIRED";
        content = `<section class="studio-detail-summary"><h2>${esc(finalFailure ? "最终评估未通过" : labels[result.status] || "验证正在推进")}</h2><p>运动由物理引擎计算，原形运动按 10 Hz 及精确首末状态采样检查。</p>${result.test_coverage ? `<p>本轮已尝试或复用 ${esc(result.test_coverage.started_test_ids?.length || 0)} / ${esc(result.test_coverage.declared_test_ids?.length || 0)} 项。${result.unrun_tests?.length ? `尚未运行：${esc(result.unrun_tests.join("、"))}` : result.test_coverage.all_tests_visited ? "全部任务已获得首轮执行机会。" : "尚无完整的首轮执行记录。"}</p>` : ""}${tests.map((test) => `<details open><summary>${esc(test.test_id)} · ${esc(studioTestOutcome(test))}</summary><p>时长 ${esc(test.requested_duration_s)} 秒 · 检查 ${esc(test.geometry_audit?.checked_solver_samples ?? "待完成")} 个姿态</p><ul>${(test.assertions || []).map((item) => `<li>${item.passed ? "✓" : "×"} ${esc(item.purpose || item.id)}${item.observed_operands ? `：${esc(JSON.stringify(item.observed_operands))}` : ""}</li>`).join("")}</ul></details>`).join("")}${(result.review?.observations || []).map((item) => `<p>${item.verdict === "met" ? "✓" : "·"} ${esc(item.observed_fact)}</p>`).join("")}${!tests.length && result.issues?.length ? `<details><summary>当前问题</summary><pre>${esc(JSON.stringify(result.issues, null, 2))}</pre></details>` : ""}${result.assembly_warnings?.length ? `<details open class="studio-floating-parts"><summary>装配检查（建议）：可能有悬空零件或未接触的关节</summary><ul>${result.assembly_warnings.map((item) => `<li translate="no">${esc(item)}</li>`).join("")}</ul></details>` : ""}<small>检查覆盖实际采样的姿态；不声称未采样时刻或连续时间内无干涉。</small></section>`;
      } else {
        content = `<div class="studio-stage-selector"><label for="studio-stage-select">查看阶段</label><select id="studio-stage-select">${(data.stages || []).map((item) => `<option value="${esc(item.id)}" ${item.id === state.stageId ? "selected" : ""}>${esc(item.title)} · ${esc(statusLabel(item.status, true))}</option>`).join("")}</select></div>${missionPanel(data).replace('id="mission-workspace"', 'id="stage-workspace"')}${evidencePanel(data)}`;
      }
    } else if (state.studioDrawer === "parts") {
      content = partsPanel();
    } else if (state.studioDrawer === "config") {
      content = configurationPanel();
    } else if (state.studioDrawer === "source") {
      content = historySourcePanel();
    } else if (state.studioDrawer === "activity") {
      content = data.live_workflow ? trajectoryPanel() : roleSessionsPanel("details");
    } else if (state.studioDrawer === "usage") {
      const usage = data.live_usage;
      content = usage ? `<section class="studio-detail-summary"><h2>实际使用记录</h2><dl><div><dt>输入 Token</dt><dd>${usage.input_tokens == null ? "待更新" : esc(formatProjectTokens(usage.input_tokens))}</dd></div><div><dt>缓存输入</dt><dd>${usage.cached_input_tokens == null ? "待更新" : esc(formatProjectTokens(usage.cached_input_tokens))}</dd></div><div><dt>输出 Token</dt><dd>${usage.output_tokens == null ? "待更新" : esc(formatProjectTokens(usage.output_tokens))}</dd></div><div><dt>已完成轮次</dt><dd>${esc(usage.completed_turn_count ?? "待更新")}</dd></div></dl><p>${usage.active_tokens_pending ? "当前轮次的 Token 尚未返回，已收到的用量继续保留。" : "显示运行时实际返回的用量。"}${usage.partial ? " 部分历史用量缺失。" : ""}</p><small>${esc(usage.source || "")}</small></section>` : projectUsagePanel(data.project_usage);
    } else if (state.studioDrawer === "controls") {
      content = `${pauseBanner(data)}${harnessDefectBanner(data)}<div data-worker-recovery-slot data-worker-recovery-key="${esc(workerRecoveryKey())}">${workerRecoveryZone()}</div>${loopApprovalZone(data)}`;
    }
    return `<aside class="studio-drawer" aria-label="${esc(titles[state.studioDrawer])}"><div class="history-resizer" role="separator" aria-label="调整历史面板宽度" aria-orientation="vertical" tabindex="0" data-history-resize></div><header><h2>${esc(titles[state.studioDrawer])}</h2><button type="button" data-studio-close aria-label="关闭详情">×</button></header><nav class="studio-detail-tabs" aria-label="详情类别">${Object.entries(titles).filter(([id]) => id !== "controls" || !data.live_workflow).map(([id, title]) => `<button type="button" data-studio-drawer="${id}" aria-pressed="${state.studioDrawer === id}">${esc(title)}</button>`).join("")}</nav><div class="studio-drawer-body">${content}</div></aside>`;
  }

  function studioMain() {
    const preview = studioPreview();
    const hasPhysics = Boolean((state.data?.evidence || []).some((item) => item.details?.visual_role === "physics-evidence-replay"));
    const motion = studioMotion();
    return `<main id="mission-workspace" class="studio-main" tabindex="-1"><div class="studio-canvas-heading"><div><span data-studio-phase>${esc(studioStatus())}</span><h1>${esc(state.studioMode === "physics" ? "碰撞诊断回放" : state.studioMode === "motion" ? "原形运动回放" : preview.title)}</h1></div><span class="studio-preview-label">${state.studioMode === "physics" ? "诊断视图 · 碰撞表示" : state.studioMode === "motion" ? "原始外形 · 已记录的物理运动" : preview.artifact ? "已保存产物" : "制作预览 · 原始几何"}</span>${state.studioMode === "design" && motionPlayable(motion) ? watchMotionButton(motion) : ""}<div class="studio-live-tests" data-studio-tests ${studioLiveTests() ? "" : "hidden"}>${studioLiveTests()}</div></div>${studioViewport()}${historyStatus()}<div class="studio-canvas-bottom"><div class="studio-view-modes" role="group" aria-label="查看模式"><button type="button" data-studio-mode="design" aria-pressed="${state.studioMode === "design"}">${studioIcons.cube}原始模型</button>${motion ? `<button type="button" class="studio-mode-motion" data-studio-mode="motion" aria-pressed="${state.studioMode === "motion"}">▶ 运动回放</button>` : ""}${hasPhysics ? `<button type="button" data-studio-mode="physics" aria-pressed="${state.studioMode === "physics"}">诊断回放</button>` : ""}</div>${state.studioMode === "motion" ? `<select id="studio-motion-select" aria-label="选择原形运动工况">${studioMotionChoices().map((item) => `<option value="${esc(item.id)}" ${item.id === motion?.id ? "selected" : ""}>${esc([(state.data?.load_cases || []).find((load) => load.id === item.load_case_id)?.label || item.label || item.load_case_id, item.revision].filter(Boolean).join(" · "))}${motionPlayable(item) ? "" : "（1 帧）"}</option>`).join("")}</select>` : state.studioMode === "physics" ? `<select id="studio-load-select" aria-label="选择回放工况">${scopedLoads().map((item) => `<option value="${esc(item.id)}" ${item.id === state.loadId ? "selected" : ""}>${esc(item.label || item.id)}</option>`).join("")}</select>` : `<span class="studio-navigation-hint">拖动旋转 · 滚轮缩放 · 右键平移</span>`}<button class="studio-details-button" type="button" data-studio-drawer="activity">Agent 轨迹 ↗</button><button class="studio-details-button" type="button" data-studio-drawer="evidence">验证与证据 ↗</button></div></main>`;
  }

  function studioFooter() {
    return `<footer class="studio-footer"><span class="studio-sync" data-studio-sync data-connected="${state.progressConnected}"><i></i><span data-studio-sync-label>${boot.mode !== "live" ? "离线快照" : state.progressConnected ? "自动同步中" : "正在连接"}</span></span><span data-studio-update>${esc(fmtTime(state.data?.live_workflow?.updated_at || state.data?.run?.updated_at))}</span><div></div><span data-studio-revision>${esc(state.data?.run?.revision || state.data?.live_workflow?.revision || "")}</span><button type="button" data-studio-drawer="activity">运行记录 ↗</button></footer>`;
  }

  function showStudioMediaFailure(surface, message = "预览文件未能载入") {
    if (!surface?.isConnected || surface.dataset.mediaFailed === "true") return;
    surface.dataset.mediaFailed = "true";
    window.Quest3D?.disposeAll(surface);
    const available = studioAssets().length;
    surface.innerHTML = `<div class="studio-empty studio-media-error">${studioIcons.cube}<h2>${esc(message)}</h2><p>当前预览暂时无法读取。可以查看其他已保存产物，或重新载入预览。</p><div class="studio-empty-actions"><button type="button" class="studio-primary" data-studio-retry-preview>重新载入预览</button><button type="button" class="studio-secondary" data-studio-show-assets>${available ? `查看 ${available} 项已有产物` : "查看运行记录"} ↗</button></div><span class="studio-empty-hint">模型源文件与运行结果保持不变</span></div>`;
    surface.querySelector("[data-studio-retry-preview]")?.addEventListener("click", () => { surface.remove(); render(); });
    surface.querySelector("[data-studio-show-assets]")?.addEventListener("click", () => {
      if (available) { state.studioPanel = "assets"; state.studioSidebarOpen = true; }
      else state.studioDrawer = "activity";
      render();
    });
  }

  function bindStudioMediaEvents() {
    app.querySelectorAll("[data-studio-thumbnail]").forEach((image) => {
      if (image.dataset.fallbackBound) return;
      image.dataset.fallbackBound = "true";
      const fallback = () => {
        image.hidden = true;
        image.style.display = "none";
        const initials = image.parentElement?.querySelector(".studio-project-initials");
        if (initials) initials.hidden = false;
      };
      image.addEventListener("error", fallback, { once: true });
      if (image.complete && !image.naturalWidth) fallback();
    });
    const surface = app.querySelector("[data-studio-view-key]");
    if (!surface || surface.dataset.mediaFailed === "true") return;
    surface.querySelectorAll("img, video").forEach((media) => {
      if (media.dataset.fallbackBound) return;
      media.dataset.fallbackBound = "true";
      const fallback = () => {
        media.hidden = true;
        media.style.display = "none";
        const model = media.closest(".evidence-3d");
        if (model && model.dataset.viewerStatus !== "error") return;
        showStudioMediaFailure(surface, media.tagName === "VIDEO" ? "回放未能载入" : "模型预览未能载入");
      };
      media.addEventListener("error", fallback, { once: true });
      if (media.tagName === "IMG" && media.complete && !media.naturalWidth) fallback();
    });
  }

  function checkStudioViewerLoad() {
    const surface = app.querySelector("[data-studio-view-key]");
    const model = surface?.querySelector(".evidence-3d");
    if (model?.dataset.viewerStatus !== "error") return;
    const fallback = model.querySelector("img:not([hidden])");
    if (fallback?.naturalWidth > 0) return;
    showStudioMediaFailure(surface, "3D 模型暂未载入");
  }

  async function prepareStudioMotion() {
    const node = app.querySelector(".studio-motion-model");
    if (!node || node.dataset.replayReady || node.studioPreparing) return;
    node.studioPreparing = true;
    try {
      const [scene, replay] = await Promise.all([fetchJson(node.dataset.sceneSrc), fetchJson(node.dataset.replaySrc)]);
      if (!node.isConnected) return;
      if (scene.source_kind !== "original_geometry" || replay.source_kind !== "original_geometry") throw new Error("运动记录未绑定原始几何");
      node.questSceneManifest = scene;
      node.questReplay = replay;
      node.questInitialState = { branch: "physical", frame: state.physicsFrame, layer: "off", part: state.physicsPart };
      bindPhysicsState(node);
      node.dataset.replayReady = "true";
      node.classList.add("evidence-3d");
      await window.Quest3D?.mountAll(node);
      checkStudioViewerLoad();
      // Start playing: the motion is what people come to this view for.
      if (node.isConnected && (replay.branches?.physical?.frames?.length || 0) > 1) {
        node.closest(".studio-motion-workspace")?.querySelector("[data-physics-play]")?.click();
      }
    } catch (error) {
      showStudioMediaFailure(node.closest("[data-studio-view-key]"), `运动回放未能载入：${error.message}`);
    }
  }

  function harnessApprovalZone() {
    const frameUrl = surfaceHref("/maintenance/proposals");
    const readOnlyNotice = humanDecisionCapability
      ? ""
      : `<div class="human-approval-locked"><strong>当前是只读旁观链接</strong><span>你可以查看提案与审核进度；请使用服务启动日志中的 Human maintenance dashboard 链接提交最终决定。</span></div>`;
    return `<section class="human-approval-zone harness-approval-zone" aria-labelledby="harness-approval-title">
      <header class="human-approval-head"><div><span>人类审批</span><h2 id="harness-approval-title">Harness 修改提案</h2><p>操作员提出、改动审核员复核后的意见在此等待人类最终决定。</p></div><a href="${esc(frameUrl)}" target="_blank" rel="noopener">独立打开 ↗</a></header>
      ${readOnlyNotice}
      <iframe class="proposal-approval-frame" title="Harness 修改提案审批" src="${esc(frameUrl)}"></iframe>
    </section>`;
  }

  function renderMaintenance() {
    if (!state.data) return;
    captureRoleSessionScroll();
    window.Quest3D?.disposeAll();
    ensureEvidence(state.data);
    const replayClass = state.surface === "loop" && state.revisionReplayOpen ? " has-revision-replay" : "";
    const pauseNotice = pauseBanner(state.data);
    const loopBody = `<div class="surface-body loop-surface"><div class="workspace has-human-approval${pauseNotice ? " has-pause" : ""}">${pauseNotice}${loopApprovalZone(state.data)}${questPanel(state.data)}${missionPanel(state.data)}${evidencePanel(state.data)}</div>${roleSessionsPanel("dock")}</div>`;
    const loopLoopBody = `<main class="looploop-workspace" id="mission-workspace"><div class="looploop-heading"><div><span>Harness 维护层</span><h2>四角色运行拓扑与人类审批</h2><p>这里只监督流程、编排和 Harness 修改；项目机械设计仍留在 Loop 内完成。</p></div><a href="${esc(surfaceHref("/"))}">返回 Loop 项目视图</a></div><div class="looploop-grid">${roleSessionsPanel("board")}<aside class="looploop-approval-rail"><div class="looploop-action-stack"><div data-worker-recovery-slot data-worker-recovery-key="${esc(workerRecoveryKey())}">${workerRecoveryZone()}</div>${loopApprovalZone(state.data)}</div>${harnessApprovalZone()}</aside></div></main>`;
    app.innerHTML = `<div class="unified-dashboard" data-surface="${esc(state.surface)}">${globalSidebar()}<div class="app-shell${replayClass}">${topbar(state.data)}${state.surface === "loop" ? revisionReplayPanel() + loopBody : loopLoopBody}${statusbar(state.data)}</div></div>`;
    if (state.surface === "loop") preparePhysicsViewer();
    bindEvents();
    restoreRoleSessionScroll(activeRoleName());
    requestAnimationFrame(() => {
      if (state.surface === "loop") window.Quest3D?.mountAll(document);
      const list = document.querySelector(".quest-list");
      const chosen = document.querySelector(".quest-button.selected")?.closest(".quest-item");
      if (!list || !chosen) return;
      if (matchMedia("(max-width: 920px)").matches) {
        list.scrollTo({ left: Math.max(0, chosen.offsetLeft - (list.clientWidth - chosen.clientWidth) / 2), behavior: "auto" });
      } else {
        list.scrollTo({ top: Math.max(0, chosen.offsetTop - (list.clientHeight - chosen.clientHeight) / 2), behavior: "auto" });
      }
    });
  }

  function render() {
    if (state.surface === "loop-loop") return renderMaintenance();
    if (!state.data) return;
    captureRoleSessionScroll();
    const focusId = document.activeElement?.id;
    const selection = document.activeElement?.selectionStart;
    const oldDrawerScroll = app.querySelector(".studio-drawer-body")?.scrollTop || 0;
    const sourceDisclosure = [...app.querySelectorAll(".history-source details")].map(node => node.open);
    ensureEvidence(state.data);
    const retained = app.querySelector("[data-studio-view-key]");
    const template = document.createElement("template");
    template.innerHTML = `<div class="studio-shell ${state.studioSidebarOpen ? "" : "is-sidebar-collapsed"}" data-surface="loop">${studioHeader()}<div class="studio-body ${state.studioDrawer ? "has-history" : ""} ${state.projectRailCollapsed ? "projects-collapsed" : ""}" style="--history-width:${state.historyWidth}px" data-mobile-history="${state.mobileHistoryPane}"><nav class="history-mobile-tabs" aria-label="制作历史"><button data-history-mobile="model">模型</button><button data-history-mobile="activity">轨迹</button><button data-history-mobile="source">源码</button><button data-history-mobile="config">配置</button><button data-history-mobile="parts">零件</button></nav>${studioProjectRail()}${studioNavigator()}${studioMain()}${studioDrawer()}</div>${studioFooter()}${state.revisionReplayOpen ? `<div class="studio-history-overlay">${revisionReplayPanel()}</div>` : ""}</div>`;
    const priorRail = app.querySelector(".studio-project-rail");
    const priorProjectScroll = priorRail?.querySelector(".studio-project-scroll")?.scrollTop || 0;
    const nextRail = template.content.querySelector(".studio-project-rail");
    if (priorRail && nextRail && priorRail.dataset.railKey === nextRail.dataset.railKey) nextRail.replaceWith(priorRail);
    const priorSource = app.querySelector(".history-source");
    const nextSource = template.content.querySelector(".history-source");
    if (priorSource?.dataset.sourceKey && priorSource.dataset.sourceKey === nextSource?.dataset.sourceKey) nextSource.replaceWith(priorSource);
    const replacement = template.content.querySelector("[data-studio-view-key]");
    const reuseViewer = retained && replacement && retained.dataset.studioViewKey === replacement.dataset.studioViewKey;
    const sameHistoryProject=state.historyScope==="project" && retained?.dataset.historyProject && retained.dataset.historyProject===replacement?.dataset.historyProject;
    const oldModel = retained?.querySelector(".studio-model");
    const newModel = replacement?.querySelector(".studio-model");
    if (oldModel && newModel && (retained.dataset.studioRun === state.runId || sameHistoryProject) && state.studioMode === "design") {
      oldModel.dataset.modelSrc = newModel.dataset.modelSrc;
      oldModel.style.visibility = ""; oldModel.removeAttribute("aria-hidden");
      newModel.replaceWith(oldModel);
    } else if (oldModel && replacement && (retained.dataset.studioRun === state.runId || sameHistoryProject) && state.studioMode === "design") {
      oldModel.style.visibility = "hidden"; oldModel.setAttribute("aria-hidden", "true");
      replacement.prepend(oldModel);
    } else if (reuseViewer) replacement.replaceWith(retained);
    // The retained viewport is now outside the old shell: dispose only removed views.
    window.Quest3D?.disposeAll(app);
    app.replaceChildren(template.content);
    const projectScroll = app.querySelector(".studio-project-scroll");
    if (projectScroll) projectScroll.scrollTop = priorProjectScroll;
    syncProjectSelection();
    if (["project-search","trajectory-search"].includes(focusId)) { const input=app.querySelector(`#${focusId}`); input?.focus({preventScroll:true}); if (input && selection != null) input.setSelectionRange(selection,selection); }
    if (Array.isArray(state.data?.studio_preview?.up_axis)) {
      app.querySelectorAll(".studio-model, .studio-motion-model").forEach((node) => { node.dataset.upAxis = state.data.studio_preview.up_axis.join(","); });
    }
    if (!reuseViewer || state.studioMode !== "physics") preparePhysicsViewer();
    bindEvents();
    bindStudioEvents();
    bindTrajectoryEvents();
    bindHistoryEvents();
    bindPartsEvents();
    refreshConfig(); updateActivity();
    const drawerScroll = app.querySelector(".studio-drawer-body"); if (drawerScroll) drawerScroll.scrollTop = oldDrawerScroll;
    app.querySelectorAll(".history-source details").forEach((node,index) => {if (index < sourceDisclosure.length) node.open = sourceDisclosure[index];});
    if (state.studioDrawer === "activity" && state.trajectoryRun !== state.runId) refreshTrajectory();
    bindStudioMediaEvents();
    restoreRoleSessionScroll(activeRoleName());
    updateStudioStats();
    requestAnimationFrame(() => {
      Promise.resolve(window.Quest3D?.mountAll(app)).then(() => { checkStudioViewerLoad(); refreshPartsPanel(); prefetchNearbyHistory(); });
      prepareStudioMotion();
    });
  }

  async function selectHistoryPosition(position, frame, fromPlayback=false) {
    const seekSerial=++historySeekSerial;
    if (!fromPlayback) stopHistoryPlay();
    state.historyPosition = position;
    if (!frame) {
      await refreshHistory();
      frame = designTimeline()[position - (state.history?.offset || 0)];
    }
    if(frame && seekSerial===historySeekSerial)await commitDesignFrame(frame.id,seekSerial,fromPlayback);
  }

  function followLatestHistory() {
    historySeekSerial++;
    stopHistoryPlay(); state.compareOpen=false;compareController?.abort();
    state.designFramePinned = ""; state.historyPosition = null; state.trajectoryTurn = "";
    state.trajectoryBefore = null; state.trajectoryTimeWindow = null; state.trajectoryFollow = true;
    if(state.historyScope!=="project")state.history = state.data?.studio_preview?.history || null;
    state.historySource = null; state.historySourceName = ""; state.historySourceOffset = 0;
    state.historyViewerNotice = ""; state.markerNotice=null;
    updateUrl(); render(); refreshHistory(); refreshTrajectory(); refreshHistorySource();
  }

  function bindHistoryEvents() {
    app.querySelector("[data-history-scope]")?.addEventListener("click",toggleHistoryScope);
    app.querySelectorAll("[data-history-chapter]").forEach(button=>button.addEventListener("click",()=>{const ordinal=Number(button.dataset.historyChapter);selectHistoryPosition(ordinal,designTimeline()[ordinal-(state.history?.offset || 0)]);}));
    app.querySelector("[data-revision-events]")?.addEventListener("click",()=>{state.studioDrawer="evidence";state.mobileHistoryPane="evidence";render();});
    app.querySelectorAll("[data-history-pin]").forEach(button=>button.addEventListener("click",()=>pickComparison(button.dataset.historyPin)));
    app.querySelector("[data-history-compare-close]")?.addEventListener("click",()=>{state.compareOpen=false;compareController?.abort();render();});
    app.querySelector("[data-history-compare-view]")?.addEventListener("change",event=>{state.compareView=event.target.value;state.compareData=null;render();refreshComparison();});
    app.querySelector("[data-history-wipe]")?.addEventListener("input",event=>{state.compareWipe=Number(event.target.value);app.querySelector(".history-comparison .revision-compare-viewport")?.style.setProperty("--revision-wipe",state.compareWipe+"%");});
    app.querySelectorAll("[data-history-marker]").forEach(button=>button.addEventListener("click",async()=>{
      const ordinal=Number(button.dataset.historyMarker);
      const markers=(state.history?.markers || []).filter(marker=>marker.ordinal===ordinal);
      const selection=selectHistoryPosition(ordinal,designTimeline()[ordinal-(state.history?.offset || 0)]);
      const seek=historySeekSerial;await selection;
      if(seek!==historySeekSerial || designFrame()?.ordinal!==ordinal && Math.max(0,designFrameIndex())+(state.history?.offset || 0)!==ordinal)return;
      state.markerNotice=markers.map(marker=>marker.label+(marker.test_id ? " · "+marker.test_id : "")).join("；");
      state.studioDrawer="activity";state.mobileHistoryPane="activity";render();refreshTrajectory();
    }));
    const model = app.querySelector(".studio-model");
    if (model && !model.dataset.historyNoticeBound) {
      model.dataset.historyNoticeBound = "true";
      model.addEventListener("quest-model-state", event => {
        state.historyViewerNotice = event.detail.message;
        const notice = app.querySelector("[data-history-viewer-notice]"); if (notice) notice.textContent = event.detail.message;
      });
    }
    app.querySelectorAll("[data-history-mobile]").forEach(button => button.addEventListener("click", () => {
      state.mobileHistoryPane = button.dataset.historyMobile;
      if (state.mobileHistoryPane !== "model") state.studioDrawer = state.mobileHistoryPane;
      render(); refreshTrajectory(); refreshHistorySource();
    }));
    const sourceRoot = app.querySelector(".history-source");
    if (sourceRoot && !sourceRoot.dataset.historyBound) {
      sourceRoot.dataset.historyBound = "true";
    app.querySelector("[data-history-source-file]")?.addEventListener("change", event => {
      state.historySourceName = event.target.value; state.historySourceOffset = 0; refreshHistorySource();
    });
    app.querySelector("[data-history-source-next]")?.addEventListener("click", () => {
      state.historySourceOffset = state.historySource?.next_offset || 0; refreshHistorySource();
    });
    app.querySelector("[data-history-source-first]")?.addEventListener("click", () => {
      state.historySourceOffset = 0; refreshHistorySource();
    });
    }
    const scroll = app.querySelector(".studio-drawer-body");
    const pause = () => {
      if (state.studioDrawer !== "activity") return;
      state.trajectoryFollow = false;
      const checkbox = app.querySelector("[data-trajectory-follow]"); if (checkbox) checkbox.checked = false;
    };
    scroll?.addEventListener("wheel", pause, {passive:true}); scroll?.addEventListener("touchmove", pause, {passive:true});
    const resizer = app.querySelector("[data-history-resize]");
    const width = value => {
      state.historyWidth = Math.max(300, Math.min(innerWidth * .55, value));
      app.querySelector(".studio-body")?.style.setProperty("--history-width", state.historyWidth + "px");
    };
    resizer?.addEventListener("keydown", event => {
      if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); width(state.historyWidth + (event.key === "ArrowLeft" ? 20 : -20)); }
    });
    resizer?.addEventListener("pointerdown", event => {
      event.preventDefault(); const start = event.clientX, initial = state.historyWidth;
      const move = e => width(initial + start - e.clientX);
      const end = () => {document.removeEventListener("pointermove", move); document.removeEventListener("pointerup",end);};
      document.addEventListener("pointermove",move); document.addEventListener("pointerup",end);
    });
  }

  function bindProjectRailEvents(rail) {
    if (!rail || rail.dataset.projectControlsBound) return;
    rail.dataset.projectControlsBound = "true";
    rail.querySelector("[data-project-search]")?.addEventListener("input",event=>{state.projectSearch=event.target.value;updateProjectFilter();});
    rail.querySelector(".project-status-chips")?.parentElement?.addEventListener("click",event=>{
      const button=event.target.closest("[data-project-status]");if(!button)return;
      state.projectStatusFilter=button.dataset.projectStatus;updateProjectFilter();
    });
    rail.querySelector("[data-project-collapse]")?.addEventListener("click",()=>{state.projectRailCollapsed=!state.projectRailCollapsed;render();});
  }

  function bindStudioEvents() {
    bindProjectRailEvents(app.querySelector(".studio-project-rail"));
    app.querySelector("[data-enable-notifications]")?.addEventListener("click",enableNotifications);
    app.querySelector("[data-copy-config]")?.addEventListener("click",()=>navigator.clipboard?.writeText(JSON.stringify(state.config,null,2)));
    app.querySelector("[data-history-play]")?.addEventListener("click",()=>{state.historyPlaying ? stopHistoryPlay() : startHistoryPlay();});
    app.querySelector("[data-history-speed]")?.addEventListener("change",event=>{state.historyInterval=Number(event.target.value);});
    app.querySelectorAll("[data-studio-panel]").forEach((button) => button.addEventListener("click", () => { state.studioPanel = button.dataset.studioPanel; render(); }));
    app.querySelector("[data-studio-sidebar]")?.addEventListener("click", () => { state.studioSidebarOpen = !state.studioSidebarOpen; render(); });
    app.querySelectorAll("[data-studio-drawer]").forEach((button) => button.addEventListener("click", () => { state.studioDrawer = button.dataset.studioDrawer; state.mobileHistoryPane = state.studioDrawer; render(); if (state.studioDrawer === "activity") refreshTrajectory(); if (state.studioDrawer === "source") refreshHistorySource(); if (state.studioDrawer === "config") refreshConfig(); }));
    app.querySelector("[data-studio-close]")?.addEventListener("click", () => { state.studioDrawer = ""; state.mobileHistoryPane = "model"; render(); });
    app.querySelectorAll("button[data-studio-mode]").forEach((button) => button.addEventListener("click", () => { state.studioMode = button.dataset.studioMode; state.compareOpen=false;compareController?.abort(); state.studioAssetId = ""; updateUrl(); render(); }));
    app.querySelectorAll("[data-studio-asset]").forEach((button) => button.addEventListener("click", () => { state.studioAssetId = button.dataset.studioAsset; state.studioMode = "design"; render(); }));
    app.querySelectorAll("[data-studio-version]").forEach((button) => button.addEventListener("click", () => { state.selectionMode = "pinned"; state.studioAssetId = ""; loadRun(button.dataset.studioVersion, { announce: true }); }));
    bindProjectButtons(); observeProjectCovers();
    app.querySelector("[data-studio-follow]")?.addEventListener("click", () => {
      state.selectionMode = state.selectionMode === "follow_current" ? "pinned" : "follow_current";
      const project = projectForRun(state.runId);
      const runId = project?.default_run_id || project?.current_run_id;
      if (state.selectionMode === "follow_current" && runId && runId !== state.runId) loadRun(runId); else { updateUrl(); render(); }
    });
    app.querySelector("#studio-stage-select")?.addEventListener("change", (event) => { state.stageId = event.target.value; ensureScopedSelection(); state.evidenceId = ""; updateUrl(); render(); });
    app.querySelector("#studio-load-select")?.addEventListener("change", (event) => { state.loadId = event.target.value; state.physicsFrame = 0; updateUrl(); render(); });
    app.querySelector("#studio-motion-select")?.addEventListener("change", (event) => { state.studioMotionId = event.target.value; state.studioMotionPicked = true; state.physicsFrame=0; updateUrl(); render(); });
    const timelineRange = app.querySelector("[data-timeline-frame]");
    const timelineFrameAt = (raw) => designTimeline()[Number(raw) - (state.history?.offset || 0)];
    timelineRange?.addEventListener("input", (event) => {
      const frame = timelineFrameAt(event.target.value);
      const index = designTimeline().indexOf(frame);
      const label = app.querySelector("[data-timeline-label]");
      const time = designFrameTime(frame?.created_at);
      if(label)label.textContent=`${frame?.revision || ""}${frame?.turn ? " · "+frame.turn : ""} · 记录 ${Number(event.target.value)+1}/${state.history?.total || designTimeline().length}${time ? " · "+time : ""}`;
      if (timelineCommitTimer) clearTimeout(timelineCommitTimer);
      timelineCommitTimer = setTimeout(() => { timelineCommitTimer = null; selectHistoryPosition(Number(event.target.value), frame); }, 250);
    });
    timelineRange?.addEventListener("change", (event) => {
      if (timelineCommitTimer) { clearTimeout(timelineCommitTimer); timelineCommitTimer = null; }
      selectHistoryPosition(Number(event.target.value), timelineFrameAt(event.target.value));
    });
    app.querySelector("[data-timeline-follow]")?.addEventListener("click", () => { followLatestHistory(); });
    app.querySelector("[data-timeline-trajectory]")?.addEventListener("click", () => openDesignFrameTrajectory());
  }

  function updateTimelineBar() {
    const bar = app.querySelector("[data-studio-timeline]");
    if (!bar) return;
    const frames = designTimeline();
    const range = bar.querySelector("[data-timeline-frame]");
    if (range) range.max = String((state.history?.total || frames.length) - 1);
  }

  const studioTestsMarkup = new WeakMap();
  function updateStudioStats() {
    const timelineExpected = !state.compareOpen && state.studioMode === "design" && (designTimeline().length > 0 || boot.mode==="live");
    const timelineBar = app.querySelector("[data-studio-timeline]");
    if (state.data && Boolean(timelineBar) !== timelineExpected) { render(); return; }
    if (timelineBar) updateTimelineBar();
    const seconds = studioElapsed();
    const elapsed = app.querySelector("[data-studio-elapsed]");
    if (elapsed) elapsed.textContent = seconds == null ? "待记录" : formatProjectDuration(seconds);
    const tokens = studioTokenValue();
    const counter = app.querySelector("[data-studio-tokens]");
    if (counter) counter.textContent = tokens == null || !Number.isFinite(tokens) ? "待更新" : formatProjectTokens(tokens, true);
    const pending = app.querySelector("[data-studio-token-pending]");
    if (pending) pending.textContent = state.data?.live_usage?.active_tokens_pending ? "待更新" : "";
    const phase = app.querySelector("[data-studio-phase]");
    if (phase) phase.textContent = studioStatus();
    const tests = app.querySelector("[data-studio-tests]");
    if (tests) {
      const html = studioLiveTests();
      // Compare with the markup last written, not the DOM: English mode rewrites the DOM text.
      if (studioTestsMarkup.get(tests) !== html) { tests.innerHTML = html; studioTestsMarkup.set(tests, html); }
      tests.hidden = !html;
    }
    const sync = app.querySelector("[data-studio-sync]");
    if (sync) sync.dataset.connected = String(state.progressConnected);
    const label = app.querySelector("[data-studio-sync-label]");
    if (label) label.textContent = boot.mode !== "live" ? "离线快照" : state.progressConnected ? "自动同步中" : state.progressReceivedAt ? "连接中断 · 保留最近数据" : "正在连接";
    const stamp = app.querySelector("[data-studio-update]");
    if (stamp) stamp.textContent = fmtTime(state.data?.live_workflow?.updated_at || state.data?.run?.updated_at);
  }

  function stopProgressWatch() {
    progressGeneration += 1;
    if (progressTimer) clearTimeout(progressTimer);
    progressTimer = null;
    progressController?.abort();
    progressController = null;
  }

  async function pollStudioProgress(generation) {
    if (boot.mode !== "live" || !state.runId || document.hidden || generation !== progressGeneration) return;
    const runId = state.runId;
    const controller = new AbortController();
    progressController = controller;
    let importedSnapshot = isImportedSnapshot();
    try {
      const response = await fetch(`${boot.apiBase}/progress?run=${encodeURIComponent(runId)}`, { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error(`progress ${response.status}`);
      const progress = await response.json();
      if (generation !== progressGeneration || runId !== state.runId) return;
      importedSnapshot = isImportedSnapshot(progress.live_workflow);
      const before = studioViewKey();
      state.activity = progress.activity || null; updateActivity(); refreshConfig(true);
      const motionOptions = JSON.stringify((state.data?.studio_preview?.motions || []).map((item) => [item.id, item.replay_version]));
      for (const key of ["live_workflow", "live_usage", "studio_preview"]) {
        if (Object.prototype.hasOwnProperty.call(progress, key)) state.data[key] = progress[key];
      }
      if (!importedSnapshot) {
        refreshHistory();
        if (state.studioDrawer === "activity" && state.trajectoryBefore == null && state.trajectoryFollow) refreshTrajectory();
      }
      state.progressConnected = true;
      state.progressReceivedAt = Date.now();
      if (progress.live_workflow) state.progressWorkflowReceivedAt = state.progressReceivedAt;
      if (before !== studioViewKey() || motionOptions !== JSON.stringify((state.data?.studio_preview?.motions || []).map((item) => [item.id, item.replay_version]))) render(); else updateStudioStats();
    } catch (error) {
      if (controller.signal.aborted || generation !== progressGeneration) return;
      state.progressConnected = false;
      updateStudioStats();
    } finally {
      if (generation === progressGeneration) {
        progressController = null;
        progressTimer = setTimeout(() => { progressTimer = null; pollStudioProgress(generation); }, importedSnapshot ? 15000 : 2000);
      }
    }
  }

  function startProgressWatch() {
    stopProgressWatch();
    if (!studioClockTimer) studioClockTimer = setInterval(updateStudioStats, 1000);
    if (boot.mode === "live" && state.runId && !document.hidden) pollStudioProgress(progressGeneration);
  }

  function stopRevisionReplay() {
    state.revisionReplayPlaying = false;
    if (revisionReplayTimer) {
      clearTimeout(revisionReplayTimer);
      revisionReplayTimer = null;
    }
    const button = document.getElementById("revision-replay-play");
    if (button) {
      button.textContent = `▶ 播放 ${revisionReplayPlaybackFrames().length} 个画面`;
      button.setAttribute("aria-pressed", "false");
    }
  }

  function revisionReplayStep(direction = 1, { onCommitted = null } = {}) {
    const project = revisionReplayProject();
    const frames = state.revisionReplayPlaying
      ? revisionReplayPlaybackFrames(project)
      : revisionReplayVersions(project);
    if (!frames.length) return false;
    const currentIndex = frames.findIndex((item) => item.id === state.revisionReplayRunId);
    let nextIndex = currentIndex + direction;
    if (currentIndex < 0) {
      const selectedOrdinal = revisionOrdinal(
        revisionReplayVersions(project).find(
          (item) => item.id === state.revisionReplayRunId,
        ),
      );
      if (direction > 0) {
        nextIndex = frames.findIndex(
          (item) => revisionOrdinal(item) > selectedOrdinal,
        );
      } else {
        nextIndex = -1;
        frames.forEach((item, index) => {
          if (revisionOrdinal(item) < selectedOrdinal) nextIndex = index;
        });
      }
    }
    if (nextIndex < 0 || nextIndex >= frames.length) return false;
    const previousTargetId = state.revisionReplayRunId;
    state.revisionReplayPreviousRunId = previousTargetId;
    state.revisionReplayRunId = frames[nextIndex].id;
    if (state.revisionReplayCompare) {
      const versions = revisionReplayVersions(project);
      const nextVersionIndex = versions.findIndex((item) => item.id === state.revisionReplayRunId);
      const previousTargetIndex = versions.findIndex((item) => item.id === previousTargetId);
      state.revisionReplayCompareRunId = previousTargetIndex >= 0
        && previousTargetIndex < nextVersionIndex
        ? previousTargetId
        : versions[nextVersionIndex - 1]?.id || "";
    }
    updateUrl();
    renderRevisionReplayOnly({ onCommitted });
    return true;
  }

  function queueRevisionReplayStep() {
    if (revisionReplayTimer) clearTimeout(revisionReplayTimer);
    if (!state.revisionReplayPlaying) return;
    revisionReplayTimer = setTimeout(() => {
      if (!revisionReplayStep(1, { onCommitted: queueRevisionReplayStep })) {
        stopRevisionReplay();
        renderRevisionReplayOnly();
      }
    }, 1000);
  }

  function renderRevisionReplayOnly({ onCommitted = null } = {}) {
    const current = document.getElementById("revision-replay");
    if (!current || !state.revisionReplayOpen) return;
    const renderSerial = ++revisionReplayRenderSerial;
    const holder = document.createElement("div");
    holder.innerHTML = revisionReplayPanel();
    const replacement = holder.firstElementChild;
    if (!replacement) return;
    const commit = () => {
      if (renderSerial !== revisionReplayRenderSerial || !current.isConnected) return;
      current.replaceWith(replacement);
      bindRevisionReplayEvents(replacement);
      if (typeof onCommitted === "function") onCommitted();
      requestAnimationFrame(() => {
        window.Quest3D?.mountAll(replacement);
        const selectedTick = replacement.querySelector(".revision-tick.selected");
        const timelineScroll = selectedTick?.closest(".revision-timeline-scroll");
        if (selectedTick && timelineScroll) {
          timelineScroll.scrollTo({
            left: Math.max(
              0,
              selectedTick.offsetLeft
                - (timelineScroll.clientWidth - selectedTick.offsetWidth) / 2,
            ),
            behavior: "auto",
          });
        }
      });
    };
    const stackedImages = [...replacement.querySelectorAll(".revision-replay-stack-image")];
    if (!stackedImages.length) {
      commit();
      return;
    }
    Promise.race([
      Promise.all(stackedImages.map((image) => imageReady(image).catch(() => null))),
      new Promise((resolve) => window.setTimeout(resolve, 5000)),
    ]).then(commit);
  }

  function imageReady(image) {
    if (image.complete) {
      return image.naturalWidth ? Promise.resolve(image) : Promise.reject(new Error("image unavailable"));
    }
    return new Promise((resolve, reject) => {
      image.addEventListener("load", () => resolve(image), { once: true });
      image.addEventListener("error", () => reject(new Error("image unavailable")), { once: true });
    });
  }

  function drawContainedImage(context, image, width, height) {
    const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
    const shownWidth = image.naturalWidth * scale;
    const shownHeight = image.naturalHeight * scale;
    context.drawImage(
      image,
      (width - shownWidth) / 2,
      (height - shownHeight) / 2,
      shownWidth,
      shownHeight,
    );
  }

  async function renderRevisionImageDifference(root) {
    const viewport = root.querySelector('.revision-compare-viewport[data-compare-mode="difference"]');
    const canvas = root.querySelector("#revision-difference-canvas");
    const baseImage = root.querySelector("#revision-compare-base-image");
    const targetImage = root.querySelector("#revision-compare-target-image");
    if (!viewport || !canvas || !baseImage || !targetImage) return;
    try {
      await Promise.all([imageReady(baseImage), imageReady(targetImage)]);
      if (!root.isConnected || state.revisionReplayCompareMode !== "difference") return;
      const baseRatio = baseImage.naturalWidth / baseImage.naturalHeight;
      const targetRatio = targetImage.naturalWidth / targetImage.naturalHeight;
      if (Math.abs(baseRatio - targetRatio) / Math.max(baseRatio, targetRatio) > 0.01) {
        state.revisionReplayCompareMode = "wipe";
        updateUrl();
        viewport.dataset.compareMode = "wipe";
        root.querySelectorAll("button[data-compare-mode]").forEach((button) => {
          button.setAttribute(
            "aria-pressed",
            button.dataset.compareMode === "wipe" ? "true" : "false",
          );
        });
        const wipeControl = root.querySelector(".revision-wipe-control");
        if (wipeControl) wipeControl.hidden = false;
        const note = root.querySelector(".revision-compare-controls > small");
        if (note) note.textContent = "两张固定机位画幅不同，已保留划屏并关闭热区，避免产生假变化。";
        return;
      }
      const rect = viewport.getBoundingClientRect();
      const scale = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.min(960, Math.round(rect.width * scale)));
      const height = Math.max(1, Math.min(540, Math.round(rect.height * scale)));
      const baseCanvas = document.createElement("canvas");
      const targetCanvas = document.createElement("canvas");
      baseCanvas.width = targetCanvas.width = canvas.width = width;
      baseCanvas.height = targetCanvas.height = canvas.height = height;
      const baseContext = baseCanvas.getContext("2d", { willReadFrequently: true });
      const targetContext = targetCanvas.getContext("2d", { willReadFrequently: true });
      const outputContext = canvas.getContext("2d");
      for (const context of [baseContext, targetContext]) {
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, width, height);
      }
      drawContainedImage(baseContext, baseImage, width, height);
      drawContainedImage(targetContext, targetImage, width, height);
      const before = baseContext.getImageData(0, 0, width, height);
      const after = targetContext.getImageData(0, 0, width, height);
      const output = outputContext.createImageData(width, height);
      for (let index = 0; index < output.data.length; index += 4) {
        const delta = Math.max(
          Math.abs(before.data[index] - after.data[index]),
          Math.abs(before.data[index + 1] - after.data[index + 1]),
          Math.abs(before.data[index + 2] - after.data[index + 2]),
        );
        output.data[index] = Math.round(after.data[index] * 0.32 + 173);
        output.data[index + 1] = Math.round(after.data[index + 1] * 0.32 + 173);
        output.data[index + 2] = Math.round(after.data[index + 2] * 0.32 + 173);
        output.data[index + 3] = 255;
        if (delta < 32) continue;
        const color = [112, 72, 205];
        const alpha = Math.min(0.92, 0.48 + delta / 510);
        output.data[index] = Math.round(output.data[index] * (1 - alpha) + color[0] * alpha);
        output.data[index + 1] = Math.round(output.data[index + 1] * (1 - alpha) + color[1] * alpha);
        output.data[index + 2] = Math.round(output.data[index + 2] * (1 - alpha) + color[2] * alpha);
      }
      outputContext.putImageData(output, 0, 0);
      canvas.hidden = false;
    } catch (_error) {
      canvas.hidden = true;
    }
  }

  async function refreshRevisionGeometryEvidence(root) {
    if (!state.revisionReplayCompare) return;
    const project = revisionReplayProject();
    const versions = revisionReplayVersions(project);
    const target = versions.find((item) => item.id === state.revisionReplayRunId);
    const base = revisionCompareBase(versions, target);
    if (!base || !target) return;
    const key = revisionGeometryCacheKey(base.id, target.id);
    if (revisionGeometryCache.has(key)) return;
    revisionGeometryCache.set(key, { status: "loading" });
    try {
      const data = await fetchJson(
        `${boot.apiBase}/revision-geometry?base=${encodeURIComponent(base.id)}&target=${encodeURIComponent(target.id)}`,
      );
      revisionGeometryCache.set(key, { status: "ready", data });
    } catch (error) {
      revisionGeometryCache.set(key, { status: "error", error: String(error) });
    }
    if (!root.isConnected) return;
    const currentProject = revisionReplayProject();
    const currentVersions = revisionReplayVersions(currentProject);
    const currentTarget = currentVersions.find((item) => item.id === state.revisionReplayRunId);
    const currentBase = revisionCompareBase(currentVersions, currentTarget);
    if (currentBase?.id !== base.id || currentTarget?.id !== target.id) return;
    const proof = root.querySelector(".revision-geometry-proof");
    if (!proof) return;
    const holder = document.createElement("div");
    holder.innerHTML = revisionGeometryProofMarkup(base, target);
    if (holder.firstElementChild) proof.replaceWith(holder.firstElementChild);
  }

  function revisionVideoRecorderType() {
    if (!window.MediaRecorder) return null;
    const candidates = [
      ["video/mp4;codecs=avc1.42E01E", "mp4"],
      ["video/webm;codecs=vp8", "webm"],
      ["video/webm;codecs=vp9", "webm"],
      ["video/webm", "webm"],
    ];
    return candidates.find(([mime]) => MediaRecorder.isTypeSupported(mime)) || null;
  }

  function loadRevisionVideoImage(item, sourceUrl = undefined) {
    const url = sourceUrl === undefined ? revisionPreviewUrl(item) : sourceUrl;
    if (!url) return Promise.resolve(null);
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    return imageReady(image).catch(() => null);
  }

  async function cacheRevisionVideoFrames(project, frames, onProgress) {
    await ensureRevisionFrameCache(project, onProgress);
    return frames.map(
      (item) => revisionFrameObjectUrls.get(item.id) || null,
    );
  }

  function mediaRecorderStateChange(recorder, action, eventName) {
    if (recorder.state === "inactive") return Promise.resolve();
    const changed = new Promise((resolve) => {
      recorder.addEventListener(eventName, resolve, { once: true });
    });
    recorder[action]();
    return changed;
  }

  function roundedRect(context, x, y, width, height, radius) {
    context.beginPath();
    context.roundRect(x, y, width, height, radius);
    context.fill();
  }

  function paintRevisionVideoFrame(context, {
    project,
    frames,
    index,
    image,
    previousImage,
    mix,
  }) {
    const width = context.canvas.width;
    const height = context.canvas.height;
    const item = frames[index];
    const statusColors = {
      passed: "#19a568",
      warning: "#d78a0b",
      running: "#277ac9",
      failed: "#d53253",
      blocked: "#7849ae",
      stale: "#7c8a9a",
      not_started: "#7c8a9a",
    };
    context.globalAlpha = 1;
    context.fillStyle = "#0b1422";
    context.fillRect(0, 0, width, height);
    const glow = context.createRadialGradient(width * 0.72, 60, 10, width * 0.72, 60, width * 0.7);
    glow.addColorStop(0, "rgba(50, 133, 230, 0.25)");
    glow.addColorStop(1, "rgba(11, 20, 34, 0)");
    context.fillStyle = glow;
    context.fillRect(0, 0, width, height);
    context.fillStyle = "#96aac3";
    context.font = "700 18px system-ui, sans-serif";
    context.fillText(tr("同视角图片延时 · 每个不同画面 1 秒"), 72, 50);
    context.fillStyle = "#f6f9fd";
    context.font = "800 28px system-ui, sans-serif";
    context.fillText(String(project.label || project.id || "机械项目").slice(0, 52), 72, 86);

    context.fillStyle = "#f8fbff";
    roundedRect(context, 72, 112, 1136, 492, 18);
    context.save();
    context.beginPath();
    context.roundRect(88, 128, 1104, 430, 12);
    context.clip();
    context.fillStyle = "#edf3f9";
    context.fillRect(88, 128, 1104, 430);
    if (previousImage && mix < 1) {
      context.save();
      context.translate(88, 128);
      context.globalAlpha = 1 - mix;
      drawContainedImage(context, previousImage, 1104, 430);
      context.restore();
    }
    if (image) {
      context.save();
      context.translate(88, 128);
      context.globalAlpha = previousImage ? mix : 1;
      drawContainedImage(context, image, 1104, 430);
      context.restore();
    } else {
      context.fillStyle = "#718196";
      context.font = "600 22px system-ui, sans-serif";
      context.textAlign = "center";
      context.fillText(tr("此版本没有固定机位画面"), width / 2, 350);
      context.textAlign = "left";
    }
    context.restore();

    const statusColor = statusColors[item.status] || "#7c8a9a";
    context.fillStyle = statusColor;
    roundedRect(context, 96, 570, 10, 10, 5);
    context.fillStyle = "#162336";
    context.font = "850 22px system-ui, sans-serif";
    context.fillText(String(item.revision || item.name || tr("未编号")), 118, 582);
    context.fillStyle = "#5b6c82";
    context.font = "650 17px system-ui, sans-serif";
    context.fillText(tr(`${statusLabel(item.status)} · ${revisionReason(item)}`), 210, 582);

    const trackLeft = 86;
    const trackRight = 1194;
    const trackY = 662;
    context.strokeStyle = "#33455d";
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(trackLeft, trackY);
    context.lineTo(trackRight, trackY);
    context.stroke();
    frames.forEach((frame, frameIndex) => {
      const x = frames.length === 1
        ? trackLeft
        : trackLeft + (trackRight - trackLeft) * frameIndex / (frames.length - 1);
      const inactiveRadius = frames.length > 60 ? 2.5 : 5;
      context.beginPath();
      context.arc(x, trackY, frameIndex === index ? 8 : inactiveRadius, 0, Math.PI * 2);
      context.fillStyle = frameIndex === index
        ? (statusColors[frame.status] || "#7c8a9a")
        : "#8090a5";
      context.fill();
    });
    context.fillStyle = "#90a1b6";
    context.font = "600 14px system-ui, sans-serif";
    context.fillText(`STEP ${index + 1} / ${frames.length} · ${item.revision || item.name || tr("未编号")}`, 72, 704);
    context.textAlign = "right";
    context.fillText(tr("由正式版本轨道生成"), 1208, 704);
    context.textAlign = "left";
  }

  function animateRevisionVideoWindow(startedAt, deadline, draw) {
    return new Promise((resolve) => {
      const frame = (now) => {
        const progress = Math.min(
          1,
          Math.max(0, (now - startedAt) / Math.max(1, deadline - startedAt)),
        );
        draw(progress);
        if (now < deadline) requestAnimationFrame(frame);
        else resolve();
      };
      requestAnimationFrame(frame);
    });
  }

  async function exportRevisionReplayVideo() {
    if (state.revisionReplayVideoExporting) return;
    const recorderType = revisionVideoRecorderType();
    const canvas = document.createElement("canvas");
    if (!recorderType || typeof canvas.captureStream !== "function") {
      toast("当前浏览器不能直接生成视频");
      return;
    }
    const project = revisionReplayProject();
    const frames = revisionReplayPlaybackFrames(project);
    if (!frames.length) {
      toast("没有可导出的同视角 3D 版本");
      return;
    }
    stopRevisionReplay();
    state.revisionReplayVideoExporting = true;
    const button = document.getElementById("revision-replay-export-video");
    if (button) {
      button.disabled = true;
      button.textContent = "载入画面…";
    }
    let stream = null;
    let cachedFrameUrls = [];
    try {
      cachedFrameUrls = await cacheRevisionVideoFrames(project, frames, (completed, total) => {
        if (button?.isConnected) button.textContent = `缓存 ${completed}/${total}`;
      });
      let currentImage = await loadRevisionVideoImage(frames[0], cachedFrameUrls[0]);
      let previousImage = null;
      canvas.width = 1280;
      canvas.height = 720;
      const context = canvas.getContext("2d");
      stream = canvas.captureStream(30);
      const chunks = [];
      const recorder = new MediaRecorder(stream, {
        mimeType: recorderType[0],
        videoBitsPerSecond: Math.max(
          1_500_000,
          Math.min(
            5_000_000,
            Math.round((60 * 1024 * 1024 * 8) / Math.max(1, frames.length)),
          ),
        ),
      });
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data?.size) chunks.push(event.data);
      });
      const stopped = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
      recorder.start(250);
      const recordingStartedAt = performance.now();
      let recordingPauseOffsetMs = 0;
      for (let index = 0; index < frames.length; index += 1) {
        if (button?.isConnected) button.textContent = `生成 ${index + 1}/${frames.length}`;
        let nextImageReady = index + 1 >= frames.length;
        let nextImage = null;
        const nextImagePromise = index + 1 < frames.length
          ? loadRevisionVideoImage(
            frames[index + 1],
            cachedFrameUrls[index + 1],
          ).then((image) => {
            nextImage = image;
            nextImageReady = true;
            return image;
          })
          : Promise.resolve(null);
        const frameStartedAt = recordingStartedAt + recordingPauseOffsetMs + index * 1000;
        const frameDeadline = recordingStartedAt + recordingPauseOffsetMs + (index + 1) * 1000;
        await animateRevisionVideoWindow(
          frameStartedAt,
          frameDeadline,
          (progress) => paintRevisionVideoFrame(context, {
            project,
            frames,
            index,
            image: currentImage,
            previousImage: index > 0 ? previousImage : null,
            mix: index > 0 ? Math.min(1, progress / 0.06) : 1,
          }),
        );
        previousImage = currentImage;
        let pausedForDecode = false;
        const decodePauseStartedAt = performance.now();
        if (!nextImageReady && index + 1 < frames.length) {
          await mediaRecorderStateChange(recorder, "pause", "pause");
          pausedForDecode = true;
        }
        currentImage = nextImageReady ? nextImage : await nextImagePromise;
        if (pausedForDecode) {
          await mediaRecorderStateChange(recorder, "resume", "resume");
          recordingPauseOffsetMs += performance.now() - decodePauseStartedAt;
        }
      }
      recorder.stop();
      await stopped;
      const recordedBlob = new Blob(chunks, { type: recorderType[0] });
      if (!recordedBlob.size) throw new Error("empty recording");
      let outputBlob = recordedBlob;
      let extension = recorderType[1];
      if (extension === "webm") {
        if (button?.isConnected) button.textContent = "转成 MP4…";
        try {
          const response = await fetch(`${boot.apiBase}/revision-video`, {
            method: "POST",
            headers: { "Content-Type": recordedBlob.type || "video/webm" },
            body: recordedBlob,
          });
          if (response.ok) {
            const converted = await response.blob();
            if (converted.size) {
              outputBlob = converted;
              extension = "mp4";
            }
          }
        } catch (_error) {
          // The original WebM remains a valid, recoverable fallback.
        }
      }
      const url = URL.createObjectURL(outputBlob);
      const link = document.createElement("a");
      const safeName = String(project.label || project.id || "mechanism")
        .replace(/[^\p{L}\p{N}._-]+/gu, "-")
        .replace(/^-+|-+$/g, "")
        || "mechanism";
      link.href = url;
      link.download = `${safeName}-visual-changes-timelapse.${extension}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
      toast(`去重延时视频已导出（${extension.toUpperCase()}）`);
    } catch (_error) {
      toast("视频生成失败，请换用最新版 Chrome");
    } finally {
      stream?.getTracks().forEach((track) => track.stop());
      state.revisionReplayVideoExporting = false;
      const currentButton = document.getElementById("revision-replay-export-video");
      if (currentButton) {
        currentButton.disabled = false;
        currentButton.textContent = "导出去重视频";
      }
      ensureRevisionFrameCache(revisionReplayProject()).catch(() => {});
    }
  }

  function bindRevisionReplayEvents(root = document) {
    if (!state.revisionReplayOpen) return;
    const image = root.querySelector("#revision-replay-image");
    const revealStackedImage = () => {
      const visual = image?.closest(".revision-replay-visual");
      if (visual?.classList.contains("is-stacked")) {
        requestAnimationFrame(() => visual.classList.add("is-ready"));
      }
    };
    if (image?.complete && image.naturalWidth) revealStackedImage();
    else image?.addEventListener("load", revealStackedImage, { once: true });
    image?.addEventListener("error", () => {
      const fallback = image.dataset.fallbackSrc;
      if (fallback && image.dataset.fallbackTried !== "true") {
        image.dataset.fallbackTried = "true";
        image.src = fallback;
        return;
      }
      image.hidden = true;
      image.closest(".revision-replay-visual")?.classList.add("is-ready", "has-error");
      const placeholder = root.querySelector(".revision-replay-placeholder");
      if (placeholder) placeholder.hidden = false;
    });
    root.querySelectorAll(".revision-preview-image").forEach((previewImage) => {
      previewImage.addEventListener("error", () => {
        previewImage.hidden = true;
        const placeholder = root.querySelector(".revision-replay-placeholder");
        if (placeholder) placeholder.hidden = false;
      });
    });
    root.querySelectorAll("[data-compare-mode]").forEach((button) => {
      if (!button.matches("button")) return;
      button.addEventListener("click", () => {
        state.revisionReplayCompareMode = button.dataset.compareMode === "difference"
          ? "difference"
          : "wipe";
        updateUrl();
        renderRevisionReplayOnly();
      });
    });
    root.querySelector("#revision-wipe-range")?.addEventListener("input", (event) => {
      state.revisionReplayWipe = Number(event.target.value || 50);
      const viewport = root.querySelector(".revision-compare-viewport");
      viewport?.style.setProperty("--revision-wipe", `${state.revisionReplayWipe}%`);
    });
    root.querySelector("#revision-compare-base")?.addEventListener("change", (event) => {
      state.revisionReplayCompareRunId = event.target.value;
      updateUrl();
      renderRevisionReplayOnly();
    });
    root.querySelectorAll("[data-revision-run-id]").forEach((button) => button.addEventListener("click", () => {
      stopRevisionReplay();
      const previousTargetId = state.revisionReplayRunId;
      state.revisionReplayPreviousRunId = previousTargetId;
      state.revisionReplayRunId = button.dataset.revisionRunId || state.revisionReplayRunId;
      if (state.revisionReplayCompare) {
        const versions = revisionReplayVersions();
        const nextVersionIndex = versions.findIndex((item) => item.id === state.revisionReplayRunId);
        const previousTargetIndex = versions.findIndex((item) => item.id === previousTargetId);
        state.revisionReplayCompareRunId = previousTargetIndex >= 0
          && previousTargetIndex < nextVersionIndex
          ? previousTargetId
          : versions[nextVersionIndex - 1]?.id || "";
      }
      updateUrl();
      renderRevisionReplayOnly();
    }));
    root.querySelector("#revision-replay-prev")?.addEventListener("click", () => {
      stopRevisionReplay();
      revisionReplayStep(-1);
    });
    root.querySelector("#revision-replay-next")?.addEventListener("click", () => {
      stopRevisionReplay();
      revisionReplayStep(1);
    });
    root.querySelector("#revision-replay-play")?.addEventListener("click", async (event) => {
      if (state.revisionReplayPlaying) {
        stopRevisionReplay();
        renderRevisionReplayOnly();
        return;
      }
      const frames = revisionReplayPlaybackFrames();
      if (!frames.length) return;
      event.currentTarget.disabled = true;
      event.currentTarget.textContent = `准备 ${frames.length} 个变化…`;
      await ensureRevisionFrameCache(revisionReplayProject());
      if (!state.revisionReplayOpen || !root.isConnected) return;
      state.revisionReplayCompare = false;
      const currentIndex = frames.findIndex(
        (item) => item.id === state.revisionReplayRunId,
      );
      if (currentIndex < 0) {
        const selected = revisionReplayVersions().find(
          (item) => item.id === state.revisionReplayRunId,
        );
        const nextIndex = frames.findIndex(
          (item) => revisionOrdinal(item) > revisionOrdinal(selected),
        );
        state.revisionReplayPreviousRunId = "";
        state.revisionReplayRunId = frames[nextIndex >= 0 ? nextIndex : 0].id;
      } else if (currentIndex === frames.length - 1) {
        state.revisionReplayPreviousRunId = "";
        state.revisionReplayRunId = frames[0].id;
      }
      state.revisionReplayPlaying = true;
      updateUrl();
      renderRevisionReplayOnly({ onCommitted: queueRevisionReplayStep });
    });
    root.querySelector("#revision-replay-compare")?.addEventListener("click", () => {
      stopRevisionReplay();
      state.revisionReplayCompare = !state.revisionReplayCompare;
      if (state.revisionReplayCompare) {
        const project = revisionReplayProject();
        const versions = revisionReplayVersions(project);
        const selected = versions.find((item) => item.id === state.revisionReplayRunId);
        state.revisionReplayCompareRunId = revisionCompareBase(versions, selected)?.id || "";
      }
      updateUrl();
      renderRevisionReplayOnly();
    });
    root.querySelector("#revision-replay-export-video")?.addEventListener("click", exportRevisionReplayVideo);
    root.querySelector("#revision-replay-close")?.addEventListener("click", () => {
      stopRevisionReplay();
      state.revisionReplayOpen = false;
      updateUrl();
      render();
    });
    root.querySelector("#revision-replay-open-run")?.addEventListener("click", () => {
      const runId = state.revisionReplayRunId;
      stopRevisionReplay();
      state.revisionReplayOpen = false;
      state.selectionMode = "pinned";
      updateUrl();
      loadRun(runId, { announce: true });
    });
    if (state.revisionReplayCompare) {
      refreshRevisionGeometryEvidence(root);
      requestAnimationFrame(() => renderRevisionImageDifference(root));
    }
    if (state.revisionReplayPlaying) {
      const frames = revisionReplayPlaybackFrames();
      const frameIndex = frames.findIndex((item) => item.id === state.revisionReplayRunId);
      [frames[frameIndex - 1], frames[frameIndex + 1]].filter(Boolean).forEach((item) => {
        const preload = new Image();
        preload.src = revisionPreviewUrl(item);
      });
    }
  }

  function toast(message) {
    document.querySelector(".toast")?.remove();
    const node = document.createElement("div");
    node.className = "toast";
    node.textContent = message;
    document.body.appendChild(node);
    setTimeout(() => node.remove(), 2400);
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      let payload = {};
      try { payload = await response.json(); }
      catch (_error) { payload = {}; }
      const error = new Error(payload.detail || `${response.status} ${response.statusText}`);
      error.status = response.status;
      error.code = payload.error || "";
      throw error;
    }
    return response.json();
  }

  function normalizeRoleSessions(payload) {
    const allowed = new Set(rolesForSurface());
    const rows = Array.isArray(payload?.roles) ? payload.roles : [];
    return rows.flatMap((row) => {
      const role = String(row?.role || row?.role_id || "");
      if (!allowed.has(role)) return [];
      const projectedMessages = Array.isArray(row.messages)
        ? row.messages
        : (Array.isArray(row.session?.turns) ? row.session.turns : []).flatMap(
          (turn) => (Array.isArray(turn?.messages) ? turn.messages : []).map(
            (message) => ({ ...message, type: message?.type || "agent_message" }),
          ),
        );
      const messages = projectedMessages.flatMap((message) => {
        const type = String(message?.type || message?.kind || "").toLowerCase();
        const author = String(message?.author || message?.role || "").toLowerCase();
        const isAgentMessage = ["agentmessage", "agent_message", "assistant_message"].includes(type)
          || ["assistant", "agent", role].includes(author);
        const isExplicitSafeHarnessEvent = message?.safe === true && type === "harness_event";
        if ((!isAgentMessage && !isExplicitSafeHarnessEvent) || typeof message?.text !== "string") return [];
        if (type.includes("tool") || type.includes("user") || type.includes("system")) return [];
        const text = message.text.trim();
        if (!text) return [];
        const displayText = isExplicitSafeHarnessEvent
          && typeof message?.display_text === "string"
          && message.display_text.trim()
          ? message.display_text.trim()
          : isExplicitSafeHarnessEvent && harnessPlainLanguage[message.event_kind]
            ? harnessPlainLanguage[message.event_kind]
            : text;
        const technicalCandidate = isExplicitSafeHarnessEvent
          && typeof message?.technical_text === "string"
          && message.technical_text.trim()
          ? message.technical_text.trim()
          : text;
        const technicalText = isExplicitSafeHarnessEvent
          && technicalCandidate !== displayText
          ? technicalCandidate
          : "";
        return [{
          id: String(message.id || ""),
          text: displayText,
          technicalText,
          eventKind: isExplicitSafeHarnessEvent ? String(message.event_kind || "") : "",
          createdAt: String(message.created_at || message.timestamp || ""),
          kind: isExplicitSafeHarnessEvent ? "harness_event" : "agent_message",
          direction: isExplicitSafeHarnessEvent ? "input" : "output",
        }];
      });
      return [{
        role,
        status: String(row.status || row.runtime?.state || "DORMANT"),
        displayStatus: String(row.display_status || ""),
        available: row.available === true,
        errorCode: String(row.error_code || row.session?.error_code || ""),
        providerStatus: String(
          row.provider_status
          || row.session?.turns?.[row.session.turns.length - 1]?.status
          || row.session?.status
          || "",
        ),
        updatedAt: String(row.updated_at || ""),
        messages,
      }];
    });
  }

  function captureRoleSessionScroll() {
    const stream = document.querySelector("[data-role-chat-stream]");
    const role = stream?.dataset.role || "";
    if (!stream || !role) return;
    const maxScroll = Math.max(0, stream.scrollHeight - stream.clientHeight);
    roleSessionScrollPositions.set(role, {
      top: stream.scrollTop,
      stickToBottom: maxScroll - stream.scrollTop <= 32,
    });
  }

  function restoreRoleSessionScroll(roleName) {
    window.requestAnimationFrame(() => {
      const stream = document.querySelector("[data-role-chat-stream]");
      if (!stream || stream.dataset.role !== roleName) return;
      const saved = roleSessionScrollPositions.get(roleName);
      const maxScroll = Math.max(0, stream.scrollHeight - stream.clientHeight);
      stream.scrollTop = !saved || saved.stickToBottom
        ? maxScroll
        : Math.min(saved.top, maxScroll);
    });
  }

  function selectRoleSession(roleName, { focus = false } = {}) {
    if (!rolesForSurface().includes(roleName) || roleName === activeRoleName()) return;
    captureRoleSessionScroll();
    state.roleSessionRole = roleName;
    renderRoleSessionsOnly();
    if (focus) {
      window.requestAnimationFrame(() => {
        document.querySelector(`[data-role-select="${roleName}"]`)?.focus();
      });
    }
  }

  function bindRoleSessionEvents() {
    const buttons = [...document.querySelectorAll("[data-role-select]")];
    buttons.forEach((button, index) => {
      button.addEventListener("click", () => selectRoleSession(button.dataset.roleSelect || ""));
      button.addEventListener("keydown", (event) => {
        if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
        event.preventDefault();
        const backwards = event.key === "ArrowUp" || event.key === "ArrowLeft";
        const nextIndex = (index + (backwards ? -1 : 1) + buttons.length) % buttons.length;
        selectRoleSession(buttons[nextIndex]?.dataset.roleSelect || "", { focus: true });
      });
    });
  }

  function renderRoleSessionsOnly() {
    captureRoleSessionScroll();
    const node = document.querySelector("[data-role-sessions-content]");
    if (node) node.innerHTML = roleSessionsContent();
    bindRoleSessionEvents();
    restoreRoleSessionScroll(activeRoleName());
    const button = document.querySelector("[data-role-session-refresh]");
    if (button) button.disabled = state.roleSessionsLoading;
    const stamp = document.querySelector(".role-sessions-head > div:last-child small");
    if (stamp) {
      const updated = roleSessionTime(state.roleSessionsUpdatedAt);
      stamp.textContent = updated ? `更新于 ${updated}` : "等待首次刷新";
    }
    const kernel = eventKernelPresentation();
    const badge = document.querySelector("[data-event-kernel-status]");
    if (badge) badge.dataset.status = kernel.tone;
    const kernelLabel = badge?.querySelector("[data-event-kernel-label]");
    if (kernelLabel) kernelLabel.textContent = `Harness 内核 · ${kernel.label}`;
    renderWorkerRecoverySlot();
  }

  function workerRecoveryKey() {
    const boundary = eventKernelBoundary();
    return `${String(state.eventKernel?.status || "")}:${boundary.lane}:${boundary.errorCode}:${humanDecisionCapability ? "human" : "readonly"}`;
  }

  function renderWorkerRecoverySlot() {
    const slot = document.querySelector("[data-worker-recovery-slot]");
    if (!slot) return;
    const key = workerRecoveryKey();
    if (slot.dataset.workerRecoveryKey !== key) {
      slot.dataset.workerRecoveryKey = key;
      slot.innerHTML = workerRecoveryZone();
    }
    bindWorkerRecoveryEvents();
  }

  function bindWorkerRecoveryEvents() {
    const form = document.getElementById("worker-recovery-form");
    if (!form || form.dataset.bound === "true") return;
    form.dataset.bound = "true";
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const actor = document.getElementById("worker-recovery-actor")?.value.trim() || "";
      const note = document.getElementById("worker-recovery-note")?.value.trim() || "";
      const message = document.getElementById("worker-recovery-message");
      const button = form.querySelector("[data-worker-resume]");
      if (!actor || !note) {
        setResourceReviewMessage(message, "请填写身份和给 Worker 的说明。", "error");
        return;
      }
      form.setAttribute("aria-busy", "true");
      if (button) button.disabled = true;
      setResourceReviewMessage(message, "正在唤醒同一个 Worker…", "pending");
      try {
        const workerResumeEndpoint = `${boot.apiBase}/worker-resume`;
        const response = await fetch(workerResumeEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Human-Decision-Capability": humanDecisionCapability,
          },
          body: JSON.stringify({ run: state.runId, actor, note }),
        });
        let payload = {};
        try { payload = await response.json(); }
        catch (_error) { payload = {}; }
        if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
        state.eventKernel = payload.event_kernel || { status: "QUEUED" };
        renderWorkerRecoverySlot();
        toast("说明已保存，Harness 正在唤醒同一个 Worker。");
      } catch (error) {
        form.setAttribute("aria-busy", "false");
        if (button) button.disabled = false;
        setResourceReviewMessage(message, `提交失败：${error.message}`, "error");
      }
    });
  }

  function stopRoleSessionWatch({ clear = false } = {}) {
    roleSessionGeneration += 1;
    if (roleSessionTimer) window.clearTimeout(roleSessionTimer);
    roleSessionTimer = null;
    if (roleSessionController) roleSessionController.abort();
    roleSessionController = null;
    state.roleSessionsLoading = false;
    if (clear) {
      state.roleSessions = [];
      state.roleSessionsSignature = "";
      state.roleSessionsUpdatedAt = "";
      state.roleSessionsError = "";
      state.roleSessionRole = rolesForSurface()[0];
      state.eventKernel = { status: "NOT_STARTED" };
      roleSessionScrollPositions.clear();
    }
  }

  async function pollRoleSessions(generation, { immediate = false } = {}) {
    if (
      boot.mode !== "live"
      || !state.runId
      || !features.maintenance
      || Boolean(state.data?.live_workflow?.project_id)
      || generation !== roleSessionGeneration
      || document.hidden
    ) return;
    if (roleSessionController) roleSessionController.abort();
    const controller = new AbortController();
    roleSessionController = controller;
    state.roleSessionsLoading = true;
    if (immediate || !state.roleSessions.length) renderRoleSessionsOnly();
    try {
      const query = new URLSearchParams({ run: state.runId, surface: state.surface });
      const headers = humanDecisionCapability
        ? { "X-Human-Decision-Capability": humanDecisionCapability }
        : {};
      const response = await fetch(`${boot.apiBase}/role-sessions?${query}`, {
        cache: "no-store",
        headers,
        signal: controller.signal,
      });
      let payload = {};
      try { payload = await response.json(); }
      catch (_error) { payload = {}; }
      if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
      if (generation !== roleSessionGeneration) return;
      const sessions = normalizeRoleSessions(payload);
      const signature = JSON.stringify(sessions);
      state.roleSessionsError = "";
      state.roleSessionsUpdatedAt = String(payload.updated_at || new Date().toISOString());
      state.eventKernel = payload.event_kernel || { status: "NOT_STARTED" };
      if (signature !== state.roleSessionsSignature) {
        state.roleSessions = sessions;
        state.roleSessionsSignature = signature;
      }
    } catch (error) {
      if (controller.signal.aborted || generation !== roleSessionGeneration) return;
      state.roleSessionsError = `角色会话刷新失败：${error.message}`;
    } finally {
      if (generation !== roleSessionGeneration) return;
      state.roleSessionsLoading = false;
      roleSessionController = null;
      renderRoleSessionsOnly();
      roleSessionTimer = window.setTimeout(
        () => pollRoleSessions(generation),
        3000,
      );
    }
  }

  function startRoleSessionWatch({ clear = false } = {}) {
    stopRoleSessionWatch({ clear });
    const generation = roleSessionGeneration;
    if (boot.mode !== "live" || !state.runId || document.hidden || state.data?.live_workflow?.project_id) return;
    pollRoleSessions(generation, { immediate: true }).catch(() => {});
  }

  function liveSignature(data) {
    return JSON.stringify({
      run: data.run,
      stages: (data.stages || []).map((item) => [
        item.id,
        item.status,
        item.why,
        item.metrics,
      ]),
      loads: (data.load_cases || []).map((item) => [
        item.id,
        item.branch_status,
        item.physical_status,
        item.reduced_status,
        item.execution_lane?.state,
        item.execution_lane?.raw_status,
        item.execution_lane?.lane_phase,
        item.execution_lane?.reason,
        item.execution_lane?.attempts,
      ]),
      resourceReview: data.resource_review || null,
      projectUsage: data.project_usage || null,
    });
  }

  function stopLiveWatch() {
    liveWatchGeneration += 1;
    if (liveWatchController) liveWatchController.abort();
    liveWatchController = null;
  }

  async function loadRun(runId, {
    preserve = false,
    announce = false,
    onlyIfChanged = false,
    historySelection = null,
    historyFollow = false,
    historySeek = null,
  } = {}) {
    const switchingRun = boot.mode === "live" && Boolean(state.runId) && runId !== state.runId;
    if (switchingRun) {
      stopLiveWatch();
      stopRoleSessionWatch({ clear: true });
      stopProgressWatch();
      if(!historySelection){historySeekSerial++;stopHistoryPlay();state.historyScope="revision";}
      configController?.abort(); state.config = null; state.configKey = "";
      state.studioAssetId = "";
      if(!historySelection){state.designFramePinned="";state.history=null;state.historyRun="";state.historyPosition=null;}
      state.trajectoryTimeWindow = null;
      state.trajectory=null;state.trajectoryRun="";state.trajectoryLoading=false;
      state.historySource = null;
      historyController?.abort(); sourceRequestSerial++;
      state.progressConnected = false;
      state.progressReceivedAt = 0;
      state.resourceReviewContinuation = null;
    }
    const requestSerial = ++loadRequestSerial;
    state.loading = true;
    try {
      const data = await fetchJson(`${boot.apiBase}/run?id=${encodeURIComponent(runId)}`);
      try {
        data.resource_review = data.live_workflow?.project_id || !features.maintenance ? { pending: false, request: null } : await fetchJson(
          `${boot.apiBase}/resource-approval?run=${encodeURIComponent(runId)}`,
        );
      } catch (error) {
        if (data.run?.status_raw === "RESOURCE_APPROVAL_REQUIRED") {
          data.resource_review = {
            pending: true,
            request: null,
            error: error.message,
          };
        } else {
          data.resource_review = { pending: false, request: null };
        }
      }
      if (requestSerial !== loadRequestSerial || historySeek!=null && historySeek!==historySeekSerial) return { ok: false, changed: false };
      const signature = liveSignature(data);
      if (onlyIfChanged && signature === state.liveSignature) {
        return { ok: true, changed: false };
      }
      if ([state.compareA,state.compareB].some(pin=>pin && pin.project!==data.run.project_id)) {state.compareA=null;state.compareB=null;state.compareOpen=false;compareController?.abort();}
      state.data = data;
      state.progressWorkflowReceivedAt = Date.now();
      state.liveSignature = signature;
      state.runId = runId;
      if(historySelection){state.historyRun=runId;setDesignSelection(historyFollow ? "" : historySelection.id);}
      refreshHistory();
      if (!preserve || !data.stages.some((x) => x.id === state.stageId)) state.stageId = data.run.default_stage_id;
      const preservedLoadExists = preserve && data.load_cases.some((x) => x.id === state.loadId);
      if (!preservedLoadExists) state.loadId = data.load_cases[0]?.id || "";
      if (!data.mechanisms.some((x) => x.id === state.mechanismId)) state.mechanismId = data.mechanisms[0]?.id || "";
      ensureScopedSelection();
      if (!preservedLoadExists) selectBestPhysicsEvidenceLoad();
      if (!preserve) state.evidenceId = "";
      if (!preserve) {
        state.physicsBranch = "physical";
        state.physicsFrame = 0;
        state.physicsLayer = "selected";
        state.physicsPart = "";
        state.physicsFocusEvidenceId = "";
        const cleanUrl = new URL(location.href);
        cleanUrl.searchParams.delete("hidden");
        history.replaceState(null, "", cleanUrl);
      }
      state.stale = false;
      state.loading = false;
      updateUrl();
      render();
      refreshLaunchStatus();
      if (!progressController && !progressTimer) startProgressWatch();
      if (announce) toast(`已切换：${data.run.name} · ${statusLabel(data.run.status)}`);
      if (switchingRun) {
        startLiveWatch();
        startRoleSessionWatch({ clear: true });
      }
      return { ok: true, changed: true };
    } catch (error) {
      if (requestSerial !== loadRequestSerial) return { ok: false, changed: false };
      state.stale = Boolean(state.data);
      if (!state.data) throw error;
      state.loading = false;
      render();
      toast(`刷新失败，保留上一次证据：${error.message}`);
      if (switchingRun) {
        startLiveWatch();
        startRoleSessionWatch();
      }
      return { ok: false, changed: false };
    } finally {
      if (requestSerial === loadRequestSerial) {
        state.loading = false;
        if(historySelection && state.runId!==runId){startLiveWatch();startRoleSessionWatch();startProgressWatch();}
      }
    }
  }

  async function refreshLive(change) {
    if (
      boot.mode !== "live"
      || state.loading
      || !state.runId
      || liveRefreshInFlight
      || !change?.changed
    ) return false;
    const selectedRunId = state.runId;
    const selectedProjectId = projectForRun(selectedRunId)?.id
      || state.data?.run?.project_id
      || state.data?.run?.package_id
      || "";
    const refreshRequestSerial = loadRequestSerial;
    liveRefreshInFlight = true;
    try {
      if (!change.catalog_changed) {
        if (change.run_changed) {
          const loaded = await loadRun(selectedRunId, {
            preserve: true,
            onlyIfChanged: false,
          });
          return loaded?.ok === true;
        }
        return true;
      }
      let payload;
      try {
        payload = await fetchJson(
          `${boot.apiBase}/runs?fresh=1&change=${encodeURIComponent(change.catalog_token || "")}`,
        );
      } catch (error) {
        if (error?.status !== 409) throw error;
        payload = await fetchJson(`${boot.apiBase}/runs?fresh=1`);
      }
      const rows = payload.runs || [];
      if (
        state.loading
        || refreshRequestSerial !== loadRequestSerial
        || state.runId !== selectedRunId
      ) return false;
      const catalogChanged = applyRunCatalog(payload);
      const selectedProject = state.projects.find(
        (item) => item.id === selectedProjectId
      );
      const currentStillVisible = rows.some((item) => item.id === selectedRunId);
      const projectDefaultRunId = selectedProject?.default_run_id
        || selectedProject?.current_run_id
        || selectedProject?.runs?.[0]?.id;
      const desiredRunId = state.selectionMode === "pinned" && currentStillVisible
        ? selectedRunId
        : projectDefaultRunId;
      if (desiredRunId) {
        if (!currentStillVisible && state.selectionMode === "pinned") {
          state.selectionMode = "follow_current";
        }
        const loaded = await loadRun(desiredRunId, {
          preserve: desiredRunId === selectedRunId,
          onlyIfChanged: false,
        });
        if (!loaded?.ok) return false;
        if (!loaded.changed && catalogChanged) render();
      } else {
        state.stale = true;
        render();
      }
      return true;
    } catch (_error) {
      state.stale = Boolean(state.data);
      render();
      return false;
    } finally {
      liveRefreshInFlight = false;
    }
  }

  const liveReconnectDelay = () => new Promise(
    (resolve) => window.setTimeout(resolve, 1000),
  );

  async function watchLiveChanges(generation, controller) {
    const watchedRunId = state.runId;
    let since = "";
    while (
      boot.mode === "live"
      && generation === liveWatchGeneration
      && watchedRunId === state.runId
      && !controller.signal.aborted
    ) {
      const query = new URLSearchParams({ run: watchedRunId, timeout: "25" });
      if (since) query.set("since", since);
      let response;
      try {
        response = await fetch(`${boot.apiBase}/changes?${query}`, {
          cache: "no-store",
          signal: controller.signal,
        });
      } catch (_error) {
        if (controller.signal.aborted || generation !== liveWatchGeneration) return;
        await liveReconnectDelay();
        continue;
      }
      if (generation !== liveWatchGeneration || watchedRunId !== state.runId) return;
      if (response.status === 204) {
        since = response.headers.get("X-Dashboard-Change-Token") || since;
        continue;
      }
      if (!response.ok) {
        if (response.status === 400) since = "";
        await liveReconnectDelay();
        continue;
      }
      let change;
      try {
        change = await response.json();
      } catch (_error) {
        await liveReconnectDelay();
        continue;
      }
      if (!change.changed) {
        since = change.token || since;
        continue;
      }
      const refreshed = await refreshLive(change);
      if (refreshed) {
        since = change.token || since;
      } else {
        await liveReconnectDelay();
      }
    }
  }

  function startLiveWatch() {
    if (boot.mode !== "live" || !state.runId || isImportedSnapshot()) return;
    stopLiveWatch();
    const generation = liveWatchGeneration;
    const controller = new AbortController();
    liveWatchController = controller;
    watchLiveChanges(generation, controller).catch(() => {});
  }

  function renderEvidenceOnly() {
    if (!state.data) return;
    ensureEvidence(state.data);
    const node = document.getElementById("evidence-list");
    const top = node?.scrollTop || 0;
    if (node) { node.innerHTML = evidenceList(state.data); node.scrollTop = top; }
    bindEvidenceEvents();
  }

  function bindEvidenceEvents() {
    document.querySelectorAll("[data-evidence-id]").forEach((button) => button.addEventListener("click", () => {
      state.evidenceId = state.evidenceId === button.dataset.evidenceId ? "" : button.dataset.evidenceId;
      renderEvidenceOnly();
    }));
    document.querySelectorAll("[data-penetration-focus]").forEach((button) => button.addEventListener("click", () => {
      const item = (state.data?.evidence || []).find((entry) => entry.id === button.dataset.penetrationFocus);
      const focus = item?.details?.dashboard_focus;
      if (!focus) { toast("这项证据缺少 3D 定位信息"); return; }
      state.stageId = state.data?.stages?.find((entry) => (entry.kind || entry.id) === "branch-simulation")?.id || state.stageId;
      state.loadId = focus.load_case_id || state.loadId;
      state.physicsBranch = "physical";
      state.physicsFrame = Math.max(0, Number(focus.frame) || 0);
      state.physicsLayer = "selected";
      state.physicsPart = focus.primary_node || "";
      state.physicsFocusEvidenceId = item.id;
      state.evidenceId = item.id;
      updateUrl();
      render();
      requestAnimationFrame(() => document.querySelector(".physics-evidence-workspace")?.scrollIntoView({ block: "start", behavior: "smooth" }));
    }));
    document.querySelectorAll(".copy-path").forEach((button) => button.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(button.dataset.path || ""); toast("路径已复制"); }
      catch { toast("浏览器未允许复制，请直接选择路径"); }
    }));
  }

  function bindEvents() {
    bindRoleSessionEvents();
    bindWorkerRecoveryEvents();
    document.querySelector("[data-role-session-refresh]")?.addEventListener("click", () => {
      startRoleSessionWatch();
    });
    document.querySelectorAll("[data-jump]").forEach((button) => button.addEventListener("click", () => {
      document.querySelector(button.dataset.jump || "")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    const syncButton = document.querySelector("[data-sync-branches]");
    if (syncButton) {
      const videos = [...document.querySelectorAll(".media-pair video")].slice(0, 2);
      const updateSyncLabel = () => { syncButton.textContent = videos.some((video) => !video.paused) ? "❚❚ 同步暂停" : "▶ 同步播放"; };
      syncButton.addEventListener("click", async () => {
        if (videos.length < 2) { toast("本次实验缺少一条分支视频"); return; }
        if (videos.some((video) => !video.paused)) {
          videos.forEach((video) => video.pause());
        } else {
          const start = Math.min(...videos.map((video) => video.currentTime || 0));
          videos.forEach((video) => { video.currentTime = start; });
          try { await Promise.all(videos.map((video) => video.play())); }
          catch { toast("浏览器阻止了自动播放，请分别点击视频播放"); }
        }
        updateSyncLabel();
      });
      if (videos.length === 2) {
        videos[0].addEventListener("timeupdate", () => {
          if (!videos[0].paused && Math.abs(videos[1].currentTime - videos[0].currentTime) > 0.08) videos[1].currentTime = videos[0].currentTime;
        });
        videos.forEach((video) => {
          video.addEventListener("play", updateSyncLabel);
          video.addEventListener("pause", updateSyncLabel);
          video.addEventListener("ended", updateSyncLabel);
        });
      }
    }
    document.querySelectorAll("[data-stage-id]").forEach((button) => button.addEventListener("click", () => {
      state.stageId = button.dataset.stageId;
      ensureScopedSelection();
      selectBestPhysicsEvidenceLoad();
      state.evidenceId = "";
      updateUrl();
      render();
      requestAnimationFrame(() => { const node = document.getElementById("mission-workspace"); if (node) node.scrollTop = 0; });
    }));
    document.querySelectorAll("[data-resource-decision]").forEach((button) => button.addEventListener("click", async () => {
      const request = state.data?.resource_review?.request;
      const approver = document.getElementById("resource-review-approver")?.value.trim() || "";
      const approverType = state.data?.resource_review?.standard_role_topology === true
        ? "human"
        : document.getElementById("resource-review-approver-type")?.value || "";
      const note = document.getElementById("resource-review-note")?.value.trim() || "";
      const message = document.getElementById("resource-review-message");
      const form = document.getElementById("resource-review-form");
      if (!request || !approver || !note) {
        setResourceReviewMessage(message, "请填写审核人和审核意见。", "error");
        return;
      }
      const decision = button.dataset.resourceDecision;
      const buttons = document.querySelectorAll("[data-resource-decision]");
      buttons.forEach((item) => { item.disabled = true; });
      form?.setAttribute("aria-busy", "true");
      setResourceReviewMessage(message, "正在记录决定…", "pending");
      try {
        const response = await fetch(`${boot.apiBase}/resource-approval/decide`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Human-Decision-Capability": humanDecisionCapability,
          },
          body: JSON.stringify({
            run: state.runId,
            request_id: request.request_id,
            approver,
            approver_type: approverType,
            decision,
            note,
          }),
        });
        let payload = {};
        try { payload = await response.json(); }
        catch (_error) { payload = {}; }
        if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
        state.resourceReviewContinuation = payload.continuation
          ? { ...payload.continuation, runId: state.runId }
          : null;
        render();
        await loadRun(state.runId, { preserve: true });
      } catch (error) {
        form?.setAttribute("aria-busy", "false");
        setResourceReviewMessage(message, `提交失败：${error.message}`, "error");
        buttons.forEach((item) => { item.disabled = false; });
      }
    }));
    document.querySelectorAll("[data-resource-review-mode]").forEach((button) => button.addEventListener("click", async () => {
      const message = document.getElementById("resource-review-mode-message");
      button.disabled = true;
      setResourceReviewMessage(message, "正在切换审核方式…", "pending");
      try {
        const response = await fetch(`${boot.apiBase}/resource-approval/mode`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Human-Decision-Capability": humanDecisionCapability,
          },
          body: JSON.stringify({ run: state.runId, mode: button.dataset.resourceReviewMode }),
        });
        let payload = {};
        try { payload = await response.json(); }
        catch (_error) { payload = {}; }
        if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
        await loadRun(state.runId, { preserve: true });
      } catch (error) {
        button.disabled = false;
        setResourceReviewMessage(message, `切换失败：${error.message}`, "error");
      }
    }));
    document.querySelectorAll("[data-resource-continuation-retry]").forEach((button) => button.addEventListener("click", async () => {
      const message = document.getElementById("resource-continuation-message");
      button.disabled = true;
      setResourceReviewMessage(message, "正在重试…", "pending");
      try {
        const response = await fetch(`${boot.apiBase}/resource-approval/retry`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run: state.runId,
            request_id: button.dataset.requestId,
          }),
        });
        let payload = {};
        try { payload = await response.json(); }
        catch (_error) { payload = {}; }
        if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
        state.resourceReviewContinuation = payload.continuation
          ? { ...payload.continuation, runId: state.runId }
          : null;
        render();
        await loadRun(state.runId, { preserve: true });
      } catch (error) {
        button.disabled = false;
        setResourceReviewMessage(message, `重试失败：${error.message}`, "error");
      }
    }));
    document.querySelectorAll("[data-load-id]").forEach((button) => button.addEventListener("click", () => {
      state.loadId = button.dataset.loadId; state.evidenceId = ""; state.physicsFrame = 0; state.physicsPart = ""; state.physicsFocusEvidenceId = ""; updateUrl(); render();
    }));
    document.querySelectorAll("[data-mechanism-id]").forEach((button) => button.addEventListener("click", () => {
      state.mechanismId = button.dataset.mechanismId; render();
    }));
    document.getElementById("project-select")?.addEventListener("change", (event) => {
      const project = state.projects.find((item) => item.id === event.target.value);
      const runId = project?.default_run_id || project?.current_run_id || project?.runs?.[0]?.id;
      state.selectionMode = "follow_current";
      stopRevisionReplay();
      state.revisionReplayCompare = false;
      state.revisionReplayCompareRunId = "";
      state.revisionReplayPreviousRunId = "";
      if (state.revisionReplayOpen && runId) state.revisionReplayRunId = runId;
      if (runId) loadRun(runId, { announce: true });
    });
    document.getElementById("run-select")?.addEventListener("change", (event) => {
      state.selectionMode = "pinned";
      stopRevisionReplay();
      state.revisionReplayPreviousRunId = "";
      if (state.revisionReplayOpen) state.revisionReplayRunId = event.target.value;
      loadRun(event.target.value, { announce: true });
    });
    document.getElementById("revision-replay-button")?.addEventListener("click", () => {
      stopRevisionReplay();
      state.revisionReplayOpen = !state.revisionReplayOpen;
      if (state.revisionReplayOpen) {
        state.revisionReplayRunId = state.runId;
        state.revisionReplayPreviousRunId = "";
        state.revisionReplayCompare = false;
        state.revisionReplayCompareRunId = "";
      }
      updateUrl();
      render();
    });
    document.getElementById("refresh-button")?.addEventListener("click", () => {
      const target = new URL(location.href);
      target.searchParams.set("refresh", "1");
      location.href = target.toString();
    });
    document.getElementById("export-button")?.addEventListener("click", () => {
      if (boot.mode === "live") { location.href = `${boot.apiBase}/export?id=${encodeURIComponent(state.runId)}`; return; }
      const blob = new Blob(["<!doctype html>\n", document.documentElement.outerHTML], { type: "text/html;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = `${state.data.run.name}-quest-dashboard.html`; link.click(); URL.revokeObjectURL(url);
    });
    document.getElementById("scope-toggle")?.addEventListener("click", () => {
      state.evidenceScope = state.evidenceScope === "stage" ? "all" : "stage"; state.evidenceId = ""; render();
    });
    document.getElementById("evidence-search")?.addEventListener("input", (event) => {
      state.evidenceSearch = event.target.value; state.evidenceId = ""; renderEvidenceOnly();
    });
    bindRevisionReplayEvents();
    bindEvidenceEvents();
  }

  function workspaceFromUrl(href, options) {
    const url = new URL(href, options?.origin || "https://dashboard.local/");
    const path = url.pathname.replace(/\/index\.html$/, "/").replace(/\/+$/, "") || "/";
    if (path.includes("/maintenance")) return null;
    const imported = path.match(/^(.*)\/imports\/([^/]+)$/);
    if (imported) {
      const basePath = `${imported[1]}/imports/${imported[2]}`;
      return { basePath, apiBase: `${basePath}/api` };
    }
    const localPath = String(
      options?.localUrl || `${String(options?.currentBasePath || "").replace(/\/imports\/[^/]+$/, "")}/`,
    ).replace(/\/index\.html$/, "").replace(/\/+$/, "");
    if (localPath && path === localPath) return { basePath: localPath, apiBase: `${localPath}/api` };
    return null;
  }

  function resetWorkspaceView() {
    workspaceSwitchSerial += 1;
    summaryGeneration += 1;
    stopLiveWatch();
    stopRoleSessionWatch({ clear: true });
    stopProgressWatch();
    stopHistoryPlay();
    stopRevisionReplay();
    if (summaryTimer) clearTimeout(summaryTimer);
    summaryTimer = null;
    summaryPending = false;
    if (compareTimer) clearTimeout(compareTimer);
    compareTimer = null;
    if (timelineCommitTimer) clearTimeout(timelineCommitTimer);
    timelineCommitTimer = null;
    historyController?.abort();
    configController?.abort();
    compareController?.abort();
    loadRequestSerial += 1;
    historySeekSerial += 1;
    sourceRequestSerial += 1;
    revisionFrameCacheGeneration += 1;
    revisionFrameCachePromise = null;
    revisionFrameCacheProjectId = "";
    revisionFrameCacheSignature = "";
    revisionFrameObjectUrls.forEach((url) => URL.revokeObjectURL(url));
    revisionFrameObjectUrls.clear();
    revisionGeometryCache.clear();
    visibleProjects.clear();
    thumbnailPending.clear();
    state.data = null;
    state.runs = [];
    state.projects = [];
    state.runId = "";
    state.stageId = "";
    state.loadId = "";
    state.mechanismId = "";
    state.evidenceId = "";
    state.history = null;
    state.historyRun = "";
    state.historyPosition = null;
    state.historySource = null;
    state.historySourceName = "";
    state.historyError = "";
    state.historyScope = "revision";
    state.trajectory = null;
    state.trajectoryRun = "";
    state.trajectoryLoading = false;
    state.trajectoryError = "";
    state.trajectoryTimeWindow = null;
    state.activity = null;
    state.compareOpen = false;
    state.compareA = null;
    state.compareB = null;
    state.compareData = null;
    state.revisionReplayOpen = false;
    state.revisionReplayRunId = "";
    state.revisionReplayPreviousRunId = "";
    state.revisionReplayCompare = false;
    state.revisionReplayCompareRunId = "";
    state.designFramePinned = "";
    state.liveSignature = "";
    state.catalogSignature = "";
    state.progressConnected = false;
    state.progressReceivedAt = 0;
    state.progressWorkflowReceivedAt = 0;
    state.studioAssetId = "";
    state.summaries = {};
    state.summaryReady = false;
    state.thumbnails = {};
    state.config = null;
    state.configKey = "";
    state.selectionMode = "follow_current";
    state.resourceReviewContinuation = null;
    state.launchStatus = null;
    launchStatusSerial += 1;
    clearTimeout(launchStatusTimer);
    state.stale = false;
    state.loading = false;
    state.physicsBranch = "physical";
    state.physicsFrame = 0;
    state.physicsLayer = "selected";
    state.physicsPart = "";
    state.physicsFocusEvidenceId = "";
    window.Quest3D?.disposeAll(document);
  }

  async function openWorkspace() {
    const serial = workspaceSwitchSerial;
    const q = new URLSearchParams(location.search);
    const requested = q.get("run");
    let catalogEndpoint = q.get("refresh") === "1"
      ? `${boot.apiBase}/runs?fresh=1`
      : `${boot.apiBase}/runs`;
    let payload;
    while (true) {
      if (serial !== workspaceSwitchSerial) return;
      payload = await fetchJson(catalogEndpoint);
      if (serial !== workspaceSwitchSerial) return;
      catalogEndpoint = `${boot.apiBase}/runs`;
      applyRunCatalog(payload);
      if (state.runs.length) break;
      if (payload.catalog_ready) {
        if (features.team_library && state.teamLibrary.items.length) {
          app.innerHTML = `<main class="team-library-empty">${i18n.toggleMarkup("lang-toggle-corner")}<h1>团队快照</h1><p>选择快照，查看模型、运行报告和历史。</p>${teamLibraryTree()}</main>`;
          return;
        }
        if (launchAvailable()) {
          app.innerHTML = `<main class="empty-workspace">${i18n.toggleMarkup("lang-toggle-corner")}<h1>还没有机械项目</h1><p>描述一个机构，选择 Agent 和预算，Harness 会建模、仿真并把进度实时显示在这里。</p><a href="${esc(launchHref())}">新任务</a></main>`;
          return;
        }
        throw new Error("没有找到包含运行状态的实验目录");
      }
      if (!app.querySelector(".studio-project-rail, .team-library, .team-library-empty")) {
        app.innerHTML = `<div class="fatal-screen"><strong>正在载入机械项目</strong><span>历史项目正在后台建立索引，看板没有卡住。</span></div>`;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    if (serial !== workspaceSwitchSerial) return;
    const requestedIsVisible = state.runs.some((item) => item.id === requested);
    state.selectionMode = q.get("view") === "current"
      ? "follow_current"
      : requestedIsVisible
        ? "pinned"
        : "follow_current";
    const defaultProject = state.projects[0];
    const requestedProject = requestedIsVisible ? projectForRun(requested) : null;
    const runId = requestedIsVisible && state.selectionMode === "pinned"
      ? requested
      : requestedProject
        ? requestedProject.default_run_id || requestedProject.current_run_id || requested
      : defaultProject?.default_run_id
        || defaultProject?.current_run_id
        || state.runs[0].id;
    const runProject = projectForRun(runId);
    const requestedReplayRunId = q.get("replayRun");
    const replayRunIsInProject = runProject?.runs?.some((item) => item.id === requestedReplayRunId);
    state.revisionReplayOpen = q.get("replay") === "1";
    state.revisionReplayRunId = replayRunIsInProject ? requestedReplayRunId : runId;
    state.revisionReplayPreviousRunId = "";
    const requestedCompareRunId = q.get("compareRun");
    const orderedReplayVersions = revisionReplayVersions(runProject);
    const replayTargetIndex = orderedReplayVersions.findIndex(
      (item) => item.id === state.revisionReplayRunId,
    );
    const compareTargetIndex = orderedReplayVersions.findIndex(
      (item) => item.id === requestedCompareRunId,
    );
    const compareRunIsInProject = compareTargetIndex >= 0
      && compareTargetIndex < replayTargetIndex;
    state.revisionReplayCompare = q.get("replayCompare") === "1" && Boolean(compareRunIsInProject);
    state.revisionReplayCompareRunId = compareRunIsInProject ? requestedCompareRunId : "";
    state.revisionReplayCompareMode = q.get("compareMode") === "difference" ? "difference" : "wipe";
    state.stageId = q.get("stage") || "";
    state.loadId = q.get("load") || "";
    state.designFramePinned = q.get("meshframe") || "";
    restorePhysicsState(q);
    await loadRun(runId, { preserve: true });
    if (serial !== workspaceSwitchSerial) return;
    startLiveWatch();
    startRoleSessionWatch({ clear: true });
    startProgressWatch();
    pollProjectSummary();
  }

  async function switchWorkspace(href, { historyMode = "push" } = {}) {
    const target = workspaceFromUrl(href, {
      origin: location.href,
      localUrl: state.teamLibrary?.local_url,
      currentBasePath: boot.basePath || "",
    });
    if (!target) {
      location.assign(href);
      return;
    }
    if ((boot.basePath || "") === target.basePath) return;
    const next = new URL(href, location.href);
    if (!next.hash && location.hash) next.hash = location.hash;
    if (historyMode === "push") history.pushState({ workspace: target.basePath }, "", next);
    else if (historyMode === "replace") history.replaceState({ workspace: target.basePath }, "", next);
    boot.basePath = target.basePath;
    boot.apiBase = target.apiBase;
    resetWorkspaceView();
    const serial = workspaceSwitchSerial;
    toast("正在切换机械…");
    try {
      await openWorkspace();
    } catch (error) {
      if (serial !== workspaceSwitchSerial) return;
      app.innerHTML = `<div class="fatal-screen"><strong>看板无法载入</strong><span>${esc(error.message)}</span><span>请确认实验目录中存在可读取的运行状态。</span></div>`;
    }
  }

  async function init() {
    try {
      if (boot.mode === "snapshot") {
        if (!state.data) throw new Error("快照中没有嵌入运行数据");
        state.runs = [{ id: state.data.run.id, name: state.data.run.name }];
        state.projects = buildProjectCatalog(state.runs);
        const q = new URLSearchParams(location.search);
        state.stageId = q.get("stage") || state.data.run.default_stage_id;
        const requestedLoad = q.get("load");
        state.loadId = requestedLoad || state.data.load_cases[0]?.id || "";
        restorePhysicsState(q);
        state.mechanismId = state.data.mechanisms[0]?.id || "";
        ensureScopedSelection();
        if (!requestedLoad) selectBestPhysicsEvidenceLoad();
        render();
        return;
      }
      await openWorkspace();
    } catch (error) {
      app.innerHTML = `<div class="fatal-screen"><strong>看板无法载入</strong><span>${esc(error.message)}</span><span>请确认实验目录中存在可读取的运行状态。</span></div>`;
    }
  }

  document.addEventListener("click", (event) => {
    if (boot.mode !== "live" || !features.team_library || event.defaultPrevented || event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest?.("a.team-snapshot, a.team-project, a.team-library-home");
    if (!link?.href || (link.target && link.target !== "_self")) return;
    if (!workspaceFromUrl(link.href, {
      origin: location.href,
      localUrl: state.teamLibrary?.local_url,
      currentBasePath: boot.basePath || "",
    })) return;
    event.preventDefault();
    event.stopPropagation();
    switchWorkspace(link.href);
  }, true);
  app.addEventListener("click", (event) => {
    const button = event.target.closest?.("[data-launch-action]");
    if (button && !button.disabled) launchAction(button.dataset.launchAction, button);
  });
  window.addEventListener("popstate", () => {
    if (boot.mode !== "live") return;
    switchWorkspace(location.href, { historyMode: "none" });
  });

  document.addEventListener("visibilitychange", () => {
    if (summaryTimer) clearTimeout(summaryTimer); pollProjectSummary();
    if (document.hidden) {
      stopHistoryPlay(); stopLiveWatch();
      if(compareTimer)clearTimeout(compareTimer);compareController?.abort();
      if (state.revisionReplayPlaying) stopRevisionReplay();
      stopRoleSessionWatch();
      stopProgressWatch();
    } else {
      startLiveWatch(); startRoleSessionWatch();
      startProgressWatch(); refreshLaunchStatus();
      if(state.compareOpen)refreshComparison();
    }
  });

  document.addEventListener("keydown",event=>{
    if(event.ctrlKey || event.metaKey || event.altKey || state.surface!=="loop")return;
    if(event.target.closest?.("input,textarea,select,[contenteditable=true]") && !event.target.matches?.("[data-timeline-frame]"))return;
    if(window.getSelection()?.toString())return;
    if(event.key.toLowerCase()==="f") {event.preventDefault();followLatestHistory();return;}
    if(!["ArrowLeft","ArrowRight"].includes(event.key) || state.studioMode!=="design" || state.compareOpen)return;
    event.preventDefault();const total=state.history?.total || designTimeline().length;
    const current=Math.max(0,designFrameIndex())+(state.history?.offset || 0);
    const next=Math.max(0,Math.min(total-1,current+(event.key==="ArrowLeft" ? -1 : 1)));
    if(total)selectHistoryPosition(next,designTimeline()[next-(state.history?.offset || 0)]);
  });
  try {state.notificationsEnabled=localStorage.getItem("workbench-notifications")==="true" && "Notification" in window && Notification.permission==="granted";} catch {}
  init();
})();
