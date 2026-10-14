(() => {
  "use strict";

  const boot = window.QUEST_BOOT || {};
  const root = document.getElementById("runs");
  const token = new URLSearchParams(location.hash.slice(1)).get("human") || "";
  const focus = new URLSearchParams(location.search).get("focus") || "";
  const i18n = window.i18n || { toggleMarkup: () => "", t: (text) => text };
  const STORE_KEY = "funcad-runs-refresh";
  const INTERVALS = [3, 10, 30];
  const QUIET_SECONDS = 300;
  const STALE_HEARTBEAT_SECONDS = 180;
  const STATUS = {
    running: "运行中", stopping: "正在停止", passed: "已通过", stopped: "已停止",
    budget_exhausted: "预算用完", provider_error: "Agent 服务异常", queued: "排队中", starting: "正在启动", failed: "启动失败",
    partial_delivery: "部分交付", final_evaluation_failed: "最终评估未通过", scene_untestable: "场景无法测试", harness_error: "Harness 错误",
    author_unfinished: "作者未完成交付",
  };
  // One line on how an ended run finished, in place of its last phase.
  const ENDINGS = {
    passed: "所有测试已通过",
    budget_exhausted: "预算已用完，已有结果保留",
    provider_error: "Agent 服务不可用（登录、额度或网络），可以稍后继续",
    partial_delivery: "Agent 判断剩余预算不够再完整验证，交付了当前版本；没有通过全部测试",
    final_evaluation_failed: "预算临近结束，最后一版的最终评估没有通过",
    scene_untestable: "预算临近结束，最后一版的场景还无法测试，没有进行评估",
    harness_error: "Harness 运行出错，已有结果保留，可以继续",
    author_unfinished: "预算临近结束，最后一版的测试已通过，但作者没能在预算内完成交付",
    stopped: "已停止，可以继续",
  };
  const PHASES = {
    criteria: "绑定任务", author: "作者建模", author_followup: "作者修订", construction: "作者建模",
    geometry: "准备几何", template_review: "审核填写测试", physics: "物理仿真", geometry_audit: "运动几何审计",
    harness_repair: "等待修复", revision_needed: "等待修订", evidence_collection: "补充证据",
    budget_exhausted: "预算用完", provider_error: "Agent 服务异常", recovery_wait: "等待 Agent 服务恢复", complete: "已完成", stopped: "已停止", interrupted: "已中断",
  };
  const RESULTS = {
    PASS: "通过", NEEDS_REVISION: "需要修改", HARNESS_ERROR: "Harness 错误", EVIDENCE_REQUIRED: "需要补充证据",
    BUDGET_EXHAUSTED: "预算用完", ACTIVE_BUDGET_EXHAUSTED: "预算用完", SIMULATION_TIMEOUT: "仿真超时", provider_error: "Agent 服务异常",
    INCOMPLETE_EVIDENCE: "证据不完整",
  };
  const PROVIDERS = { codex: "Codex CLI", codex_app: "Codex App", claude_code: "Claude Code", cursor: "Cursor", kimi: "Kimi", anthropic: "Anthropic API" };
  const TIERS = { light: "轻量", medium: "标准", high: "完整" };

  let data = null;
  let fetchedAt = 0;
  let fetchError = "";
  let pollTimer = null;
  let focused = false;
  const openTasks = new Set();
  const settings = loadSettings();

  function loadSettings() {
    try {
      const value = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
      return { auto: value.auto !== false, interval: INTERVALS.includes(value.interval) ? value.interval : INTERVALS[0] };
    } catch (_error) { return { auto: true, interval: INTERVALS[0] }; }
  }

  function saveSettings() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(settings)); } catch (_error) { /* per-viewer convenience only */ }
  }

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function duration(seconds) {
    if (seconds == null || !Number.isFinite(seconds)) return "—";
    const total = Math.max(0, Math.floor(seconds));
    const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }

  function age(ms) {
    const seconds = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (seconds < 60) return `${seconds} 秒前`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
    return `${Math.floor(seconds / 3600)} 小时前`;
  }

  function epochMs(value) {
    const ms = Date.parse(value || "");
    return Number.isFinite(ms) ? ms : null;
  }

  function elapsedSeconds(run) {
    const started = epochMs(run.started_at);
    const attempt = epochMs(run.attempt_started_at) || started;
    if (["running", "stopping"].includes(run.status) && attempt) return (Date.now() - attempt) / 1000;
    // Finished: working time without the gaps while a run was stopped.
    if (Number.isFinite(run.elapsed_seconds)) return run.elapsed_seconds;
    const ended = epochMs(run.ended_at);
    return started && ended ? (ended - started) / 1000 : null;
  }

  // Opens the project page on this motion, playing.
  function motionHref(motion) {
    return `${boot.basePath || ""}/?run=${encodeURIComponent(motion.run_id)}&studio=motion&motion=${encodeURIComponent(motion.id)}${location.hash}`;
  }

  function motionBlock(run) {
    const motion = run.motion;
    if (!motion) {
      const waiting = ["running", "stopping"].includes(run.status) ? "第一次仿真完成后，这里会播放机器的运动" : "这个实验没有可播放的运动";
      return `<div class="run-motion is-empty"><span>${waiting}</span></div>`;
    }
    const thumbnail = motion.thumbnail || {};
    const media = thumbnail.status === "ready"
      ? `<img src="${esc(thumbnail.url)}" alt="运动预览" data-motion-src="${esc(thumbnail.url)}">`
      : `<span class="run-motion-pending">${thumbnail.status === "failed" ? "运动预览生成失败，点开查看完整回放" : "正在生成运动预览…"}</span>`;
    return `<a class="run-motion" href="${esc(motionHref(motion))}" title="观看运动回放">${media}<span class="run-motion-caption">▶ 观看运动 · <span translate="no">${esc(motion.revision)} · ${esc(motion.label)}</span> · ${esc(motion.frames)} 帧</span></a>`;
  }

  function projectHref(run) {
    return `${boot.basePath || ""}/?run=${encodeURIComponent(run.run_id)}&view=current&refresh=1${location.hash}`;
  }

  function launchHref() {
    return `${boot.basePath || ""}/launch${location.hash}`;
  }

  async function api(path, init) {
    const response = await fetch(`${boot.apiBase}${path}`, { cache: "no-store", ...init });
    let payload = {};
    try { payload = await response.json(); } catch (_error) { payload = {}; }
    if (!response.ok) throw Object.assign(new Error(payload.detail || `${response.status} ${response.statusText}`), { code: payload.error || "" });
    return payload;
  }

  function post(path, body) {
    return api(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Human-Decision-Capability": token }, body: JSON.stringify(body) });
  }

  function canAct() {
    return Boolean(data?.launch && token);
  }

  // Markup: agent and model ids stay as they are; the tier label is translated.
  function agentLine(item) {
    const ids = [PROVIDERS[item.provider] || item.provider, item.model, item.reasoning].filter(Boolean).join(" · ");
    const tier = item.tier ? `${TIERS[item.tier] || item.tier}档` : "";
    return `<span translate="no">${esc(ids)}</span>${ids && tier ? " · " : ""}${esc(tier)}`;
  }

  function activityBlock(run) {
    const activity = run.activity || {};
    const running = ["running", "stopping"].includes(run.status);
    const lines = [];
    if (activity.available) {
      const text = activity.detail != null
        ? `<span>${esc(activity.label)}：</span><span translate="no">${esc(activity.detail)}</span>`
        : `<span>${esc(activity.summary)}</span>`;
      lines.push(`<div><small>最近活动</small> <small data-age="${activity.at_ms || ""}">${activity.at_ms ? esc(age(activity.at_ms)) : ""}</small></div><div>${text}</div>`);
      const quiet = activity.at_ms ? (Date.now() - activity.at_ms) / 1000 : 0;
      if (running && quiet > QUIET_SECONDS) {
        lines.push(`<div class="is-quiet">Agent 暂时没有新的公开输出，可能正在长时间思考或运行仿真（高推理强度时常见）。</div>`);
      }
    } else {
      lines.push(`<div><small>${running ? "正在等待第一条公开活动…" : "没有公开活动记录"}</small></div>`);
    }
    const heartbeat = epochMs(run.updated_at);
    if (running && heartbeat && (Date.now() - heartbeat) / 1000 > STALE_HEARTBEAT_SECONDS) {
      lines.push(`<div class="is-stale">已经 <span data-age="${heartbeat}">${esc(age(heartbeat))}</span> 没有收到运行心跳，请打开详情或查看日志。</div>`);
    }
    return `<div class="run-activity">${lines.join("")}</div>`;
  }

  function budgetBlock(run) {
    const budget = run.budget;
    if (!budget || !budget.limit_seconds) return "";
    const share = Math.min(100, Math.max(0, (budget.active_seconds / budget.limit_seconds) * 100));
    return `<div class="run-budget"><small>有效工作时间 <span translate="no">${esc(duration(budget.active_seconds))} / ${esc(duration(budget.limit_seconds))}</span></small><div class="run-budget-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${share.toFixed(0)}"><i style="width:${share.toFixed(1)}%"></i></div></div>`;
  }

  function floatingBlock(run) {
    const check = run.floating_parts;
    if (!check?.count) return "";
    return `<div class="run-warning" title="${esc(check.messages.join("\n\n"))}">⚠ 装配检查 · <span translate="no">${esc(check.revision)}</span> · 可能有悬空零件或未接触的关节 <span translate="no">${esc(check.count)}</span> 处</div>`;
  }

  function resultBlock(run) {
    const result = run.latest_result;
    if (!result) return `<div class="run-result"><small>还没有完成的版本结果</small></div>`;
    const finalFailure = run.status === "final_evaluation_failed" && result.status === "EVIDENCE_REQUIRED";
    const label = finalFailure ? "未通过" : RESULTS[result.status] || result.status || "—";
    const tests = result.tests_total ? ` · 测试通过 <span translate="no">${result.tests_passed}/${result.tests_total}</span>` : "";
    const negative = result.status !== "PASS";
    return `<div class="run-result ${negative ? "is-negative" : ""}"><small>最新结果 · <span translate="no">${esc(result.revision)}</span></small> <strong>${esc(label)}</strong>${tests}</div>`;
  }

  function runCard(run) {
    const running = ["running", "stopping"].includes(run.status);
    const phase = run.active_test ? `物理仿真：<span translate="no">${esc(run.active_test)}</span>`
      : !running && ENDINGS[run.status] ? esc(ENDINGS[run.status]) : esc(PHASES[run.phase] || run.phase || "—");
    const elapsed = elapsedSeconds(run);
    const actions = [`<a href="${esc(projectHref(run))}">查看详情</a>`];
    if (canAct() && run.can_stop) actions.push(`<button type="button" class="is-danger" data-action="stop" data-project="${esc(run.project)}" ${run.status === "stopping" ? "disabled" : ""}>${run.status === "stopping" ? "正在停止…" : "停止"}</button>`);
    if (canAct() && run.can_resume) actions.push(`<button type="button" data-action="resume" data-project="${esc(run.project)}">继续</button>`);
    const open = openTasks.has(run.project) ? "open" : "";
    return `<article class="run-card ${run.project === focus ? "is-focus" : ""}" data-status="${esc(run.status)}" data-run="${esc(run.project)}">
      <header><a href="${esc(projectHref(run))}" translate="no">${esc(run.project)}</a><span class="run-badge" data-status="${esc(run.status)}">${esc(STATUS[run.status] || run.status)}</span></header>
      <div class="run-meta">${agentLine(run)}</div>
      ${motionBlock(run)}
      ${floatingBlock(run)}
      <div class="run-clock"><div><span>${running ? (run.resumed ? "本次已运行" : "已运行") : "总用时"}</span> <strong data-elapsed="${esc(run.project)}" translate="no">${esc(duration(elapsed))}</strong></div><div><span>版本</span> <strong translate="no">r${String(run.revision).padStart(2, "0")}</strong></div></div>
      <div class="run-phase">${phase}</div>
      ${budgetBlock(run)}
      ${activityBlock(run)}
      ${resultBlock(run)}
      ${run.tokens ? `<div class="run-meta">Token <span translate="no">${Number(run.tokens).toLocaleString()}</span></div>` : ""}
      ${run.task ? `<details class="run-task" data-task="${esc(run.project)}" ${open}><summary>任务描述</summary><p translate="no">${esc(run.task)}</p></details>` : ""}
      <div class="run-actions">${actions.join("")}</div>
    </article>`;
  }

  function queueItem(item) {
    const failed = item.status === "failed";
    const waited = epochMs(item.queued_at);
    const remove = canAct() && ["queued", "failed"].includes(item.status)
      ? `<button type="button" data-action="cancel" data-id="${esc(item.id)}">${failed ? "移除" : "取消排队"}</button>` : "";
    return `<li class="${failed ? "is-failed" : ""}"><b>${failed ? "!" : esc(item.position)}</b><div><strong translate="no">${esc(item.project)}</strong> <span class="run-badge" data-status="${esc(item.status)}">${esc(STATUS[item.status] || item.status)}</span><small>${agentLine(item)}</small>${failed ? `<small translate="no">${esc(item.error || "")}</small>` : `<small>加入队列 <span data-age="${waited || ""}">${waited ? esc(age(waited)) : ""}</span></small>`}</div>${remove}</li>`;
  }

  function counts(runs, queue) {
    const running = runs.filter((run) => ["running", "stopping"].includes(run.status)).length;
    const waiting = queue.filter((item) => ["queued", "starting"].includes(item.status)).length;
    return `运行中 <span translate="no">${running}</span> · 排队 <span translate="no">${waiting}</span> · 已结束 <span translate="no">${runs.length - running}</span>`;
  }

  function refreshedText() {
    if (fetchError) return `刷新失败：${fetchError}（稍后重试）`;
    if (!fetchedAt) return "正在刷新…";
    return `上次刷新：${age(fetchedAt)}`;
  }

  // The toolbar is built once so a refresh never closes a control the viewer is using.
  function renderShell() {
    root.innerHTML = `
      <header class="launch-header"><nav><a href="${esc(`${boot.basePath || ""}/${location.hash}`)}">← 机械工作台</a></nav>${i18n.toggleMarkup("lang-toggle-corner")}<h1>运行监控</h1><p>所有实验的实时状态。计时每秒更新，状态按下方的间隔自动刷新；可以随时打开“新任务”追加实验。</p></header>
      <div class="runs-toolbar">
        <span class="runs-counts" data-counts></span>
        <label><input type="checkbox" data-auto ${settings.auto ? "checked" : ""}> 自动刷新</label>
        <label>间隔 <select data-interval>${INTERVALS.map((seconds) => `<option value="${seconds}" ${seconds === settings.interval ? "selected" : ""}>${seconds} 秒</option>`).join("")}</select></label>
        <span class="runs-refreshed" data-refreshed></span>
        <button type="button" data-action="refresh">立即刷新</button>
        <a class="is-primary" href="${esc(launchHref())}" data-new-task hidden>新任务</a>
      </div>
      <div class="launch-notices" data-notices></div>
      <div data-sections></div>`;
  }

  function render() {
    if (!root.querySelector("[data-sections]")) renderShell();
    const runs = data?.runs || [];
    const queue = data?.queue || [];
    const active = runs.filter((run) => ["running", "stopping"].includes(run.status));
    const ended = runs.filter((run) => !["running", "stopping"].includes(run.status));
    const notices = [];
    if (data && !data.launch) notices.push(`<p class="launch-banner">只读视图：这个看板没有开启启动功能，不能停止、继续或排队。</p>`);
    else if (data && !token) notices.push(`<p class="launch-banner is-warning"><strong>只读</strong>停止、继续和排队需要服务启动时终端打印的链接（以 <code>#human=…</code> 结尾）。</p>`);
    if (queue.some((item) => ["queued", "starting"].includes(item.status))) notices.push(`<p class="launch-banner">排队的实验会在当前所有运行结束后按顺序开始；请保持看板服务运行。</p>`);
    root.querySelector("[data-counts]").innerHTML = counts(runs, queue);
    root.querySelector("[data-new-task]").hidden = !data?.launch;
    root.querySelector("[data-notices]").innerHTML = notices.join("");
    const refreshed = root.querySelector("[data-refreshed]");
    refreshed.textContent = refreshedText();
    refreshed.classList.toggle("is-stale", Boolean(fetchError));
    // Keep the animations already on screen instead of restarting them every refresh.
    const playing = new Map([...root.querySelectorAll("img[data-motion-src]")].map((image) => [image.dataset.motionSrc, image]));
    root.querySelector("[data-sections]").innerHTML = `
      <section class="runs-section"><h2>运行中<small translate="no">${active.length}</small></h2>${active.length ? `<div class="runs-grid">${active.map(runCard).join("")}</div>` : `<div class="runs-empty">现在没有正在运行的实验。</div>`}</section>
      <section class="runs-section"><h2>排队中<small translate="no">${queue.filter((item) => item.status !== "failed").length}</small></h2>${queue.length ? `<ol class="runs-queue">${queue.map(queueItem).join("")}</ol>` : `<div class="runs-empty">队列为空。在“新任务”页选择“加入队列”，实验会在当前运行结束后自动开始。</div>`}</section>
      <section class="runs-section"><h2>已结束<small translate="no">${ended.length}</small></h2>${ended.length ? `<div class="runs-grid">${ended.map(runCard).join("")}</div>` : `<div class="runs-empty">还没有结束的实验。</div>`}</section>`;
    root.querySelectorAll("img[data-motion-src]").forEach((image) => {
      const kept = playing.get(image.dataset.motionSrc);
      if (kept) image.replaceWith(kept);
    });
    if (focus && !focused) {
      const card = root.querySelector(`[data-run="${CSS.escape(focus)}"]`);
      if (card) { card.scrollIntoView({ block: "center" }); focused = true; }
    }
  }

  function tick() {
    if (!data) return;
    for (const run of data.runs || []) {
      if (!["running", "stopping"].includes(run.status)) continue;
      const node = root.querySelector(`[data-elapsed="${CSS.escape(run.project)}"]`);
      if (node) node.textContent = duration(elapsedSeconds(run));
    }
    root.querySelectorAll("[data-age]").forEach((node) => {
      const ms = Number(node.dataset.age);
      if (ms) node.textContent = age(ms);
    });
    const refreshed = root.querySelector("[data-refreshed]");
    if (refreshed) {
      refreshed.textContent = refreshedText();
      // A page left in the background, or a server that stopped answering, shows its data's age.
      refreshed.classList.toggle("is-stale", Boolean(fetchError) || (settings.auto && Date.now() - fetchedAt > (settings.interval * 1000 + 15000)));
    }
  }

  async function refresh() {
    clearTimeout(pollTimer);
    try {
      data = await api("/runs-overview");
      fetchedAt = Date.now();
      fetchError = "";
    } catch (error) {
      fetchError = error.message;
    }
    render();
    schedule();
  }

  function schedule() {
    clearTimeout(pollTimer);
    if (settings.auto && !document.hidden) pollTimer = setTimeout(refresh, settings.interval * 1000);
  }

  async function act(button) {
    const action = button.dataset.action;
    if (action === "refresh") { refresh(); return; }
    if (action === "stop" && !window.confirm(i18n.t(`停止 ${button.dataset.project}？已有进度会保存，之后可以继续。`))) return;
    button.disabled = true;
    try {
      if (action === "stop") await post("/launch/stop", { project: button.dataset.project });
      else if (action === "resume") await post("/launch/resume", { project: button.dataset.project });
      else if (action === "cancel") await post("/launch/queue/cancel", { id: button.dataset.id });
    } catch (error) {
      window.alert(i18n.t(`操作失败：${error.message}`));
    }
    refresh();
  }

  root.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (button) act(button);
  });
  root.addEventListener("toggle", (event) => {
    const project = event.target?.dataset?.task;
    if (project) event.target.open ? openTasks.add(project) : openTasks.delete(project);
  }, true);
  root.addEventListener("change", (event) => {
    if (event.target.matches("[data-auto]")) settings.auto = event.target.checked;
    else if (event.target.matches("[data-interval]")) settings.interval = Number(event.target.value);
    else return;
    saveSettings();
    schedule();
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && settings.auto) refresh(); else schedule(); });

  setInterval(tick, 1000);
  refresh();
})();
