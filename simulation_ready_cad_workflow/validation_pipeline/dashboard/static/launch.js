(() => {
  "use strict";

  const boot = window.QUEST_BOOT || {};
  const root = document.getElementById("launch");
  const token = new URLSearchParams(location.hash.slice(1)).get("human") || "";
  const PROVIDERS = { codex: "Codex CLI", codex_app: "Codex App", anthropic: "Anthropic API", claude_code: "Claude Code", cursor: "Cursor", kimi: "Kimi" };
  const TIERS = { light: "轻量", medium: "标准", high: "完整" };
  const CHECKS = {
    cad_python: ["CAD Python", "需要能导入 build123d、OCP 和 trimesh。按 CAD_TOOLCHAIN.md 安装 CAD 环境，并用 SIMREADY_CAD_PYTHON 指向它。"],
    chrono_python: ["Chrono Python", "需要能导入 pychrono。安装 PyChrono 环境后设置 SIMREADY_PYCHRONO_EXECUTABLE，或在高级选项中填写 Chrono Python。"],
  };
  const AGENT_HINTS = {
    codex: { cli: "安装 Codex CLI：npm install -g @openai/codex", auth: "运行 codex login，或设置 OPENAI_API_KEY。" },
    codex_app: { cli: "安装 Codex CLI：npm install -g @openai/codex", auth: "运行 codex login，或设置 OPENAI_API_KEY。" },
    claude_code: { cli: "安装 Claude Code：npm install -g @anthropic-ai/claude-code", auth: "运行 claude 并完成登录，或设置 ANTHROPIC_API_KEY。" },
    anthropic: { auth: "设置 ANTHROPIC_API_KEY（或 ANTHROPIC_AUTH_TOKEN）。" },
    cursor: { cli: "实验性：安装 Cursor CLI（cursor-agent）并登录。" },
    kimi: { cli: "实验性：安装 Kimi CLI（kimi）并登录。" },
  };
  const ERRORS = {
    HUMAN_DECISION_CAPABILITY_REQUIRED: "缺少操作授权：请打开服务启动时终端打印的 New task 链接（含 #human=…）。",
    LAUNCH_CONTENT_TYPE: "请求格式不正确。",
    LAUNCH_ORIGIN: "拒绝跨站点的启动请求。",
    LAUNCH_WORKSPACE: "看板的工作区不是这份代码目录，不能从这里启动运行。",
    LAUNCH_PROJECT_INVALID: "项目 ID 只能使用字母、数字、点、下划线和连字符，并以字母或数字开头。",
    LAUNCH_PROJECT_EXISTS: "这个项目 ID 已存在。换一个 ID，或在项目页继续运行原项目。",
    LAUNCH_MODE_UNAVAILABLE: "当前工作区不提供这个模式。",
    LAUNCH_TASK_REQUIRED: "自由创作需要填写任务描述。",
    LAUNCH_FIELD_INVALID: "有字段取值不正确。",
    LAUNCH_FIELD_UNKNOWN: "请求包含未知字段。",
    LAUNCH_CHRONO_MISSING: "找不到填写的 Chrono Python。",
    LAUNCH_LIMIT: "已达到并行运行上限。可以点“加入队列”，等当前运行结束后自动开始；或启动看板时用 --max-launches 提高上限。",
    LAUNCH_PROJECT_QUEUED: "这个项目 ID 已经在队列中等待。",
    LAUNCH_QUEUE_ITEM: "这个排队项已经不在等待状态。",
    LAUNCH_EXITED: "Harness 在创建项目前就退出了。",
    LAUNCH_START_TIMEOUT: "Harness 启动 30 秒后仍未创建项目；进程仍在运行，请查看下方日志。",
  };
  // Claude Code's Bash sandbox tools and the Debian/Ubuntu packages that provide them.
  const SANDBOX_PACKAGES = { bwrap: "bubblewrap", socat: "socat" };
  const i18n = window.i18n || { toggleMarkup: () => "" };
  let options = null;
  let checksTimer = null;

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function launchRedirectUrl(basePath, runId, hash) {
    return `${basePath || ""}/?run=${encodeURIComponent(runId)}&view=current&refresh=1${hash || ""}`;
  }

  function monitorUrl(basePath, project, hash) {
    return `${basePath || ""}/runs${project ? `?focus=${encodeURIComponent(project)}` : ""}${hash || ""}`;
  }

  function defaultProjectId(prefix) {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    return `${prefix || "task"}-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
  }

  async function api(path, init) {
    const response = await fetch(`${boot.apiBase}${path}`, { cache: "no-store", ...init });
    let payload = {};
    try { payload = await response.json(); } catch (_error) { payload = {}; }
    if (!response.ok) {
      const error = new Error(payload.detail || `${response.status} ${response.statusText}`);
      Object.assign(error, { code: payload.error || "", log: payload.log_tail || "" });
      throw error;
    }
    return payload;
  }

  function field(name) { return root.querySelector(`[name="${name}"]`); }
  function value(name) { return String(field(name)?.value || "").trim(); }

  function choice(items, selected, blank) {
    return `${blank == null ? "" : `<option value="">${esc(blank)}</option>`}${items.map(([id, label]) => `<option value="${esc(id)}" ${id === selected ? "selected" : ""}>${esc(label)}</option>`).join("")}`;
  }

  function providerLabel(item) {
    return `${PROVIDERS[item.id] || item.id}${item.experimental ? "（实验性）" : ""}`;
  }

  function executionBanner() {
    const execution = options.execution || {};
    if (execution.mode === "formal") return `<p class="launch-banner">正式沙箱：Agent 编写的程序在隔离沙箱中运行。</p>`;
    if (execution.mode === "container") return `<p class="launch-banner">容器模式：Agent 编写的程序在当前容器内运行，与宿主机隔离程度取决于容器配置。</p>`;
    if (execution.mode === "local") return `<p class="launch-banner is-warning"><strong>本机模式</strong>Agent 编写的代码会直接在这台机器上以当前用户身份运行，并能读写当前用户可访问的文件。只在你信任的环境中使用，推荐使用容器。</p>`;
    return `<p class="launch-banner is-error"><strong>无法运行 Agent 程序</strong>这个分发既没有正式沙箱，也不允许本地执行；运行会在第一步失败。${esc(execution.error || "")}</p>`;
  }

  function noticeBanners() {
    const notices = [];
    if (!token) notices.push(`<p class="launch-banner is-error"><strong>缺少操作授权</strong>这个页面需要服务启动时终端打印的 New task 链接（以 <code>#human=…</code> 结尾）。请从终端复制完整链接重新打开。</p>`);
    if (options.unavailable) notices.push(`<p class="launch-banner is-error"><strong>不能从这里启动</strong>${esc(ERRORS[options.unavailable.error] || "")} ${esc(options.unavailable.detail)}</p>`);
    const active = options.concurrency?.active || [];
    const limit = options.concurrency?.limit || 1;
    const waiting = (options.queue || []).filter((item) => ["queued", "starting"].includes(item.status)).length;
    if (active.length || waiting) {
      const full = active.length >= limit;
      notices.push(`<p class="launch-banner ${full ? "is-warning" : ""}"><strong>${full ? "已达到并行上限" : "正在进行的实验"}</strong>运行中 <span translate="no">${active.length}/${limit}</span> · 排队 <span translate="no">${waiting}</span>${full ? " · 新的实验可以加入队列" : ""} · <a href="${esc(monitorUrl(boot.basePath, "", location.hash))}">查看运行监控</a></p>`);
    }
    return notices.join("") + executionBanner();
  }

  function checkRow(item) {
    let label, detail, hint;
    if (item.provider) {
      const hints = AGENT_HINTS[item.provider] || {};
      label = providerLabel({ id: item.provider, experimental: item.experimental });
      const parts = [];
      if (item.cli) parts.push(item.cli_found ? `${item.cli} ${item.version || ""}`.trim() : `未找到 ${item.cli}`);
      if (item.auth === "env") parts.push(`已设置 ${item.auth_detail}`);
      else if (item.auth === "login") parts.push(item.auth_detail || "已登录");
      else if (item.auth === "missing") parts.push("未登录或未配置凭据");
      const sandboxMissing = item.sandbox_missing || [];
      if (sandboxMissing.length) parts.push(`缺少 Claude Code 沙箱工具：${sandboxMissing.join(", ")}`);
      detail = parts.join(" · ");
      hint = item.cli && !item.cli_found ? hints.cli
        : item.auth === "missing" ? hints.auth
          : sandboxMissing.length ? `在 Linux 上安装：sudo apt install ${sandboxMissing.map((tool) => SANDBOX_PACKAGES[tool] || tool).join(" ")}` : "";
    } else {
      [label, hint] = CHECKS[item.id] || [item.id, ""];
      detail = item.ok ? item.path : item.detail === "not configured" ? "未配置" : `${item.path || ""} ${item.detail || ""}`;
      if (item.ok) hint = "";
    }
    const selected = item.provider && item.provider === value("provider");
    return `<li class="launch-check" data-ok="${item.ok}" ${selected ? 'aria-current="true"' : ""}><i aria-hidden="true">${item.ok ? "✓" : "✕"}</i><div><strong>${esc(label)}</strong><small>${esc(detail)}</small>${hint ? `<span>${esc(hint)}</span>` : ""}</div></li>`;
  }

  function checksPanel() {
    const checks = options.checks || {};
    const body = checks.ready
      ? `<ul class="launch-checks">${(checks.items || []).map(checkRow).join("")}</ul>`
      : `<p class="launch-muted">正在检查 CAD、Chrono 和 Agent 命令行…首次检查可能需要几十秒。</p>`;
    return `<header><h2>环境检查</h2><button type="button" data-refresh-checks ${checks.ready ? "" : "disabled"}>重新检查</button></header>${body}`;
  }

  function checkWarning() {
    if (!options.checks?.ready) return "";
    const failing = (options.checks.items || []).filter((item) => !item.ok && (!item.provider || item.provider === value("provider")));
    return failing.length ? `环境检查未通过：${failing.map((item) => item.provider ? providerLabel({ id: item.provider }) : (CHECKS[item.id] || [item.id])[0]).join("、")}。仍可启动，但运行可能很快失败。` : "";
  }

  function render() {
    const providers = options.providers || [];
    const provider = providers[0] || { id: "codex", default_model: "", default_reasoning: "high" };
    const tier = (options.tiers || []).find((item) => item.id === options.default_tier);
    const examples = options.examples || [];
    const reasoning = (options.reasoning || []).map((item) => [item, item]);
    const disabled = !token || options.unavailable ? "disabled" : "";
    root.innerHTML = `
      <header class="launch-header"><nav><a href="${esc(`${boot.basePath || ""}/${location.hash}`)}">← 机械工作台</a><a href="${esc(monitorUrl(boot.basePath, "", location.hash))}">运行监控</a></nav>${i18n.toggleMarkup("lang-toggle-corner")}<h1>新任务</h1><p>描述要设计的机构，选择 Agent 和预算，然后立即开始或加入队列。所有实验都能在“运行监控”中查看、停止或继续。</p></header>
      <div class="launch-notices">${noticeBanners()}</div>
      <div class="launch-layout">
        <form class="launch-form" novalidate>
          <section class="launch-section"><label class="launch-field"><span>项目 ID</span><input name="project" maxlength="128" value="${esc(defaultProjectId())}" spellcheck="false"><small>保存在 .runtime_runs/projects/&lt;项目 ID&gt;，之后不能改名。</small></label></section>
          <section class="launch-section">
            <label class="launch-field"><span>任务描述</span><textarea name="task" rows="10" placeholder="例如：设计一个手摇曲柄驱动的齿轮减速机构，输出轴转速为输入的 1/4……"></textarea></label>
            <div class="launch-task-tools"><label class="launch-file">载入 .md 文件<input type="file" data-task-file accept=".md,.txt,text/markdown,text/plain"></label>${examples.length ? `<select data-example aria-label="示例任务"><option value="">选择示例任务…</option>${examples.map((item, index) => `<option value="${index}" translate="no">${esc(item.title || item.name)}</option>`).join("")}</select>` : ""}</div>
          </section>
          <section class="launch-section launch-grid">
            <label class="launch-field"><span>Agent</span><select name="provider">${choice(providers.map((item) => [item.id, providerLabel(item)]), provider.id)}</select></label>
            <label class="launch-field"><span>模型</span><input name="model" value="${esc(provider.default_model)}" list="launch-models" autocomplete="off" spellcheck="false"><datalist id="launch-models">${modelOptions(provider.id)}</datalist><small>可从列表选择，也可填写其他模型 ID。</small></label>
            <label class="launch-field"><span>推理强度</span><select name="reasoning">${choice(reasoning, provider.default_reasoning)}</select></label>
          </section>
          <section class="launch-section launch-grid">
            <label class="launch-field"><span>预算档位</span><select name="tier">${choice((options.tiers || []).map((item) => [item.id, TIERS[item.id] || item.id]), options.default_tier)}</select><small data-tier-note>${tier ? esc(tierNote(tier)) : ""}</small></label>
            <label class="launch-field"><span>总运行时长上限</span><select name="budget_hours">${hourOptions(tier)}</select><small>Agent 的有效工作时间，不含仿真；用完后保留已有结果。</small></label>
            <label class="launch-field"><span>单次仿真上限</span><select name="simulation_timeout_seconds">${simulationOptions(tier)}</select><small>仿真到这个时间还没完成会继续运行：先延长到两倍，最长 30 分钟，仍未完成才记为超时。</small></label>
          </section>
          <details class="launch-section launch-advanced"><summary>高级选项</summary>
            <div class="launch-grid">
              <label class="launch-field"><span>审核 Agent</span><select name="reviewer_provider">${choice(providers.map((item) => [item.id, providerLabel(item)]), "", "默认（Codex 或 Cursor 作者用 Codex，其他与作者相同）")}</select></label>
              <label class="launch-field"><span>审核模型</span><input name="reviewer_model" placeholder="默认" spellcheck="false"></label>
              <label class="launch-field"><span>审核推理强度</span><select name="reviewer_reasoning">${choice(reasoning, "", `默认（${options.reviewer?.default_reasoning || "medium"}）`)}</select></label>
            </div>
            <label class="launch-field"><span>Chrono Python</span><input name="chrono_python" placeholder="${esc(options.chrono_python ? `默认：${options.chrono_python}` : "默认使用 SIMREADY_PYCHRONO_EXECUTABLE")}" spellcheck="false"></label>
          </details>
          <div class="launch-error" data-error hidden></div>
          <footer class="launch-actions"><small data-check-warning></small><button type="button" class="launch-secondary" data-queue ${disabled}>加入队列（串行）</button><button type="submit" data-submit ${disabled}>立即开始（并行）</button></footer>
          <p class="launch-start-note"><span>立即开始：与正在运行的实验同时进行（最多 ${esc(options.concurrency?.limit || 1)} 个）。</span> <span>加入队列：等当前所有运行结束后，按加入顺序逐个开始。</span></p>
        </form>
        <aside class="launch-side" data-checks>${checksPanel()}</aside>
      </div>`;
    bind();
    updateChecks();
    syncReasoning(provider.default_reasoning);
  }

  function modelOptions(provider) {
    return ((options.models || {})[provider] || []).map((item) => `<option value="${esc(item.id)}" translate="no">${esc(item.label)}</option>`).join("");
  }

  // Only offer the reasoning levels the chosen model supports, when the CLI lists them.
  function syncReasoning(preferred) {
    const known = ((options.models || {})[value("provider")] || []).find((item) => item.id === value("model"));
    const allowed = known && known.reasoning.length ? known.reasoning : options.reasoning || [];
    const select = field("reasoning");
    const wanted = preferred || select.value;
    select.innerHTML = choice(allowed.map((item) => [item, item]), allowed.includes(wanted) ? wanted : allowed.includes("high") ? "high" : allowed[0]);
  }

  // The tier's recommended limits are preselected and marked; any listed value may be chosen.
  function hourOptions(tier) {
    return choice((options.budget_hours || []).map((hours) => [String(hours), `${hours} 小时${hours === tier?.hours ? "（推荐）" : ""}`]), String(tier?.hours ?? ""));
  }

  function simulationOptions(tier) {
    return choice((options.simulation_timeouts || []).map((seconds) => [String(seconds),
      `${seconds / 60} 分钟${seconds === tier?.simulation_timeout_seconds ? "（推荐）" : ""}`]), String(tier?.simulation_timeout_seconds ?? ""));
  }

  function tierNote(tier) {
    return `每版最多 ${tier.max_tasks} 项测试 · 最多 ${tier.max_parts} 个零件`;
  }

  function updateChecks() {
    const panel = root.querySelector("[data-checks]");
    if (panel) {
      panel.innerHTML = checksPanel();
      panel.querySelector("[data-refresh-checks]")?.addEventListener("click", () => pollChecks(true));
    }
    const warning = root.querySelector("[data-check-warning]");
    if (warning) warning.textContent = checkWarning();
  }

  async function pollChecks(refresh) {
    clearTimeout(checksTimer);
    try {
      const next = await api(`/launch/options${refresh ? "?refresh=1" : ""}`);
      options.checks = next.checks;
      options.concurrency = next.concurrency;
    } catch (_error) { /* keep the last result; the next poll retries */ }
    updateChecks();
    if (!options.checks?.ready) checksTimer = setTimeout(() => pollChecks(false), 1500);
  }

  function requestBody() {
    const body = { mode: "free", project: value("project"), provider: value("provider"), model: value("model"), reasoning: value("reasoning"), tier: value("tier"), task: field("task").value };
    if (value("budget_hours")) body.budget_hours = Number(value("budget_hours"));
    if (value("simulation_timeout_seconds")) body.simulation_timeout_seconds = Number(value("simulation_timeout_seconds"));
    for (const name of ["chrono_python", "reviewer_provider", "reviewer_model", "reviewer_reasoning"]) if (value(name)) body[name] = value(name);
    return body;
  }

  function showError(error) {
    const box = root.querySelector("[data-error]");
    const known = ERRORS[error.code];
    box.innerHTML = `<strong>${esc(known || "启动失败")}</strong><span>${esc(error.message)}</span>${error.log ? `<pre>${esc(error.log)}</pre>` : ""}`;
    box.hidden = false;
    box.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  async function send(path, button, busyText) {
    const buttons = [root.querySelector("[data-submit]"), root.querySelector("[data-queue]")];
    const idle = button.textContent;
    root.querySelector("[data-error]").hidden = true;
    buttons.forEach((item) => { item.disabled = true; });
    button.textContent = busyText;
    try {
      const result = await api(path, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Human-Decision-Capability": token },
        body: JSON.stringify(requestBody()),
      });
      location.assign(monitorUrl(boot.basePath, result.project, location.hash));
    } catch (error) {
      showError(error);
      buttons.forEach((item) => { item.disabled = false; });
      button.textContent = idle;
    }
  }

  function submit(event) {
    event.preventDefault();
    send("/launch", root.querySelector("[data-submit]"), "正在启动 Harness…（最多约 30 秒）");
  }

  function bind() {
    const form = root.querySelector("form");
    form.addEventListener("submit", submit);
    root.querySelector("[data-queue]").addEventListener("click", (event) => send("/launch/queue", event.currentTarget, "正在加入队列…"));
    field("model").addEventListener("change", () => syncReasoning());
    field("provider").addEventListener("change", () => {
      const provider = (options.providers || []).find((item) => item.id === value("provider"));
      if (provider) {
        field("model").value = provider.default_model;
        root.querySelector("#launch-models").innerHTML = modelOptions(provider.id);
        syncReasoning(provider.default_reasoning);
      }
      updateChecks();
    });
    field("tier").addEventListener("change", () => {
      const tier = (options.tiers || []).find((item) => item.id === value("tier"));
      root.querySelector("[data-tier-note]").textContent = tier ? tierNote(tier) : "";
      field("budget_hours").innerHTML = hourOptions(tier);
      field("simulation_timeout_seconds").innerHTML = simulationOptions(tier);
    });
    root.querySelector("[data-task-file]").addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => { field("task").value = String(reader.result || ""); };
      reader.readAsText(file, "utf-8");
      event.target.value = "";
    });
    root.querySelector("[data-example]")?.addEventListener("change", (event) => {
      const example = (options.examples || [])[Number(event.target.value)];
      if (example) field("task").value = example.text;
      event.target.value = "";
    });
  }

  async function init() {
    try {
      options = await api("/launch/options");
    } catch (error) {
      root.innerHTML = `<div class="fatal-screen"><strong>无法读取运行选项</strong><span>${esc(error.message)}</span></div>`;
      return;
    }
    render();
    if (!options.checks?.ready) pollChecks(false);
  }

  init();
})();
