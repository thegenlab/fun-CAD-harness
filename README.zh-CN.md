# FunCAD-Harness

[English](README.md)

**编码智能体设计功能性 CAD 机构，物理测试框架（harness）告诉它们这些机构是否真的能用。**

FunCAD-Harness 是论文 *Can It Actually Work? Synthesizing Functional CAD
Mechanisms* 的配套测试框架。编码智能体（Codex CLI、Claude Code 等）根据自然语言任务，
以 CAD 源码和 STEP 几何的形式构建机构。框架把原始几何转换为 PyChrono 仿真，由独立的审查智能体把任务的功能测试绑定到这一具体设计上，
测得的结果再反馈给智能体用于下一版修改——直到机构通过测试或预算耗尽。

[项目主页](https://thegenlab.ai/funcad/) · 论文（arXiv 链接即将发布） ·
[引用](#引用)

```
task prompt ─▶ author agent ─▶ CAD source + STEP ─▶ harness prepares geometry
                    ▲                                        │
                    │                                        ▼
         feedback & evidence ◀── PyChrono tests ◀── reviewer binds tests
```

本仓库包含：

- 测试框架本体：运行循环、预算、几何准备、接触与关节建模、物理运行时与证据
  （`simulation_ready_cad_workflow/validation_pipeline`）；
- 智能体编写测试时使用的通用任务模板库（`benchmark_templates/*.py`）；
- 用于启动运行并实时观看的 Web GUI（智能体轨迹、CAD 历史、3D 查看器、仿真回放）；
- 针对 **Codex CLI** 和 **Claude Code** 的智能体集成，以及实验性的
  Cursor CLI 和 Kimi CLI 支持。

FunCADBench 基准（46 个机构族、任务模板和结果）不包含在本次发布中；
描述新机构的*自由创作*模式则完整支持。

## 快速开始

运行实验有三种方式。三种方式运行的是同一条框架命令行，因此可以混用——
例如从编码智能体启动运行，再在 GUI 中观看。

| 方式 | 做法 | 适用场景 |
|---|---|---|
| **1. 让你的编码智能体来做** | 在本仓库中打开 Claude Code 或 Codex，说 *"Read README.md and run a folding stool."* | 最快上手：智能体自行安装、启动运行并汇报结果 |
| **2. GUI** | `scripts/funcad-gui`（或 Docker），然后进入 **New task** 页面 | 实时观看运行、并行运行与队列、停止/恢复 |
| **3. 命令行** | `scripts/funcad create --mode free --project … --task …` 并带参数 | 脚本和批量实验 |

### 1. 让你的编码智能体来做

克隆仓库，在其中打开你的编码智能体（Claude Code、Codex CLI 等），然后提出要求，例如：

> Read README.md and run a folding stool.

（或者「看看 README.md，跑一个折叠凳」）。智能体会按照
[通过编码智能体运行实验](#通过编码智能体运行实验)执行：
必要时安装环境，用命令行启动运行，启动 GUI 供你观看，并告诉你运行如何结束。
你的编码智能体只负责操作框架；机构本身由框架启动的作者智能体和审查智能体设计
（默认使用 Codex CLI），因此它们的 CLI 必须已登录。

### 安装

你需要一个编码智能体的账号：[Codex CLI](https://github.com/openai/codex)
（ChatGPT 登录或 `OPENAI_API_KEY`）或 [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
（Claude 登录或 `ANTHROPIC_API_KEY`）。然后从两种安装方式中选**一种**。
两者最终得到同一个 GUI。

#### 方式 A — Docker（推荐）

Docker 把所有内容放进一个镜像，同时隔离智能体编写并运行的程序。在 Linux 或 macOS 上：

```bash
git clone https://github.com/thegenlab/fun-CAD-harness.git
cd fun-CAD-harness
scripts/funcad-docker codex-login      # once; or: scripts/funcad-docker claude-login
scripts/funcad-docker                  # builds the image and starts the GUI
```

打开日志中打印的 `http://127.0.0.1:8765/launch#human=…` 链接。
运行结果保存在 `./runs`。登录信息存放在 `~/.codex` / `~/.claude`，
并与宿主机上的智能体 CLI 共享，因此宿主机上已有的登录同样可用
（macOS 上的 Claude Code 除外，它把登录信息保存在钥匙串中：请使用
`claude-login` 或 `ANTHROPIC_API_KEY`）。若改用 API 密钥，请在启动前导出
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`。首次构建会下载约 10 GB
（主要是 PyChrono 及其数值计算库）。在 Apple Silicon Mac 上，x86-64 镜像通过
Docker Desktop 的模拟运行，速度较慢但可以使用。

在 Windows（Docker Desktop、PowerShell）上：执行 `mkdir runs`，设置
`$env:HOME = $env:USERPROFILE`，然后执行 `docker compose up --build`。

#### 方式 B — conda

Linux x86-64（已测试）或 Windows x86-64；macOS 请使用 Docker。需要
[Miniforge](https://conda-forge.org/download/)、conda 或 micromamba，
并且智能体 CLI 在 `PATH` 中（`npm install -g @openai/codex` 和/或
`npm install -g @anthropic-ai/claude-code`）。在 Linux 上，Claude Code 在自己的沙箱中运行
作者智能体的 shell 命令，这需要 `bubblewrap` 和 `socat`
（`sudo apt install bubblewrap socat`）。

```bash
git clone https://github.com/thegenlab/fun-CAD-harness.git
cd fun-CAD-harness
scripts/setup_conda.sh      # creates funcad-cad and funcad-chrono, writes .env
scripts/funcad-gui          # starts the GUI on http://127.0.0.1:8765
```

在 Windows 的 PowerShell 中使用 `scripts\setup_conda.ps1` 和 `scripts\funcad-gui.ps1`。
打开 GUI 打印的 `/launch#human=…` 链接。

> 在 conda 模式下，智能体及其编写的程序直接在你的机器上以你的用户权限运行。
> 如果无法接受，请使用 Docker 或专用机器/虚拟机——参见 [docs/security.md](docs/security.md)。

### 2. 在 GUI 中启动运行

![The New task page](docs/images/new-task.png)

1. 在 **New task** 页面，选择一个示例（如 *cam follower toy*），或自己撰写机构描述。
2. 选择智能体（Codex CLI 或 Claude Code）、模型和推理级别、预算档位、总时间限制以及单次仿真限制。
   推荐的默认值是 *high* 档位，5 小时和 10 分钟；*light* 档位加 1 小时适合作为快速的首次尝试。
   仿真时间不计入总时间限制，到达限制时仍在运行的仿真会继续运行——最长为限制的两倍，
   并以 30 分钟为上限——之后才算作超时。
3. 点击 **Start now (parallel)** 与已有的运行并行启动，或点击 **Add to queue (serial)**，
   在当前运行全部结束后按顺序自动启动。你可以随时回到 New task 页面添加更多实验。
4. **Run monitor** 页面（`/runs`）列出每个实验的已运行时长、阶段、预算使用情况、
   最新的智能体活动和结果，并每隔几秒自动刷新。每张卡片会播放最新记录动作的短动画；
   点击即可打开完整的 3D 动作回放。页面提供 **Stop**、**Resume** 和队列控制；
   **Open details** 进入实时项目视图，可查看智能体的轨迹、每个 CAD 版本、3D 模型、仿真结果和回放。

New task 页面上的环境面板会显示 CAD 和 PyChrono 解释器以及智能体 CLI 是否就绪，
并为缺失的项目给出修复提示。
语言切换（中文 / English）位于页面顶部。

### 3. 命令行

GUI 启动的就是你可以自己运行的同一条命令。`scripts/funcad` 使用 `.env` 中的环境运行它
（Windows：`scripts\funcad.ps1`；Docker：`scripts/funcad-docker cli …`，
你自己的任务文件放在 `./tasks`）：

```bash
scripts/funcad --worker-provider claude_code --reasoning high --tier light \
  create --mode free --construction --project cam_toy --task examples/tasks/cam_follower_toy.md

scripts/funcad resume cam_toy
```

命令会一直运行到实验结束。可以在 GUI（`scripts/funcad-gui`）中观看，
其 Run monitor 会列出每个项目，包括从命令行启动的运行。

## 命令行参考

不指定 `--budget-hours` 和 `--simulation-timeout` 时，运行使用其档位的推荐限制
（`high`：总计 5 小时，每次仿真 10 分钟）。
`--construction`（GUI 使用它，也是 `high` 档位的默认值）在 construction 监督器下运行项目，
使循环在智能体出错后仍能继续。项目位于 `.runtime_runs/projects/<project>/`
（任务、工作流状态、智能体回合、每个版本的设计、测试运行和证据）。

| 智能体 | `--worker-provider` | 登录 | 状态 |
|---|---|---|---|
| Codex CLI | `codex`（默认） | `codex login` 或 `OPENAI_API_KEY` | 论文中使用 |
| Claude Code | `claude_code` | `claude` 登录或 `ANTHROPIC_API_KEY` | 已支持 |
| Cursor CLI | `cursor` | `cursor-agent login` | 实验性 |
| Kimi CLI | `kimi` | Kimi CLI 登录 | 实验性 |

独立审查者默认：作者为 Codex 和 Cursor 时使用 Codex，否则使用作者自己的智能体；
可通过 `--reviewer-provider` 和 `--reviewer-model` 选择其他项。预算、档位、超时、模型和环境变量
见 [docs/configuration.md](docs/configuration.md)。

## 通过编码智能体运行实验

以下步骤面向被用户要求在本仓库中运行实验的编码智能体（Claude Code、Codex 等）。
你负责操作框架；不要自己设计机构，也绝不要编辑 `.runtime_runs/` 或框架下的文件来改变结果。

1. **环境。** 如果缺少 `.env`，运行 `scripts/setup_conda.sh`
   （Windows：`scripts\setup_conda.ps1`；需要 10–20 分钟）。如果用户更喜欢 Docker，
   则下文出现 `scripts/funcad …` 的地方改用 `scripts/funcad-docker cli …`，
   GUI 改用 `scripts/funcad-docker`。检查作者智能体的 CLI 是否已登录
   （`codex login status` 或 `claude auth status`）；未登录则请用户登录。
2. **任务。** 如果用户指定了示例，使用 `examples/tasks/<name>.md`
   （`folding_stool`、`cam_follower_toy`、`pulley_hoist`、`screw_jack`）。
   否则把用户的描述用英文写成带有可度量功能需求的文本，保存到 `tasks/<project>.md`。
   项目 id 由小写字母、数字和下划线组成，且在
   `.runtime_runs/projects/` 中必须尚不存在（使用 Docker 时为 `runs/projects/`）。
3. **启动**运行——它会持续数小时，所以放到后台，输出写入日志文件：
   ```bash
   mkdir -p .runtime_runs/logs
   nohup scripts/funcad --worker-provider codex --reasoning high --tier high \
     create --mode free --construction --project folding_stool \
     --task examples/tasks/folding_stool.md > .runtime_runs/logs/folding_stool.log 2>&1 &
   ```
   使用用户要求的智能体、模型、推理级别和预算（参见
   [命令行参考](#命令行参考)和
   [docs/configuration.md](docs/configuration.md)）；否则保持这些默认值。
4. **让用户观看。** 在后台启动 GUI
   （`nohup scripts/funcad-gui > .runtime_runs/logs/gui.log 2>&1 &`），并把该日志中的
   `Run monitor: http://127.0.0.1:8765/runs#human=…` 链接交给用户。
   如果端口 8765 上已有 GUI 在运行，则复用它的链接。
5. **检查进度**：在被询问时，查看
   `.runtime_runs/projects/<project>/workflow.json`（使用 Docker 时为
   `runs/projects/<project>/workflow.json`；字段 `phase`、`revision`、
   `stop_reason`）和每个版本的 `runs/rNN/result.json`（`status`、
   各项测试结果），或者看日志末尾。命令退出后——在其最终动作预览和摘要之后——
   运行即已结束，其日志的最后一行是 JSON 状态。
6. **汇报**：运行结束后，说明结果、共多少个版本、哪些测试通过，以及用于动作回放的 Run monitor 链接。
   `scripts/funcad resume <project>` 会用剩余预算继续已停止的运行。

## 运行如何工作

1. **构建。** 作者智能体把提示分解为所需功能，构建机构（任意 CAD 工具；已提供 build123d），
   并使用[模板 API](benchmark_templates/AUTHORING_API.md)起草 `task_template.py`。
   在其回合中，它可以用
   [作者工具](simulation_ready_cad_workflow/validation_pipeline/slim_harness/AUTHOR_TOOLS.md)
   构建、预览、检查和仿真草稿。
2. **准备。** 框架导入原始 STEP 实体，构建碰撞几何，并检查零件、关节和支撑。
   它还会告知作者在真实装配中会悬空的零件——销钉够不到孔的关节、不接触任何东西的零件——
   这类提示只是建议，绝不决定结果。
3. **审查与绑定。** 独立的审查智能体审查模板，
   并把其中的操作和测量绑定到本设计的实际零件上。
4. **仿真。** PyChrono 运行每一次试验；对测量信号的固定表达式决定每项功能检查的结果。
5. **反馈。** 结果、失败证据和剩余预算返回给作者，
   作者据此修改设计、调整允许的建模选择或难度参数（会被记录，绝不悄悄修改），或者结束。

物理结果始终来自程序，绝不来自智能体。当某个版本因智能体交付的内容而无法准备、绑定或仿真时，
问题会作为版本反馈返回给作者。运行在以下情况结束：通过测试；预算用尽
（在此之前，最后一个版本会得到一次最终评估，作者可以选择交付现有成果）；
需要你处理智能体登录或配额；或者你按下 Stop。如果运行结束时没有任何可播放的仿真，
框架会导出已记录的位姿，或让最终设计在重力下稳定片刻，
因此总有动作可看；这类动作会标注为预览，而非测试。

## 测试

```bash
python scripts/run_tests.py
```

请使用 CAD 环境的 Python 运行。需要 PyChrono 的测试在 `SIMREADY_TEST_CHRONO_PYTHON`
指向物理解释器时运行，否则跳过。列在
`simulation_ready_cad_workflow/validation_pipeline/tests/known_failures.txt`
中的测试在本次发布所基于的开发分支上就已失败（主要是关于预算检查点格式的过时断言）；
在修复之前，它们会被报告为预期失败。

## 引用

```bibtex
@article{zhang2026funcad,
  title   = {Can It Actually Work? Synthesizing Functional CAD Mechanisms},
  author  = {Zhang, Wenqian and Huang, Yangyi and Xu, Zigeng and Liu, Weiyang and Liu, Zhen},
  year    = {2026}
}
```

## 许可证

MIT——见 [LICENSE](LICENSE)。仪表盘中捆绑的 three.js 构建保留其自身的 MIT 声明
（`dashboard/static/viewer3d.LICENSE.txt`）。
