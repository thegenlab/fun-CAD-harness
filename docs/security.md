# Security model

A harness run executes code written by an AI agent: the agent edits files and
runs commands during its turns, and the harness then builds the agent's CAD
source and runs its simulation callbacks. Choose where that happens.

## Docker (recommended)

The Docker image is the isolation boundary. Agents and authored programs run as
your (unprivileged) user inside the container. The harness code and the Python
environments in the image are read-only for them, so they cannot change the
programs that judge their designs. They can write to the mounted `./runs`
directory — including other projects' files there — and to the mounted agent
login directories. The GUI port is published on `127.0.0.1` only. Because nested
sandboxes usually cannot start in a container, the image sets
`SIMREADY_CONTAINER_ISOLATION=1` and the agent CLIs run without their own
sandbox inside it. Run one container per project if runs must not see each
other.

## Local conda installation

Without a container, everything runs with your user's permissions:

- Codex CLI author turns use Codex's own workspace-write sandbox where the host
  supports it (writes limited to the project and the author-tool directory,
  network disabled for tool commands); reviewer turns are read-only.
- Claude Code author turns may edit files in the project directories and run
  shell commands inside Claude Code's own sandbox (on Linux it needs
  `bubblewrap` and `socat`; without them the turn fails at launch instead of
  running unconfined). On Linux the sandbox does not filter Unix sockets,
  because the filter cannot start under Ubuntu's AppArmor: commands can reach
  local services such as your systemd/D-Bus session bus and, through them, run
  outside the sandbox. It guards against mistakes, not against a hostile agent.
  Reviewer turns get read-only tools.
- Cursor CLI and Kimi CLI run without a sandbox.
- CAD builds and simulations of agent code are bounded in time and memory but
  not sandboxed.

Agents can read files your user can read. Prefer a dedicated user account, VM
or the Docker image when that matters.

## The GUI

The dashboard binds to `127.0.0.1` by default. The New task page and the
Stop/Resume actions exist only with `--enable-launch` and require the random
per-process token printed at startup, a JSON request and a matching `Origin`.
Do not publish a launch-enabled dashboard on a shared or public network.

## Formal benchmark sandbox

The paper's benchmark runs used an additional fail-closed sandbox with offline
execution and per-run data grants. It depends on site-specific provisioning and
is not included; this release runs in the local or container mode described
above.
