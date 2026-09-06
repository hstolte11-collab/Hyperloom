<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Native Codex runtime selection

Set `INFERENCE_OPTIMIZER_CODEX_AUTH_MODE=native_oauth` to select Codex OAuth
transport. TraceLens, automatic Forge selection and kernel candidate review
recognize this transport without endpoint or API-key variables. Explicit Forge
`FORGE_AGENT_BACKEND=claude` remains authoritative. Gateway deployments keep
their existing provider preferences, including candidate review's OpenAI
preference when both endpoint sides are configured.

Set `INFERENCE_OPTIMIZER_CODEX_BIN` to the executable inside a **complete**
operator-provided Codex CLI bundle. A lone copied `bin/codex` is insufficient for
bundles requiring `codex-code-mode-host` and other adjacent runtime components.
Hyperloom does not install or copy that bundle, inspect its version, or claim
model availability from a path check.

SDK sessions resolve a nonempty per-call `codex_bin` first, then this deployment
override, then the existing SDK default. Configuring the deployment override
opts into eager validation: the selected path must resolve to a regular
executable before SDK startup, and bare names resolve against the effective
`PATH`. Without the new deployment override, existing explicit-only SDK and
subprocess configuration remains late-bound as before; the actual SDK/process
launch rejects an unusable executable rather than selecting a fallback. This
preserves command construction for worker-local paths. The shared session seam
covers the coordinator, one-shot clients and in-process specialists. Specialist
subprocesses apply explicit/deployment precedence before their existing
PATH/bundled-runtime lookup. Invalid selections never select another binary.

For KernelForge children, an explicit `FORGE_AGENT_CLI` wins over the shared
variable. Hyperloom validates and forwards the result through KernelForge's
existing `Config.agent_cli` / `AgentRuntimeConfig.executable` / `CodexConfig.codex_bin`
seam. Ray environment forwarding preserves these runtime and provider settings;
operators must mount the same complete bundle and OAuth home at those paths on
each worker. Set KernelForge's existing native auth options separately, for
example `FORGE_AGENT_OPTIONS_JSON` with `auth_mode=native_oauth` and the mounted
`home`. No credentials are synthesized.

The native quantization prelude now forwards the selected provider and its CLI
model (`--codex-model` or `--claude-model`) into the existing faithful Quark
adapter. Standalone use retains `quantization-agent --provider codex --model-id
<model>`. Existing quantization result validation and acceptance are unchanged.

Model selection remains explicit and independent of executable selection. A
campaign requiring one model must pin coordinator, specialists, critic,
scorer, RCA, audit, reports, TraceLens, candidate review, Forge and Quark inputs.
Request-level model overrides remain authoritative; environment pins alone do
not enforce a campaign-wide model allowlist. Separate review and live tool-call
validation are required before a campaign launch.
