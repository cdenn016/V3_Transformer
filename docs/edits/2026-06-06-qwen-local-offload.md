# 2026-06-06 — Local model offload: routing policy + launcher

Wired up automatic use of the already-configured `qwen-local` MCP server (a thin proxy
to a local `llama-server` serving `Qwen3.6-27B-Q8_0.gguf` at `http://127.0.0.1:8080/v1`)
so bulk, low-stakes text work can be delegated off Claude.

## Changes

- **`CLAUDE.md`** — added a `**Local model offload**` operational-policy item in the
  bold-prefixed cluster. It instructs Claude to delegate voluminous/mechanical low-stakes
  text generation (condensing long docs/logs, first-draft boilerplate, bulk reformatting,
  classify/tag-at-volume) to the `qwen-local` `offload_to_local_model` tool and review the
  output, while NEVER offloading anything that must be correct (math/theory verification,
  gauge/KL/free-energy derivations, code-correctness judgments). Notes the tool is stateless
  (pass it context; map-reduce inputs that exceed its window) and points at the launcher.
  The clause is phrased to self-disable on machines where the tool is absent.

- **`F:\qwen-mcp\start.ps1`** (outside this repo, not tracked here) — new click-to-run
  launcher for the `llama-server` backend. Probes `/health` first and refuses to
  double-launch the ~27 GB model when already serving; otherwise starts `llama-server`
  (`-ngl 99 -c 8192 --host 127.0.0.1 --port 8080`) in its own window and polls until ready.
  Context size is a config var at the top with a comment on the VRAM tradeoff.

## Verification

- `claude mcp list` shows `qwen-local` connected; a `PONG` round-trip through
  `offload_to_local_model` confirmed the backend is live.
- The launcher's `/health` probe was run inline against the live server and returned
  `{"status":"ok"}`, so the "already serving" guard fires correctly. The cold-launch
  branch was not exercised (would require unloading the running model).

## Not done

- Not committed — left in the working tree per the commit-only-when-asked policy.
- The `CLAUDE.md` change is in this repo's checked-in instructions; if the policy should
  not travel to clones, it could instead live in `~/.claude/CLAUDE.md` (user-global).

## Update — LAN binding (0.0.0.0)

Per request, `start.ps1` now binds `--host 0.0.0.0` (config var `$BindHost`) so the model is
reachable from other devices on the LAN, not just this machine. Two correctness/clarity edits
went with it: (1) the health probe and report URL were split onto a separate `$ProbeHost =
'127.0.0.1'` because `0.0.0.0` is a bind address, not a valid connect target; (2) the ready
message now prints the detected LAN URL. A SECURITY NOTE in the config block flags that the
endpoint is unauthenticated (add `--api-key` for a token gate) and needs a Windows Firewall
inbound-allow rule for TCP 8080; never port-forward it to the public internet without auth.
The MCP proxy and the health probe are unaffected — binding `0.0.0.0` still listens on loopback.

Verified after restart: listener `LocalAddress 0.0.0.0:8080 (Listen)`; `/health` → `{"status":"ok"}`;
process persists across separate calls (PID 8872); and a `READY` round-trip through the
`offload_to_local_model` MCP tool confirms the full loopback path. LAN URL reported as
`http://192.168.1.107:8080`. The Windows Firewall rule for TCP 8080 was NOT created — do that
on the machine if other devices can't connect.

## Update — coding context / KV settings (config-only, NOT applied)

`start.ps1` config changed for better coding-length context, after `/props` on the running
server showed `n_ctx: 8192` split across `total_slots: 4` (so a single chat was getting only
~2048 tokens). Flag names were read from this build's `llama-server --help` (build b9538):
`-c`/`--ctx-size`, `-np`/`--parallel` (default -1 = auto), `-fa`/`--flash-attn [on|off|auto]`,
`-ctk`/`-ctv`/`--cache-type-k|v TYPE`. Changes: `$Parallel = 1` (full window to one request),
`-fa on`, `-ctk q8_0 -ctv q8_0` (near-lossless KV-cache quant, ~half the f16 footprint), and
`$Ctx` 8192 → 16384.

NOT applied and NOT GPU-tested: the user is running a training experiment on the 5090, so the
server was deliberately not restarted (a reload could OOM the training run). The earlier
`nvidia-smi` reading (31.7/32.6 GB) was training + qwen combined, not a clean baseline, so the
fit was not measured. Takes effect on the next manual `start.ps1` launch. 16384 is the
VRAM-safe default; 32768 is worth a fit-test on a free GPU and should be lowered if the load
OOMs. An "abandoned Option B" `llama-swap` binary downloaded earlier was deleted.
