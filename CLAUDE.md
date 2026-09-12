# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## AgentMemory

At the start of work in this repo, check AgentMemory for prior context about `Sparse_Additive_HDBO`.

When durable repo context, decisions, workflows, file roles, or non-obvious fixes emerge, save concise memories via AgentMemory. Tag them with:

- `Sparse_Additive_HDBO`

Use AgentMemory REST as a fallback when MCP memory tools are unavailable. In this environment, the reachable host is usually `agentmemory-google:3111`; `localhost:3111` may be unavailable from Codex. Every endpoint except `/agentmemory/livez` requires `Authorization: Bearer $AGENTMEMORY_SECRET`.

Preferred endpoint: `https://banter-supermom-overpay.ngrok-free.dev` (ngrok tunnel to the `agentmemory-server` GCE VM). Try this first for REST calls. Note: ngrok free-tier URLs are ephemeral and can rotate when the tunnel restarts — if it stops responding (connection refused / ngrok interstitial / different repo's data), ask the user for the current URL rather than assuming AgentMemory is down, then fall back to `agentmemory-google:3111` or the SSH tunnel below.

Auth: `AGENTMEMORY_SECRET` is exported from `~/.bashrc` (mode 600). Only `/agentmemory/livez` is unauthenticated — with no header, or an empty variable, every other endpoint returns `HTTP 401 {"error":"unauthorized"}`. **A 401 is not an outage.** Never report AgentMemory as unreachable on a 401: test `[ -n "$AGENTMEMORY_SECRET" ]` first, and never echo the value (`${VAR:-...}` expands to it). Shells that skip `~/.bashrc` get no secret — notably an sbatch script with a plain `#!/bin/bash` shebang; use `#!/bin/bash -l`.

From Triton login nodes the ngrok URL is the only endpoint that works: `agentmemory-google` does not resolve there (no Tailscale), and `localhost:3211`/`:3111` are refused unless you started the tunnel yourself. Four failures in a row is the expected signature of a missing secret, not of a dead service.

The `/remember` payload field is `content`, not `text` — `text` returns `HTTP 400 {"error":"content is required"}`. The server derives its own `title` from the content and ignores a supplied one, and there is no fetch-by-id route; read back with `smart-search`.

Special case: when the user is connected to Aalto VPN / Cisco AnyConnect, `agentmemory-google:3111` may time out because AnyConnect can hijack Tailscale `100.x.x.x` routes. In that case, treat the SSH tunnel URL as the primary AgentMemory endpoint:

- Use `http://localhost:3211` for REST.
- Do not report AgentMemory as unavailable after only testing `agentmemory-google:3111`.
- First run `Invoke-RestMethod -TimeoutSec 5 http://localhost:3211/agentmemory/livez`.
- If `localhost:3211` fails, check `Get-NetTCPConnection -LocalPort 3211` before concluding the tunnel is down.
- Do not start a local AgentMemory server unless the user explicitly asks; `localhost:3211` should be an SSH tunnel to the Google VM's `127.0.0.1:3111`.

Tunnel command:

```powershell
gcloud compute ssh agentmemory-server `
  --zone=us-east1-d `
  --ssh-flag=-N `
  --ssh-flag=-L `
  --ssh-flag=3211:127.0.0.1:3111

$env:AGENTMEMORY_URL = "http://localhost:3211"
```

- ngrok health: `GET https://banter-supermom-overpay.ngrok-free.dev/agentmemory/livez`
- ngrok save: `POST https://banter-supermom-overpay.ngrok-free.dev/agentmemory/remember`
- ngrok search: `POST https://banter-supermom-overpay.ngrok-free.dev/agentmemory/smart-search`
- Health: `GET http://agentmemory-google:3111/agentmemory/livez`
- Save: `POST http://agentmemory-google:3111/agentmemory/remember`
- Search: `POST http://agentmemory-google:3111/agentmemory/smart-search`
- Aalto VPN health: `GET http://localhost:3211/agentmemory/livez`
- Aalto VPN save: `POST http://localhost:3211/agentmemory/remember`
- Aalto VPN search: `POST http://localhost:3211/agentmemory/smart-search`
- Secondary local fallback: replace `agentmemory-google` with `localhost` if the service is exposed locally.

Do not save secrets, credentials, private personal details, or transient scratch notes.


## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Project-Local Skills

This repo vendors 119 agent skills in `.agents/skills/`. Only 8 are auto-loaded
(symlinked into `.claude/skills/`); the other 111 are **available but not
surfaced automatically**. You must look them up.

Before starting any non-trivial task, search the index:

```bash
grep -i "<keyword>" .agents/skills/index.txt   # one "name :: description" per line
cat .agents/skills/<name>/SKILL.md             # read the one that fits, then follow it
```

Try a few keywords, not one — e.g. for a plotting task: `plot`, `visual`,
`figure`, `matplotlib`. Some skills ship extra reference files alongside
`SKILL.md`; `ls .agents/skills/<name>/` before assuming `SKILL.md` is all of it.

Notes:

- `index.txt` is generated. Regenerate after adding or updating skills:
  `python3 .agents/skills/mkindex.py`
- `.agents/skills/` is the Codex discovery path. Claude Code reads
  `.claude/skills/`, which holds relative symlinks into it — edit files under
  `.agents/skills/`, never through the symlink path.
- `find-skills` searches the **public** skills.sh ecosystem via `npx`; it does
  not index this directory. Don't use it to find local skills, and don't run
  `npx skills add` without asking.
- `using-agent-skills` and `using-superpowers` are dispatcher skills covering
  only their own subsets (23 and 14 skills). Treat their "invoke a skill before
  ANY response" mandates as advisory — this file takes precedence.
- Several skills overlap the built-in `/code-review` and `/simplify`. Prefer the
  built-ins unless the task specifically calls for the vendored variant.

## Project-Specific Guidelines
---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.