# Deploying unison-evals to eval.unison.ai

This document is the complete walkthrough for moving the benchmark harness from
local-only to production at `eval.unison.ai`. It covers infrastructure choices,
step-by-step commands, the SQLite → Postgres migration, auth wiring, CI
integration, DNS, and cost controls.

---

## 1. Overview

Three tiers, all independent. You can ship each one separately.

```
                           eval.unison.ai
                                  ↓
                      ┌──────────────────────┐
                      │   Vercel (Next.js)   │
                      │   - /                │
                      │   - /runs/*          │
                      │   - /request-access  │
                      └──────────┬───────────┘
                                 │ /api/* rewrite (server-side)
                                 ↓
                      ┌──────────────────────┐
                      │  Fly.io (FastAPI)    │
                      │  - /health           │
                      │  - /api/runs/*       │
                      │  - /api/registry     │
                      │  - SSE streams       │
                      └──────────┬───────────┘
                                 │
                      ┌──────────────────────┐
                      │  Supabase / Neon     │
                      │  Postgres            │
                      │  - runs              │
                      │  - eval_access_*     │
                      │  - users (Supabase)  │
                      └──────────────────────┘
```

The Next.js server rewrites `/api/*` to the FastAPI host at build time via the
`UNISON_EVALS_API` env var (see `web/next.config.ts`). The browser never crosses
origins — all API calls appear same-origin to the client.

---

## 2. Local dev (the path that works today)

### Option A — native processes (fastest inner loop)

```bash
# 1. Clone and install
git clone https://github.com/Unison-Workspace/Unison-evals.git
cd Unison-evals
uv sync --all-extras          # Python 3.12 via .python-version

# 2. Configure
cp .env.example .env
$EDITOR .env                  # set UNISON_JWT, ANTHROPIC_API_KEY at minimum

# 3. Install web dependencies
cd web && bun install && cd ..

# 4. Two terminals in parallel:
#   Terminal 1 — FastAPI on :8001
make server
#   (or: uv run unison-evals-server)

#   Terminal 2 — Next.js on :3000
make web
#   (or: cd web && bun dev)

# 5. Open http://localhost:3000/runs/new
```

Minimum `.env` for local dev:
- `UNISON_API_URL` — URL of the local Unison API (default `http://localhost:3001`)
- `UNISON_JWT` — a valid JWT for the Unison eval-turn endpoint
- `ANTHROPIC_API_KEY` — for the LLM judge (Claude)

### Option B — Docker Compose (closest to prod)

```bash
# prereqs: .env populated, Docker running
docker compose up
```

This starts both `server` (FastAPI on :8001) and `web` (Next.js on :3000).
The compose file mounts `./results` for SQLite persistence and `hf-cache` for
HuggingFace dataset caching so re-runs are fast.

**When to prefer Docker Compose:** when you want to test the prod image build,
when your local Python env diverges from 3.12, or when you need `host.docker.internal`
routing to a Unison API in another container. For tight iteration on Python code,
native processes (Option A) are faster because you skip the image rebuild.

---

## 3. Infrastructure choices

| Component | Recommended | Alternatives | Cost at v0.5 scale |
|---|---|---|---|
| FastAPI server | Fly.io (1 × `shared-cpu-1x`, 256 MB) | Render, Railway | ~$5/mo |
| Next.js web | Vercel Hobby (free) | Cloudflare Pages, Netlify | $0 |
| Postgres | Supabase free tier (500 MB) | Neon free tier, RDS t4g.micro | $0 |
| Domain | Cloudflare DNS | any registrar | ~$10/yr |
| Anthropic keys | per-eval-run usage only | — | variable |
| OpenAI keys | pgvector_naive embeddings only | — | variable |

**Total fixed: ~$5/mo + variable API costs.** The judge model choice dominates
variable cost: Claude Opus at ~$0.005/judgment vs. Claude Haiku at ~$0.0005/judgment.
A 200-question full run against two systems with Opus costs roughly $2 in judge
calls; with Haiku it costs $0.20. Reserve Opus for published leaderboard numbers;
use Haiku in CI (see Section 12).

---

## 4. Postgres migration (SQLite → Postgres)

`storage.py` uses SQLModel, which supports both SQLite and Postgres via
SQLAlchemy. The only change needed is the connection URL.

**The diff (do not apply in this PR — document only):**

```python
# src/unison_evals/server/storage.py

class Storage:
-   def __init__(self, db_path: Path) -> None:
-       db_path.parent.mkdir(parents=True, exist_ok=True)
-       self.engine = create_engine(
-           f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
-       )
-       SQLModel.metadata.create_all(self.engine)
+   def __init__(self, db_path: Path, database_url: str | None = None) -> None:
+       if database_url:
+           # Postgres — DATABASE_URL set in environment
+           self.engine = create_engine(database_url)
+       else:
+           # SQLite fallback for local dev
+           db_path.parent.mkdir(parents=True, exist_ok=True)
+           self.engine = create_engine(
+               f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
+           )
+       SQLModel.metadata.create_all(self.engine)
```

And in `server/main.py`, thread the env var through:

```python
import os

storage = Storage(
    db_path=settings.results_dir / "runs.db",
    database_url=os.getenv("DATABASE_URL"),
)
```

**Schema notes for v0.5:**
- `SQLModel.metadata.create_all` (current approach) is fine for v0.5 — it creates
  missing tables but doesn't run migrations. Safe as long as you don't rename
  existing columns.
- For v1.0, adopt Alembic: `uv add alembic`, `alembic init alembic`, configure
  `target_metadata = SQLModel.metadata` in `env.py`. Generate a baseline migration
  from the existing schema with `alembic revision --autogenerate -m "initial"`.

**Connection string format for Supabase pooler (recommended for serverless):**

```
postgresql://postgres.[ref]:[password]@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require
```

Note the port `6543` (Supabase transaction pooler) rather than `5432` (direct).
Set `connection_limit=1` in the query string if you hit pool exhaustion:
`...postgres?sslmode=require&connection_limit=1`

---

## 5. Fly.io deploy (FastAPI server)

### 5.1 Install and authenticate

```bash
# macOS
brew install flyctl
# Linux
curl -L https://fly.io/install.sh | sh

flyctl auth signup   # or: flyctl auth login (if you already have an account)
```

### 5.2 Launch the app

Run this from the repo root. It reads `Dockerfile.server` automatically.

```bash
flyctl launch \
  --name unison-evals-api \
  --dockerfile Dockerfile.server \
  --region iad \
  --no-deploy
```

This writes a `fly.toml` to the repo root. The file Fly generates is close to
correct; replace it with the version below for the right port and health check:

```toml
# fly.toml — commit this to the repo
app = "unison-evals-api"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile.server"

[env]
  PYTHONUNBUFFERED = "1"

[http_service]
  internal_port = 8001
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

  [[http_service.checks]]
    path = "/health"
    interval = "30s"
    timeout = "5s"

[[vm]]
  size = "shared-cpu-1x"
  memory = "256mb"
```

`auto_stop_machines = true` + `min_machines_running = 0` means the machine idles
to zero between runs — important for cost control at v0.5 scale where runs are
infrequent. The machine cold-starts in ~3 seconds, which is acceptable for eval
jobs that run for minutes anyway.

### 5.3 Set secrets

```bash
flyctl secrets set \
  ANTHROPIC_API_KEY="sk-ant-..." \
  OPENAI_API_KEY="sk-..." \
  DATABASE_URL="postgresql://postgres.[ref]:..." \
  UNISON_API_URL="https://api.unison.ai" \
  UNISON_JWT="eyJ..." \
  JUDGE_MODEL="claude-haiku-4-5" \
  MAX_CONCURRENT_QUESTIONS="2"
```

Secrets are encrypted at rest and injected as env vars at runtime. They never
appear in image layers.

### 5.4 Deploy

```bash
flyctl deploy
```

### 5.5 Custom domain

```bash
flyctl certs add api.eval.unison.ai
```

Fly prints the CNAME target. Add it to Cloudflare (see Section 10).

### 5.6 Verify

```bash
curl https://api.eval.unison.ai/health
# → {"status": "ok", "service": "unison-evals"}
```

---

## 6. Vercel deploy (Next.js web)

### 6.1 Install and authenticate

```bash
npm i -g vercel    # or: bun add -g vercel
vercel login
```

### 6.2 Initial deploy

```bash
cd web
vercel
```

Follow the interactive prompts: link to the `Unison-Workspace` Vercel team,
set project name to `unison-evals-web`, root directory to `web` (Vercel detects
Next.js automatically).

### 6.3 Environment variables

The rewrite in `next.config.ts` runs server-side and reads `UNISON_EVALS_API`.
This is the only variable Vercel needs for the API proxy:

```bash
vercel env add UNISON_EVALS_API production
# enter: https://api.eval.unison.ai
```

If you later add client-side fetches that bypass the proxy, add:

```bash
vercel env add NEXT_PUBLIC_API_URL production
# enter: https://api.eval.unison.ai
```

For the Supabase auth flow (Section 8), also add:

```bash
vercel env add NEXT_PUBLIC_SUPABASE_URL production
vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY production
```

### 6.4 Production deploy

```bash
vercel --prod
```

### 6.5 Custom domain

In the Vercel dashboard → Project → Settings → Domains, add `eval.unison.ai`.
Vercel shows the CNAME or A record to add in Cloudflare (see Section 10).

### 6.6 CORS notes

The Next.js rewrite means the browser only talks to `eval.unison.ai` (same
origin). The FastAPI CORS policy in `main.py` currently allows `*`. For v0.5,
lock it down:

```python
allow_origins=["https://eval.unison.ai"],
```

Keep `allow_credentials=True` if you forward the Supabase JWT as a cookie;
leave `False` if you pass it as a `Authorization: Bearer` header (recommended).

---

## 7. Supabase Postgres setup

### 7.1 Create a project

1. Go to [supabase.com](https://supabase.com), create a new organization (or use
   an existing Unison org — keep eval users separate from product users).
2. Create project: name `unison-evals`, region `us-east-1` (matches Fly `iad`).
3. Note the project reference ID (looks like `abcdefghijklmnop`).

### 7.2 Get the DATABASE_URL

In the Supabase dashboard → Project Settings → Database → Connection string, pick
**Transaction pooler** (port 6543):

```
postgresql://postgres.[ref]:[db-password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres?sslmode=require
```

This is what you set as `DATABASE_URL` in `flyctl secrets set` above.

### 7.3 Run schema migration

For v0.5, `SQLModel.metadata.create_all` (called at FastAPI startup) creates the
`runs` table automatically on first boot. No manual SQL needed.

To verify the table was created:

```bash
# using psql
psql "$DATABASE_URL" -c "\dt"
# should show: runs
```

### 7.4 Supabase auth (optional for v0.5, required for v1.0)

Enable the **Email (magic link)** provider in Supabase dashboard → Authentication
→ Providers. This powers the request-access and login flows described in
Section 8.

---

## 8. Auth / request-access flow

This section is the spec for the next subagent that implements auth. The v0.0
deployment has no auth (localhost only). v0.5 adds gated access without a full
auth overhaul.

### Tables needed (create in Supabase SQL editor)

```sql
-- Gating table — submitted via the public /request-access form
create table if not exists eval_access_requests (
  id uuid primary key default gen_random_uuid(),
  email text not null,
  name text,
  org text,
  reason text,
  submitted_at timestamptz default now(),
  approved boolean default false,
  approved_at timestamptz
);

-- Per-evaluator config (rate limits, monthly budget cap)
create table if not exists evaluators (
  user_id uuid primary key references auth.users(id),
  email text not null,
  monthly_budget_usd numeric(8,2) default 50.00,
  runs_this_month int default 0,
  approved_at timestamptz,
  notes text
);
```

### Request-access flow (web → Supabase)

1. `/request-access` page submits `{email, name, org, reason}` via the Supabase
   JS client (anon key). Anon key has INSERT-only RLS on `eval_access_requests`.
2. Supabase Webhook (or a simple cron) emails the maintainer when a new row arrives.
3. An admin approves: `UPDATE eval_access_requests SET approved = true WHERE id = '...'`.
4. An admin creates the Supabase auth user:
   ```sql
   -- Supabase dashboard → Authentication → Users → Invite user
   -- (or via the Supabase admin API)
   ```
5. A Supabase database function (trigger on `auth.users` insert) creates the
   corresponding `evaluators` row.

### Login flow (web → Supabase magic link)

1. `/login` page takes the user's email and calls `supabase.auth.signInWithOtp({email})`.
2. Supabase sends a magic link. User clicks it, lands back at `eval.unison.ai/runs/new`.
3. The Supabase JS client stores the session JWT in `localStorage` (or a cookie).
4. All fetch calls to `/api/*` include `Authorization: Bearer <jwt>`.

### FastAPI JWT validation

FastAPI validates the JWT using the Supabase JWKS endpoint:

```python
# server/routes/auth.py (to be implemented)
import httpx
from jose import jwt, JWTError

SUPABASE_JWKS_URL = "https://<ref>.supabase.co/auth/v1/.well-known/jwks.json"

async def get_current_user(authorization: str = Header(...)):
    token = authorization.removeprefix("Bearer ")
    # fetch JWKS (cache in production)
    jwks = httpx.get(SUPABASE_JWKS_URL).json()
    try:
        payload = jwt.decode(token, jwks, algorithms=["RS256"], audience="authenticated")
        return payload["sub"]  # user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

Use `user_id` to look up the `evaluators` row and enforce the monthly budget cap
before queuing the job.

---

## 9. CI integration on the Unison side

Nightly smoke runs catch regressions in the Unison adapter before they reach the
published leaderboard.

### 9.1 Service account

1. In the eval Supabase project, create a user `evals-ci@<your-domain>` via the
   Supabase admin UI.
2. Extend the JWT lifetime for this account (Supabase dashboard →
   Authentication → Settings → JWT expiry → set to `604800` for 7 days, or use
   a long-lived service-role token scoped to this account).
3. Store the JWT as `UNISON_JWT` in your deployment repo's GitHub Actions secrets.

### 9.2 Nightly workflow

Create `.github/workflows/nightly-eval.yml` in your CI repo:

```yaml
# .github/workflows/nightly-eval.yml.example
# Copy to .github/workflows/nightly-eval.yml in your CI repo.
# Runs a 50-question smoke eval nightly and posts results to gh-pages.

name: Nightly eval

on:
  schedule:
    - cron: "0 3 * * *"   # 03:00 UTC daily
  workflow_dispatch:        # allow manual trigger

jobs:
  smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    env:
      UNISON_EVALS_API: https://api.eval.unison.ai
      JUDGE_MODEL: claude-haiku-4-5   # cheap; Opus only for published numbers

    steps:
      - name: Trigger eval run
        id: trigger
        run: |
          RUN_ID=$(curl -sX POST "$UNISON_EVALS_API/api/runs" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${{ secrets.UNISON_JWT }}" \
            -d '{
              "dataset": "longmemeval",
              "track": "agent-oracle",
              "systems": ["unison-agent"],
              "limit": 50,
              "judge_model": "claude-haiku-4-5"
            }' | jq -r '.run_id')
          echo "run_id=$RUN_ID" >> "$GITHUB_OUTPUT"

      - name: Wait for completion
        run: |
          RUN_ID="${{ steps.trigger.outputs.run_id }}"
          for i in $(seq 1 60); do
            STATUS=$(curl -s "$UNISON_EVALS_API/api/runs/$RUN_ID" \
              -H "Authorization: Bearer ${{ secrets.UNISON_JWT }}" \
              | jq -r '.status')
            echo "[$i] $STATUS"
            if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then break; fi
            sleep 30
          done
          [ "$STATUS" = "completed" ] || exit 1

      - name: Fetch results
        run: |
          curl -s "$UNISON_EVALS_API/api/runs/${{ steps.trigger.outputs.run_id }}" \
            -H "Authorization: Bearer ${{ secrets.UNISON_JWT }}" \
            > results.json

      - name: Publish to gh-pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: .
          publish_branch: gh-pages
          keep_files: true
          destination_dir: nightly/${{ github.run_id }}
```

This posts a `results.json` to a `gh-pages` branch. The leaderboard page at
`eval.unison.ai` reads recent nightly runs from this branch.

---

## 10. Domain + DNS (`eval.unison.ai`)

Assumes Cloudflare DNS is managing `unison.ai`.

| Record | Type | Name | Value | Proxy |
|---|---|---|---|---|
| Web (Vercel) | CNAME | `eval` | `cname.vercel-dns.com` | Off (DNS only) |
| API (Fly.io) | CNAME | `api.eval` | `<app>.fly.dev` | Off (DNS only) |

**Why Proxy = Off:** Both Vercel and Fly.io handle TLS termination themselves.
Enabling Cloudflare proxy (orange cloud) can break Fly's TLS certificate
issuance and Vercel's custom domain verification. Use DNS-only (grey cloud).

**TLS:**
- Vercel issues a Let's Encrypt cert for `eval.unison.ai` automatically after
  the CNAME propagates (typically 5–30 minutes).
- Fly issues a cert for `api.eval.unison.ai` after `flyctl certs add api.eval.unison.ai`
  — this also triggers Let's Encrypt and takes 1–5 minutes. Check status with
  `flyctl certs show api.eval.unison.ai`.

**HSTS:** Vercel enables HSTS by default on custom domains. No action needed.

**Apex redirect (optional):** If you want `unison.ai` bare apex to redirect to
the app, add a Cloudflare Page Rule or redirect rule. Not needed if `eval.unison.ai`
is the canonical URL.

---

## 11. Env var checklist

Cross-reference with `.env.example`. This table shows what each service needs.

### FastAPI server (Fly.io secrets)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | LLM judge + Claude Code adapter |
| `OPENAI_API_KEY` | No* | — | Required only for `pgvector_naive` adapter |
| `DATABASE_URL` | Yes (prod) | SQLite fallback | Supabase pooler URL |
| `UNISON_API_URL` | Yes | — | The deployed Unison API |
| `UNISON_JWT` | Yes | — | Long-lived eval service-account JWT |
| `JUDGE_MODEL` | No | `claude-opus-4-5-20250101` | Pin per release; use Haiku in CI |
| `ADAPTER_TIMEOUT` | No | `120` | Seconds per adapter call |
| `JUDGE_TIMEOUT` | No | `60` | Seconds per judge call |
| `MAX_CONCURRENT_QUESTIONS` | No | `3` | Lower to `2` on the small Fly machine |
| `MEM0_API_KEY` | No | — | Required only for `mem0` adapter |
| `LETTA_API_KEY` | No | — | Required only for `letta` adapter |
| `LETTA_BASE_URL` | No | — | Required only for self-hosted Letta |
| `SERVER_PORT` | No | `8001` | Must match `fly.toml` `internal_port` |

### Next.js web (Vercel env vars)

| Variable | Required | Notes |
|---|---|---|
| `UNISON_EVALS_API` | Yes | FastAPI URL; used server-side by `next.config.ts` rewrite |
| `NEXT_PUBLIC_SUPABASE_URL` | Yes (v0.5) | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes (v0.5) | Supabase anon key (safe to expose) |
| `NEXT_PUBLIC_API_URL` | No | Only needed if you add client-side fetch bypassing the proxy |

---

## 12. Cost ceiling controls

### Per-run: judge model selection

The judge model is the dominant cost lever. Set `JUDGE_MODEL` at the right level
for each context:

| Context | Model | Est. cost / 200-Q run |
|---|---|---|
| Published leaderboard numbers | `claude-opus-4-5-20250101` | ~$2.00 |
| Nightly CI smoke (50 Q) | `claude-haiku-4-5` | ~$0.05 |
| Local dev iteration | `claude-haiku-4-5` | ~$0.01 |

### Per-evaluator: monthly budget cap

The `evaluators.monthly_budget_usd` column (Section 8) is the enforcement point.
FastAPI reads this before queuing a run and rejects with `HTTP 402` if the
evaluator has exceeded their cap. The `runs_this_month` counter resets via a
Supabase cron or a simple `date_trunc('month', now())` check at queue time.

### `--max-cost-usd` flag (backlog)

A future `--max-cost-usd N` CLI flag will abort a run mid-stream if accumulated
API spend (tracked in `cost.py`) exceeds the cap. Add to the backlog as a v1.0
feature. For now, set `--limit` to bound question count.

### Fly.io machine autoscale

The `fly.toml` above uses `auto_stop_machines = true` and `min_machines_running = 0`.
The machine stops between runs, so you pay only for actual wall-clock eval time
(typically 10–30 minutes per run). At the `shared-cpu-1x` rate (~$0.000008/second),
even 10 hours of active eval time per month costs ~$0.30 in compute. The $5/mo
estimate is the Fly minimum monthly charge, not actual usage cost.

### HuggingFace dataset caching

Datasets are downloaded once and cached in the `hf-cache` volume (local) or in
the container's filesystem (Fly). On Fly, the cache is lost on each deploy. Use
a Fly volume for persistence if dataset downloads become a meaningful cost or
latency issue:

```bash
flyctl volumes create hf_cache --region iad --size 5
# then mount it in fly.toml:
# [[mounts]]
#   source = "hf_cache"
#   destination = "/root/.cache/huggingface"
```
