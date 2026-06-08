"""Trajectory X-ray: run ONE LongMemEval question through the real agent and dump
its FULL trajectory — every Bash/search/grep the agent ran, what it retrieved, and
its answer vs gold. Usage: python traj.py <question_id>"""

import os
import re
import subprocess
import sys

import httpx
from datasets import load_dataset

QID = sys.argv[1]
SECRET = open(".env").read().split("UNISON_EVAL_SECRET=")[1].split("\n")[0].strip().strip('"')
PSQL = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"


def s2t(s):
    if isinstance(s, list):
        return "\n".join(
            (f"{m.get('role')}: {m.get('content')}" if isinstance(m, dict) else str(m)) for m in s
        )
    return str(s)


ds = load_dataset("xiaowu0162/longmemeval-cleaned", split="longmemeval_s_cleaned", streaming=True)
row = next(r for r in ds if str(r.get("question_id")) == QID)
seed = [
    {"path": f"/private/sources/eval/dbg/sess{i}.md", "body": s2t(s), "kind": "raw"}
    for i, s in enumerate(row.get("haystack_sessions") or [])
]
q = f"Today's date is {row.get('question_date')}.\n\n{row['question']}"

c = httpx.Client(base_url="http://localhost:3020", headers={"x-unison-eval": SECRET}, timeout=400)
tid = c.post("/api/rest/agents/eval/provision", json={"label": "traj"}).json()["tenantId"]
r = c.post(
    "/api/rest/agents/eval-turn",
    json={"question": q, "seedDocs": seed, "tenantId": tid, "memoryMode": "fresh"},
).json()

print("=" * 80)
print("Q:", row["question"])
print("GOLD:", row["answer"])
print("ANSWER:", (r.get("answer") or "")[:400])
print("steps:", r.get("totalSteps"), "tenant:", tid)
print("=" * 80)
print("TRAJECTORY (agent tool calls + results):")
# Pull every assistant/tool message for this tenant, in order.
out = subprocess.run(
    ["psql", PSQL, "-tAF", "\t", "-c",
     f"select role, left(content::text, 1200) from agent_messages where tenant_id='{tid}' order by created_at"],
    capture_output=True, text=True,
).stdout
for line in out.split("\n"):
    if "\t" not in line:
        continue
    role, content = line.split("\t", 1)
    # extract Bash commands (searches/greps) and brief result markers
    cmds = re.findall(r'"(?:command|input)"\s*:\s*"([^"]{1,160})"', content)
    for cmd in cmds:
        print(f"  [{role} CMD] {cmd}")
    txt = re.findall(r'"text"\s*:\s*"([^"]{1,200})"', content)
    for t in txt[:2]:
        if len(t) > 30:
            print(f"  [{role} TXT] {t[:160]}")
