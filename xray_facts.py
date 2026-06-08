"""Per-failure X-ray: seed a question through the real (count-verify) server, then
dump GOLD vs the full seeded cortex_facts vs the agent ANSWER + its trajectory tool
calls. Decisive test: is a missing instance ABSENT from facts (extraction-recall
miss) or PRESENT but miscounted (gate miss)? Usage: python xray_facts.py <qid> [qid...]"""

import re
import subprocess
import sys

import httpx
from datasets import load_dataset

SECRET = open(".env").read().split("UNISON_EVAL_SECRET=")[1].split("\n")[0].strip().strip('"')
PSQL = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
QIDS = sys.argv[1:]


def s2t(s):
    if isinstance(s, list):
        return "\n".join(
            (f"{m.get('role')}: {m.get('content')}" if isinstance(m, dict) else str(m)) for m in s
        )
    return str(s)


ds = load_dataset("xiaowu0162/longmemeval-cleaned", split="longmemeval_s_cleaned", streaming=True)
wanted = {q: None for q in QIDS}
for r in ds:
    qid = str(r.get("question_id"))
    if qid in wanted and wanted[qid] is None:
        wanted[qid] = r
    if all(v is not None for v in wanted.values()):
        break

c = httpx.Client(base_url="http://localhost:3022", headers={"x-unison-eval": SECRET}, timeout=400)

for qid in QIDS:
    row = wanted[qid]
    if row is None:
        print(f"!! {qid} not found"); continue
    seed = [
        {"path": f"/private/sources/eval/xr/sess{i}.md", "body": s2t(s), "kind": "raw"}
        for i, s in enumerate(row.get("haystack_sessions") or [])
    ]
    q = f"Today's date is {row.get('question_date')}.\n\n{row['question']}"
    tid = c.post("/api/rest/agents/eval/provision", json={"label": "xr"}).json()["tenantId"]
    resp = c.post(
        "/api/rest/agents/eval-turn",
        json={"question": q, "seedDocs": seed, "tenantId": tid, "memoryMode": "fresh"},
    ).json()
    print("=" * 90)
    print(f"QID {qid}  | sessions={len(seed)}  tenant={tid}")
    print("Q   :", row["question"])
    print("GOLD:", row["answer"])
    print("ANS :", (resp.get("answer") or "")[:600].replace("\n", " "))
    facts = subprocess.run(
        ["psql", PSQL, "-tAF", "\t", "-c",
         f"select to_char(valid_from,'YYYY-MM-DD'), fact_text from cortex_facts "
         f"where tenant_id='{tid}' order by valid_from"],
        capture_output=True, text=True,
    ).stdout.strip()
    flines = [l for l in facts.split("\n") if "\t" in l]
    print(f"--- SEEDED FACTS ({len(flines)}) ---")
    for l in flines:
        d, t = l.split("\t", 1)
        print(f"  [{d}] {t[:140]}")
    # trajectory: did the agent re-read raw sessions (cat sessN) or only trust facts?
    traj = subprocess.run(
        ["psql", PSQL, "-tAF", "\t", "-c",
         f"select left(content::text,4000) from agent_messages where tenant_id='{tid}' and role='assistant' order by created_at"],
        capture_output=True, text=True,
    ).stdout
    cmds = re.findall(r'"(?:command|input)"\s*:\s*"([^"]{1,120})"', traj)
    print(f"--- AGENT TOOL CALLS ({len(cmds)}) ---")
    for cmd in cmds[:25]:
        print(f"  $ {cmd}")
