"""Dump the FAILS of the newest (or given) result file, grouped by category, with
gold vs answer vs judge reasoning — the per-iteration analysis view for the prompt
loop. Usage: python fails.py [result.json]"""

import glob
import json
import sys
from collections import defaultdict

from datasets import load_dataset

rf = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("results/longmemeval-*.json"), key=lambda p: p)
d = json.load(open(rf))
ds = load_dataset("xiaowu0162/longmemeval-cleaned", split="longmemeval_s_cleaned", streaming=True)
meta = {str(r.get("question_id")): (r.get("question_type"), r.get("question"), r.get("answer")) for r in ds}

byc = defaultdict(lambda: [0, 0])
fails = defaultdict(list)
for r in d["results"]:
    qid = r["question_id"]
    m = meta.get(qid) or meta.get(qid.replace("gpt4_", ""))
    cat = (m[0] if m else "?") or "?"
    byc[cat][1] += 1
    if r["judge"].get("passed"):
        byc[cat][0] += 1
    else:
        fails[cat].append((qid, m, r["adapter"].get("answer", ""), r["judge"].get("reasoning", "")))

n = len(d["results"])
p = sum(v[0] for v in byc.values())
print(f"FILE {rf.split('/')[-1]}  | {p}/{n} = {100 * p / n:.1f}%\n")
for c, (pp, tt) in sorted(byc.items(), key=lambda x: -x[1][1]):
    print(f"  {c:26} {pp:2}/{tt:<2} {100 * pp / tt:5.1f}%")
print()
for c, items in fails.items():
    for qid, m, ans, jr in items:
        q = m[1] if m else ""
        gold = m[2] if m else ""
        print("=" * 80)
        print(f"[{c}] {qid}")
        print("Q   :", str(q)[:200])
        print("GOLD:", str(gold)[:160])
        print("ANS :", str(ans)[:340].replace("\n", " "))
        print("JUDGE:", str(jr)[:240])
