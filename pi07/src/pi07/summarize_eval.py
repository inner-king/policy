#!/usr/bin/env python3
"""π0.7 평가 실행 요약 — runs/eval/<id>/json/*.json 과 trace/conn*/ep*.jsonl 을 읽는다.

    python3 src/pi07/summarize_eval.py runs/eval/pi07ll-...-ll-hl runs/eval/pi07ll-...-ll

인스턴스별: q_score, 성공, 스텝, 손 이동 거리. HL 조건이면 trace 에서 낸 subtask 순서(연속 중복 제거)와
전환 횟수. 표본이 작아(조건당 10) 평균 차이를 결론으로 쓰지 말고 인스턴스별로 본다.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def load_results(d: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(d, "json", "*.json"))):
        r = json.load(open(f))
        out.append(dict(instance=r["instance_id"], q=r["q_score"]["final"], success=bool(r["success"]),
                        steps=r["steps"], base=r["agent_distance"]["base"], left=r["agent_distance"]["left"],
                        right=r["agent_distance"]["right"]))
    return out


def load_traces(d: str) -> list[list[str]]:
    seqs = []
    for f in sorted(glob.glob(os.path.join(d, "trace", "conn*", "ep*.jsonl"))):
        seq = []
        for line in open(f):
            r = json.loads(line)
            if "hl" in r and (not seq or seq[-1] != r["hl"]):
                seq.append(r["hl"])
        if seq:
            seqs.append(seq)
    return seqs


def main():
    for d in sys.argv[1:]:
        rs = load_results(d)
        print(f"== {os.path.basename(d.rstrip('/'))}  (n={len(rs)})")
        for r in sorted(rs, key=lambda z: z["instance"]):
            print(f"  inst {r['instance']:4d}  q={r['q']:.3f}  success={int(r['success'])}  steps={r['steps']:6d}  "
                  f"base={r['base']:6.1f}m  hands={r['left']:6.1f}/{r['right']:6.1f}m")
        if rs:
            qs = [r["q"] for r in rs]
            print(f"  mean q={statistics.mean(qs):.3f}  success {sum(r['success'] for r in rs)}/{len(rs)}  "
                  f"q>0 {sum(q > 0 for q in qs)}/{len(rs)}")
        seqs = load_traces(d)
        if seqs:
            print(f"  HL subtask 순서 (에피소드 {len(seqs)}개, 연속 중복 제거):")
            for s in seqs:
                print(f"    [{len(s) - 1} 전환] " + " → ".join(s[:12]) + (" …" if len(s) > 12 else ""))


if __name__ == "__main__":
    main()
