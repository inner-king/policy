"""HDD 미러가 HF 저장소 파일 목록과 일치하는지 크기로 대조한다.

hf download 는 중간에 죽어도 받은 만큼은 남기므로 "디렉토리가 있다"는 게 완료의 증거가
못 된다. 실제로 어노테이션 3400개 중 1357개만 받힌 채 끝난 적이 있다 (httpx.ReadTimeout).

실행:  vla/.venv/bin/python src/common/verify_b1k_mirror.py
반환:  전부 일치하면 0, 하나라도 빠지면 1
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "behavior-1k/2026-challenge-demos"
MIRROR = Path(os.environ.get("B1K_HDD", "/data/b1k")) / "2026-challenge-demos"
CHUNKS = [4, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 62, 70, 72, 95, 96]


def wanted(rel: str) -> bool:
    if rel.startswith(("meta/", "README", "LICENSE")):
        return True
    for c in CHUNKS:
        if f"chunk-{c:03d}/" in rel or rel.startswith(f"annotations/task-{c:04d}/"):
            return True
    return rel.startswith("annotations/") and rel.endswith(".csv")


sizes = {
    f.rfilename: (f.size or 0)
    for f in HfApi().repo_info(REPO, repo_type="dataset", files_metadata=True).siblings
    if wanted(f.rfilename)
}

missing, wrong = [], []
for rel, size in sizes.items():
    p = MIRROR / rel
    if not p.exists():
        missing.append(rel)
    elif p.stat().st_size != size:
        wrong.append((rel, p.stat().st_size, size))

total = sum(sizes.values())
print(f"대상 {len(sizes)}개 파일 · {total / 1e9:.1f} GB")
print(f"  일치 {len(sizes) - len(missing) - len(wrong)} · 없음 {len(missing)} · 크기불일치 {len(wrong)}")
for rel in missing[:10]:
    print(f"    없음  {rel}")
for rel, got, want in wrong[:10]:
    print(f"    크기  {rel}  {got} != {want}")
if len(missing) > 10 or len(wrong) > 10:
    print("    ... (앞의 10개만 표시)")

sys.exit(1 if missing or wrong else 0)
