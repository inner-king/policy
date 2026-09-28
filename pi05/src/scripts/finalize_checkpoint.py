"""학습이 끝난 **직후** 마지막 체크포인트가 Drive 에 온전히 올라갔는지 확정한다.

    python scripts/finalize_checkpoint.py <run 폴더>
    python scripts/finalize_checkpoint.py <run 폴더> --repair     # 망가진 것 되살리기 시도

## 🔴 왜 필요한가 — 8차를 여기서 잃었다 (HANDOFF 39.12)

학습 로그의 `Finished saving checkpoint` 와 `No errors found` 는 **Drive 에 온전히
올라갔다는 뜻이 아니다.** 학습 프로그램은 `/content/drive` 에 쓴 것까지만 보고,
실제 업로드는 Drive 연결 프로그램이 뒤에서 따로 한다. 런타임이 반납되면 그 틈에
올라가다 만 파일은 되살릴 수 없다.

2026-09-15 `run8/20999` 가 정확히 그랬다. 21시간을 학습해 놓고 목차 파일
(`params/d/d572eb...`, 2.5 KB) 하나가 빠져서 **마지막 체크포인트를 못 열었다.**
쓸 수 있었던 것은 그 전의 18000 이었다.

## 무엇이 다른가 — `verify_checkpoint.py` 와의 관계

`verify_checkpoint.py` 는 **폴더 하나**를 검사한다. 이 스크립트는

1. run 폴더의 **스텝 전부**를 새것부터 검사하고,
2. **쓸 수 있는 가장 최근 스텝**을 딱 집어 알려 주고,
3. `--repair` 면 망가진 것을 되살려 본다.

사람이 없어도 되도록 만든 것이다 — 학습 셀 끝에서 자동으로 부른다. 8차 때는
검사 셀이 따로 있었지만 **세션이 먼저 끝나서 아무도 못 눌렀다.**

## `--repair` 가 하는 일 (HANDOFF 39.12-4)

빠진 것은 보통 `params/d/` 안의 작은 목차 파일이다. Drive 는 "만들자마자 이름
바꾸기" 를 FUSE 위에서 놓치는 일이 있다. 그래서 **원래 폴더를 건드리지 않고**
`<step>/params/` 를 `<step>_repair/params/` 로 복사해 업로드를 새로 일으킨다.
복사가 끝나면 다시 검사한다.

⚠️ **이 절차는 아직 실제 사고에서 성공한 적이 없다.** 39.12 를 쓸 때 런타임이
없어 해 보지 못했다. 되살아나면 좋고, 안 되면 그 전 스텝을 쓰면 된다 — 어느
쪽이든 **이 스크립트가 "쓸 수 있는 스텝"을 알려 주는 것이 본래 값어치다.**

## 종료 코드

    0  마지막 스텝이 온전하다
    1  마지막은 깨졌지만 더 이전 스텝은 쓸 수 있다  (그 번호를 찍는다)
    2  쓸 수 있는 스텝이 하나도 없다
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERIFY = ROOT / "scripts" / "verify_checkpoint.py"


def step_dirs(run_dir: pathlib.Path):
    """`<run>/<숫자>` 폴더를 **큰 것부터**. 숫자가 아닌 폴더는 건너뛴다."""
    out = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            out.append((int(p.name), p))
    return [p for _, p in sorted(out, reverse=True)]


def verify(step_dir: pathlib.Path, read_params: bool) -> tuple[bool, str]:
    cmd = [sys.executable, str(VERIFY), str(step_dir)]
    if read_params:
        cmd.append("--read-params")
    r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return r.returncode == 0, (r.stdout + r.stderr)


def repair(step_dir: pathlib.Path) -> pathlib.Path | None:
    """`<step>` 을 `<step>_repair` 로 복사해 업로드를 새로 일으킨다.

    원래 폴더는 **건드리지 않는다** — 손대면 그나마 올라간 것까지 잃을 수 있다.
    `train_state/` 는 뺀다. 이어서 학습할 때만 쓰는 것이고 v3/v5 기준 10.9 GiB
    라 올리는 데만 16분이 더 든다. 평가에는 `params/` 만 있으면 된다.
    """
    dst = step_dir.parent / f"{step_dir.name}_repair"
    if dst.exists():
        print(f"  이미 있다: {dst}")
        return dst
    src = step_dir / "params"
    if not src.exists():
        print(f"  🔴 {src} 자체가 없다 — 되살릴 것이 없다")
        return None
    try:
        _gb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 2**30
        print(f"  복사 {src} -> {dst/'params'}  ({_gb:.1f} GiB)")
        print(f"  ⚠️ Drive 업로드 11.5 MB/s 기준 약 {_gb*1024/11.5/60:.0f}분 걸린다. 끝까지 기다릴 것.")
    except Exception:
        print(f"  복사 {src} -> {dst/'params'}   (업로드를 새로 일으킨다)")
    try:
        shutil.copytree(src, dst / "params")
    except Exception as e:
        print(f"  🔴 복사 실패: {type(e).__name__}: {e}")
        return None
    # assets 도 같이 — 정규화 통계가 여기 들어 있다
    for extra in ("assets",):
        s = step_dir / extra
        if s.exists():
            try:
                shutil.copytree(s, dst / extra)
            except Exception:
                pass
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--repair", action="store_true",
                    help="깨진 마지막 스텝을 되살려 본다 (새 폴더로 복사)")
    ap.add_argument("--max-check", type=int, default=3,
                    help="새것부터 몇 개까지 검사하나 (기본 3). 전부 보려면 크게")
    ap.add_argument("--no-read-params", action="store_true",
                    help="목차만 열고 본문은 안 읽는다 (빠르지만 덜 확실하다)")
    a = ap.parse_args()

    run_dir = pathlib.Path(a.run_dir)
    if not run_dir.is_dir():
        print(f"🔴 폴더가 없다: {run_dir}")
        return 2
    steps = step_dirs(run_dir)
    if not steps:
        print(f"🔴 스텝 폴더가 하나도 없다: {run_dir}")
        return 2

    print(f"{'=' * 70}\n{run_dir}\n스텝 {len(steps)}개: "
          f"{', '.join(p.name for p in steps[:8])}{' ...' if len(steps) > 8 else ''}\n{'=' * 70}")

    ok_steps, bad_steps = [], []
    for p in steps[:a.max_check]:
        ok, out = verify(p, not a.no_read_params)
        print(f"{'✅' if ok else '🔴'} {p.name}")
        if not ok:
            bad_steps.append(p)
            for line in out.strip().splitlines()[-4:]:
                print(f"     {line}")
        else:
            ok_steps.append(p)
            for line in out.strip().splitlines()[-2:]:
                print(f"     {line}")

    newest = steps[0]
    if ok_steps and ok_steps[0] is newest:
        print(f"\n✅ 마지막 체크포인트 {newest.name} 가 온전하다.")
        print("⚠️ 그래도 **런타임을 끄기 전에** 로컬에서 rclone 으로 받아 한 번 더")
        print("   검사할 것. 다시 마운트해도 가상머신 캐시가 보일 수 있어서, Drive")
        print("   서버에 실제로 있는지는 여기서 확정되지 않는다 (HANDOFF 39.12).")
        return 0

    print(f"\n🔴 마지막 체크포인트 {newest.name} 를 열 수 없다.")
    if a.repair and newest in bad_steps:
        print("\n── 되살리기 시도 ──")
        dst = repair(newest)
        if dst is not None:
            ok, out = verify(dst, not a.no_read_params)
            print(f"{'✅' if ok else '🔴'} {dst.name} 재검사")
            for line in out.strip().splitlines()[-4:]:
                print(f"     {line}")
            if ok:
                print(f"\n✅ 되살렸다 — **{dst}** 를 쓸 것 (params 만 들어 있다).")
                return 0

    if ok_steps:
        use = ok_steps[0]
        lost = int(newest.name) - int(use.name)
        print(f"\n➡️ **쓸 수 있는 가장 최근 스텝은 {use.name} 다.** "
              f"({lost:,} 스텝 = 약 {lost*3.0/3600:.1f}시간을 잃었다)")
        print("   평가·다음 차수 이어받기에 이 스텝을 쓸 것.")
        print("🔴 런타임을 끄지 말고 39.12 의 복구 절차를 먼저 시도할 것.")
        return 1

    print(f"\n🔴 검사한 {a.max_check}개가 전부 깨졌다. --max-check 를 늘려 더 뒤까지 볼 것.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
