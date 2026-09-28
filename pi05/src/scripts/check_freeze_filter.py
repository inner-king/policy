"""v3 의 `freeze_filter` 가 실제로 무엇을 얼리는지 센다 (CU 0, GPU 불필요).

    JAX_PLATFORMS=cpu third_party/openpi/.venv/bin/python scripts/check_freeze_filter.py

`nnx.eval_shape` 로 모델을 **모양만** 만들어서 실제 가중치를 안 잡는다.
3.4B 모델이지만 메모리를 거의 안 쓰고 몇 초에 끝난다.

여기서 확인하는 것 (기대값은 docs/HANDOFF.md 15.6):

    v2: 학습 467M (SigLIP 414.8 + LoRA 50.0 + 선형 2.2)
    v3: 학습 480M (Action expert 427.9 + LoRA 50.0 + 선형 2.2), SigLIP 은 동결

**필터가 어긋나면 에러가 안 난다.** 엉뚱한 것을 학습하거나 아무것도 학습하지
않는 채로 CU 를 태우게 되므로, 학습 전에 반드시 이 숫자를 눈으로 볼 것.
"""
import pathlib
import re
import sys

import flax.nnx as nnx
import jax

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))

import cooking_config

from openpi.training import config as _config


def group(path: str) -> str:
    if re.fullmatch(r"PaliGemma/img/.*", path):
        return "SigLIP 시각 인코더"
    if "lora" in path:
        return "LoRA (전문가 1)" if re.fullmatch(r".*llm.*_1.*", path) else "LoRA (전문가 0)"
    if "llm" not in path:
        return "선형 변환 (state/action proj 등)"
    return "Action expert 본체" if re.fullmatch(r".*llm.*_1.*", path) else "Gemma 2B 본체"


def report(name: str) -> None:
    cfg = _config.get_config(name)
    model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
    params = nnx.state(model, nnx.Param)

    def sizes(state):
        out = {}
        for path, leaf in dict(state.flat_state()).items():   # flax 0.10.2 API
            p = "/".join(str(x) for x in path)
            v = getattr(leaf, "value", leaf)
            out[group(p)] = out.get(group(p), 0) + int(v.size)
        return out

    tr = sizes(params.filter(cfg.trainable_filter))
    fz = sizes(params.filter(cfg.freeze_filter)) if cfg.freeze_filter is not nnx.Nothing else {}
    total_tr, total_fz = sum(tr.values()), sum(fz.values())

    print(f"\n{'=' * 68}\n{name}\n{'=' * 68}")
    print(f"{'부분':<34}{'학습':>13}{'동결':>13}")
    for g in sorted(set(tr) | set(fz)):
        print(f"{g:<34}{tr.get(g,0)/1e6:>11.1f}M{fz.get(g,0)/1e6:>12.1f}M")
    print(f"{'-'*60}")
    print(f"{'합계':<34}{total_tr/1e6:>11.1f}M{total_fz/1e6:>12.1f}M"
          f"   (전체 {(total_tr+total_fz)/1e6:.1f}M, 학습 비중 {100*total_tr/(total_tr+total_fz):.1f}%)")
    # 동결 bf16 2바이트 / 학습 fp32 파라미터4 + Adam m,v 8 + 기울기 4
    mem = (total_fz * 2 + total_tr * 16) / 2**30
    print(f"{'상태+기울기 (계산값)':<34}{mem:>11.1f} GiB")
    # ⚠️ EMA 는 **동결분까지 포함한 전체 사본 하나**를 더 든다
    # (`train.py:112` 의 `ema_params=None if ... else params`). 학습 대상만이 아니다.
    # 이걸 빼고 여유를 재면 스모크에서 배치 상한을 잘못 잡는다.
    if cfg.ema_decay is not None:
        ema = (total_fz * 2 + total_tr * 4) / 2**30
        print(f"{f'  + EMA 사본 (ema_decay={cfg.ema_decay})':<34}{ema:>11.1f} GiB")
        print(f"{'  = 합계':<34}{mem + ema:>11.1f} GiB")
        # 체크포인트도 커진다: EMA 가 있으면 checkpoints.py:145-152 가 train_state 에서
        # params 를 안 비워서 raw params 가 한 벌 더 저장된다.
        ck = (total_fz * 2 + total_tr * 4) * 2 / 2**30 + total_tr * 8 / 2**30
        print(f"{'체크포인트 1개 (계산값)':<34}{ck:>11.1f} GiB"
              f"   <- EMA 없으면 {(total_fz*2+total_tr*4+total_tr*8)/2**30:.1f} GiB")


def main() -> None:
    cooking_config.register()
    for name in ("pi05_cooking_top20", "pi05_cooking_top20_v3",
                 "pi05_cooking_top20_v4", "pi05_cooking_top32_v5"):
        report(name)


if __name__ == "__main__":
    main()
