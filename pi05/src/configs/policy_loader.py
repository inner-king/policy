"""8GB GPU 에서도 학습된 정책을 올릴 수 있게 하는 로더.

openpi 의 `create_trained_policy` 는 `restore_params(..., restore_type=jax.Array)`
를 쓴다. orbax 가 읽는 동안 중간 버퍼까지 **GPU 위에** 잡기 때문에, 최종
가중치가 6.81GB 로 카드에 들어가는데도 읽는 도중 피크가 넘쳐서 죽는다:

    RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2415919104 bytes
    (= 가장 큰 텐서 gating_einsum, 18x2x2048x16384 bfloat16)

여기서는 **먼저 호스트 RAM 으로 전부 읽은 뒤**(numpy), 텐서를 하나씩 GPU 로
옮긴다. 그러면 GPU 피크가 최종 크기(6.81GB)를 넘지 않는다.

RAM 은 그만큼 더 쓰지만 이 노트북은 54GB 라 여유가 있다.
"""
import pathlib

import jax
import jax.numpy as jnp
import numpy as np


def load_policy(cfg, ckpt_dir: pathlib.Path):
    """학습된 체크포인트로 추론용 Policy 를 만든다 (GPU 메모리 절약판)."""
    from openpi.models import model as _model
    from openpi.policies import policy as _policy
    from openpi.training import checkpoints as _checkpoints
    from openpi import transforms as _tf

    ckpt_dir = pathlib.Path(ckpt_dir)
    # 1) 호스트 RAM 으로 읽는다 (GPU 를 건드리지 않는다)
    params = _model.restore_params(ckpt_dir / "params",
                                   restore_type=np.ndarray, dtype=jnp.bfloat16)
    # 2) 텐서를 하나씩 옮긴다 -> GPU 피크 = 최종 크기
    params = jax.tree.map(jnp.asarray, params)
    model = cfg.model.load(params)

    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    # 정규화 통계는 반드시 **체크포인트 안의 것**을 쓴다. 학습 때 쓴 것과
    # 다르면 에러 없이 엉뚱한 크기의 액션이 나온다.
    norm_stats = _checkpoints.load_norm_stats(ckpt_dir / "assets", dc.asset_id)

    return _policy.Policy(
        model,
        transforms=[*dc.data_transforms.inputs,
                    _tf.Normalize(norm_stats, use_quantiles=dc.use_quantile_norm),
                    *dc.model_transforms.inputs],
        output_transforms=[*dc.model_transforms.outputs,
                           _tf.Unnormalize(norm_stats, use_quantiles=dc.use_quantile_norm),
                           *dc.data_transforms.outputs],
    ), dc
