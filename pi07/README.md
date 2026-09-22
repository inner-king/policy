# pi07 — π0.7 재현

π0.7([arXiv:2604.15483](https://arxiv.org/abs/2604.15483))의 학습 방법을 24 GB GPU 2장 서버에 맞춰 옮기고,
BEHAVIOR-1K 2026 챌린지 데모 18개 태스크로 학습·평가한 실험의 코드와 설정이다.

- 모델·데이터·하이퍼파라미터·평가 프로토콜: **[SETTINGS.md](SETTINGS.md)**
- 결과와 해석: [이슈](https://github.com/inner-king/policy/issues)

| 모듈 | 하는 일 | 구현 |
|---|---|---|
| HL (고수준 정책) | 관측 3뷰 + 태스크 문장 + 지시 이력 → 지금 유효한 subtask 문장 | π0.5 base 의 PaliGemma + LoRA 39.2M |
| LL (저수준 VLA) | 관측 3뷰 + π0.7 프롬프트(Task/Subtask/Speed/Quality/Mistake/Control Mode) (+ 서브골 3뷰) → 50 스텝 액션 청크 | π0.5 base, VLM 동결, action expert 430M 학습 |
| WM (서브골 월드 모델) | 현재 3뷰 + subtask → 현재 세그먼트 끝의 3뷰 | FLUX.2 [klein] base 4B + LoRA 96.3M |

---

## 구성

이 디렉토리가 코드의 루트(`CAPSTONE`)다. 셸 스크립트는 `src/env.sh` 를, 파이썬은 `__file__` 을 기준으로
경로를 정하므로 `pi07/` 을 어디에 두어도 된다. `models/` `data/` `runs/` 의 산출물은 실행하면서 생기고 커밋하지 않는다.

```
pi07/
├── src/
│   ├── env.sh              경로 단일 출처. 셸 스크립트가 첫 줄에서 source 한다
│   ├── pi07/
│   │   ├── build_index.py      데모 → data/pi07/{index.json, lowdim.npy, stats.json}
│   │   ├── subtask_text.py     B1K 스킬 주석 → 영어 subtask 문장
│   │   ├── dataset.py frames.py   HL·LL·WM 데이터셋, 영상 디코딩
│   │   ├── models.py           LL·HL (π0.5 base 위)
│   │   ├── world_model.py      WM (FLUX.2 klein + LoRA), `cache-text` 로 텍스트 캐시 생성
│   │   ├── train.py            HL·LL·WM 학습, 자동 재개, GPU 2장 데이터 병렬
│   │   ├── launch.sh           학습을 systemd 사용자 유닛으로 띄운다 (세션과 무관하게 돈다)
│   │   ├── chain.sh stage3.sh  앞 실행이 끝나면 다음 실행을 띄운다 / 서브골 생성 → genmix
│   │   ├── gen_subgoals.py     WM 으로 서브골 이미지 생성
│   │   ├── serve_pi07.py       정책 서버 (websocket, HL+LL[+WM] 을 한 GPU 에)
│   │   ├── run_eval.sh         서버 + 시뮬레이터 1회 평가
│   │   ├── run_eval_client.sh  떠 있는 서버에 붙는 로컬 시뮬레이터 클라이언트
│   │   ├── run_remote.sh eval_remote.py   다른 머신의 시뮬레이터 클라이언트 (관측 224 축소 + JPEG)
│   │   └── status.sh serve_autostop.sh summarize_eval.py   상태 보기, 서버 자동 정지, 평가 요약
│   ├── common/
│   │   ├── b1k_wrappers.py     평가 환경 래퍼 (원본 해상도 RGB, 적용 액션 기록)
│   │   └── fetch_cooking_demos.sh verify_b1k_mirror.py make_task_views.py   데모 다운로드·검증·뷰 생성
│   ├── vla/serve_smolvla.py    websocket 와이어 포맷 (serve_pi07.py 가 import)
│   └── analysis/               개루프 진단(probe_pi07_ll.py), 롤아웃 액션 분석, 그림 스크립트
├── runs/{train,eval}/<run-id>/run.yaml   실행별 설정 기록 (결과 필드는 뺐다)
└── env/requirements-vla.txt   학습·서빙 환경 패키지 목록 (pip freeze)
```

실행 기록은 run-id 로 찾는다. 형식은 `<모델>-<초기값>-<총스텝>[-at<체크포인트>][-<변형>]` 이다.

| run-id | 내용 |
|---|---|
| `pi07hl-pi05base-8k` | HL 8,000 스텝 |
| `pi07ll-pi05base-30k` | LL 30,000 스텝 — 평가에 쓴 LL |
| `pi07wm-flux2kleinbase-6k` | WM 6,000 스텝 |
| `pi07ll-pi05base-30k-genmix-5k` | LL 30k 에서 이어 생성 서브골을 섞어 5,000 스텝 |
| `pi07ll-pi05base-200k` | LL 200,000 스텝, GPU 2장 (2026-09-22 기준 학습 중) |
| `pi07ll-pi05base-30k-at30k-<task>-ll[-hl]` | 폐루프 평가. `-ll` = LL 단독(태스크 문장만), `-ll-hl` = LL + HL |

---

## 환경

### 하드웨어

| | 학습·로컬 평가 | 원격 평가 |
|---|---|---|
| GPU | RTX 3090 Ti 24 GB (GPU0) + RTX 3090 24 GB (GPU1) | RTX 3060 12 GB |
| RAM | 48 GB + 스왑 64 GB | 31 GB |
| 드라이버 | 580.178.04 | — |

NVIDIA 드라이버 595.84 에서는 Isaac Sim 5.1.0 이 뜨지 않는다. 580 계열을 쓴다.

### 학습·서빙 환경

uv 로 만든 Python 3.12 venv 하나(`vla/.venv`)에 다음을 설치했다. 전체 목록은 [env/requirements-vla.txt](env/requirements-vla.txt) 에 있고,
설치 순서는 그 파일의 머리 주석에 있다.

| 패키지 | 버전 |
|---|---|
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 (PyTorch cu128 인덱스) |
| lerobot | 0.5.2 — `git+https://github.com/wensi-ai/lerobot@436812bd8ee39b768c645c248c91f1330834e687` (release/b1k) |
| transformers | 5.5.4 |
| diffusers | 0.40.0 — WM 용. 다른 패키지를 바꾸지 않도록 `--no-deps` 로 추가했다 |
| av, pyarrow, msgpack, websockets | 15.1.0, 25.0.1, 1.2.1, 17.0.1 |

같은 환경을 다시 만들 때는 목록을 그대로 맞춘다 (`uv pip sync` 는 목록에 없는 패키지를 설치하지 않는다).
이 명령으로 새 머신에 설치해 보지는 않았다.

```bash
cd pi07
uv venv vla/.venv --python 3.12
uv pip sync --python vla/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  env/requirements-vla.txt
```

그림 스크립트(`src/analysis/plot_*.py`)는 matplotlib 과 `fonts-noto-cjk` 가 필요하다. `vla/.venv` 에는 matplotlib 이 없다.

### 평가 환경

BEHAVIOR-1K 챌린지 하네스(`python -m omnigibson.eval.eval`)를 쓴다.

| | 로컬 | 원격 |
|---|---|---|
| OmniGibson | 3.9.1 (BEHAVIOR-1K 태그 `v3.9.1`, uv venv, Python 3.11) | 3.9.2 (이 저장소의 `setup/install_behavior.sh` 로 만든 conda env `behavior`) |
| Isaac Sim | 5.1.0 | 5.1.0 |

`src/env.sh` 의 기본값은 원래 머신 배치(`pi07/omnigibson/.venv`, `pi07/behavior`)다. 이 저장소의 설치 방식을 따랐다면 환경변수로 덮어쓴다.

```bash
export OG=~/BEHAVIOR-1K                                  # 하네스를 실행할 디렉토리
export OG_PY=~/miniforge3/envs/behavior/bin/python       # 하네스 인터프리터
export BEHAVIOR=~/BEHAVIOR-1K/datasets                   # OMNIGIBSON_DATA_PATH
```

### 모델 가중치

`pi07/models/` 아래에 받는다. 저장소에는 넣지 않는다.

| 디렉토리 | HF 저장소 | 커밋 | 조건 |
|---|---|---|---|
| `models/pi05_base` | `lerobot/pi05_base` | `b211f3d4` | Gemma 약관 (PaliGemma 파생) |
| `models/paligemma_tokenizer` | `google/paligemma-3b-pt-224` (토크나이저 파일만) | `35e4f464` | HF 게이트, Gemma 약관 동의 필요 |
| `models/flux2_klein_base_4b` | `black-forest-labs/FLUX.2-klein-base-4B` | `a3b4f484` | Apache-2.0 |

```bash
vla/.venv/bin/hf auth login      # paligemma 는 HF 에서 게이트 승인을 받은 계정으로
vla/.venv/bin/hf download lerobot/pi05_base --local-dir models/pi05_base
vla/.venv/bin/hf download google/paligemma-3b-pt-224 tokenizer.json tokenizer.model tokenizer_config.json \
  special_tokens_map.json added_tokens.json --local-dir models/paligemma_tokenizer
vla/.venv/bin/hf download black-forest-labs/FLUX.2-klein-base-4B --local-dir models/flux2_klein_base_4b
```

### 데이터

HF 데이터셋 [`behavior-1k/2026-challenge-demos`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)(LeRobot v3)에서
요리 17개 태스크와 `clean_a_keyboard`(task 93)를 쓴다. 태스크당 200 에피소드, 합계 3,600 에피소드다. 태스크 목록은 [SETTINGS.md §5](SETTINGS.md#5-데이터).

1. **데모 다운로드** — `bash src/common/fetch_cooking_demos.sh` 가 요리 17개 태스크를 `$B1K_HDD/2026-challenge-demos` 로 받는다
   (RGB·depth·parquet 약 470 GB). `verify_b1k_mirror.py` 로 HF 파일 목록과 크기를 대조한다.
2. **태스크 뷰** — `src/common/make_task_views.py` 가 `data/b1k/<task>/` 심링크 뷰와 `data/b1k/manifest.json`(요리 17개)을 만든다.
3. **clean_a_keyboard (task 93)** — 요리 17개와 따로 받고 뷰도 손으로 만들었다. `build_index.py` 는 manifest 와 별개로 task 93 을 항상 붙이고,
   주석은 `behavior/task-0093-clean_a_keyboard/annotations/episode_0093XXXX.json` 에서 읽는다(고정 경로).
   요리 미러(`$B1K_HDD/2026-challenge-demos`)에 chunk-093 을 받으면 `make_task_views.py` 가 manifest 에 넣어 에피소드가 두 번 들어가므로 다른 곳에 받는다.
   원래 머신의 주석 폴더를 어디서 받았는지는 기록이 없다. 아래는 HF 의 `annotations/task-0093/` 을 쓰는 방법이고(파일명 규칙은 같다), 이 순서로 돌려 보지는 않았다.

   ```bash
   source src/env.sh     # B1K_HDD 등 경로 변수
   K=$B1K_HDD/clean_a_keyboard/2026-challenge-demos
   vla/.venv/bin/hf download behavior-1k/2026-challenge-demos --repo-type dataset --local-dir $K \
     --include "meta/*" --include "data/chunk-093/*" --include "videos/*/chunk-093/*" --include "annotations/task-0093/*"
   mkdir -p data/b1k/clean_a_keyboard/{data,meta/episodes} behavior/task-0093-clean_a_keyboard
   ln -s $K/data/chunk-093 data/b1k/clean_a_keyboard/data/
   ln -s $K/meta/episodes/chunk-093 data/b1k/clean_a_keyboard/meta/episodes/
   cp $K/meta/info.json data/b1k/clean_a_keyboard/meta/
   for c in zed_link left_realsense_link right_realsense_link; do
     v=observation.rgb.${c}_camera_0; mkdir -p data/b1k/clean_a_keyboard/videos/$v
     ln -s $K/videos/$v/chunk-093 data/b1k/clean_a_keyboard/videos/$v/
   done
   ln -s $K/annotations/task-0093 behavior/task-0093-clean_a_keyboard/annotations
   ```
4. **학습 인덱스** — `vla/.venv/bin/python src/pi07/build_index.py` (약 2분). `data/pi07/` 에 인덱스·저차원 배열(memmap 7.3 GB)·정규화 통계가 생긴다.
   `index.json` 에 절대경로가 들어가므로 디렉토리를 옮기면 다시 만든다. 정책 서버 머신에도 `data/pi07/` 이 있어야 한다.
5. **WM 텍스트 캐시** — `vla/.venv/bin/python src/pi07/world_model.py cache-text` (GPU 8 GB, 1~2분). WM 학습·서브골 생성·WM 서빙 전에 한 번.

BEHAVIOR 에셋과 `omnigibson.key` 는 이 저장소 루트의 [README](../README.md) 대로 설치한다(재배포 금지).

---

## 실행

명령은 모두 `pi07/` 에서 실행한다. 학습은 `launch.sh` 가 systemd 사용자 유닛으로 띄운다.
유닛이라 세션이 끊겨도 돌고, 크래시 뒤 2분 후 자동 재개하며, `systemctl --user stop <run-id>` 로 멈추면 체크포인트를 쓰고 끝난다.
같은 명령을 다시 실행하면 최신 체크포인트에서 이어 돈다. 로그인하지 않은 상태에서도 돌게 하려면 `loginctl enable-linger $USER` 가 필요하다.
systemd 가 없으면 `launch.sh` 가 조립하는 `train.py` 명령을 직접 실행해도 된다.

### 학습

아래 인자는 각 실행의 `runs/train/<id>/run.yaml` 에 기록된 그대로다. 두 번째 인자가 GPU 번호다.

```bash
# HL — GPU1
bash src/pi07/launch.sh pi07hl-pi05base-8k 1 hl 8000 --micro-batch 16 --accum 2 --warmup 300 --workers 4 \
  --save-every 500 --val-every 1000 --milestone 2000

# LL 30k — GPU0
bash src/pi07/launch.sh pi07ll-pi05base-30k 0 ll 30000 --micro-batch 16 --accum 2 --goal-per-micro 4 --warmup 1000 \
  --workers 5 --save-every 1000 --val-every 2000 --milestone 10000

# WM — HL 이 끝나면 같은 GPU1 에서 시작 (chain.sh 가 앞 실행의 COMPLETED 를 기다린다)
systemd-run --user --unit=pi07-chain-wm --working-directory=$PWD bash src/pi07/chain.sh \
  pi07hl-pi05base-8k pi07wm-flux2kleinbase-6k 1 wm 6000 --micro-batch 8 --accum 2 --warmup 300 --workers 3 \
  --save-every 500 --val-every 1000 --milestone 2000

# 서브골 4,000장 생성(GPU1) → LL 30k 에서 genmix 5k(GPU0). 단계별 대기·재개는 stage3.sh 가 한다
systemd-run --user --unit=pi07-stage3 --working-directory=$PWD -p Restart=on-failure -p RestartSec=300 \
  -p MemoryMax=16G -p MemorySwapMax=4G bash src/pi07/stage3.sh

# LL 200k — GPU 2장 데이터 병렬 (GPU 인자에 쉼표). 유효 배치는 30k 와 같은 32
PI07_MEM_MAX=26G bash src/pi07/launch.sh pi07ll-pi05base-200k 0,1 ll 200000 --micro-batch 16 --accum 1 \
  --goal-per-micro 4 --warmup 1000 --workers 4 --save-every 2000 --val-every 2000 --milestone 10000

bash src/pi07/status.sh      # 유닛 상태, 마지막 로그, 검증 결과, 체크포인트, GPU, 메모리
```

체크포인트는 `runs/train/<id>/checkpoints/step-*/` 에 저장된다. 최근 2개와 milestone 배수만 남고, milestone 은 옵티마이저 상태를 지운다.

### 서빙과 폐루프 평가

기록된 평가는 정책 서버 하나를 띄워 두고 포트로 조건을 나눴다. 8010 은 HL 켬, 8011 은 HL 끔(태스크 문장만)이다.

```bash
# 정책 서버 (HL+LL 이 PaliGemma 하나를 나눠 쓴다. --wm 을 주면 WM 까지 한 GPU 에 최대 19 GB)
CUDA_VISIBLE_DEVICES=0 vla/.venv/bin/python src/pi07/serve_pi07.py \
  --ll runs/train/pi07ll-pi05base-30k/checkpoints/step-030000 \
  --hl runs/train/pi07hl-pi05base-8k/checkpoints/step-008000 \
  --port 8010 --trace-dir runs/eval/pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl/trace \
  --flat-port 8011 --flat-trace-dir runs/eval/pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll/trace

# 같은 머신의 시뮬레이터 (public_test 인덱스 0-4, 두 조건을 순서대로)
TASK=thawing_frozen_food HOST=127.0.0.1 INSTANCES="0 1 2 3 4" SIM_GPU=0 \
  CONDS="8010:pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll-hl 8011:pi07ll-pi05base-30k-at30k-thawing_frozen_food-ll" \
  bash src/pi07/run_eval_client.sh

# 다른 머신의 시뮬레이터 (인덱스 5-9). $E 에 eval_remote.py 와 b1k_wrappers.py 를 복사해 둔다
HOST=<서버 주소> bash src/pi07/run_remote.sh

# 서버와 시뮬레이터를 한 번에 (GPU 두 장이 비어 있을 때). 결과는 runs/eval/<LL run>-at30k-<task>-ll-hl[-TAG]/ 에 쓰고
# 그 run.yaml 을 덮어쓰므로, 기록된 실행과 겹치지 않게 TAG 를 준다
TASK=thawing_frozen_food TAG=rerun LL=runs/train/pi07ll-pi05base-30k/checkpoints/step-030000 \
  HL=runs/train/pi07hl-pi05base-8k/checkpoints/step-008000 bash src/pi07/run_eval.sh
```

서버는 인증 없이 `--host`(기본 0.0.0.0)에 열린다. 신뢰할 수 있는 네트워크에서만 띄운다.
평가 결과는 `runs/eval/<id>/` 에 `json/`(점수), `videos/`, `trace/`(서버 기록 — LL 프롬프트, HL 켠 조건은 HL 결정), `actions-*/`(적용된 액션)로 남는다.
`python3 src/pi07/summarize_eval.py runs/eval/<id>` 로 요약한다.

### 분석

```bash
vla/.venv/bin/python src/analysis/probe_pi07_ll.py --ckpt runs/train/pi07ll-pi05base-30k/checkpoints/step-030000   # GPU 1장
vla/.venv/bin/python src/analysis/probe_pi07_rollout.py runs/eval/<id>   # 롤아웃 액션 로그 vs 데모 액션 (CPU, dataset.py 때문에 torch 필요)
```

`probe_pi07_ll.py` 는 결과를 `runs/train/<id>/analysis/ll_probe_<step>.json` 으로 쓰는데, `plot_pi07_ll.py` 는 `ll_probe.json` 을 읽는다.
그림을 다시 그릴 때는 해당 파일을 `ll_probe.json` 으로 복사해 둔다. 그림 스크립트는 matplotlib 이 있는 파이썬으로 돌린다
(`plot_pi07_ll.py` 독스트링의 `vla/.venv` 는 틀렸다 — matplotlib 이 없다).

---

## 코드 버전과 결과의 대응

이 디렉토리의 코드는 2026-09-20 수정본이고, `pi07ll-pi05base-200k` 가 이 코드로 돈다. 나머지 학습 4개와 평가는 그 전 코드로 돌았다.
학습 표본 생성, 손실, 옵티마이저, 서빙의 관측→액션 경로는 바뀌지 않았다. 바뀐 것은 아래와 같다.

| 변경 | 이전 결과에 주는 영향 |
|---|---|
| `train.py` LL 검증: 이전에는 배치마다 같은 키를 덮어써서 스트림별 **마지막 16표본** 값만 남았다. 지금은 표본 수 가중 평균, 고정 샘플링 잡음(시드 4321), 관절군별·앞 5/25 스텝 L1, "직전 액션 반복" 기준선(`*_rep_l1`)을 기록한다 | 30k·genmix 의 `val_*` 는 16표본 값이라 200k 의 `val_*` 와 직접 비교하지 않는다 |
| `train.py`·`dataset.py` 서브골 유/무 검증: 서브골 있는 32표본을 없는 쪽 앞 32표본과 같은 (에피소드, t) 로 짝지었다(`val_goal_minus_nogoal_l1`) | 이전 기록의 서브골 유/무 비교는 다른 표본끼리의 비교다 |
| `models.py` `sample_actions` 에 `noise=`·`generator=` 인자 | 서빙은 인자 없이 불러 동작이 같다 |
| `train.py`·`launch.sh` GPU 2장 데이터 병렬 | GPU 1장이면 이 분기가 꺼진다 |
| `serve_pi07.py` proprio 길이 검사, 모르는 task_id 경고, HL 예외 시 LoRA 끄기, trace 에 state·추론 시간 기록 | 이전 평가에서는 발동하지 않았다. 이전 trace 에는 state 가 없다 |
| `b1k_wrappers.py` 마지막 에피소드 액션 로그를 끝까지 기록, `instance_id` 기록 | 이전 평가의 `actions-*/` 는 스트림마다 마지막 에피소드가 500 스텝 단위로 잘렸고 `instance_id` 가 전부 0 이다 |
| `run_eval_client.sh` run.yaml 자동 작성 | 없음 |

공개하면서 바꾼 것 (기본값에서의 동작은 원래와 같다):

- `src/env.sh`: 환경 루트(`GROOT` `OG` `VLA_VENV` `BEHAVIOR` `B1K_HDD`)를 환경변수로 덮어쓸 수 있게 했고, 하네스 인터프리터 `OG_PY` 를 추가했다.
- `run_eval.sh`·`run_eval_client.sh`: 인터프리터를 `$OG_PY` 로 바꿨다. `run_eval_client.sh` 는 `HOST` 를 필수로 했고(사설 IP 기본값 제거), 추적되는 run.yaml 에 서버 주소를 쓰지 않는다.
- `make_task_views.py`·`verify_b1k_mirror.py` 의 미러 경로를 `$B1K_HDD` 기준으로, `fetch_cooking_demos.sh` 의 네트워크 인터페이스를 `IFACE` 로 덮어쓸 수 있게 했다.
- `run_remote.sh`: 원격 머신에 있던 스크립트를 옮기면서 `HOST` 를 필수로, 인터프리터·데이터 경로를 `$HOME` 기준 기본값으로, `EXP_PATH`·`OMNIGIBSON_APPDATA_PATH` 를 선택 사항으로 바꿨다. 기록된 실행에서 준 값은 머리 주석에 있다.
- `launch.sh`: 주석의 계정 이름.

그 밖의 코드는 원래 파일과 같다.

---

## 라이선스 주의

- 가중치: `lerobot/pi05_base`·PaliGemma 토크나이저는 Gemma 약관을, FLUX.2 klein base 4B 는 Apache-2.0 을 따른다. 학습한 체크포인트는 이 저장소에 넣지 않는다.
- 데이터: `behavior-1k/2026-challenge-demos` 는 HF 에서 받는다. BEHAVIOR 에셋(BEHAVIOR Data Bundle)은 비상업적 학술 연구 용도로만 쓸 수 있고 재배포할 수 없다.
- 평가 스크립트는 `OMNI_KIT_ACCEPT_EULA=YES` 를 설정한다. 실행하면 NVIDIA Isaac Sim EULA 에 동의하는 것이다.
