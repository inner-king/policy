# pi05 — π0.5 요리 스킬 정책 (스킬 문장 조건화)

π0.5 base(openpi, JAX)를 BEHAVIOR-1K 2026 챌린지 데모의 **스킬 구간**으로 파인튜닝해, 스킬 문장
(`open the door of the fridge` 같은)을 받아 그 스킬을 수행하는 **정책 하나**를 만드는 실험이다.
태스크당 문장 하나를 쓰는 공식 baseline 과 달리 같은 장면에서 문장으로 동작을 골라야 하므로,
매 차수 "정책이 문장을 읽는가"를 오프라인 검사와 시뮬레이터 폐루프로 잰다.

- 모델·데이터·하이퍼파라미터·평가 프로토콜: **[SETTINGS.md](SETTINGS.md)**
- 코드·실행 기록: **[`ers345/pi05`](https://github.com/inner-king/policy/tree/ers345/pi05/pi05) 브랜치** (main 에는 이 문서와 SETTINGS.md 만 둔다 — [AGENTS.md](../AGENTS.md))
- 결과와 해석: [이슈](https://github.com/inner-king/policy/issues) (`[pi05]`)

| 항목 | 내용 |
|---|---|
| 시작 가중치 | openpi `pi05_base` (openpi 커밋 `215abfb`) |
| 데이터 | `behavior-1k/2026-challenge-demos` 의 요리 15 태스크 × 200 에피소드를 스킬 구간 66,892개로 자른 것. RGB 3뷰 224×224 |
| 지시문 | B1K 스킬 주석을 영어 명령문으로 바꾼 것. 학습 범위 466종(1차) → 20종(2~8차) → 32종(9~10차) |
| 학습 | Colab A100(40/80 GB) 한 장, 차수당 10k~26k 스텝. 10차 체크포인트의 계보는 2 → 3 → 4 → 5 → 6 → 7 → 9 → 10 (1차·8차는 갈래가 끊겼다) |
| 평가 | 로컬 RTX 5060 Laptop 8 GB. 정책은 CPU 추론, OmniGibson 3.9.1 에서 스킬 단위 폐루프 |

모듈은 π0.7 식으로 나누면 LL(저수준 정책) 하나뿐이다. 상위 정책(다음 스킬 문장 예측)과 서브골 생성은 없고,
평가에서는 사람이 쓴 스킬 주석 문장을 그대로 넣는다.

---

## 차수와 run-id

run-id 는 `<모델>-<초기값>-<총스텝>` 이다. 모델 자리는 설정 판(v1~v5, [SETTINGS §6](SETTINGS.md#6-학습-설정)),
초기값 자리는 이어받은 앞 차수다. 원래 저장소에서는 `run1`~`run10` 으로 불렀고, 체크포인트 경로도
`<설정 이름>/<runN>/<스텝>` 이라 이슈와 코드 주석에는 그 이름이 나온다.

| 차수 | run-id | 설정 (`src/configs/cooking_config.py`) | 이어받은 가중치 | 스텝 |
|---|---|---|---|---:|
| 1 | `pi05v1-pi05base-10k` | `pi05_cooking` | π0.5 base | 10,000 |
| 2 | `pi05v2-pi05base-13k` | `pi05_cooking_top20` | π0.5 base | 13,000 |
| 3 | `pi05v3-run2-12.5k` | `pi05_cooking_top20_v3` | run2/12999 | 12,500 |
| 4 | `pi05v3-run3-11.5k` | 〃 | run3/12499 | 11,500 |
| 5 | `pi05v3-run4-12.5k` | 〃 | run4/11500 | 12,500 |
| 6 | `pi05v3-run5-25k` | 〃 | run5/12499 | 25,000 |
| 7 | `pi05v3-run6-26k` | 〃 | run6/25000 | 26,000 |
| 8 | `pi05v4-run7-21k` | `pi05_cooking_top20_v4` | run7/25999 | 21,000 (평가는 18000) |
| 9 | `pi05v5-run7-24k` | `pi05_cooking_top32_v5` | run7/25999 | 24,000 |
| 10 | `pi05v5-run9-24k` | 〃 | run9/23999 | 24,000 |

실행별 설정은 브랜치의 `runs/train/<run-id>/run.yaml`, 마지막 평가(10차)는 `runs/eval/pi05v5-run9-24k-at24k/run.yaml` 에 있다.

---

## 구성

아래 파일은 `ers345/pi05` 브랜치에 있다. **코드 루트는 `pi05/src/` 이고 명령은 모두 여기서 실행한다.**
스크립트는 자기 위치의 두 단계 위(`src/`)를 루트로 보므로 `pi05/` 을 어디에 두어도 된다.
`third_party/` `data/` `checkpoints/` `outputs/` `assets/` 는 실행하면서 `src/` 안에 생기고 커밋하지 않는다.

```
pi05/
├── src/
│   ├── configs/
│   │   ├── cooking_config.py     학습 설정 v1~v5 를 openpi 설정 목록에 런타임 등록 (openpi 는 고치지 않는다)
│   │   ├── r1pro_policy.py       R1Pro 입출력 변환: state 61 → 25차원, 팔·몸통 18차원 델타 액션
│   │   ├── task_subset.py        지시문 부분집합과 지시문당 프레임 상한 (데이터를 다시 만들지 않고 인덱스만 거른다)
│   │   ├── pi0_patch.py          다표본 flow matching(N=8)과 차원별 손실 로깅 (런타임 패치)
│   │   ├── lerobot_patch.py      6.7만 에피소드 규모에서 LeRobot 로더 기동 비용 우회
│   │   ├── policy_loader.py      8 GB GPU·CPU 에서 체크포인트를 읽는 로더
│   │   └── segment_index.py      학습 세그먼트 ↔ 원본 에피소드·스킬 번호 대응 복원
│   └── scripts/
│       ├── segment_lib.py build_segments.py        스킬 주석 → 세그먼트·지시문 (data/segments.parquet)
│       ├── build_dataset.py verify_dataset.py       원본 → 224×224 LeRobot 데이터셋, 무결성 전수 점검
│       ├── compute_norm_stats.py compute_norm_stats_perstep.py   정규화 통계 (v3 부터 청크 위치별 (32, 23))
│       ├── train.py preflight.py                    openpi 학습 래퍼(시드·lr 일정 방어벽 포함), 학습 직전 점검
│       ├── check_freeze_filter.py check_pi0_patch.py   학습/동결 파라미터 집계, 다표본 패치 값 검증
│       ├── finalize_checkpoint.py verify_checkpoint.py   받은 체크포인트가 온전하고 열리는지 확인
│       ├── serve_policy_b1k.py                      정책 websocket 서버 (openpi 서버 형식, CPU/GPU)
│       ├── rollout_b1k.py                           OmniGibson 폐루프 하네스: 데모 재생 인계 → 정책 구간 → 스킬 판정·영상·설명 파일
│       ├── export_demo_actions.py                   롤아웃 인계용 데모 액션을 npz 로 내보낸다
│       ├── pick_top32.py plan_rollout32.py          32종 선정 검사, 롤아웃 계획(데모·인스턴스·스킬 번호) 뽑기
│       ├── rollout_all20.sh rollout_new12.sh rollout_unseen.sh   평가 스위트: 기존 20종 21판 / 신규 12종 12판 / 일반화 13판
│       ├── rollout_null_control.sh                  대조군 — 정책에 묻지 않고 같은 무대를 돌린다
│       ├── eval_chain.sh rollout_watchdog.sh        한 차수 평가 연쇄 실행(0~5단계), 멈춘 판 감시
│       └── probe_language.py ablate_policy.py compare_rollout_actions.py   언어 조건화 검사 (오프라인 2종, 폐루프 1종)
└── runs/{train,eval}/<run-id>/run.yaml             실행별 설정 기록 (결과 필드는 넣지 않는다)
```

---

## 환경

| | 데이터 변환·평가 (로컬) | 학습 |
|---|---|---|
| 기기 | Ubuntu 22.04, RTX 5060 Laptop 8 GB (sm_120), RAM 54 GB | Colab A100 40 GB 또는 80 GB 한 장 (차수별 카드는 run.yaml) |
| 정책 환경 | openpi `215abfb` 의 uv venv (JAX 0.5.3, Python 3.11) | 같음 |
| 시뮬레이터 | BEHAVIOR-1K `v3.9.1`, conda env `behavior` | — |

- 8 GB 카드에는 시뮬레이터(약 5.5 GiB)와 정책(약 6.8 GiB)이 같이 안 올라가서 **정책은 CPU 로 추론한다**(호출당 약 9.5 초).
  롤아웃 한 판은 40~47분이고 대부분이 추론 시간이다. 정책 서버는 websocket 이라 GPU 가 있는 다른 머신으로 옮기면 코드 수정 없이 빨라진다(돌려 보지는 않았다).
- 롤아웃 스크립트는 시뮬레이터 인터프리터를 `BEH` 로 받는다. 기본값은 `~/miniconda3/envs/behavior/bin/python` 이다.
  이 저장소의 `setup/install_behavior.sh`(v3.9.2, miniforge)로 만든 환경이면 `export BEH=~/miniforge3/envs/behavior/bin/python`.
  **v3.9.2 로는 돌려 보지 않았다.**

```bash
cd pi05/src
git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi
git -C third_party/openpi checkout 215abfb
(cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)
export HF_LEROBOT_HOME=$PWD/data/lerobot HF_HUB_OFFLINE=1   # HF_HUB_OFFLINE 이 없으면 로컬 데이터가 있어도 Hub 를 기다리다 멈춘다
```

---

## 데이터

요리 15 태스크: 4 · 40 · 41 · 42 · 43 · 45 · 46 · 48 · 49 · 51 · 62 · 70 · 72 · 95 · 96 (이름은 [SETTINGS §5](SETTINGS.md#5-데이터)).

1. **메타·주석 받기** — `data/meta/`(에피소드 인덱스, `tasks.jsonl`)와 `data/annotations/task-00XX/`(스킬 주석).
   아래 명령은 파일 배치를 보고 적은 것이고 **이대로 다시 받아 보지는 않았다.**
   ```bash
   hf download behavior-1k/2026-challenge-demos --repo-type dataset --local-dir data \
     --include "meta/*" --include "annotations/*"      # 주석은 100 태스크 것을 다 받는다. 쓰는 것은 15 태스크뿐
   ```
2. **세그먼트·지시문** — `uv run --project third_party/openpi python scripts/build_segments.py` → `data/segments.parquet`
   (에피소드 3,000개 → 세그먼트 81,278개, 고유 지시문 472종).
3. **변환** — `uv run --project third_party/openpi python scripts/build_dataset.py`. 태스크 단위로 원본을 받아 224×224 H.264(crf 23)로
   바꾸고 원본을 지운다. `move to` 세그먼트는 시드 0 으로 25%만 남긴다. 결과 66,892 에피소드(= 세그먼트) / 26,579,202 프레임 / 65.6 GB,
   원래 기기에서 16.0시간. 끝나면 `data/lerobot/cooking-vla/behavior-cooking-224` 를 `data/lerobot/jiu121/behavior-cooking` 으로 옮긴다
   (`cooking_config.REPO_ID` 가 이 이름을 본다).
4. **점검** — `scripts/verify_dataset.py` (개수·프레임 수·인덱스 연속성 전수 점검).
5. **정규화 통계** — 설정마다 따로 낸다. 부분집합은 분포가 달라 **전체 통계를 쓰면 에러 없이 정규화가 어긋난다.**
   ```bash
   uv run --project third_party/openpi python scripts/compute_norm_stats_perstep.py --config-name pi05_cooking_top32_v5 \
     --max-frames 400000 --batch-size 256 --num-workers 6
   ```
   통계가 없으면 openpi 는 경고만 찍고 정규화 없이 학습한다. `scripts/preflight.py` 가 이것을 막는다.

BEHAVIOR 데이터·에셋과 `omnigibson.key` 는 저장소 루트의 [README](../README.md) 대로 설치한다(재배포 금지).

---

## 실행

### 학습

10차(`pi05v5-run9-24k`)와 같은 인자다. 다른 차수의 인자는 각 run.yaml 에 있다. 원래는 Colab 노트북이 이 명령을 조립했다(노트북은 저장소·Drive 경로가 박혀 있어 브랜치에 넣지 않았다).

```bash
export COOKING_INIT_PARAMS_V5=checkpoints/pi05_cooking_top32_v5/run9/23999/params   # v3·v4 는 COOKING_INIT_PARAMS
uv run --project third_party/openpi python scripts/preflight.py pi05_cooking_top32_v5
uv run --project third_party/openpi python scripts/train.py pi05_cooking_top32_v5 \
  --exp-name=run10 --seed=49 --batch-size=32 --num-workers=8 \
  --num-train-steps=24000 --lr-schedule.decay-steps=24000 --lr-schedule.warmup-steps=1000 \
  --lr-schedule.peak-lr=1e-4 --checkpoint-base-dir=checkpoints \
  --keep-period=2000 --save-interval=2000 --no-wandb-enabled
```

- `--lr-schedule.decay-steps` 를 스텝 수와 같이 바꾸지 않으면 lr 이 끝까지 안 식는다. `--lr-schedule.peak-lr` 을 빼면 설정 기본값 5e-5 로 돌아간다(4차부터 1e-4).
- 이어받을 때는 **실행 이름과 시드를 둘 다 바꾼다.** openpi 는 재개할 때 데이터 순서를 복원하지 않아서 같은 시드면 앞 차수가 본 샘플을 같은 순서로 다시 본다. `train.py` 의 방어벽은 시드가 기본값 42 인지만 본다.
- 받은 체크포인트는 `scripts/finalize_checkpoint.py checkpoints/<설정>/<runN>` 으로 확인한다. Drive 에서 받다 잘리거나 목차 파일이 빠져 **안 열리는** 경우가 실제로 있었다.

### 평가

```bash
# 인계용 데모 액션 (스위트 머리말에 필요한 데모 목록과 명령이 있다)
third_party/openpi/.venv/bin/python scripts/export_demo_actions.py --task slicing_vegetables --episode 8652

# 스위트 하나 — 체크포인트와 설정을 같이 넘긴다 (안 맞으면 멈춘다)
COOKING_CKPT=checkpoints/pi05_cooking_top32_v5/run10/23999 COOKING_CONFIG=pi05_cooking_top32_v5 \
  OUT=outputs/run10_base20 bash scripts/rollout_all20.sh

# 한 차수 전체 (0~5단계: 체크포인트 검사 → 언어 검사 → 대조군 → 신규 12 → 기존 20 → 일반화 13, 약 34시간)
setsid nohup bash scripts/eval_chain.sh run10 >> outputs/run10_eval_chain.log 2>&1 < /dev/null &
setsid nohup bash scripts/rollout_watchdog.sh >> outputs/run10_eval/watchdog.log 2>&1 < /dev/null &
```

- `eval_chain.sh` 의 0단계는 원래 기기의 rclone Drive 원격(`gdrive:`)에서 중간 체크포인트를 받는다. 다른 환경에서는 이 단계가 실패하고 넘어가므로 체크포인트를 미리 `checkpoints/` 에 둔다.
- 롤아웃 스크립트는 판마다 정책 서버를 `pkill -f serve_policy_b1k` 로 끄고 새로 띄운다. **스위트 두 개를 동시에 돌리지 않는다**(서로의 서버를 죽인다). 스크립트를 띄우는 명령줄에 이 문자열을 넣으면 래퍼 자신이 죽는다.
- 결과 폴더마다 `README.md`(재생 구간과 정책 구간의 경계, 스킬이 해야 할 일, 판정 결과)가 생긴다. 무대가 정상이었는지는 로그의 `[무대 검사]` 줄로 센다.

언어 조건화 검사(오프라인, CPU 로 체크포인트당 약 15~20분):

```bash
HF_LEROBOT_HOME=$PWD/data/lerobot HF_HUB_OFFLINE=1 third_party/openpi/.venv/bin/python scripts/probe_language.py \
  --config pi05_cooking_top32_v5 --ckpt checkpoints/pi05_cooking_top32_v5/run10/23999 --batch 16 --device cpu \
  --json-out outputs/probe_run10.json
```

---

## 코드 버전과 결과의 대응

이 디렉토리는 원래 저장소(비공개)의 커밋 `4c9abe6`(2026-09-28)에서 필요한 파일만 옮긴 것이다. 이슈의 결과는 그때그때의 코드로 냈고,
결과를 해석할 때 영향이 있는 변경은 아래와 같다.

| 변경 (시점) | 이전 결과에 주는 영향 |
|---|---|
| 롤아웃을 **데모의 장면 인스턴스**(`task_instance_id`)에서 돌리게 함 (09-03) | 그 전 롤아웃(2~4차, 20회)은 인스턴스 0 에서 돌아 인계 자세가 데모와 달랐다. 비교에 쓰지 않는다 |
| 바퀴 고정(`--zero-base`)을 기본에서 뺌 (09-03) | 그 전의 "폐루프에서 손을 안 쥔다"는 바퀴를 묶어 생긴 현상이었다 |
| 스킬 단위 판정기 도입 (09-05), 20종 전부에 붙음·`half` 물체 구분 (09-07) | 09-05 전 기록의 `성공: False` 는 태스크 전체 BDDL 목표라 스킬 하나로는 원리적으로 참이 될 수 없었다 |
| 정책 구간 1분 → 75초 → **90초(2,700 스텝)**, 조기 종료 기본 끔 (09-06~09) | 60초판에서는 65~71초에 나온 성공을 못 봤다. 조기 종료가 켜진 판은 문을 "5%만" 연 순간 15초 뒤 잘렸다 |
| `probe_language.py` 가 **지시문만** 바꿔 끼우게 함 (09-25) | 그 전 값(예: 7차 +277.9%)은 지시문과 함께 로봇 자세 토큰까지 바꾼 값이다. 새 값과 직접 견주지 않는다 |
| 결과 폴더 `README.md` 의 머리 경고를 무대 문제/부수 변화로 나눔 (09-26) | 9차 결과 폴더의 README 는 둘을 섞어 적었다. 9차 판은 로그 `[무대 검사]` 줄로 센다 |

공개하면서 바꾼 것 (기본값에서의 동작은 원래와 같다):

- `build_dataset.py`·`verify_dataset.py`·`build_segments.py`: 로컬 절대경로 `ROOT` 를 스크립트 위치 기준으로 바꿨다.
- `rollout_null_control.sh`: 절대경로 `cd` 를 스크립트 위치 기준으로, 시뮬레이터 인터프리터를 `BEH` 로 덮어쓸 수 있게 했다.
  `rollout_watchdog.sh`·`eval_chain.sh`: 롤아웃 프로세스를 찾는 패턴이 `BEH` 를 따르게 했다.

넣지 않은 것:

- 원래 저장소의 문서(`docs/HANDOFF.md` 등). 코드 주석의 `HANDOFF 40.16` 같은 절 번호가 가리키는 곳이다. 요지는 SETTINGS.md 와 이슈에 옮겼다.
- 일회성 진단·과거 스위트 스크립트(1차 진단 `eval_offline.py`·`ablate_by_frequency.py`, 7차 냉장고 실험 `run7_*.sh`, `run9_eval_chain.sh` 등), Colab 노트북, 데이터 업로드 스크립트, 정규화 통계 파일.
  이슈에서 이 스크립트로 낸 수치에는 그렇다고 적었다.
- 스냅샷 시점에 커밋되지 않았던 작업 중 변경(언어부 LoRA 절제용 `--zero-lang-lora` 등).

---

## 라이선스 주의

- 데이터: `behavior-1k/2026-challenge-demos` 는 HF 에서 받는다. BEHAVIOR 에셋(BEHAVIOR Data Bundle)은 비상업적 학술 연구 용도로만 쓸 수 있고 재배포할 수 없다.
- 가중치: π0.5 base 는 openpi 저장소의 약관을 따른다. 학습한 체크포인트는 이 저장소에 넣지 않는다.
- 롤아웃은 Isaac Sim 을 띄운다. 실행하면 NVIDIA Isaac Sim EULA 에 동의하는 것이다.
