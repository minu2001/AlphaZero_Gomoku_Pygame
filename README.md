# Gomoku (Omok) AI using AlphaZero-style Self-Play

### 11×11 Board | Policy-Value Network | Monte Carlo Tree Search



---
### 시연 영상
 https://youtu.be/ge36JB_28YQ
---

## 1. 연구 개요 (Abstract)

본 프로젝트는 **11×11 오목(Gomoku)** 환경에서 인간 수준 이상의 의사결정을 수행하는 인공지능 에이전트를 구현하는 것을 목표로 한다. AlphaZero 프레임워크를 기반으로, **정책-가치 신경망(Policy-Value Network)** 과 **Monte Carlo Tree Search(MCTS)** 를 결합한 자기대국(Self-play) 학습 구조를 설계하였다.

모델은 사전 지식이나 인간 기보 데이터 없이 **완전한 자기대국(Self-play)** 만을 통해 학습되었으며, 실제 학습 과정은 **총 13일 이상** 지속되었다. 이 과정에서 수만 회의 MCTS 시뮬레이션과 반복적인 신경망 갱신이 수행되었다.

---

## 2. 프로젝트 구조

```
├── agents.py            # ZeroAgent (MCTS + PVNet), 탐색 로직
├── model.py             # Policy-Value Network (Residual CNN)
├── main.py              # Self-play 및 학습 루프
├── eval_main.py         # 평가/전시용 실행 코드(별도 UI)
├── utils.py             # 보드 상태, 승리 판정, 데이터 증강
├── pygame_env.py        # 11x11 오목 GUI 환경(대시보드/효과 포함)
├── play_with_ai.py      # 학습된 모델과 인간 대국(스레드로 UI 프리징 방지)
├── webapi.py            # 웹 대시보드 API(정책/방문횟수/메시지 노출)
├── agent_info.py        # 에이전트 정책/방문횟수/가치 기록
├── game_info.py         # 게임 상태 관리
├── logs/
│   ├── log_250918.txt
│   ├── log_250922.txt
│   └── log_250923.txt
```

---

## 3. 코드 레벨 상세 설명 (Module Walkthrough)

이 섹션은 “무슨 코드가 어디서 무엇을 하고, 왜 그렇게 했는지”를 설명한다. 특히 **13일 이상 학습이 가능했던 이유(속도/안정성 최적화 포인트)** 를 함께 기술한다.

### 3.1 `main.py` — 전체 학습 파이프라인(자기대국 → 리플레이버퍼 → 학습 → 저장)

`main.py`는 프로젝트의 엔진이다.

#### (1) 하이퍼파라미터/실험 세팅

* `BOARD_SIZE = 11`, `N_MCTS = 400`, `TAU_THRES = 6`
* 네트워크: `N_BLOCKS = 10`, `IN_PLANES = 5`, `OUT_PLANES = 128`
* 학습: `N_SELFPLAY = 100`, `MEMORY_SIZE = 30000`, `BATCH_SIZE = 32`, `LR = 2e-4`

#### (2) 자기대국(self_play)

* 각 턴마다 `Agent.get_pi(root_id, tau)`를 호출하여 **MCTS 기반 정책 π** 를 얻는다.
* π에서 action을 샘플링하여 게임을 진행하고, 상태/정책/결과를 `(state, π, z)` 형태로 저장한다.
* 게임 종료 시 흑/백의 승패를 `reward_black`, `reward_white`로 정리해 **각 턴의 z(label)** 로 역전파한다.
* 마지막에 `utils.augment_dataset(...)`로 회전/대칭 증강을 수행해 데이터 효율을 높인다.

#### (3) 학습(train)

* **데이터가 부족하면 학습을 건너뛰는 안정장치**가 존재한다.

  * `if len(rep_memory) < BATCH_SIZE: ... Skipping.`
  * 초기 단계에서 학습이 ‘빈 데이터’로 터지는 것을 방지한다.
* `rep_memory`에서 표본을 뽑아 `train_memory`를 만든 뒤 `DataLoader`로 학습한다.
* 손실함수는 AlphaZero 표준 구성:

  * Value loss: MSE `(v - z)^2`
  * Policy loss: Cross-Entropy `-(π · log p)`
  * Total: `loss = v_loss + p_loss`

#### (4) 체크포인트/재시작 설계

* `save_model`, `save_dataset`로 **모델과 리플레이버퍼를 주기적으로 저장**한다.
* `load_data(model_path, dataset_path)`로 **중단 후 재개(resume training)** 가 가능하다.

---

### 3.2 `agents.py` — ZeroAgent(MCTS + PVNet) 구현의 핵심

`ZeroAgent`는 AlphaZero의 “두뇌”에 해당한다. 실제 연산량(=학습시간 13일의 대부분)을 차지하는 부분이므로, 여기서의 안정성과 최적화가 가장 중요하다.

#### (1) `get_pi(root_id, tau)` — 정책 π 생성

* `self._init_mcts(root_id)`로 루트 노드를 준비하고,
* `self._mcts(self.root_id)`로 **N_MCTS만큼 시뮬레이션**한다.
* 루트의 child 방문 횟수(`n`)를 모아 `pi = visit / visit.sum()`로 정책을 만든다.
* `tau == 0`일 때는 argmax(원핫)으로 바꿔 “가장 좋은 수”를 선택한다.

**안정성/버그 방지 최적화**

* `child_id in self.tree` 체크를 넣어 KeyError를 방지한다.
* `visit.sum() == 0`일 때 0으로 나누는 문제를 방지한다.

#### (2) `_init_mcts` — Dirichlet Noise로 탐색 다양성 확보

* 루트가 “새 루트(실제 루트)”일 때와 아닐 때를 구분한다.
* `noise=True`이면, 루트의 자식 prior `p`에 Dirichlet noise를 섞는다.

  * `p = 0.75 * p + 0.25 * noise`
* 자기대국 데이터가 한 쪽으로 쏠리는 것을 줄여, 장기적으로 학습이 더 안정적으로 진행되게 한다.

#### (3) `_selection` — PUCT 기반 탐색 + 예외 케이스 처리

* 일반적으로는 `Q + U`가 최대인 자식을 선택한다.
* **최적화/안정화 포인트:** 모든 자식의 방문 횟수 합 `total_n == 0`인 특수 케이스가 존재한다.

  * 이때 `Q+U` 계산이 무의미해지므로,
  * 정책 prior `p`가 가장 큰 자식을 선택하도록 처리하여 탐색이 멈추지 않게 한다.

#### (4) `_expansion_evaluation` — GPU 추론 + 합법수 마스킹

* `torch.no_grad()`로 gradient 추적을 끄고 추론한다.

  * MCTS에서 이 부분은 수만 번 호출되므로 **매우 큰 속도 최적화 지점**이다.
* 합법 수만 prior_prob에 남기고, 확률 합이 0이면 **균등분배 fallback**을 둔다.

---

### 3.3 `model.py` — PVNet(Residual CNN)

* 입력 (IN_PLANES=5) → 잔차 블록(N_BLOCKS=10) → Policy head + Value head
* 정책 헤드는 121(=11×11) 위치에 대한 확률을 출력
* 가치 헤드는 현재 상태의 승률을 -1~1로 출력

이 구조는 오목처럼 “국소 패턴 + 장기 연결(포석)”이 동시에 중요한 게임에서
CNN의 공간적 특성과 Residual의 깊이를 동시에 활용하기 위한 설계다.

---

### 3.4 `utils.py` — 게임 규칙/상태 인코딩/증강

* `check_win(board, win_mark)`: 5목 승리 판정(가로/세로/대각)
* `legal_actions(node_id, board_size)`: 현재까지 둔 수를 제외한 행동 인덱스 반환
* `get_state_pt(...)`: 현재 보드를 **채널 형태(state tensor)** 로 변환
* `augment_dataset(...)`: 회전/대칭을 이용한 데이터 증강

규칙/상태 처리를 utils에 모아둔 이유는,
에이전트/환경/UI가 모두 동일한 규칙을 공유해야 하기 때문이다.

---

### 3.5 `play_with_ai.py` — UI 프리징 방지를 위한 스레딩 최적화

학습된 모델과 사람이 대국할 때, AI는 한 수를 위해 MCTS 400회를 돌린다.
이를 메인 스레드에서 수행하면 Pygame 창이 멈춘 것처럼 보이기 때문에,
**AI 연산을 별도 스레드로 분리**했다.

* AI 턴이면 `threading.Thread(target=ai_worker, ...)`로 실행
* 결과는 `queue.Queue()`로 메인 루프에 전달
* 메인 루프는 계속 `env.render()`를 돌 수 있어 “렉/멈춤” 체감이 크게 줄어든다.

또한 콘솔 출력 가독성을 위해 `np.set_printoptions(precision=2, suppress=True)`를 사용하여
정책 분포와 승률 출력이 사람이 읽기 쉬운 형태로 유지된다.

---

### 3.6 `pygame_env.py` / `eval_main.py` — 전시/시연용 시각적 완성도 최적화

전시 환경에서는 "AI가 실제로 생각하는 느낌"을 전달하는 연출이 중요하다.

* 폰트 로딩을 함수로 분리해 OS별(윈도/맥) 폰트 폴백을 제공
* `SRCALPHA` 서피스를 사용해 **Glow/Win Halo** 같은 알파 블렌딩 효과 구현
* 최근 착수에만 Glow를 적용해 매 프레임 계산량을 제한
* 애니메이션 중에는 hover hint를 비활성화하여 화면 혼잡/연산량을 줄임
* FPS를 60으로 고정하고 `clock.tick(FPS)`로 렌더 루프를 안정화

---

## 4. 성능 최적화 포인트(어디서, 왜 빨라졌나)

본 프로젝트에서 실제 체감 성능에 큰 영향을 준 최적화 지점은 다음과 같다.

### 4.1 MCTS 추론에서의 `torch.no_grad()`

MCTS는 한 수에서 400회, 전체 학습에서는 그 수십/수백 배로 신경망 추론을 호출한다.
이때 `no_grad()`를 사용하지 않으면 불필요한 그래프가 쌓여 속도와 메모리가 크게 악화된다.

### 4.2 예외 케이스 처리로 “탐색 멈춤/NaN” 방지

* `visit.sum()==0` 방지
* `total_n==0`일 때 prior 기반 선택
* prior 합이 0이면 균등분배

이런 방지 장치가 없으면 학습 초기에 정책이 평평하거나 0에 가까워지는 구간에서
MCTS가 깨지거나(0/0, NaN) 아예 진행이 멈출 수 있다.

### 4.3 UI는 스레딩으로 분리해서 사용자 체감 개선

학습 성능 최적화는 아니지만, “전시/시연”에서 가장 중요한 UX 최적화다.
AI가 생각하는 동안에도 UI가 살아있는 것이 매우 큰 체감 차이를 만든다.

### 4.4 (현재 코드 기준) 병목 지점도 명확히 기록해둠

`train()` 내부에 다음 코멘트가 있다:

* “이 부분이 나중에 데이터가 쌓이면 느려지는 원인이 됩니다.”

원인은 `random.sample(list(rep_memory), sample_size)`에서
매 iteration마다 `deque → list` 변환 + 샘플링이 발생하는 구조이기 때문이다.

---

## 5. 개선 제안(보고서용 Future Work)

### 5.1 리플레이 버퍼 샘플링 최적화

* `rep_memory`를 list로 매번 변환하지 않도록 캐싱하거나,
* numpy 인덱싱 기반 샘플링, 또는 fixed-size ring buffer 구조로 변경

### 5.2 학습 스케줄 개선

현재는 0 iteration 이후 `N_SELFPLAY_ITER = 1`로 매우 작은 자기대국만 수행한다.
학습 품질과 안정성을 위해 "self-play와 train 비율"을 더 공격적으로 조정할 수 있다.

### 5.3 분산 Self-play

멀티프로세싱/멀티GPU를 사용하면 13일 학습 시간을 크게 줄일 수 있다.

---

## 6. AlphaGo · AlphaZero · MuZero 비교 (이론적 배경)

본 프로젝트의 이론적 기반을 명확히 하기 위해, AlphaGo / AlphaZero / MuZero의 차이를 정리한다.

### 6.1 AlphaGo

AlphaGo는 **지도학습 + 강화학습 + MCTS** 를 결합한 최초의 대규모 바둑 AI이다.

* **사전 기보 데이터(인간 전문가 기보)** 로 정책망을 지도학습
* 이후 자기대국 강화학습으로 성능 향상
* 정책망(Policy Network)과 가치망(Value Network)이 **분리**되어 존재
* 환경 규칙(바둑 룰)을 명시적으로 알고 있음

👉 핵심 한계: 인간 데이터 의존, 복잡한 파이프라인

---

### 6.2 AlphaZero

AlphaZero는 AlphaGo의 구조를 단순화·일반화한 모델이다.

* **인간 기보 데이터 없이**, 완전한 자기대국(Self-play)만 사용
* 정책망과 가치망을 **단일 네트워크(PVNet)** 로 통합
* 체스/쇼기/바둑 등 여러 게임에 동일 구조 적용 가능
* 환경 규칙은 여전히 **명시적으로 알고 있음**

👉 본 프로젝트는 **AlphaZero 구조를 오목 도메인에 적용**한 사례이다.

---

### 6.3 MuZero

MuZero는 AlphaZero에서 한 단계 더 나아간 모델이다.

* **환경 규칙(전이 함수)을 알지 못함**
* 게임 상태 → 잠재 상태(latent state)로 변환하는 **Dynamics Model** 학습
* 내부적으로 환경을 “상상(imagination)”하며 MCTS 수행

| 구분    | AlphaZero | MuZero                                 |
| ----- | --------- | -------------------------------------- |
| 규칙 지식 | 명시적       | 없음                                     |
| 상태    | 실제 상태     | 잠재 상태                                  |
| 모델 구성 | PVNet     | Representation + Dynamics + Prediction |

👉 MuZero는 일반 강화학습 문제에 강력하지만, 구현 복잡도와 학습 비용이 매우 큼

---

### 6.4 본 프로젝트의 위치

* AlphaGo ❌ (인간 기보 미사용)
* MuZero ❌ (환경 모델 학습 없음)
* **AlphaZero ⭕ (Self-play + MCTS + PVNet)**

본 프로젝트는 **AlphaZero 계열 중에서도 가장 정석적인 구조**를 충실히 구현하는 데 초점을 맞췄다.

---

## 7. 학습 로그 기반 타임라인 분석 (13일)

실제 로그(`logs/log_250918.txt`, `log_250922.txt`, `log_250923.txt`)를 기반으로 학습 흐름을 정리한다.

### Day 1–2 : 실험 초기 (9×9 → 11×11 전환)

* 초기에는 9×9 보드로 실험 시작
* CUDA 비활성 상태(CPU)로 실행됨
* MCTS/모델 구조 정상 동작 여부 검증이 목적

---

### Day 3–5 : GPU 활성화 및 본격 학습 진입

* CUDA=True 확인
* BOARD_SIZE를 11로 고정
* N_MCTS=400, N_SELFPLAY=100 유지
* 자기대국 반복 수행 시작

이 시점부터 연산량이 급격히 증가하며 학습 시간이 길어짐

---

### Day 6–10 : 장기 자기대국 안정화 구간

* 반복 로그: `Playing Episode 10, 20, …, 100`
* 학습이 중단 없이 지속됨 → **안정성 확보 확인**
* Dirichlet noise, 예외 처리 로직이 효과적으로 작동

---

### Day 11–13 : 모델 저장 및 전시 연동 단계

* 학습된 모델을 `.pickle` 형태로 저장
* `play_with_ai.py`에서 모델 로딩 후 인간 대국 가능 확인
* UI/전시 환경에서 실시간 정책·승률 출력 테스트

---

## 8. 전시 및 대시보드 구조

본 프로젝트는 “학습 결과를 보여주는 것”까지를 목표로 설계되었다.

### 8.1 Pygame 전시 UI 구조

* `pygame_env.py`

  * 보드 렌더링
  * 착수 애니메이션 / 승리 효과
  * 최근 착수 강조(Glow)

* `play_with_ai.py`

  * AI 연산을 별도 스레드에서 실행
  * 정책 분포 및 예상 승률 콘솔 출력

👉 관람객 입장에서 **AI가 생각하고 결정하는 과정**을 직관적으로 체감 가능

---

### 8.2 웹 대시보드 구조 (`webapi.py`)

웹 대시보드는 “AI 내부 상태를 시각화”하기 위한 용도이다.

* `/periodic_status`

  * 현재 보드 상태
  * 정책 분포(p)
  * 방문 횟수(visit)
  * 에이전트별 승률 히스토리

* `/prompt_status`

  * MCTS 진행 중 메시지(예: simulation count)

이 구조는 **디버깅 + 전시 설명용**으로 동시에 활용 가능하다.

---

## 9. 핵심 코드만 요약 설명 (중요 부분만)

### `ZeroAgent.get_pi()`

* 한 수를 결정하는 핵심 함수
* 내부에서 MCTS → 방문 횟수 → 정책 π 생성

### `_selection / _expansion_evaluation / _backup`

* AlphaZero MCTS의 표준 3단계 구현
* 예외 처리로 장기 학습 안정성 확보

### `PVNet`

* 정책과 가치를 동시에 예측
* 오목의 공간적 패턴 학습에 적합한 Residual CNN

### `play_with_ai.py`의 스레딩

* 전시/시연에서 UX를 좌우하는 핵심 최적화

---

## 10. 결론

본 프로젝트는 단순한 게임 AI 구현이 아니라,

* AlphaZero 이론 이해
* 장기 자기대국 학습을 가능하게 하는 코드 안정성
* 전시·시연까지 고려한 시스템 설계

를 모두 포함한 **엔드투엔드 강화학습 프로젝트**이다.

특히 **13일간의 실제 학습 로그**는 이 프로젝트가 단순 예제가 아닌,
현실적인 계산 비용과 엔지니어링 난이도를 수반한 연구·전시용 시스템임을 보여준다.
