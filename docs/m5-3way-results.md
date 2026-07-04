# M5 — 3-way(2-way) HTTP 서빙 실측 결과 (dgx-serving vs llama.cpp, GB10)

> `docs/m5-3way-methodology.md`(공정 조건 설계) 실행. 챕터9부터 인프라-gated로 미룬 숙제 — M5 온라인 배칭 서버가
> 서면서 처음 조건 참. **성패가 인프라(통제 밖)라 조사-우선·되는 만큼 정직 측정**(강행 아님).
> 선례: `docs/ch6-3way-27b-results.md`(27b·B=1·CLI 기반). 이번은 **HTTP 서버·연속 배칭·26b MoE** 신규 좌표.

## 0. 인프라 조사 결과 (되는 조합 확정)

| 엔진 | GB10 상태 (이 세션 확인) | 이 2-way 투입 |
|---|---|---|
| **dgx-serving** | M5 온라인 배칭 서버 4모델 완성·API 엔진 천장(+0.4~1%) | ✅ (serve_api :8055) |
| **llama.cpp** | llama-server **sm_121 빌드 성공**·`gemma-4-26B-A4B` gguf(Q4_K_M 16GB) 확보·`-np 8 -c 4096` HTTP 起動 | ✅ (llama-server :8080) |
| **vLLM** | GB10 셋업 성공(`stage2-vllm-blueprint`: docker `vllm-openai:v0.17.0`, Qwen3.5-0.8B 로드·생성 확인) | ❌ 이 2-way 미투입 |

- **매칭 모델 = `gemma-4-26B-A4B`가 유일한 클린 2-way**: 4모델 중 로컬 gguf가 정확히 대응하는 건 26b뿐
  (27b=qwen 3.5 vs gguf 3.6 버전 불일치·31B=31B gguf 부재). → **26b MoE로 공정 2-way**.
- **vLLM 3-way는 별도 트랙(정직)**: vLLM GB10 구동 자체는 `stage2-vllm-blueprint`서 실증(과장 아님). 단 이 26b 2-way에
  넣으려면 (a)26b AWQ/GPTQ-4bit 체크포인트 (b)docker vllm-openai HTTP 서버 起動 — 시간 박스 넘겨 후속 세션으로.
  ★즉 "vLLM이 GB10서 안 돈다"가 아니라 "이 세션 2-way 범위 밖"(정직).

## 1. 공정 조건 (methodology §2 실행)

| 축 | 이 측정 | 캐비엇(정직) |
|---|---|---|
| 모델 | 양쪽 `gemma-4-26B-A4B`(동일 아키텍처·동일 활성 파라미터) | — |
| 양자화 | dgx q4(uq=1) ↔ llama Q4_K_M | 둘 다 ~4bit이나 스킴 다름 → **속도만 대조·품질 별도** |
| trace | 같은 프롬프트 8종·같은 max_tokens(G=48)·같은 동시성 N | — |
| 측정 지점 | **HTTP 클라이언트 관측** TTFT·tpot·throughput(단일 부하 생성기 `bench_3way.py`) | 서버 내부 타이머 아님 = 배포 현실 |
| 토크나이저 | dgx=HF gemma-4-E4B(V=262144) · llama=gguf 내장 | max_tokens 고정 → 생성 work 동일(속도 공정) |
| warm | 양쪽 throwaway 후 측정·SM clock 2398→2470MHz(warm 확인) | cold-클럭 교란 방지(헌법) |
| 배치 정책 | dgx Bmax=8(연속 배칭) ↔ llama `-np 8`(연속 배칭) | **동일 부하·유사 정책**(공정≠동일 구현) |
| rope | dgx=dummy rope(gemma4) | 품질 별도·**속도 대표**(계산량 동일) |

## 2. 결과 — 배치 스케일링 곡선 (throughput, warm, HTTP, 같은 26b·q4급)

| N(동시) | dgx t/s | llama t/s | dgx/llama | dgx TTFT | llama TTFT | dgx tpot | llama tpot |
|---|---|---|---|---|---|---|---|
| 1 | 14.7 | 52.6 | **0.28×** | 132ms | 46ms | 66.5ms | 16.7ms |
| 2 | 27.6 | 54.6 | 0.51× | 193ms | 70ms | 69.8ms | 28.5ms |
| 4 | 48.4 | 65.3 | 0.74× | 315ms | 126ms | 77.6ms | 59.6ms |
| 8 | 80.6 | 168.3 | **0.48×** | 567ms | 197ms | 89.3ms | 43.8ms |

- **dgx 연속 배칭 = near-linear**: 14.7→80.6 = **5.5×**(N=1→8). 슬롯 격리 amortize 양호(온라인 배칭 작동).
- **llama**: 52.6→168.3 = 3.2×(저-N 저조 52→65 → N=8 점프 168 — llama -np 8 저부하서 슬롯 미활용).
- **완전배치 N=8**: dgx 0.48×(2.1× 느림). 단일 N=1: 0.28×(3.6× 느림).

## 3. 정직 분석 (STEP C — overclaim 금지)

**dgx가 진다(measured). 하지만 왜 지는지가 정직의 핵심:**

1. **26b는 우리 최신·미최적 MoE.** decode graph를 이 M4에서 갓 구현(correctness-first, C-1/C-2는 "성능 관찰만").
   perf 튜닝 0. llama의 gemma-4-26B-A4B 커널은 성숙(수년 최적화·fused MoE·tiled GEMM). → **커널 성숙도 갭**.
2. **엔진 천장이 아님.** 27b dense는 roofline 62.5% 달성(measured). 26b MoE는 구조적으로 낮은 달성률
   (토큰당 640 expert-gemv launch + V=262144 거대 vocab unembed + dense∥MoE 이중 MLP). = 이 모델의 미최적, 엔진 한계 아님.
3. **배칭이 갭을 좁힘(긍정 신호).** 0.28×(N=1)→0.48×(N=8). dgx 연속 배칭이 near-linear로 amortize —
   서빙 스택(온라인 배칭·슬롯 격리)은 작동. 남은 갭은 **per-token 커널 속도**(FP4/fused/tiled = 기존 lever, 착수 전).
4. **spec = 구조적 차별점(이 2-way 미노출).** llama.cpp도 draft-model spec은 있으나, 우리 **MTP-head + tree-spec +
   연속 배칭 결합(곱 서빙)**은 다름(27b measured chain K=3 1.40×·tree 1.11×). 단 26b는 plain MoE(MTP 없음)라
   이 비교엔 안 걸림 — 27b spec 2-way는 매칭 gguf 부재로 별도(사실만 기록, 과장 금지).

**ch6 선례 대비**: ch6(27b·B=1·CLI) decode 0.71×·prefill 25-38× 뒤짐. 이 세션(26b MoE·HTTP·배칭) N=8 0.48×.
모델·측정 방식 달라 직접 비교 불가하나, **두 좌표 모두 "llama raw 커널이 빠르다"에 일관**(honest ⑦). 과장 없음.

## 4. 하네스 수정 (이 측정서 발견·수정)

- **`bench_3way.py` 인자 파싱 버그**: `if N==8` 센티넬이 첫 위치 인자가 8일 때 충돌(`"8 48"`→N=48로 오파싱).
  위치 인자 리스트로 수정. → 최초 N=8 실패는 실은 **N=48 초과구독**이 원인(측정 오류, 엔진 아님).
- **`serve_api.py` `request_queue_size`=5→128**: 동시 연결 폭주(초과구독)서 SYN backlog 초과 시 커널 리셋 방지
  (`daemon_threads`·`allow_reuse_address` 동반). llama-server(C++)는 큰 backlog라 견딤 — 프론트 견고성 정합.

## 5. 초과구독(부하 > Bmax) — backlog 수정 검증

| N(동시) | dgx t/s | llama t/s | dgx/llama | dgx TTFT | llama TTFT |
|---|---|---|---|---|---|
| 8 (=Bmax) | 80.5 | 164.5 | 0.49× | 579ms | 223ms |
| 16 (2×Bmax) | 81.4 | 144.5 | 0.56× | 2862ms | 1267ms |
| 32 (4×Bmax) | 81.7 | 163.2 | 0.50× | 7629ms | 3576ms |

- **★backlog 수정 검증됨**: N=16·32(구 backlog 5 초과)가 **ConnectionReset 없이** 전부 성공. (수정 전엔 N=48
  오파싱이 backlog 5 초과 → 리셋. 이제 128로 견딤.)
- **dgx throughput ~81 포화**(Bmax=8 cap 정확·80.5→81.4→81.7 flat) — 과부하를 **큐잉으로 흡수**(크래시 아님).
  초과 요청은 대기 → TTFT만 증가(579→2862→7629ms). 서빙 견고성 확인.
- **비율 초과구독서도 안정**(0.49~0.56×). dgx TTFT가 llama의 ~2×(aggregate throughput 절반 → 큐 배수 느림).
- N=8 재현: 0.48×(1차)·0.49×(2차) — **측정 재현성 확인**(헌법 결정성).

## 6. 결론

- **최소 보장 달성**: dgx vs llama 2-way(26b, 공정, HTTP, warm) measured. dgx N=8 0.48×(2.1× 느림) — 정직.
- **갭 = 최신 미최적 MoE 커널 성숙도**(엔진 천장 아님·배칭은 작동·near-linear).
- **vLLM 3rd leg = 후속**(GB10 구동은 실증됨·26b 체크포인트+docker HTTP만 남음 — 이 세션 범위 밖, 정직).
- overclaim 없음: 이기는 축 없음(속도). 우리 차별점(spec 곱 서빙·GDN·bit-id 정확성)은 이 plain-26b 비교에 안 걸림.
