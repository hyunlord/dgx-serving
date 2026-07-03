# M4 STEP B — Gemma4 31B 성능 지도 (실측 vs roofline)

> 정직 스코프: spec 無(MTP 부재) → 27b **plain**과만 비교. 긴 context(>수천)는 sliding 링버퍼 전 제외.
> chunked prefill 미구현 → 프리필은 통짜 참고치(TTFT 주장 아님). roofline은 weight-이동 상한(KV/act 미계상)
> = 예측 도구 아님(달성률 17~63% 밴드가 이를 방증). 도구: tools/bench_gemma4.cu.

## STEP 0 — warm-클럭 측정 프로토콜 (명문화)

★ **모든 성능 측정 전 고정 워밍 런(throwaway) 필수.** GB10은 idle 시 SM 클럭 208MHz로 떨어지고, cold 상태
측정은 실측을 크게 왜곡한다(교훈: 이전 세션 qwen plain 동일 바이너리가 **cold 40.4 → warm 50.8 tok/s**, +26%).
- 절차: throwaway 벤치 1회 → `nvidia-smi --query-gpu=clocks.sm` 로 클럭 확인(2400MHz대) → 본 측정.
- 비교는 항상 **controlled back-to-back**(같은 GPU 열/클럭 상태). cold 앵커 금지.
- bench_gemma4는 내부 8스텝 warm-up도 하지만, GPU 클럭 ramp는 프로세스 밖 상태라 throwaway 런이 별도 필요.

## STEP B-1 — single decode (roofline 대조)

측정: n_past=0 시작(attention over ≤ngen keys=미미 → 순수 matmul 바운드), ngen=64 스텝, warm.

| 경로 | t/s | roofline 17.4 대비 | 커널 | 비고 |
|---|---|---|---|---|
| **W4A8 single** | **9.28** | **53.3%** | gemv_w4a8 | 27b와 동일 gemv족. ★perf-preview — gemma4 W4A8 correctness golden 미검증(성능만) |
| q4 single | 6.95 | 40.0% | gemm_tiled_q4 | 검증 경로. gemm_tiled가 gemv보다 비효율 |
| batch M=1 | 6.20 | — | gemm_batch_q4 | 서빙 경로의 B=1. gemv 대비 **-33%**(M=1 비최적) |

**대조: 27b single non-spec = 62.5%**(roofline 19.0, 실측 11.9, W4A8 gemv). Gemma4 W4A8 single **53.3%**로
27b에 근접 — **엔진이 두 번째 모델에도 일반화**(같은 gemv 커널족). 격차(53.3 vs 62.5%)는 Gemma4 구조가
weight-only roofline을 초과하는 bytes/특성 탓: V=262144(lm_head 5376×262144≈1.4B params=4.5%), hd_f=512
full attention(10층), 60층. **matmul-바운드 확정**(n_past=0 순수 matmul이 곧 이 수치).

## STEP B-2 — batch decode (B 스윕, gemm_batch_q4 q4)

| B | dec/stream | dec/aggr(총처리량) |
|---|---|---|
| 1 | 6.20 | 6.20 |
| 2 | 6.17 | 12.34 |
| 4 | 6.02 | **24.07** |
| 8 | 3.17 | 25.38 |
| 16 | 2.34 | 37.48 |

**포화점: B4.** aggregate가 B1→B4 준선형(3.9×) 후 B4→B8 정체(24→25, per-stream 6.0→3.2 절벽), B16=37.5.
GDN state 비용이 없는데도(27b plain은 B8 포화) **gemm_batch_q4가 B4에서 포화** — 27b의 MMQ(tensor-core)급
batch 커널 최적화가 Gemma4 batch-path엔 아직 없음(gemm_batch canonical만). = **batch 커널이 서빙 lever**.
메모리: per-seq KV(선형 캡, max_len=328) 591MB → B16=9.5GB, MemAvailable 94GB 예산 내 여유(가드 정상).

## STEP B-3 — plain 서빙 / CYC 병목

- **서빙 throughput proxy = batch-path aggregate**(서빙은 batch 커널 사용): B4 24 / B16 37 t/s.
  27b plain 서빙(warm 50.8 tok/s, B8 포화)보다 낮음 — 원인은 B-2와 동일(gemm_batch B4 포화 + M=1 비최적).
  ※ 전용 Poisson 서빙 실측(dummy-rope로 현실 길이)은 golden rope 17행 제약 우회 가능하나, throughput 상한은
    batch-path 수치가 지배하므로 CYC 결론 불변 — 별도(마진 작음).
- **CYC 병목 지도(측정+추론)**: 프리필 75 t/s(256토큰 통짜, compute-bound 병렬) vs decode 9.28 t/s =
  **8× 비율 → decode는 memory-bound**(weight 스트리밍 지배). n_past=0에서 이미 9.28 = attention 아닌 matmul이
  천장. → **다음 lever = FP4(텐서코어, toolkit-blocked) 또는 gemv/batch 커널 최적화 = 27b와 동일 결론.**
  Gemma4 고유 추가 lever: (a) batch 커널 MMQ화(B4 포화 해소), (b) B=1 슬롯을 gemv로(서빙 M=1 -33% 회수),
  (c) W4A8 correctness golden(현 q4 40%→W4A8 53% 확정 시 서빙 전환).

## STEP B-4 — llama.cpp 대조 (스킵, 정직 기록)

로컬 Gemma4 gguf는 **12B만**(gemma-4-12B-it-*.gguf) 존재, 우리 엔진은 **31B**. 크기 불일치(다른 param=다른
roofline)라 31B-엔진 vs 12B-llama 비교는 apples-to-oranges → **강행 금지, 스킵**. 31B gguf 확보 시 27b 방식
(single/prefill/batch 축별, 같은 q4급) 재개.

## 결론 — 엔진 일반성 & 다음 lever

1. **일반성 실증**: Gemma4 W4A8 single 53.3%가 27b 62.5%에 근접 = 커널 라이브러리가 두 번째 모델에 일반화.
2. **matmul-지배**(memory-bound) = 27b와 동일 좌표 → **FP4가 최대 미사용 lever**(toolkit-blocked, [[fp4-mma-toolkit-blocked]]).
3. **Gemma4 고유 갭**: batch 커널 B4 포화(MMQ 미적용) + M=1 비최적 + W4A8 미검증 — STEP C 튜닝/최적화 후보.
4. 정직: q4=검증 40%, W4A8=perf-preview 53%(correctness gate 별도). Poisson 서빙 실측·per-layer ncu 분해는
   결론(matmul-지배)에 불필요하나 확보 시 정밀화 가능.
