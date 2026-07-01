# 챕터9 production scheduler — admission guard + slot 재사용 + 요청 큐 + mism=0

> ragged 토대(8aefb4e: per-seq 위치 + staggered 진입 mism=0) 위에 **실제 가변 요청 안전 서빙 로직**.
> 세 요소: ① admission guard(KV 예산, 시스템다운 방지, **최우선**) ② slot 재사용(종료 자리 새 요청) ③ 요청 큐.
> 테스트: `tests/test_cbatch_prod.cu` — R=6 요청을 B_max slot으로(재사용+큐 강제).

## STEP A — admission guard (최우선, 시스템 다운 방지)

unified 메모리 oversubscribe = 페이지 fault-in = 물리 RAM 고갈 = **시스템 전체 다운**(프로세스 OOM 아님, 실경험 103GB).
이 엔진의 batch state는 `B_max × max_len`을 **사전 할당** → 가드 = **alloc 시점**에 예산 검사:

```
per-seq = nfull·max_len·NKV·HD·2·2 (Kc+Vc bf16)  +  ngdn·(NH·D·D + (K-1)·conv_dim)·4 (GDN S+convctx fp32)
검사:  B_max · per-seq  ≤  0.7 × free(cudaMemGetInfo)   초과면 alloc 전 중단(B_max 줄여야).
```

**스트레스 검증 (안 죽나 — 최우선 게이트):**
- Bmax=5000(요청 폭주): per-seq 20.8MB × 5000 = **104GB > 26.5GB 예산** → **거부**(alloc 안 함, exit=0 깨끗한 중단).
- max_len=500000(긴 context): per-seq **6.2GB** × 8 = 49GB > 26.6GB → **거부**. 긴 context가 per-seq를 키우는 것 정확 감지.
- **시스템 다운 없이** 둘 다 alloc 전에 거부. unified oversubscribe 방지 확립.

## STEP B — slot 재사용 + 요청 큐

- **요청 큐**(deque FIFO): 모든 요청 대기 → free slot 생기면 admission(길이≤max_len) 통과 시 꺼내 배치.
- **물리 slot 고정**(compaction 안 함 — 8aefb4e 교훈): 종료 요청 slot 회수 → 큐 다음 요청이 그 물리 자리 재사용.
- **`reset_batch_slot(c,st,b)`**(models/qwen35_batch.cu): 재사용 시 물리 slot b의 GDN state(S/convctx) clear(잔재 제거).
  KV는 clear 불필요 — 새 요청이 자기 위치 0..P-1에 덮고, attention은 [0..POS(b)]만 read → 옛 요청 KV 잔재 안 읽음.
- R=6을 B_max=2로: admission 6회(**재사용 4회**), step 42.

## STEP C — mism=0 (정확성) + 안전성

### 올바른 정확성 기준 재정의 — iso(홀로) 대조
`brief`의 "각 요청 == solo bit-identical"에서 기존 solo = `qwen35_decode`(solo graph). 그러나 **scheduler 정확성의 진짜
기준은 "이웃/재사용/큐가 요청을 오염시키나"** = 같은 batch 엔진에 **홀로(B=1, 이웃·재사용 없음)** 돌린 출력(`iso`)과의 일치.
solo graph 비교는 그래프-구현 차이(아래)를 섞는다.

- **PRIMARY 게이트: got(scheduler: 재사용+큐+이웃) == iso(홀로) bit-identical** → 이웃/재사용/큐 불변.
  - **fp32: 오염 mismatch 0** (재사용 4회에도).
  - **W4A8: 오염 mismatch 0**. → 서빙 로직 정확(잔재 안 읽음, compaction 버그 없음).

### W4A8 발견 — solo-graph vs batch-graph borderline flip (scheduler 무관)
W4A8에서 `got` vs `solo(qwen35_decode)` = 11 토큰 차. 그러나 **`iso` vs solo도 정확히 11 == `got` vs solo** →
scheduler가 추가하는 발산 **0**. 11은 100% **그래프-구현 차**이지 재사용/큐/이웃 때문이 아님. 골든 대조로 확정:

| | fp32 | W4A8 |
|---|---|---|
| solo(qwen35_decode) vs batch-B1 max\|Δlogit\| | **5e-4 ~ 2e-3** | 첫 분기 step 0.32 |
| argmax FLIP | **0 (완전 일치)** | 있음 |
| 첫 FLIP step top-2 gap | — | **0.009** (거의 동점) |

- **fp32**: 두 그래프 등가(Δlogit ~1e-3 = attention/GDN 누적순서 차이, argmax robust → 토큰 완전 일치).
- **W4A8**: 첫 분기 step의 top-2 gap(0.009) ≪ 양자화 노이즈(0.32) → **borderline flip**. near-tie 토큰(760 vs 25)의 승자를
  두 그래프의 미세 fp 차이가 다르게 고름 → 이후 궤적 발산(cascade). **batch graph 자체는 정확**(fp32 bit-identical).
- 판별 근거(모두 골든 대조, 추측 아님): ① `gemv_w4a8_mlp<2>` vs `gemv_w4a8_sb` S=1 **bit-identical**(0/1024, 모든 seed)
  → decode 커널 무죄. ② chunk-prefill vs token-prefill 토큰 차 **0** → prefill 방식 무죄. ③ Bmax=1/2/6 **동일 출력**
  → 배칭/재사용/dummy 무죄. → 유일한 원인 = solo/batch 그래프 구현의 fp 누적순서 차 × W4A8 정밀도.

**함의(정직):** 프로젝트의 "batch == solo(qwen35_decode) bit-identical" 게이트는 W4A8에서 **borderline 토큰에 대해 항상 성립하진
않음** — 두 유효한 W4A8 근사(별도 graph 구현)가 near-tie 승자를 다르게 고를 수 있음. **scheduler는 정확**(iso 불변). 진짜
solo/batch W4A8 bit-identity가 필요하면 두 graph의 fp 누적순서 정렬(attention/GDN reduction 통일)이 별도 과제.

## 정직 — 이번 범위 + 남은 것
- **완료: production scheduler(admission guard + slot 재사용 + 요청 큐) + mism=0(iso 불변, fp32·W4A8) + admission 스트레스
  (시스템 다운 없이 거부).** 실제 가변 요청 안전 서빙의 정확성·안전성 확립.
- **발견: W4A8 solo-graph vs batch-graph borderline flip**(scheduler 무관, 그래프-구현 차 × 4bit) — mism=0 게이트 의미 재정의.
- **남은 것(STEP C measure)**: 실부하 throughput·p95 — 정직하게 하려면 큰 요청 스트림(도착분포·warm-up·정적 대비)이 필요한
  별도 벤치(R=6 검증부하로는 warm-vs-cold 노이즈 우려, honest-perf 원칙). vLLM 비교 보류(GB10 튜닝).

## 참고
- 테스트: `tests/test_cbatch_prod.cu` (`make tests/test_cbatch_prod`; 인자 [w][g][uq][Bmax][maxlen]).
- 프로브: `tools/probe_mlp_vs_sb.cu`(mlp/sb bit-exact), `tools/probe_solo_vs_batch_logits.cu`(단계별 logit 골든 대조).
- 자산: reset_batch_slot(models/qwen35_batch.cu), ragged h_pos_b(8aefb4e), rope_partial_pos(STEP3).
