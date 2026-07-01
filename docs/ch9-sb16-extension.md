# 챕터9 gemv_w4a8_sb SMAX 8→16 확장 — B=9~15 collapse 제거

> 실부하 벤치(1a7ee9e) 발견: 연속 배칭 active B 요동, **B=9~15가 `gemm_batch_w4a8` 비특수화({1,2,4,8,16}만)
> → default 루프(B× B=1 재적재) → collapse**. SB(1적재)는 SMAX=8에 갇혀 B≤8만 매끄러움.
> 이번 = SB SMAX 8→16 확장으로 9~15도 1적재. de-risk 먼저(occupancy 트레이드오프, gemv-MLP SB 역효과 전례).

## STEP A — de-risk (occupancy, CUDA API 결정적 + 격리 타이밍) `tools/probe_sb16.cu`
| 커널 | regs | occ | S=2 | S=4 | S=8 | S=12 | S=16 |
|---|---|---|---|---|---|---|---|
| SB<8> | 40 | **100%** | 최적 | 최적 | 최적 | — | — |
| SB<16> | 48 | **83%** | 1.00× | 0.91× | 0.92× | **5.85×** | 0.85× |

- **occupancy 100→83%**(acc[16]으로 +8 regs) → S≤8서 SB16이 ~8% 느림 → **S≤8은 SB8 유지 필수**.
- **S=12: SB16이 현행(gemm_batch default 루프) 대비 5.85× 빠름**(314 vs 1840us) — collapse 정확히 해결.
- **S=16: gemm_batch<16>(특수화)가 SB16보다 빠름**(340 vs 398us) → 16은 gemm_batch 유지.
- bit-exact: SB16 == gemm_batch 모든 S(0-diff), SB16 == SB8 (S≤8).

## STEP B — 정식 dispatch (surgical, 회귀 0)
`launch_gemv_w4a8_sb`(gemv_w4a8.cuh): **S≤8 → SMAX8(occ100%) / 9~16 → SMAX16**. batch 경로(qwen35_batch.cu):
`B≤8 → SB8` / **`9~15 → SB16`(신규)** / `≥16 → gemm_batch<16>`(특수화 최적). B≤8·B=16 불변 = 회귀 0 구조적 보장.

## STEP C — before/after (in-engine per-B, `test_batch_decode`, 0.8b W4A8)
| B | BEFORE(fix전) | AFTER(SB16) | 개선 |
|---|---|---|---|
| 8 | 808 (SB8) | 815 | 불변 |
| 10 | **406** (collapse) | 790 | **1.95×** |
| 12 | **399** (collapse) | 819 | **2.05×** |
| 14 | **406** (collapse) | 817 | **2.01×** |
| 16 | 840 (gemm_batch) | 844 | 불변 |

- **collapse 실재 확정**: B=10~15가 fix 전 ~400 tok/s(1적재 절반), SB16으로 ~800(**2×**). B=8·16 불변(회귀 0).
- **이제 B=8~16 total throughput 평탄(~790~844)** — 매끄러운 batch 곡선.

## iso 게이트 (`tests/test_sb16_iso.cu`, B=12 SB16 + B=16 gemm_batch)
- batch decode == B개 single decode: **fp32·W4A8 둘 다 argmax ALL match**(토큰 동일=mism=0).
- W4A8 max-abs 0.319 / MEAN-rel 1.5e-2는 **pre-existing W4A8 batch-vs-single 계통오차**(bf16 attn+양자화; test_batch_decode
  B=4도 동일 0.319 — **SB16 무관·B무관**). B=12(SB16)·B=16(gemm_batch) 값 동일 = SB16이 오차 추가 0(probe bit-exact 재확인).

## e2e 서빙 (`bench_serve`, 정직 nuance)
- **Bmax=12(옛 collapse 영역)**: avgB 8.9~9.66(9~12 dwell)서 연속 **415~443 tok/s 매끄러움**(Bmax=8 459·16 419 동급).
  fix 전이면 이 9~12 스텝 collapse(~400 in-engine)로 aggregate 훨씬 낮았을 것 → **비-2거듭제곱 Bmax 운영점 매끄럽게**.
- **Bmax=16 포화(λ=32)**: 416→419로 거의 불변 — **엔진이 B=16(이미 효율적 gemm_batch<16>)에 머물고 prefill 지배**라
  9~15 transient 짧음. fix 진가는 **9~15 dwell**(중간부하·비pow2 Bmax), pow2 포화선 안 보임(정직).

## 정직 좌표 + 남은 것
- **순 이득, 다운사이드 0**: B=9~15 in-engine 2×(collapse 제거), B≤8·B=16 불변, bit-exact, iso 게이트. 1a7ee9e "cap Bmax=8"
  **완화 → Bmax≤16 전부 매끄러움**. occupancy 83%(S≤8은 SB8이라 무관).
- **남은 것**: B>16(17~31)은 여전히 gemm_batch default collapse(SB SMAX 16까지) — 필요시 SMAX↑(occupancy 더↓ 트레이드오프
  재de-risk). chunked prefill(bench가 prefill 지배 — 서빙 aggregate 개선의 별도 큰 lever). 27B 재측정.
- 자산: probe_sb16(occupancy+bit-exact+타이밍), test_sb16_iso(B=12·16 iso), gemv_w4a8_sb SMAX dispatch.
