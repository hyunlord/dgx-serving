# 챕터9 continuous batching — ragged-batch(per-seq 위치) 지원 + 검증

> odd-batch 수정(2959c17)으로 가변 B 커널 준비. 연속배칭은 요청이 매 스텝 진입/이탈 → **요청마다 다른 위치(ragged)**.
> 현 batch decode는 same-length(단일 n_past). STEP A의 진짜 전제 = **ragged-batch 지원**(커널/graph 층). 이번 = 그것 + 최소 scheduler로 검증.

## 발견 — 현 batch decode는 same-length 전용
`qwen35_decode_batch`: 단일 `base=st.n_past`를 모든 seq가 공유(rope, KV append, attention 모두 base). `st.n_past+=1`
(전 seq 동시). 연속배칭(요청마다 다른 위치)엔 부적합 → ragged 지원 필요.

## ragged-batch 지원 (per-seq 위치)
`qwen35_decode_batch`에 `h_pos_b[B]`(seq별 위치, null=기존 same-length) 추가:
- **rope**: ragged면 `launch_rope_partial_pos(d_pos)`(STEP3 자산, per-row 위치) — row b = seq b 위치. else 기존 batch rope.
- **KV append + attention**: 이미 per-b 루프 → `base` → `POS(b)`(seq b 위치). 각 seq 자기 KV[0..POS(b)] attend.
- **GDN/conv**: 위치무관(recurrence state carry) → 변경 없이 ragged 호환.
- `st.n_past`: ragged면 불변(호출자/scheduler가 per-seq 위치 관리).
커널 재사용(SB·B·rope_pos), 위치만 per-seq. same-length 경로(h_pos_b=null) 회귀 PASS(불변).

## 최소 scheduler + 검증 (`tests/test_cbatch.cu`)
2 요청: A(step0 시작), **B(step Kenter=6 staggered 진입)** → A는 pos~6, B는 pos0 = **다른 위치(ragged)**. st.B 동적(1→2).
각 slot: 프롬프트 1토큰/step ingest(=prefill, 청크==단발 검증됨) 후 decode. h_pos_b=slot별 위치.
**검증: batch-생성 == solo greedy bit-identical.** → **A=0/10, B=0/10 mism (fp32·W4A8 둘 다).**
즉 **ragged 위치 정확 + staggered 진입서 이웃/진입시점 무관**(A가 B 진입에 안 흔들리고, B가 A KV 안 읽음).

**버그 교훈**: 첫 시도 B=8/10 실패 — slot compaction 오류(A 종료 후 st.B=1로 줄이니 B[물리 idx1]가 idx0=A KV 읽음).
수정: st.B가 시작된 물리 slot 다 덮게 유지(종료 slot은 dummy·무시). 물리 slot 고정이 핵심.

## 정직 — 이번 범위 + 남은 것
- **완료: ragged-batch graph 지원(per-seq 위치) + 최소 scheduler(staggered 진입) + mism=0 검증(fp32·W4A8).** 연속배칭 핵심 정확성.
- **남은 것(production scheduler)**: ① **KV 메모리 예산 admission guard**(unified oversubscribe=시스템 다운 — 최우선; 이번 최소 테스트는 B=2 소규모라 미스트레스) ② slot 재사용/compaction(요청 큐, 종료 slot에 새 요청) ③ chunked prefill(현 1토큰/step ingest는 긴 프롬프트 느림) ④ STEP C: 실부하 throughput·p95 latency(정적 대비).
- vLLM 비교 보류(GB10 튜닝 필요).

## 다음
production scheduler(KV 예산 가드 최우선 + slot 재사용 + 요청 큐) → e2e throughput·p95. ragged 커널 토대는 검증됨.
