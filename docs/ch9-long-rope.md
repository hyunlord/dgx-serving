# 챕터9 long-rope 테이블 — 긴 context(pos≥128) 정확 서빙

> chunked prefill(7e70dc1) caveat: rope 테이블 **128행**(reference forward 캡처 길이) → pos≥128 timing-only(부정확).
> 목표 모델 native context **262144**(RAG·긴 대화·코드). long-rope = chunked prefill 정확성 완성 + 긴 context 서빙.

## STEP A — rope 정의 파악 (추측 금지)
- **manifest**: rope_theta=**1e7**, rotary_dim=**64**(partial_rotary_factor 0.25 × head_dim 256), scale 없음.
- **Qwen3.5/3.6 config 전부 `rope_scaling = null`** (YaRN/NTK 아님), max_position_embeddings=262144.
- 기존 128행 테이블은 HF 모델 forward의 `position_embeddings` 캡처(golden). scaling 한계 아니라 **reference 길이 = 128**.
- rope는 **위치만 의존**(토큰 무관) → 위치 0..max_len 계산으로 확장 가능.

## STEP B — 공식==모델 rope 증명 (골든, `tools/gen_rope_long.py`)
표준 RoPE(rope_scaling=None): `inv_freq[i]=1/theta^(2i/rot)`, `emb=cat([pos⊗inv_freq, pos⊗inv_freq])`, cos/sin, attention_scaling=1.0.
**transformers와 동일 fp32 계산** 필수(fp64면 3.4e-6 차 — fp32면 일치).
- **pos<128 검증: max|Δcos|=max|Δsin|=5.96e-08 (1 ULP) → BIT-IDENTICAL** = 공식==모델 캡처 golden.
- **rope_scaling=None(위치-균일) + pos<128 1-ULP 일치 → pos≥128 확장 정확(golden by formula-equivalence).** 어떤 scaling이라도
  저위치 perturb → 1 ULP 일치가 scaling 없음의 실증. (YaRN이면 이 공식 수정 필요했을 것 — config·측정 둘 다 None 확인.)
- 0.8b·27b 동일(같은 theta/rotary_dim). ext 테이블 8192행 덤프(rope_cos_ext.bin, 2.1MB — *.bin gitignore, 생성물).

## STEP C — 엔진 long-context 검증 (`tests/test_longrope.cu`)
ext 테이블로 **P=256(>128)** prefill: chunked prefill(prefill_batch_slot) == 1토큰/step prefill == (같은 ext rope, KV).
- **fp32·W4A8 0/8 mismatch** → 엔진 long-context 경로(attention/KV pos≥128) **자기일관 + 정확한 rope**.
- rope 테이블 정확(1 ULP 증명) + 엔진 pos≥128 자기일관 = **chunked prefill P>128 이제 정확**(timing-only → 정확).
- bench_serve가 ext 테이블 자동 로드(있으면 maxlen행, pos≥128 정확; 없으면 long 128 타일 근사+경고).

## 정직 좌표 + 남은 것
- **완료**: rope 정의 파악(scaling=None) + 공식==모델 1 ULP 증명(golden) + ext 테이블(8192행, 0.8b·27b) + 엔진 pos≥128
  자기일관(test_longrope P=256, chunked==1토큰). chunked prefill 긴 프롬프트 **빠르게+정확하게**.
- **골든의 성격**: rope 테이블 정확성은 **formula-equivalence**(공식==캡처 golden 1 ULP + scaling=None config)로 증명 —
  transformers >128 forward 샘플 대조보다 정밀(1 ULP, 전 위치). 
- **남은 것**: ① 절대 e2e golden — transformers를 P>128 프롬프트로 forward해 엔진 전체(rope+attention at scale) 대조
  (attention 대규모 numerics 확인, rope와 별개; HF 파이프라인 재실행 필요) ② **attention 12K shared 벽**([[attn-12k-shared-wall]]):
  현 attn 커널 context >~12K launch 실패 — 262K native엔 FA 필수(전제, 8192까진 검증됨) ③ **엔진 런타임 rope 생성**
  (현재 파일 로드 — production은 config에서 init 시 생성) ④ 긴 context KV admission 예산(max_len↑ → per-seq KV↑, guard 갱신).
- 자산: gen_rope_long.py(공식+검증+덤프), test_longrope.cu, bench_serve ext 로드, rope.cuh(launch_rope_partial_pos).
