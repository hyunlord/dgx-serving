# 챕터5 STEP 2.4/2.5 — 비트폭 일반화 + 측정 하네스 정교화 설계 메모 (진단)

> 구현은 다음 챕터. 이번엔 "어디를 어떻게 바꿔야 하나" 진단만. (헌법: 토대 진단, 구현 분리.)

## ⚠️ 27B q4 검증 상태 (선결) — GDN 보편화 갭 (STEP 2.3 완료=갭 발견)
추출(q4 400 tensors)·골든(fp32-CPU err-0)은 완료. 검증 `test_27b`는 **L1(첫 GDN)서 NaN** — 정체는
**GDN GQA 그룹핑**(27B k=16/v=48, 0.8B k=v=16 우연). graph+gdn_step이 `num_k_heads==num_v_heads` 암묵 가정.
→ **27B q4 err-0 완결은 GDN 그룹핑 수정(커널/graph 변경) 필요** = 이 의뢰서의 "커널 수정=멈추고 보고"에 걸림.
직전 (가) 승인(GDN 그룹핑 지원)과 이 의뢰서(진단-only)가 충돌 — 사용자 결정 필요(아래 게이트).

## STEP 2.4 — 비트폭 일반화 진단 (q2/q1 토대)

### 현 q4의 비트폭(4) 하드코딩 위치 (3곳)
| 위치 | 4비트 박힌 곳 |
|---|---|
| **추출**(extract_weights_qwen35 quantize_int4) | `scale=maxabs/7`(7=2³−1), `q=round(w/s)+8`(8=2³), `clip(0,15)`, `pack 2/byte`(low/high nibble) |
| **커널**(gemv_q4/gemm_batch_q4/w4a8) | `Wq+o*(IN/2)`(2 weight/byte), `w[i>>1]`+`&0xF`/`>>4`(nibble), `(q−8)` |
| **골든**(make_reference dequant_roundtrip) | 동일 unpack(nibble, −8) |
- config엔 비트폭 필드 없음(QBLK만 컴파일타임). 비트폭은 q4로 암묵 고정.

### q2/q1 추가 시 바뀌는 것 (파라미터 `nbits`)
- pack: 8/nbits weight per byte (q4=2, q2=4, q1=8). `IN*nbits/8` bytes.
- zero-point: `2^(nbits−1)` (q4=8, q2=2, q1=1). scale denom: `2^(nbits−1)−1` (q4=7, q2=1, q1→대칭불가).
- unpack: mask `(1<<nbits)−1`, shift `nbits*k`. 커널이 nibble(4) 대신 nbits로.
- W4A8 DP4A: int4×int8 dp4a는 4비트 전제 — q2는 dp4a 재설계 또는 fp 경로(복잡).

### 구조 제안 (다음 챕터)
- `QMat`에 `int bits` + manifest `weight_bits`. 커널을 `template<int BITS>` 또는 runtime nbits dispatch.
- pack/unpack 헬퍼를 `kernels/quant_pack.cuh`로 분리(추출/커널/골든 공유 = 단일 진실).
- **q2 품질 경고(필수 기록)**: 순진한 per-block round q2(4단계)는 대부분 망가짐. 실용 서빙 품질엔
  **GPTQ/AWQ/importance-aware**(2차정보·outlier 보호) 필요 = **큰 별도 챕터**. q1은 post-training 보통 불가(한계 측정용).
  → q2 "구현"은 쉽지만 "실용 품질"은 정교한 기법이 본체.

## STEP 2.5 — 측정 하네스 정교화 진단 (서빙 현실)

### 현 측정의 구멍
- `test_batch_decode`: **prompt 15토큰(짧은 context) + N=96 decode**, total/per-seq tok/s만.
- **TTFT 없음**(prefill 시간 미측정), **context 스윕 없음**, **출력길이 고정(96)**.
- ⚠️ **rope 테이블 128 상한**(chapter3) — 현재 긴 context 측정 **불가**(maxlen≤128). 긴 context 전제조건.
- ⇒ "B=8 980 t/s"는 **짧은 context 한 점**. 실서빙(긴 프롬프트/대화)·3-way 공정 비교엔 부족.

### 서빙 현실 측정축 (설계, 다음 챕터 구현)
| 축 | 의미 | 왜 |
|---|---|---|
| **TTFT** | prefill 시간(context C) | 첫 토큰 지연 = UX. context에 민감(prefill은 C에 선형~). |
| **TPOT / decode t/s** | 토큰당 시간(context C) | KV 커지면 attention이 긴 KV 읽어 느려짐 — C별 측정 필수. |
| **context 스윕** | 128/2K/8K/32K | 27B는 긴 context 가능. 엔진별 long-context 강약. |
| **출력 길이** | 96/512/2K | decode 길면 KV 자람 → TPOT 변화. |

### 전제조건/구현 메모 (다음 챕터)
- **rope 테이블 확장**: 128 → 32K position (현 maxlen 병목 해소). config max_seq_len 주도.
- **prefill 타이밍**: qwen35_forward(또는 batch prefill)을 타이밍 + TTFT 분리.
- **KV 긴 context**: KV 캐시 max_len을 context 스윕 상한으로. 27B KV 메모리(긴 context×B) 예산 확인.
- 측정 구조: `bench` 하네스 — (모델, context C, 출력 N, B) → (TTFT, TPOT, total t/s). 3-way 공통.
- **3-way 재비교는 같은 (C,N,B)** 조건으로 — chapter3 "같이 뽑기" 교훈을 context까지 확장.

## 게이트 (사용자 결정)
1. **27B q4 완결 vs 충돌**: GDN 그룹핑 수정(직전 가 승인, 커널/graph 변경)을 **지금** 하나, 아니면 이 의뢰서대로
   진단-only로 두고 GDN은 별도 STEP? (q4 err-0은 GDN 수정 없이는 불가.)
2. 비트폭 일반화 구조(QMat.bits + quant_pack.cuh + nbits dispatch) — 방향 동의? (구현은 다음.)
3. 측정 하네스(TTFT/TPOT/context 스윕 + rope 32K 확장) — 방향 동의? (구현은 다음.)
