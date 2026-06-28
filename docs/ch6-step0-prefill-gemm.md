# 챕터6 STEP 0 — tiled q4 prefill GEMM (엔진의 빠진 절반)

> 2차 크래시 규명에서 드러난 근본 갭: 엔진에 **prefill GEMM이 없다**. gemv_q4(decode용)가 토큰마다
> 가중치 재read → 32K prefill = 1.2PB 트래픽 = GPU 74분 풀로드 → 발열 폭발 + 행/크래시.
> 발열이 "메모리 아니라 컴퓨트"의 물리적 증거(OOM은 즉사, 컴퓨트 폭발은 오래 갈리며 과열).
> 추론 = prefill + decode. decode만 만들었음 → 실서빙(긴 프롬프트/RAG/코드) 불가(첫 토큰까지 74분).

## 문제 (측정 확정)
- `gemv_q4`: 출력당 1워프가 weight row를 read → **토큰마다 weight 전체 재read**(재사용 0).
  prefill Sq 토큰 = weight를 Sq번 read. 27B q4 weight ~37GB × Sq:
  - Sq=2048 → 4.6분, 8192 → 18분, **32768 → 74분**(=2차 크래시). 청크해도 총 트래픽 동일.
- decode(Sq=1)는 weight 1회 read = 정상 → 엔진이 decode 전용으로 최적화돼 있었음(숨은 가정 #3).

## 자산 — 이미 있는 weight-reuse 커널 (챕터3/4, 검증됨)
`kernels/gemm_batch.cuh`:
- **`gemm_batch_q4<B>`**: weight row 1회 로드 + **unpack 1회** → **B행 공유**(acc[B], `wv` 재사용).
  검증됨(test_gemm_batch, 챕터3 isolated 4.3×→in-engine 6.5×). B∈{1,2,4,8,16} 템플릿.
- **`gemm_mw_q4<B,NW,ROWS>`**: 멀티워프(챕터4) — NW워프가 IN 분할 협력 + ROWS 출력행. B≥4서 q4 이득.
- **`gemm_batch_w4a8`**: W4A8 경로(use_q4=2).
- batch decode가 이미 이 디스패치 사용(qwen35_batch.cu:81-83): w4a8/q4(mw→batch)/f32.

**통찰**: batch GEMM의 "B개 시퀀스가 weight 공유" ≡ prefill의 "한 시퀀스의 Sq 토큰이 weight 공유".
**같은 weight-reuse 원리.** prefill = batch GEMM을 "토큰 타일"에 적용. 완전 새 기술 아니라 가진 것의 확장.

## 설계

### 핵심: tiled prefill (GEMM 재사용 + 청크 안전성 결합)
prefill Sq 토큰을 **타일(chunk) 단위**로 처리. 각 타일에서:
- **행렬곱** = batch GEMM(weight 1회 read → 타일 토큰 전부 재사용). 트래픽 Sq → Sq/타일.
- **GDN** = gdn_step이 타일 토큰만 순차(짧은 커널 — 32768회 단일커널 폭발 방지).
- **workspace** = 타일 크기 비례(O(Sq) 폭발 방지). **2번(청크 prefill)이 여기 통합됨.**
- **attention** = 타일 query × (committed KV + 타일) — KV는 decode_state에 누적.

→ tiled prefill = **컴퓨트(GEMM 재사용) + 메모리/안전(청크) 동시 해결.** 2차 크래시의 두 양상 모두 제거.

### 타일 크기 (2축)
- **GEMM 재사용 폭**: 레지스터 acc[B] 제약 → B≤16(검증 커널). 16× 트래픽↓ (74분→4.6분 @32K).
  - 더 큰 재사용(>16)은 shared-memory 타일 GEMM 신규 필요(Phase 2, 필요시).
- **청크(GDN/workspace) 크기**: 더 클 수 있음(예 256~512). 청크 내부서 GEMM은 16씩 sub-tile.
  - 시작: 청크=GEMM폭=16 (단순, 검증커널 그대로). 이후 청크 키우고 GEMM은 16 sub-tile.

### 어디 바꾸나
- prefill 경로의 `mm()`(qwen35_decode.cu, qwen35.cu)가 큰 S에서 gemv_q4 대신 batch GEMM 디스패치.
  decode(S=1)는 gemv_q4 유지(이미 최적). 즉 mm()에 "S 크면 GEMM, 작으면 GEMV" 분기 — 또는 tiled 루프.
- bench prefill = 단발 호출 → 청크 루프(qwen35_decode를 타일씩, commit). decode_state가 KV/state 누적.

## 검증 계획 (헌법)
1. **GEMM 정확성**: tiled-GEMM prefill == gemv prefill == golden, 오차0 (작은 Sq, 안전).
2. **청크 동등성**: 단발 prefill(타일) == decode 누적(1토큰씩) == golden, 오차0 (GDN/conv/KV 청크간 연결 검증).
3. **ncu**: weight 트래픽 sectors/req가 Sq× → Sq/타일× 감소 확인(시간 아닌 지표 — 헌법). occupancy/bank.
4. **0.8B/27B 회귀 0**: 기존 18/18 + 27B/35B forward.
5. **안전**: 단발 Sq 상한 유지하며 작은 Sq부터. 검증 후에야 상한 올리고 큰 context 측정.

## STEP 0 게이트 — 결정 필요
1. **방향: tiled prefill (batch GEMM 재사용 + 청크 결합)** 동의? (컴퓨트+안전 동시, 2번 통합)
2. **Phase 1 = 검증된 gemm_batch_q4를 16-타일로 prefill에 라우팅**(16× 트래픽↓, 저위험) 먼저.
   Phase 2(>16 재사용, shared-mem 타일 GEMM)는 16×로 부족하면. 동의?
3. **타일 크기 시작값 = 16**(GEMM폭=청크=16), 이후 청크 키움. 동의? 다른 시작값?
4. **검증 = GEMM 오차0 + 청크 동등성 오차0 + ncu 트래픽↓ + 회귀0**. 동의?
5. 구현 단위: STEP 1(mm GEMM 라우팅 + 정확성) → STEP 2(청크 prefill 통합 + 동등성) → STEP 3(ncu + 상한 올려 27B/35B context 측정). 동의?
