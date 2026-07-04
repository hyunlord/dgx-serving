# M5 spec 코어 API 설계 (정립 노트 2C — spec 경로 API 노출)

> STEP B = 설계 먼저. spec(speculative decode) 코어를 온라인 서빙 daemon 뒤에 노출하는 방법을 확정한다.
> STEP C(구현)는 이 설계 위에서. ★무리한 통합 금지 — 스텝 구조가 근본적으로 다르면 안전한 쪽을 택한다.

## 문제: plain 루프 전제 = "1 스텝 = 슬롯당 +1 토큰"

방금 만든 모델-중립 온라인 루프 `runtime/serving_online.h`의 `run_online_daemon(OnlineBackend&, Bmax)`:
- admit(reset_slot+prefill_ids+첫 토큰 argmax) → `decode_step`(Bmax ragged) → **슬롯별 argmax 1개** → greedy 피드백 → pos+1·gen+1 → 완료.
- ★핵심 전제: **decode_step 1회 = 슬롯당 정확히 +1 토큰**. 그래서 루프가 슬롯마다 1토큰씩 스트림·전진.

spec 코어(`tools/bench_serve_spec.cu`)는 스텝 구조가 다르다:
- 슬롯별 **MTP draft(tree K,W)** → 슬롯당 K+1개 후보 토큰(트리).
- **`qwen35_verify_batch`**(ragged concat — 활성 슬롯 노드를 off[b]/base_b로 concat, 한 forward) → 검증.
- 슬롯별 **accept/promote** → **슬롯마다 가변 개수(1~K+1) 토큰 확정**(ragged accept). = 스텝당 슬롯별 전진량이 다름.
- 슬롯 상태 = per-slot `DecodeState`(main KV/GDN) + `MtpKV`(draft KV, 수명=요청). plain의 `BatchDecodeState` 슬롯보다 무겁다.

즉 spec은 "1 스텝 = 슬롯당 1~K+1 토큰(가변)". plain 루프의 "슬롯당 +1"과 안 맞는다.

## 두 설계 옵션

### 옵션 A — 통합 인터페이스 (decode_step이 가변 accept를 반환)
`serving_backend.h`의 `has_spec()` 주석 확장점: "spec backend는 decode_step 내부에서 사이클 캡슐화, 슬롯당 가변
accept를 out-배열로 반환 → 코어 slot-advance가 그 배열을 읽음."
- `OnlineBackend.decode_step`을 확장: 슬롯별 **확정 토큰 리스트**(1~K+1)를 out-배열로 반환. argmax를 backend 내부로.
  - plain backend: 항상 1토큰/슬롯(내부 argmax) 반환 → 특수 케이스(accept=1).
  - spec backend: draft→verify→accept 사이클 내부 캡슐화 → 1~K+1토큰/슬롯 반환.
- 루프: 슬롯별로 확정 토큰 N개 스트림 + pos+=N·gen+=N. **스텝-종류 무지**(plain=spec=같은 루프).
- 장점: 루프 완전 통합(모델-중립 극대화), 확장점 설계와 정합. 단점: **plain backend도 리팩터**(argmax를 backend로 이동),
  prefill 첫토큰도 인터페이스에 맞춰야, spec의 무거운 상태(MtpKV·per-slot DecodeState·draft/verify 머신)를 decode_step 뒤로
  전부 숨겨야 = 인터페이스가 spec 복잡도를 다 흡수. 리스크: 검증된 plain 루프를 건드림.

### 옵션 B — spec 전용 daemon 경로 (권장)
별도 `run_online_daemon_spec(SpecBackend&, Bmax)` 루프(spec 전용 스텝). reader 스레드·큐·태그드 I/O·admission은
`serving_online.h`의 공유 헬퍼로 재사용, **스텝 본체만 spec 사이클**(draft→verify→accept, 가변 전진). API front 공통
(태그드 프로토콜 동일). daemon 바이너리가 분기: plain 모델→`run_online_daemon` / spec 모델→`run_online_daemon_spec`.
- 장점: **검증된 plain 루프 무손상**(리스크 격리), spec 사이클이 자기 루프에 명시적(가독), 인터페이스 변경 최소.
- 단점: reader/큐/스케줄 인프라 일부 중복(공유 헬퍼로 완화 가능), 루프 2개 유지.

## 추천: 옵션 B (spec 전용 경로)

근거(헌법 "무리한 통합 금지·spec 신중·정확성 불변"):
1. spec 스텝 구조가 plain과 **근본적으로 다름**(가변 accept·MtpKV·ragged verify concat) — 통합(A)은 검증된 plain 루프에
   spec 복잡도를 억지로 얹어 회귀 위험. B는 plain 무손상.
2. **정확성 최우선**: spec은 accept "곱" 이득 보존 + bit-id가 걸려있어(borderline flip 위험, honest ⑥) 격리된 경로에서
   신중히 검증하는 게 안전. A는 plain/spec 공유 코드라 한쪽 버그가 양쪽 오염.
3. 인프라(reader/큐/태그드)는 `serving_online.h` 헬퍼로 공유 → B도 중복 최소. 나중에 안정화되면 A로 통합 가능(역은 어려움).

## STEP C 구현 시 핵심 위험 지점 (막히면 보고 대상)

1. **실 greedy 피드백**: bench_serve_spec은 dummy 토큰(trace-구동). 실서빙은 accept된 마지막 토큰을 다음 draft 입력으로
   피드백해야(MTP draft는 마지막 committed 토큰→K+1 draft). = spec의 실 greedy 루프(bench엔 없음).
2. **per-slot MtpKV + DecodeState 수명**: 요청마다 자기 MtpKV(draft KV)+DecodeState(main KV) 필요. admit=alloc·
   complete=free. plain(BatchDecodeState 슬롯)보다 상태 관리 복잡.
3. **동적 배치 하 ragged verify concat**: `qwen35_verify_batch`의 Nnodes_b/base_b/off[b]를 활성 슬롯 변동마다 재구성
   (실시간 도착/완료로 활성 집합 변화 → 스텝마다 concat 레이아웃 재계산).
4. **accept 곱 보존 bit-id**: API 경유 accept == 직접 spec 서빙 accept(스킴 내). 게이트는 토큰뿐 아니라 **슬롯별 accept 수**도 대조.
5. **borderline flip(honest ⑥)**: 게이트는 spec 스킴 **내** bit-id(API spec==직접 spec), spec==plain 아님(다른 graph=flip 가능).

## 정립 노트 2C 갱신
- 공개 API 표면: plain(gemma4·qwen 27b/35b) = `run_online_daemon`(검증됨). spec(27b) = `run_online_daemon_spec`(별도 경로, STEP C).
- front(serve_api.py)는 태그드 프로토콜 공통 — plain/spec daemon 바이너리만 다름(--model 확장). 계산 불변(front 무지).
- 미해결→해소: "spec 코어 API 노출"(정립 노트 2C) = 옵션 B 확정. 곱(accept) 보존이 게이트 핵심.
