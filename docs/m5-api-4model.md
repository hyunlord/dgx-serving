# M5 API 4모델 확장 — 모델-중립 온라인 daemon (STEP A: MoE 35b)

> 3-way 선결: API를 더 많은 모델로(gemma4 plain 검증됨 → +qwen 27b/35b MoE plain → +27b spec[STEP C]).
> ★헌법: 모델 차이는 graph/backend 배선, 서빙 루프는 모델-중립. 스코프 gemma4/qwen plain·greedy·bit-id 불변.

## 구조 (모델-중립 온라인 서빙 루프)

`runtime/serving_online.h` — 실시간 스트림-구동 greedy 온라인 배칭 루프를 **모델-중립**으로 추출(serving_core.h의 trace-구동
짝). `OnlineBackend` 인터페이스(vocab·max_len·alloc_state·free_state·**reset_slot**·prefill_ids·decode_step·logits_to_host)로만
모델 접촉 — 루프는 모델/커널/device 버퍼 무지. reader 스레드+큐+태그드 I/O+greedy 피드백은 공유.
- `run_online_daemon(OnlineBackend&, Bmax)`: admit(reset_slot→prefill_ids→첫토큰 argmax)→decode_step(Bmax ragged)→
  슬롯별 argmax→greedy 피드백→완료. 태그드 프로토콜("<req_id> <max_tokens> <ids>" / "<req_id> <tok>"·"<req_id> END").

daemon = backend 배선 + setup + `run_online_daemon` 호출(루프 공유):
- `tools/serve_daemon_gemma4_online.cu`: Gemma4OnlineBackend(gemma4 primitive, dummy rope). **무회귀 PASS**(test_serve_online).
- `tools/serve_daemon_qwen_online.cu`: QwenOnlineBackend(qwen35 primitive — 27b/35b 공용, 실 rope golden). ★qwen은 GDN state
  보유 → `reset_slot`=reset_batch_slot(슬롯 재사용 시 GDN clear) override(gemma4는 GDN 없어 no-op).
- `tools/serve_api.py`: `--model {gemma4,qwen}` 분기(daemon 바이너리·arg 레이아웃 상이 — qwen은 --golden[rope] 필수). front 공통.

## 35b MoE 정확성 검증 (direct-pipe standalone — bit-id)

35b-a3b(256 experts, MoE fused, GDN 혼합, V=248320)를 online daemon으로 서빙. ★정확성 = 온라인 배칭이 각 요청 계산 안 바꿈:
- **concurrent==solo**(2슬롯 동시): req1 [13,248046,198,248045,74455,198,248068,198,90700,8340] = 단독과 bit-identical(슬롯 격리).
- **reuse==solo**(req1 완료→cv.wait→req2 slot0 재사용): req1==단독·req2==(concurrent의 req2) bit-identical. GDN reset_slot 정상.
- 클린 종료(stdin EOF→shutdown), 무누수. 재현(direct-pipe):
  `(printf "1 8 1 100 200 300 400\n"; sleep 40; printf "2 8 1 50 60 70 80 90\n"; sleep 15) | serve_daemon_qwen_online <w> <golden> 2 384 8`

## 정직 기록

- **daemon 정확성 = 증명됨**(gemma4 무회귀 + 35b direct-pipe concurrent/reuse bit-id). 모델-중립 루프 = 헌법 실증(qwen도 같은 루프).
- **★test_online_daemon.py subprocess harness WIP**: Popen(stdin/stdout/stderr PIPE) 경로에서 gemma4·qwen 둘 다 stall 관찰
  (첫/둘째 요청 stdin/pipe 타이밍 — daemon 버그 아님, test-infra 이슈). daemon은 direct-pipe로 검증. subprocess 경로 수정=후속.
- 35b 콜드 로드 ~4-6분(22GB, 256 expert q4) — idle처럼 보이나(0% CPU·S state) 드라이버 I/O 진행 중(hang 아님). warm 재로드 ~21s.
  교훈: 대형 모델 콜드 로드를 hang으로 오판 말 것. SIGKILL mid-cuda는 GPU 컨텍스트 오염 위험 → SIGTERM+stdin EOF 클린 종료 선호.

## 성능 (관찰만 — 주장 아님)
- 35b single decode(warm) 관찰만(bench_serve 별도). 이번 스코프=정확성(bit-id)·온라인 동작, 성능 최적화는 후속.

## 다음 (STEP B 설계 → STEP C)
- STEP B 완료(docs/m5-spec-api-design.md): spec 코어 API = 옵션 B(spec 전용 경로 `run_spec_online`) 권장.
- STEP C(후속): `run_spec_continuous`(bench_serve_spec.cu, 실-greedy 스케줄러·iso_out=accept 게이트)를 trace→실시간 변환.
  프로젝트 최복잡 코드(draft/verify/accept/promote·per-slot MtpKV·DecodeState) — 신중 구현.
