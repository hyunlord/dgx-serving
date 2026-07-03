# M4 MoE — Qwen3.5-35B-A3B Phase 0 (유연성 3라운드) — 검증 + roofline

> ★발견: MoE 35B-A3B Phase 0(해부·변환·golden)은 **챕터5에서 이미 구현·검증 완료**(M4 dense 트랙 이전).
> 이 세션 = greenfield 재구현이 아니라 **현재 상태 검증(챕터7-10+M4 이후 무회귀) + 진짜 갭(plan.py MoE roofline) 보강**.
> (헌법: 대상을 보고 설명과 모순되면 표면화 — 브리프의 greenfield 가정과 실제 자산 상태가 달랐음.)

## 기존 자산 (챕터5, 6/28)

- `artifacts/qwen35-35b-a3b/`: weights(22GB q4) + golden(fp32_hs_0..40 층-스트리밍 + router_logits + moe_in/out + shared).
- `kernels/router_topk.cuh`(router top-k, 모델독립) + `models/moe_block.cuh`(MoE MLP graph, config 주도).
- engine `MK_QWEN3_5_MOE` 통합(백본 커널 공유 + MoE graph 분기). 골든 게이트 3종.

## 해부 (manifest, 추측 0)

`qwen3_5_moe`: H2048 L40 V248320(untied), **experts 256·top_k 8·moe_int 512·shared_int 512·norm_topk_prob=true**.
GDN 혼합: linear_attention 30 : full_attention 10 (interval-4, 27B와 동형). GDN k16/v32·hd128·conv4.
**MTP head 없음** → spec 불가(Gemma4처럼 plain 서빙 — "spec 곱 스토리" 미해당, 정직).
profile: `tools/autotune/profiles/qwen35-35b-a3b.json`.

## 골든 게이트 — 현재 전부 PASS (챕터7-10+M4 이후 무회귀)

| 게이트 | 결과 | 의미 |
|---|---|---|
| test_router (expert 선택 격리) | **idx 불일치=0** / weight err 1.49e-8 / sum-to-1 5.96e-8 | ★브리프의 expert-index 게이트 — 커널 top8→softmax8 == HF softmax256→top8→renorm |
| test_moe_expert (expert FFN) | rel 6.99e-7 (max 3.38e-9) | gate_up [gate;up] 블록분리 + SwiGLU 재사용 정확 |
| test_35b (full forward) | **argmax ALL** / last-tok 11751=gold / MEAN-rel 6.64e-2@L40(bf16 40층 bound) | MoE 전체 forward 보편화 3번째 |

**★expert-index 게이트(브리프 핵심)는 test_router로 이미 존재·통과** — idx 불일치=0 = router top-k 선택이 HF와 완전 일치
(argmax만으론 가려질 라우팅 오류를 직접 배제). tie-break: norm_topk_prob renorm 규약 일치(sum-to-1 5.96e-8).

## roofline — MoE active-bytes (★수배 상한, 이 세션 신규)

plan.py에 `active_params()` + MoE-aware roofline 추가. **decode bytes = 활성분만**(전체 상주 ≠ 이동):
- 활성 = lm_head(V×H) + Σlayer(attn/GDN + router H×E + shared 3·H·SI + **top_k · 3·H·MI**).
- **active_params_b = 3.11B** (/ total 35.0B) → single_decode 상한 **165.3 t/s**.

**대조**: dense 27B roofline 19.0(실측 11.9=62.5%), Gemma4 9.26(53.3%). MoE 165.3 = dense의 **8.7×** —
"decode bytes=활성 파라미터만"의 GB10 정합(128GB unified가 전체 256 expert 상주, 이동은 top-8만). ※상한이며
예측 아님(GDN state·expert scatter·라우팅 오버헤드로 하회 — 실측은 서빙 세션). 27b/gemma4 roofline 회귀 유지(19.0/17.4).

## ③ 유연성 3라운드 판정

| 라운드 | 대상 | ③(커널 내부 모델분기) | 비고 |
|---|---|---|---|
| 1 | 커널 라이브러리 | 0 | dense(27B/Gemma4) |
| 2 | 서빙 코어 | 0 | serving_core.h |
| 3 | **MoE(35B-A3B)** | **0** | moe_block=config 주도(E/K/MI/SI), router_topk=모델독립 |

**문서화된 ② 1건**: router_topk renorm 가정(Qwen3.5-MoE는 top_w/=sum 함 → 항등 성립). 다음 MoE(Gemma4 26B-A4B)가
renorm 안 하면 config 분기 필요(router_topk.cuh:8 주석에 명시) — ③ 아니라 ②(예상된 config 확장). ROUTER_MAX_E 512는 컴파일 상한(QBLK류).

## 정직 스코프 / 다음

- Phase 0 = 정확성만(성능 측정 없음). 골든 스위트 PASS + roofline 좌표(165.3)가 산출물.
- MTP 없음 → spec 불가(plain). 다음: MoE 서빙/성능 세션(active-bytes 실측 vs 165.3, expert scatter batch 처리),
  레시피. 26B-A4B(Gemma MoE — router renorm 규약 확인 필요). 유연성 3라운드 = 헌법 유지(③ 커널·서빙코어·MoE 전부 0).
