# M4 4호 — Gemma4 26B-A4B Phase 0 (유연성 4라운드) — 해부 + 판정 + 자산 blocker

> M4 완주 마지막(Gemma family × MoE). ★핵심 질문: 기존 부품 **조립만으로 서는가**(신규 커널 0). 결과:
> **커널 0은 유지되나 "조립만"은 부분 반증** — 26B-A4B는 제3의 아키텍처(dense∥MoE 병렬)라 새 graph + config 3종 필요.
> ★가중치 미다운로드(config만 해부) → 변환/골든/서빙은 50GB 다운로드-gated 후속.

## STEP A — 해부 (HF config만, 추측 0)

가중치 로컬 부재(캐시 refs 스텁만) → `hf_hub_download config.json`으로 config만 확보 + 설치된 transformers 5.12.1
`modeling_gemma4.py`로 router/expert 규약 확정(다운로드 불요).

**구조(gemma4_text MoE)**: H2816 L30(full 5 / sliding 25, 31B와 동형 혼합) V262144 **tied**, NQ16 hd_s256/hd_f512
nkv_s8/nkv_f2, **k_eq_v**·**GeGLU**·**softcap 30**·**dual RoPE**(full θ1e6 proportional / sliding θ1e4)·embed×√H.
**GDN 없음**(순수 sliding/full attention — Gemma). **MTP 없음 → plain**(spec 불가).
**MoE**: E=128, top_k=8, moe_int=704. **dense MLP intermediate=2112(병렬)**.

### ★router 규약 (예고된 분기 — 확정)
`modeling_gemma4.py Gemma4TextRouter.forward`:
```
h = RMSNorm(no-scale)(h); h = h * self.scale * H^-0.5     # ★router 자체 norm + scale (Qwen엔 없음)
scores = proj(h); probs = softmax(scores, 128)
top_w, top_i = topk(probs, 8)
top_w /= top_w.sum()                                       # ★renorm — Qwen(norm_topk_prob)과 동일 (router_topk.cuh 가정 성립!)
top_w = top_w * self.per_expert_scale[top_i]               # ★per-expert 학습 스칼라 (NEW — Qwen엔 없음)
```
- **renorm: ✓ Qwen과 동일** → router_topk.cuh 재사용 가능(예고된 "renorm 안 하면 config 분기" 우려는 기우).
- **NEW 2종**: (a) router 자체 norm+scale(RMSNorm no-scale + ×scale×H^-0.5), (b) per_expert_scale[128] 곱(topk 후).

### ★디코더 층 위상 (제3의 구조 — 예상 밖)
`Gemma4TextDecoderLayer.forward`: **dense MLP + MoE를 병렬 실행 후 합산**:
```
h = mlp(h)                                    # dense GeGLU (intermediate 2112) — 항상 실행
if enable_moe_block:
    h1 = post_ff_norm_1(h)                     # dense 경로 post-norm
    h2 = experts(pre_ff_norm_2(residual), router(...))   # MoE 경로 (128e top8, shared 없음)
    h2 = post_ff_norm_2(h2)
    h = h1 + h2                                # ★dense + MoE 합산 (Qwen: MoE가 MLP 대체 / 31B: dense만)
```
- **shared expert 없음**(Qwen 35b엔 있었음) — 대신 병렬 dense MLP.
- **추가 norm sandwich**: pre_ff_norm + post_ff_norm_1 + pre_ff_norm_2 + post_ff_norm_2 + post_ff_norm.

### roofline (active-bytes, config 기반)
active = lm_head(0.74B) + attn(0.97B) + **dense_mlp(0.54B, 병렬)** + router(0.01B) + experts(1.43B) = **3.68B**
→ single 상한 **139.7 t/s** (35b-a3b 165.3·active 3.11B보다 낮음 — dense MLP가 experts와 병렬로 active↑).
※ plan.py active_params는 Qwen-MoE 형상(full+GDN, MoE-replace)이라 이 모델(sliding-attn + 병렬-dense)엔 수동 산출.

## ★유연성 4라운드 판정 (③ + 조립 가설)

| 요소 | 판정 |
|---|---|
| 신규 커널 | **0**(전부 재사용: rmsnorm_noscale·scale·moe_fused·router_topk·GeGLU·softcap·dual-rope·attn sliding/full) |
| ③ 커널 내부 모델분기 | **0**(설계 — 전부 config/graph/인자) |
| ★"조립만으로 선다" | **부분 반증** — 부품(커널)은 다 있으나 **새 graph 필요**(dense∥MoE 병렬 위상) |
| 필요한 config/② 확장 | 3종: (a) router-own-norm+scale (b) per_expert_scale 곱(dw에) (c) no-shared MoE variant(moe_fused 조건화) + 병렬 dense combine |

**결론**: "커널 라이브러리 + 얇은 graph + config" 헌법은 **HOLDS**(③=0, 신규 커널 0) — 모델 차이가 graph/config에 격리됨.
단 26B-A4B의 graph는 **순수 조립이 아닌 새 위상**(dense∥MoE)이라 3호(35b, 기존 graph에 moe 분기 1줄)보다 graph 작업 큼.
즉 유연성 곡선: same-family+same-paradigm(27b→35b)=조립, **new-paradigm(dense∥MoE)=새 graph(커널은 재사용)**.

## ★자산 blocker — 변환/골든/서빙은 다운로드-gated (stop-report)

- 가중치 로컬 **미다운로드**(config만 3KB 확보, 본체 ~50GB bf16 미다운로드). STEP B(변환)·C(골든)·D(서빙)는:
  1. 50GB HF 다운로드(gemma-4-26B-A4B-it — config 확보로 접근 가능 확인, gated 아님).
  2. q4 변환(extract_weights Gemma MoE expert 텐서 — prefix 자동감지 확장).
  3. **새 graph 구현**(gemma4 MoE 디코더 층: dense∥MoE + router-norm + per_expert_scale + no-shared).
  4. 골든(HF reference — 26B MoE, 메모리 가드 필수 — 31B fp32가 118GB 근접한 전례).
- = deliberate 대형 세션(다운로드 + 새 graph + 메모리-가드 골든). 헌법(막히면 보고·무리한 작업 몰지 않기)대로
  **STEP A 완결로 이번 종료**, B/C/D는 자산 준비 + graph 구현을 다음 세션으로 정직히 분리.

## 파이프 곡선 (4호 개입 — 정직)
- 3호(35b, qwen-family same-paradigm): 서빙 코드 0(조립).
- 4호(26b, gemma-family new-paradigm dense∥MoE): 해부/router는 기존 도구로 즉시(개입 소), **but graph는 새로**
  (dense∥MoE 위상) + 가중치 다운로드. = **개입이 3호보다 다시 오름**(new-paradigm 비용) — M7 곡선의 정직한 데이터점:
  자동화 절감은 **paradigm 공유 시**, 새 paradigm(구조 위상)은 여전히 graph 비용.

## 다음
- STEP B/C/D: 50GB 다운로드 → 변환 → gemma4 MoE graph(dense∥MoE·per_expert_scale·no-shared·router-norm) → 골든(메모리 가드).
- 완주 시 M4 4모델 완결. 이후: 서빙 스케줄러(전 모델 63≪195 lever)·M3.
