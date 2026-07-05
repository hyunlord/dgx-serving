# 1단계 블루프린트 — 공용 커널 + SmolLM3 graph

> 레퍼런스(`<REF_ENGINE_REPO>` = 레퍼런스/이식 출처 repo(사용자 로컬), 특히 `full_engine.cu`, `quantize_*.py`, `dump_full.py`)에서
> 추출한 정확한 사양. 이식의 기준 문서. compaction 견디게 여기 박아둠.

## SmolLM3-3B config (HF에서 확정)

| H | FFN | L | NQ | NKV | HD | V | rms_eps | rope_theta | tie |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 11008 | 36 | 16 | 4 | 128 | 128256 | 1e-6 | 5e6 | True |

- GQA group = NQ/NKV = 4. q_dim = NQ*HD = 2048 = H. kv_dim = NKV*HD = 512.
- `no_rope_layers = [1,1,1,0]×9` (길이 36). **1 = RoPE 적용, 0 = NoPE(skip)**. (의미는 골든이 최종 판정)
- attention scale = 1/sqrt(HD). attention_bias=false, mlp_bias=false, hidden_act=silu. qk-norm 없음(그건 Qwen3).
- lm_head는 embedding과 tied. embed/norm은 fp32 유지, 선형(q/k/v/o/gate/up/down/lm_head)은 int4.

## int4 양자화 포맷 (`quantize_ffn.py`/`quantize_rest.py`와 한 치도 안 어긋나게)

weight는 row-major [OUT, IN] (IN은 32배수). flatten 후 QBLK=32 단위 블록(블록은 한 OUT row 내부에 머묾):
- `scale = maxabs(block)/7.0`, scale==0 → 1e-8.  (signed int4 -7..+7, zero-point 8)
- `q = clip(round(w/scale) + 8, 0, 15)` → uint8
- 패킹: 2개씩 `packed = q[even] | (q[odd] << 4)` → **low nibble=짝수 인덱스, high nibble=홀수 인덱스**
- dequant: `(q - 8) * scale`

디스크 레이아웃 (행렬 [OUT, IN]):
- `{name}_q4.bin`: uint8, 크기 `OUT*IN/2`, row-major
- `{name}_scale.bin`: fp32, 크기 `OUT*IN/QBLK`, row-major
- 커널 인덱싱: row o → `w = Wq + o*(IN/2)`, `s = sc + o*(IN/QBLK)`.
  원소 i → byte `i>>1`, nibble `(i&1)? hi:lo`, 블록 `i/QBLK`.

## 공용 커널 사양 (full_engine.cu에서 이식, config 인자화)

1. **rmsnorm** `(X,W,Y,hh,eps)`: 블록당 1 row. `ss=Σx²; inv=rsqrt(ss/hh+eps); y=x*inv*W[i]`. block-reduce.
2. **gemv_q4** `(X,Wq,sc,Y,IN,OUT)`: warp당 1 output row(`m0` 방식). lane이 i를 32 stride로 누적,
   `acc += X[i]*((q-8)*scale[i/QBLK])`, warp-reduce → Y[o]. (1단계는 정확성만 — 단순 m0 방식)
3. **gemv_f32** `(X,W,Y,IN,OUT)`: 위와 동형이나 fp32 weight 직접. (검증경로 A는 q4만 쓰므로 선택적)
4. **rope** `(T,cos,sin,S,NH,HD)`: HD=128을 두 절반(i, i+64)으로 회전.
   `t1=base[i],t2=base[i+64]; base[i]=t1*c−t2*sn; base[i+64]=t2*c+t1*sn`, `c=cos[s*HD+i]` (i:0..63).
   cos/sin 테이블은 위치당 HD 길이, 앞 64 = 뒤 64 (HF cat(freqs,freqs)).
5. **attention(GQA,causal)** `(Q,K,V,O,S,...)`: (head h, query qi) 블록. kv=h/GRP.
   `score[j]=（q·k_j)/sqrt(HD)` for j≤qi, softmax, `O=Σ prob_j*V_j`. KV는 [pos, NKV, HD] 레이아웃.
6. **swiglu** `(g,u,Y,n)`: `silu(g)*u`, `silu(x)=x/(1+e^-x)`.
7. **embedding** `(ids,W,Y,S,H)`: row 복사.
8. **add** residual.

## forward graph (레이어 l), config로 H/L/FFN/heads 받음

```
x = embed(ids)
for l in 0..L-1:
    n  = rmsnorm(x, input_ln[l])
    q,k,v = gemv(n, Wq/Wk/Wv[l])
    if use_rope[l]: rope(q); rope(k)        # ★ NoPE 처리 — 레퍼런스와 다름(레퍼런스는 무조건 rope)
    a  = attention(q,k,v)                   # KV 캐시 [0..pos]
    o  = gemv(a, Wo[l])
    x  = x + o
    n2 = rmsnorm(x, post_ln[l])
    g,u= gemv(n2, Wgate/Wup[l])
    act= swiglu(g,u)
    d  = gemv(act, Wdown[l])
    x  = x + d
logits = gemv(rmsnorm(x, final_norm), Wlm_head)   # lm_head = tied embed
```

## 검증 전략 (헌법: 단계별 격리, 오차 0 기준)

프롬프트 "The capital of France is" (S=5). `make_reference.py`가 HF로 골든 2종 생성:
- **golden_fp32**: HF fp32 forward. per-layer 입력 hidden `[S,H]`, final logits `[S,V]`.
- **golden_q4**: HF 각 Linear.weight를 dequant(quant(W))로 교체 후 forward. (q4와 동일 가중치의 fp32 수학)

| 검증 | 비교 | 판정 |
|---|---|---|
| A. graph/커널 수학 | engine(q4) vs **golden_q4** | per-layer + logits < 1e-2 → 통과 |
| B. quant budget(정보) | golden_q4 vs golden_fp32 | 누적 오차 관찰 |
| C. 단위 | 각 커널 작은 입력 vs torch | 오차 0 |

→ 어느 레이어/연산에서 갈라지는지 per-layer 대조로 즉시 특정. 끝값만 보지 않는다.

## 레퍼런스 대비 의도적 차이 (그대로 복사 안 함)

1. **NoPE**: `use_rope[l]`로 제어 (레퍼런스는 전 레이어 rope — pos0 단일토큰만 검증해 무탈했던 것).
2. **다토큰 prefill 검증**: pos>0에서 rope/attn/NoPE 실제 검증 (레퍼런스 pos0은 rope=항등이라 미검증).
3. **config 인자화**: `#define H/L/FFN/...` 제거, ModelConfig 런타임 주입. QBLK=32만 컴파일 타임.

## 산출물 경로 (gitignore)

`artifacts/smollm3-3b/weights/` (q4+scale+embed+norm), `artifacts/smollm3-3b/golden/` (golden_fp32/q4).

## 1단계 결과 (완료, PASS)

프롬프트 "The capital of France is" (5토큰, NoPE 9/36 레이어 = rope 27/36).

| 검증 | 결과 |
|---|---|
| 커널 단위 (vs CPU 오라클) | rmsnorm 2.4e-7, gemv_q4 9.5e-7, rope 0, attn 1.2e-7, swiglu 6e-8 — 전부 OK |
| graph 레이어별 hidden (vs golden_q4) | 전 37개 < 1e-3 (worst 9.2e-4 @ L10) |
| 최종 logits (vs golden_q4) | **1.34e-4** (< 1e-2 통과) |
| 최종 logits (vs golden_fp32) | 21.4 (= q4 quant budget, 정보용) |
| next-token (마지막 위치) | engine=12366 = golden_fp32=12366 (Paris) MATCH |

검증 중 확인된 사실:
- **HF `output_hidden_states` 마지막 원소(hs[L])는 final-norm 적용 후 값**. 엔진도 fn(post-norm) 덤프해 맞춤.
- **extract의 rope_theta 버그**: transformers 5.x SmolLM3는 theta를 `cfg.rope_parameters['rope_theta']`(=5e6)에
  저장(top-level 아님). 수정 완료. 1단계는 HF 골든 rope 테이블을 로드하므로 영향 없었음.

## 다음 단계로 넘긴 것 (stage 2+)

- 엔진 자체 rope 테이블 생성(theta=5e6) — 지금은 golden 테이블 로드.
- KV 캐시 관리(runtime/kvcache), 단일토큰 decode 스텝(현재 prefill만), batch/spec.
- gemv 성능 변형(m1/mMO/mSP) + ncu 분석 (지금은 정확성용 m0 방식).
- 성능 최적화 일체.
