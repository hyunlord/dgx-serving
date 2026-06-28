# 챕터6 — 27B 3-way 실측 비교 (GB10, Qwen3.5-27B, B=1)

> 이 세션 가장 중요한 측정. "우리 엔진 빠르다"는 믿음을 정면 비교가 기각.
> measurement-first의 절정 — 마지막 기각이 가장 큼(우리 강점 믿음 자체).

## 결과 (같은 GB10, 같은 Qwen3.5-27B, B=1)

| 엔진 | prefill tok/s | decode tok/s | 정밀도 | 비고 |
|---|---|---|---|---|
| **우리** | **36** (q4 16-타일 GEMM) | **10** (W4A8) | ~4bit | use_q4=1 prefill / use_q4=2 decode 각 최선 |
| **llama.cpp** | **~900** (Q4_0, flash-attn) | **~14** | Q4_0 | 공정 q4 비교 |
| **vLLM** | **~1365** (bf16, cuBLAS) | 4.5 (bf16) | bf16 | decode는 bf16 대역폭 한계(불공정) |

측정:
- 우리: tools/bench.cu (cudaEvent, 청크 prefill). ctx 2048 기준 발췌.
- llama.cpp: llama-bench -p 2048 -n 128 -fa 1 (build/bin, CUDA).
- vLLM: docker vllm/vllm-openai:v0.17.0, bench latency --input-len 2048 --output-len 128.
  total 29.9s = prefill ~1.5s(2048→1365 t/s) + decode 28.4s(128@4.5 t/s, bf16 54GB 대역폭).

## 정직한 결론 — 우리가 둘 다 진다

### prefill: 25~38× 뒤짐 (압도적 #1 갭)
- llama.cpp 900, vLLM bf16 1365 vs 우리 36. 둘 다 **제대로 된 tiled GEMM**(cuBLAS/tensor-core, weight 1회 read).
- 우리 Phase-1 16-타일 GEMM = 1.84 TFLOP/s(ncu). gemv보다 4.1× 빨라졌으나 cuBLAS 상대 25× 부족.
- ncu가 Phase1 후 compute-bound 확인 → Phase2(X타일)로도 안 됨. **진짜 tiled GEMM(shared-mem/tensor-core) 필요.**
- → **(B) prefill GEMM이 명백한 최우선.** 25× 격차라 효과 큼.

### decode (공정 q4): 1.4× 뒤짐
- 우리 10(W4A8) vs llama.cpp 14. 대역폭 효율 우리 52%(10/19) vs llama.cpp 73%(14/19).
- llama.cpp의 fused 커널 + 최적 q4 dot이 우위. **챕터4 "W4A8 980 이미 최적" 결론 재검토 필요** —
  그건 우리 내부 비교였고, 외부 좌표가 "21%p 더 갈 수 있다"(52→73%) 보여줌. "최적"이 아니라 "우리 한계"였음.
- vLLM은 GPTQ-Int4 미측정(base bf16만) — 공정 decode 비교엔 GPTQ 필요(다음).

## 측정-우선의 교훈 (확정)
- 과거 "B=8 980으로 llama.cpp 912 추월"은 **짧은 context·특정 배치 한정 반쪽 진실**. 같은 조건 head-to-head선 llama.cpp가 둘 다 앞섬.
- 측정 하네스 없었으면 반쪽 믿음 유지했을 것 — 비교 고집이 정확히 옳음.
- 숨은 가정 4개가 새 조건마다 노출: GDN head(모델 확장), snap full-Sq(메모리), prefill 없음(컴퓨트), 그리고 **"우리 빠르다"(외부 비교)**.

## 다음
1. **(B) prefill GEMM** (STEP 0 설계부터, 큰 작업): 진짜 tiled q4 GEMM(shared-mem X·W 타일 + DP4A/tensor-core). 25× 최우선.
2. decode 1.4×: llama.cpp fused/q4-dot 따라가기(챕터4 "최적" 재검토, 21%p 여지).
3. (선택) vLLM GPTQ-Int4로 공정 decode 3-way + 35B 3-way.
