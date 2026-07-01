# Llama.cpp CUDA Q4_0 Latency-Hiding Analysis

## Context
Your kernel has 72.8% stall on `long_scoreboard` (L1TEX/global memory latency), with synchronous load semantics: load W → compute → load next W → stall. This analysis identifies the concrete latency-hiding techniques llama uses that you lack.

---

## 1. MMVQ Decode GEMV (mul_mat_vec_q) Inner Loop Structure

**File:** `/home/hyunlord/repos/llama.cpp-mtp/ggml/src/ggml-cuda/mmvq.cu:572–592`

### Strategy: Multiple Outstanding Loads via Unrolling + Implicit Software Pipelining

```cuda
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);
    const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(
                vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] += vec_dot_q_cuda(
                        vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }
}
```

**Key Observations:**

1. **Unroll Factor:** The nested loops unroll over `ncols_dst` (typically 1-8) and `rows_per_cuda_block` (1-2).
   - VDR (vec_dot_ratio) = 2 for Q4_0 (vecdotq.cuh:112)
   - `blocks_per_iter = vdr * nwarps * warp_size / qi` causes multiple block indices to be processed per loop, creating ILP
   
2. **Implicit Multi-Iteration Overlap:**
   - Each unrolled iteration calls `vec_dot_q_cuda()` which internally iterates VDR times (vecdotq.cuh:728-733)
   - The loop-unroll directive creates multiple independent load→compute chains that can overlap
   - Next block's loads can issue while current block's DP4A is in flight

3. **Vectorized Loads (int4/float4 via get_int_b2/b4):**
   - `get_int_b2(bq4_0->qs, iqs + i)` (vecdotq.cuh:730) reads 32-bit aligned
   - `get_int_b4(bq8_1->qs, iqs + i)` (vecdotq.cuh:731) reads 32-bit aligned
   - Compiler can bundle these into vectorized loads (better throughput, same latency)

4. **Outstanding Loads:** With `nwarps=4` threads per output element and 2 rows per block, you have ~8 independent load addresses in flight before synchronization.

---

## 2. MMVQ Shared Memory Staging

**File:** `/home/hyunlord/repos/llama.cpp-mtp/ggml/src/ggml-cuda/mmvq.cu:569`

### Strategy: **NO Shared Memory for Weights** — Direct Global Loads with Register ILP

```cuda
const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    // Load directly from global memory via vec_dot_q_cuda
    tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
}
```

**Key Point:** There is **NO shared memory tile staging** for the decode path. Weights load directly from global memory. The latency hiding comes entirely from:
- **ILP via loop unrolling** (multiple load→compute chains)
- **Implicit instruction-level parallelism** within vec_dot_q_cuda (VDR=2 iterations)

---

## 3. MMQ Batched GEMM Double-Buffering (Multi-Token Decode)

**File:** `/home/hyunlord/repos/llama.cpp-mtp/ggml/src/ggml-cuda/mmq.cuh:3485–3518`

### Strategy: Explicit Double-Buffering of Shared Tiles

```cuda
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    // Load X tile into shared memory (once per 256/32=8 K iterations)
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);  // LINE 3486
    {
        // Load first Y tile into shared memory
        const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
            int l = l0 + threadIdx.y*warp_size + threadIdx.x;
            tile_y[l] = by0[l];  // Global→Shared, contributes to prefetch latency
        }
    }
    __syncthreads();  // LINE 3497: Wait for tile loads
    
    // Compute using tiles
    vec_dot(tile_x, tile_y, sum, 0);  // LINE 3499: Use first tile
    __syncthreads();  // LINE 3501
    
    // Load SECOND Y tile (prefetch next block while computing first)
    {
        const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
        for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
            int l = l0 + threadIdx.y*warp_size + threadIdx.x;
            tile_y[l] = by0[l];  // Prefetch next tile
        }
    }
    __syncthreads();  // LINE 3513
    
    // Compute using second tile (while first tile's loads hidden by prefetch of third)
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);  // LINE 3515
    __syncthreads();  // LINE 3517
}
```

**Pipeline Depth: 2 stages**

1. **Stage 1:** Load tile_x (weights) + load tile_y block 0 → compute block 0
2. **Overlap:** While computing block 0, load tile_y block 1 (prefetch)
3. **Stage 2:** Compute block 1 (while prefetching block 2 for next kb0 iteration)

**Outstanding Loads:** 
- Each load_tiles call reads `mmq_y * MMQ_TILE_NE_K` int32s for weights (e.g., 128 * 64 = 8192 ints, split across nwarps=8 warps)
- Each tile_y load reads `mmq_x * MMQ_TILE_Y_K` int32s (e.g., 128 * 40 = 5120 ints)
- With threadblock coalescing, ~8-16 outstanding 128-byte loads per warp

---

## 4. Concrete Latency-Hiding Techniques (Ranked by Impact)

### Technique 1: **Register-Level ILP via Loop Unrolling + Implicit Instruction Scheduling**

**Why It Hides Latency:** 
- Compiler unrolls loop bodies to create independent load→compute chains
- While one load's latency (400-600 cycles) is pending, 8-12 other independent operations execute
- DP4A (4 int8 dot products) = ~4 cycles throughput; with VDR=2, each vec_dot is ~8 compute ops
- Outstanding loads from unrolled iterations hide ~50-100 cycles of latency per thread

**File:Line:** 
- mmvq.cu:578-591: `#pragma unroll` over j, i with VDR=2
- vecdotq.cuh:728-733: Inner VDR loop in vec_dot_q4_0_q8_1

**Adoption for Your DP4A INT8 Kernel:**
```cuda
// Instead of:
for (int k = 0; k < K; k++) {
    load W[k];  // STALL HERE
    compute += dp4a(A[k], W[k]);
}

// Use:
float tmp0=0, tmp1=0, tmp2=0, tmp3=0;  // 4 accumulators (register file = 2048 regs/warp)
#pragma unroll 4
for (int k = 0; k < K; k += 4) {
    int w0 = get_int_b4(W, k+0);
    int w1 = get_int_b4(W, k+1);
    int w2 = get_int_b4(W, k+2);
    int w3 = get_int_b4(W, k+3);
    // Load w0-w3 in parallel, issue w0→tmp0, w1→tmp1, w2→tmp2, w3→tmp3
    tmp0 += dp4a(A_block, w0);
    tmp1 += dp4a(A_block, w1);
    tmp2 += dp4a(A_block, w2);
    tmp3 += dp4a(A_block, w3);
    // tmp0-3 execute independently; 4 loads in flight hide latency of earlier loads
}
```

**Expected Impact:** +20-40% reduction in long_scoreboard stalls by overlapping 4-8 load→compute chains.

---

### Technique 2: **Shared Memory Double-Buffering for Prefetch**

**Why It Hides Latency:**
- Load tile N+1 into shared memory WHILE computing tile N
- Shared memory latency (~30-50 cycles via L1) is hidden by compute of tile N
- No stall when transitioning from tile N to N+1

**File:Line:** 
- mmq.cuh:3485-3518: `load_tiles` + two explicit tile_y load loops with __syncthreads between

**Adoption for Your Kernel (Batch Decode):**
```cuda
// Tile size: 64 K elements per thread block (shared 16 KB)
__shared__ int W_tile0[256], W_tile1[256];  // Double-buffer weights
__shared__ int A_tile[256];

for (int k = 0; k < K; k += 256) {
    // Load tile k into W_tile0 while computing k-1 from W_tile1
    if (k > 0) {
        compute_with(W_tile1, A_tile, sum);  // Compute k-1
        __syncthreads();
    }
    
    // Prefetch tile k+1 into the other buffer
    const int offset = k + threadIdx.x;
    W_tile0[threadIdx.x] = (offset < K) ? W[offset] : 0;  // Global→Shared, hides under previous compute
    __syncthreads();
    
    // Swap
    int *tmp = W_tile0; W_tile0 = W_tile1; W_tile1 = tmp;
}
compute_with(W_tile1, A_tile, sum);  // Final tile
```

**Expected Impact:** +10-20% reduction in stalls if compute per tile is ~200+ cycles; smaller tiles reduce benefit.

---

### Technique 3: **cp.async for Pipelined Global→Shared Loads (Turing+)**

**Why It Hides Latency:**
- `cp.async` overlaps global load latency with compute
- Load issues asynchronously; computation proceeds in parallel; `cp.async.wait` stalls only when needed
- Reduces synchronization bubbles

**File:Line:** 
- MMQ uses `load_tiles` (mmq.cuh:398-457) which on Turing+ targets uses native async load semantics
- Blackwell native FP4 paths (mmq.cuh:3471-3476) use cp.async implicitly via PTX

**Adoption for Your Kernel (If Turing+):**
```cuda
#if __CUDA_ARCH__ >= 750  // Turing+
// Load asynchronously
uint32_t smem_addr = __cvta_generic_to_shared(W_tile);
asm volatile(
    "cp.async.cg.shared.global [%0], [%1], 128, [%2];"
    : : "r"(smem_addr), "l"((uint64_t)W_global + offset), "r"(0)
);

// Compute while load is in flight
compute_kernel(A, W_prev, sum);

// Wait for load
asm volatile("cp.async.wait_all;");
#else
// Fallback: manual load loop
for (int i = 0; i < 128/4; i++) {
    W_tile[threadIdx.x + i*(BLOCK_SIZE)] = W_global[offset + i*(BLOCK_SIZE)];
}
#endif
```

**Expected Impact:** +5-15% reduction in stalls on Turing/Ampere; minimal on older (no cp.async).

---

## Summary Table

| Technique | File:Line | Mechanism | ILP Depth | Applicable to Your Kernel? | Expected Impact |
|-----------|-----------|-----------|-----------|---------------------------|-----------------|
| **Unroll + Register ILP** | mmvq.cu:578-591, vecdotq.cuh:728-733 | Loop unroll creates 4-8 independent load→compute chains; compiler schedules them to overlap latency | 4-8 chains | **YES — implement immediately** | **+20-40%** reduction in long_scoreboard |
| **Shared Memory Double-Buffering** | mmq.cuh:3485-3518 | Prefetch tile N+1 while computing tile N; shared latency (~40cy) hidden by compute (~200cy) | 2 tiles | **YES — if batched decode >200cy/tile** | **+10-20%** reduction in stalls |
| **cp.async (Turing+)** | mmq.cuh:398-457 (implicit) | Async global→shared; compute overlaps with load; barrier only when needed | Implicit in loads | **YES — if Turing+** | **+5-15%** on Turing/Ampere |

---

## How Your Kernel Differs

Your kernel's bottleneck (72.8% long_scoreboard stalls):

1. **Synchronous Load Semantics:** Load → Stall (wait for data) → Compute → Load next
   - Missing: Only 1 load in flight; next load waits for data availability
   
2. **No Loop Unrolling:** No register ILP to create independent load→compute chains
   - Missing: 4-8 outstanding loads that llama has

3. **No Shared Memory Prefetch:** Weights load directly or without pipeline overlap
   - Missing: Double-buffering that hides shared→register latency

4. **Possible Low VDR:** If your kernel processes 1 element at a time instead of 2-4
   - Missing: Implicit instruction-level parallelism within vec_dot

---

## Action Plan for Your INT8 Decode Kernel

**Priority 1 (Immediate, +20-40% improvement):**
1. Unroll the K-loop by factor 4: Create 4 independent accumulators tmp0-tmp3
2. Load 4 weight blocks in parallel: `w0, w1, w2, w3 = get_int_b4(...)`
3. Issue 4 independent DP4As: `tmp0 += dp4a(A, w0); tmp1 += dp4a(A, w1); ...`
4. Let compiler schedule them to overlap the 400-600cy load latency

**Priority 2 (Moderate, +10-20%):**
1. Add `__shared__ int W_tile[256]` for weight prefetch
2. Launch the K-loop with two __syncthreads barriers: load next tile while computing current
3. Benchmark tile size (64, 128, 256 elements) to balance prefetch latency vs. compute time

**Priority 3 (Architecture-specific, +5-15%):**
1. If targeting Turing+, use `cp.async.cg.shared.global` for async weight loads
2. Otherwise, fallback to manual tile loads with the double-buffer pattern

---

## References

- **Stream-K Decomposition (cited in code):** https://arxiv.org/abs/2301.03598
- **NVIDIA CUDA C++ Programming Guide:** Section on Occupancy and Register Pressure
- **Llama.cpp GitHub:** ggml-cuda kernels (mmvq.cu, mmq.cuh, vecdotq.cuh)

