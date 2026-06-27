#ifndef DGX_ENGINE_H
#define DGX_ENGINE_H
// 챕터5: 다중모델 서빙 엔진 공통 인터페이스 + 모델 레지스트리.
//   새 모델 추가 = model_type → graph 엔트리(레지스트리) + config. 커널/graph 불변(헌법).
//   현재 등록: qwen3_5 (0.8B/4B/27B 동일 패밀리 graph 공유). Gemma 등은 새 ModelKind로 확장.
#include <cstdio>
#include <cstring>
#include "config.h"
#include "weights_qwen.h"
#include "manifest_qwen.h"
#include "../models/qwen35.h"
#include "../models/qwen35_batch.h"
#include "decode_state.h"

enum ModelKind { MK_QWEN3_5 = 0, MK_QWEN3_5_MOE = 1, MK_UNKNOWN = -1 };

// 레지스트리: manifest의 "model_type" → ModelKind. (없으면 qwen3_5 기본 — 현 단일 패밀리.)
//   주의: "qwen3_5_moe" 가 "qwen3_5" 를 부분포함 → 더 구체적인 moe 를 먼저 체크.
static inline ModelKind model_kind_from_manifest(const char *manifest_path) {
    FILE *f = fopen(manifest_path, "rb");
    if (!f) return MK_QWEN3_5;
    char buf[4096]; size_t n = fread(buf, 1, sizeof(buf) - 1, f); buf[n] = 0; fclose(f);
    if (strstr(buf, "\"model_type\"")) {
        if (strstr(buf, "qwen3_5_moe") || strstr(buf, "qwen3.5_moe")) return MK_QWEN3_5_MOE;  // MoE 먼저
        if (strstr(buf, "qwen3_5") || strstr(buf, "qwen3.5") || strstr(buf, "qwen3_next")) return MK_QWEN3_5;
        return MK_UNKNOWN;   // 미등록 model_type — 새 graph 필요(보편화 신호)
    }
    return MK_QWEN3_5;       // 태그 없으면 기존 단일 패밀리
}

struct Engine {
    ModelConfig c;
    QwenWeights w;
    ModelKind   kind;
};

// 공통 로드: weights_dir/manifest.json → config + kind + weights. 모델 무관 진입점.
static inline Engine engine_load(const char *weights_dir, int load_q4 = 1, int load_fp32 = 1) {
    Engine e{};
    char mani[2048]; snprintf(mani, sizeof(mani), "%s/manifest.json", weights_dir);
    e.kind = model_kind_from_manifest(mani);
    e.c = load_config_qwen(mani);
    e.w = load_weights_qwen(weights_dir, e.c, load_q4, load_fp32);
    return e;
}
static inline void engine_free(Engine &e) { free_weights_qwen(e.w); free_config_qwen(e.c); }

// ---- 공통 batch decode 인터페이스 (모델 무관, kind로 graph 디스패치) ----
static inline BatchDecodeState engine_alloc_batch(const Engine &e, int B, int max_len) {
    return alloc_batch_decode_state(e.c, B, max_len);   // 현 graph 공통(qwen3_5 패밀리)
}
static inline void engine_free_batch(BatchDecodeState &st) { free_batch_decode_state(st); }

static inline void engine_decode_batch(const Engine &e, BatchDecodeState &st, const int *d_tokens,
                                       const float *d_cos, const float *d_sin, float *d_logits, int use_q4) {
    switch (e.kind) {
        case MK_QWEN3_5: qwen35_decode_batch(e.c, e.w, st, d_tokens, d_cos, d_sin, d_logits, use_q4); break;
        default: fprintf(stderr, "engine: 미등록 ModelKind %d — graph 레지스트리 확장 필요\n", e.kind); break;
    }
}

#endif // DGX_ENGINE_H
