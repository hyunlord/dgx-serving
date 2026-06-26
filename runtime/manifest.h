#ifndef DGX_MANIFEST_H
#define DGX_MANIFEST_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "config.h"

// extract_weights.py 가 쓴 manifest.json 을 읽어 ModelConfig 구성 (controlled 포맷, naive 파서).
// use_rope 는 malloc 으로 할당 (free_config 로 해제).

static inline char *mf_read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "manifest 못 엶: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "manifest read err\n"); exit(1); }
    buf[n] = 0; fclose(f); return buf;
}

static inline const char *mf_find(const char *s, const char *key) {
    char pat[128]; snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(s, pat);
    if (!p) { fprintf(stderr, "manifest key 없음: %s\n", key); exit(1); }
    p = strchr(p + strlen(pat), ':');
    if (!p) { fprintf(stderr, "manifest ':' 없음: %s\n", key); exit(1); }
    return p + 1;
}

static inline double mf_num(const char *s, const char *key) { return strtod(mf_find(s, key), nullptr); }
static inline int mf_int(const char *s, const char *key) { return (int)strtol(mf_find(s, key), nullptr, 10); }
static inline int mf_bool(const char *s, const char *key) {
    const char *p = mf_find(s, key);
    while (*p == ' ' || *p == '\n' || *p == '\t') p++;
    return strncmp(p, "true", 4) == 0 ? 1 : 0;
}

static inline ModelConfig load_config(const char *manifest_path) {
    char *s = mf_read_file(manifest_path);
    ModelConfig c{};
    c.hidden_size       = mf_int(s, "hidden_size");
    c.intermediate_size = mf_int(s, "intermediate_size");
    c.num_layers        = mf_int(s, "num_layers");
    c.num_q_heads       = mf_int(s, "num_q_heads");
    c.num_kv_heads      = mf_int(s, "num_kv_heads");
    c.head_dim          = mf_int(s, "head_dim");
    c.vocab_size        = mf_int(s, "vocab_size");
    c.rms_norm_eps      = (float)mf_num(s, "rms_norm_eps");
    c.rope_theta        = (float)mf_num(s, "rope_theta");
    c.tie_word_embeddings = mf_bool(s, "tie_word_embeddings");
    c.max_seq_len       = 0;

    // use_rope 배열 파싱
    int *ur = (int *)malloc(sizeof(int) * c.num_layers);
    const char *p = strchr(mf_find(s, "use_rope"), '[');
    if (!p) { fprintf(stderr, "use_rope '[' 없음\n"); exit(1); }
    p++;
    for (int i = 0; i < c.num_layers; i++) {
        char *end; long v = strtol(p, &end, 10);
        if (p == end) { fprintf(stderr, "use_rope 파싱 부족 (i=%d)\n", i); exit(1); }
        ur[i] = (int)v; p = end;
        while (*p && *p != ',' && *p != ']') p++;
        if (*p) p++;
    }
    c.use_rope = ur;
    free(s);
    return c;
}

static inline void free_config(ModelConfig &c) { free((void *)c.use_rope); c.use_rope = nullptr; }

#endif // DGX_MANIFEST_H
