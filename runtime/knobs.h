#pragma once
// Knob registry v1 — the only place the engine reads its environment.
//
// Every env-driven setting is one row of g_knobs below: name, type, default, accepted range, class,
// scope and the invariants/gates a change must re-run. Call sites read values with knob_raw(name),
// which keeps the old getenv() contract (nullptr = not set, caller applies its documented default),
// so moving a call site onto the registry does not change behavior for any accepted value.
//
// Resolution order: explicit env > recipe file (DGX_KNOBS_FILE, lines "NAME=VALUE", '#' comments) >
// default. On first use the registry validates every set value against its range (exit 2 with the
// allowed range on failure) and warns about env names under a registry prefix that are not knobs
// (typo guard). knobs_init(tool) additionally prints the effective configuration as one JSON line
// on stderr ("KNOBS {...}"); bench, daemon and ppl mains call it.
//
// Classes: tunable = output bits unchanged (performance only; the listed invariant is the reason),
// numeric = output values change (quality gate §2 + golden before adoption), diagnostic = test/bench
// harness or debugging only (not a search target), fixed = functional serving setting (not a search
// target). tools/knobs_dump prints this table as JSON (autotune search space) or Markdown (docs).
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

extern char **environ;

enum KnobType { KNOB_ENUM, KNOB_INT, KNOB_FLOAT, KNOB_STRING, KNOB_INT_LIST, KNOB_FLOAT_LIST, KNOB_PRESENCE };
enum KnobClass { KNOB_TUNABLE, KNOB_NUMERIC, KNOB_DIAGNOSTIC, KNOB_FIXED };

struct KnobDef {
    const char *name;
    KnobType type;
    const char *dflt;        // effective value when unset ("" = off/none)
    const char *range;       // ENUM "a|b|c"; INT/FLOAT "min..max" (either side may be empty); otherwise free text
    KnobClass cls;
    const char *scope;       // models / paths that read it
    const char *gates;       // invariants and gates to re-run after changing it
    const char *desc;
};

// clang-format off
static const KnobDef g_knobs[] = {
// ---- Gemma 4 dense decode / prefill ----------------------------------------------------------------------------------
{"G4_ATTN_SPLIT",       KNOB_ENUM,  "1",   "0|1",           KNOB_NUMERIC, "gemma4 dense decode/verify/batch attention",
 "golden, ppl, spec==greedy, batch==single, fastpath==batch, graph_on==off", "1 = split-KV decode attention (R15 default); 0 = per-query-head kernels and fp32 KV default"},
{"G4_ATTN_CHUNK",       KNOB_ENUM,  "64",  "32|64|128|256", KNOB_NUMERIC, "gemma4 split-KV attention",
 "golden, ppl, spec==greedy, batch==single, fastpath==batch, graph_on==off", "split-KV chunk length in keys (changes the combine order)"},
{"G4_KV_DTYPE",         KNOB_ENUM,  "f16", "fp32|f16|bf16", KNOB_NUMERIC, "gemma4 dense KV storage (MoE forces fp32)",
 "golden, ppl, spec==greedy, batch==single, fastpath==batch, graph_on==off", "KV cache element type; unset = f16, or fp32 when G4_ATTN_SPLIT=0"},
{"G4_MMQ_PREFILL",      KNOB_ENUM,  "1",   "0|1",           KNOB_NUMERIC, "gemma4 prefill linear layers (W4A8)",
 "golden, ppl, prefill_slot", "1 = MMQ tensor-core prefill GEMM for S >= G4_MMQ_MIN_S; 0 = SB/GEMV prefill"},
{"G4_MMQ_MIN_S",        KNOB_INT,   "16",  "16..",          KNOB_NUMERIC, "gemma4 prefill linear layers",
 "golden, ppl, prefill_slot", "smallest prefill row count that uses MMQ (values below 16 were clamped to 16 before v1)"},
{"G4_FLASH_PREFILL",    KNOB_ENUM,  "1",   "0|1",           KNOB_NUMERIC, "gemma4 prefill attention",
 "golden, ppl, prefill_slot", "1 = tensor-core flash attention for prefill chunks with S >= G4_FLASH_MIN_S"},
{"G4_FLASH_MIN_S",      KNOB_INT,   "16",  "16..",          KNOB_NUMERIC, "gemma4 prefill attention",
 "golden, ppl, prefill_slot", "smallest prefill row count that uses flash attention"},
{"G4_NORM_FUSED",       KNOB_ENUM,  "1",   "0|1",           KNOB_NUMERIC, "gemma4 dense decode/verify/batch/concat/prefill norms",
 "golden, ppl, spec==greedy, batch==single, fastpath==batch, graph_on==off", "parallel RMSNorm row function + q/k norm+RoPE(+KV store) and post-norm+residual fusions (R16 default); 0 = per-kernel norms"},
{"G4_MMQ_UNPACK",       KNOB_ENUM,  "0",   "0|1",           KNOB_TUNABLE, "gemma4 prefill MMQ (W4A8, S >= G4_MMQ_MIN_S)",
 "golden, prefill_slot (same mma fragments and accumulation)", "unpack the int4 B tile to int8 once per K-tile in shared memory"},
{"G4_MMQ_SCALE_ASYNC",  KNOB_ENUM,  "0",   "0|1",           KNOB_TUNABLE, "gemma4 prefill MMQ",
 "golden, prefill_slot (same scale values)", "row/column scales loaded with cp.async into double buffers with the tiles"},
{"G4_MMQ_CP16",         KNOB_ENUM,  "0",   "0|1",           KNOB_TUNABLE, "gemma4 prefill MMQ",
 "golden, prefill_slot (copy width only)", "16-byte cp.async for the A tile (and B with G4_MMQ_UNPACK=1)"},
{"G4_CUDA_GRAPH",       KNOB_ENUM,  "2",   "0|1|2",         KNOB_TUNABLE, "gemma4 single-stream decode (generate session, gemma4_decode)",
 "graph_on==off, fastpath==batch", "decode launch mode: unset/2 = Graph v2 session, 1 = per-step v1 capture, 0 = eager"},
{"G4_GRAPH_OVERLAP",    KNOB_ENUM,  "1",   "0|1",           KNOB_TUNABLE, "gemma4 Graph v2 session",
 "graph_on==off", "overlap host bookkeeping with the next replay (v2 only)"},
{"G4_GRAPH_NODE_UPDATE",KNOB_ENUM,  "0",   "0|1",           KNOB_TUNABLE, "gemma4 Graph v2 session",
 "graph_on==off", "direct kernel-node parameter patching instead of per-step capture + exec update (v2 only)"},
{"G4_SERVE_FASTPATH",   KNOB_ENUM,  "1",   "0|1",           KNOB_TUNABLE, "online daemon, single-stream backends",
 "fastpath==batch", "serve a lone greedy request through the Graph v2 generation session"},
{"BM16",                KNOB_ENUM,  "0",   "0|1",           KNOB_NUMERIC, "gemma4/qwen batch decode B >= 14",
 "golden, ppl, batch==single", "opt-in BM16 split-K GEMM for large batches (scheme change)"},
{"SPLITK",              KNOB_INT,   "0",   "0..",           KNOB_NUMERIC, "BM16 split-K GEMM (only with BM16=1)",
 "golden, ppl, batch==single", "split-K factor; 0 = automatic (about 96 blocks)"},
{"MMQ256",              KNOB_ENUM,  "0",   "0|1|2",         KNOB_TUNABLE, "MMQ prefill/verify GEMM tile (gemma4, qwen)",
 "golden (claimed bitwise: same K-reduction order)", "MMQ tile: 0 = 128x128, 1 = 256x64, 2 = per-(N,K) table"},
{"MLP_U",               KNOB_ENUM,  "2",   "2|3|4|6",       KNOB_TUNABLE, "qwen W4A8 MLP GEMV",
 "golden (claimed bitwise: same compute order)", "MLP GEMV unroll factor"},
{"MOE_FUSED",           KNOB_ENUM,  "1",   "0|1",           KNOB_TUNABLE, "MoE block (qwen 35B-A3B, gemma4 26B-A4B)",
 "golden, batch==single (claimed bit-identical k-order)", "1 = device-fused grouped expert GEMV; 0 = naive host loop"},
// ---- Gemma 4 speculative serving (serve_daemon_gemma4_spec_online) ---------------------------------------------------
{"G4_SPEC_K",           KNOB_INT,   "3",   "1..8",          KNOB_TUNABLE, "gemma4 spec daemon draft length",
 "spec==greedy (K only changes speed)", "draft tokens per cycle; overrides the daemon's K argument (serve_api --spec-k)"},
{"G4_SPEC_DRAFT_UQ",    KNOB_ENUM,  "2",   "0|2",           KNOB_TUNABLE, "gemma4 spec daemon drafter precision",
 "spec==greedy (drafter only changes proposals)", "assistant drafter weights: 2 = W4A8, 0 = fp32; overrides the daemon's ast_uq argument"},
{"G4_SPEC_SERVE",       KNOB_ENUM,  "1",   "0|1",           KNOB_TUNABLE, "gemma4 online daemon with a drafter (serve_api --model gemma4 --assistant)",
 "spec==greedy, fastpath==batch (switch at cycle boundaries)", "serve a lone greedy request with speculative cycles (K = G4_SPEC_K) when a drafter is loaded; 0 = plain"},
// ---- Qwen speculative serving (bench_serve_spec / serve_daemon_qwen_spec_online) --------------------------------------
{"MTP_Q4",              KNOB_ENUM,  "0",   "0|1",           KNOB_TUNABLE, "qwen MTP draft head",
 "spec==greedy (draft only changes proposals)", "q4 MTP draft weights"},
{"DRAFT_GEMV",          KNOB_ENUM,  "0",   "0|1",           KNOB_TUNABLE, "qwen draft matmul",
 "spec==greedy", "GEMV draft matmul instead of gemm_batch"},
{"DRAFT_SEQ",           KNOB_ENUM,  "0",   "0|1",           KNOB_TUNABLE, "qwen draft scheduling",
 "spec==greedy", "sequential per-slot draft"},
{"DRAFT_MW",            KNOB_ENUM,  "0",   "0|1|2",         KNOB_TUNABLE, "qwen draft q4 matmul tile",
 "spec==greedy", "draft multi-warp tile variant (1 = NW4x2 rows, 2 = NW2x4 rows)"},
{"DQ4",                 KNOB_ENUM,  "",    "0|1",           KNOB_TUNABLE, "qwen draft lm_head",
 "spec==greedy", "draft lm_head in q4; unset = on when uq != 0"},
{"HOTSWITCH",           KNOB_STRING,"",    "K1,W1,K2,W2,thr", KNOB_NUMERIC, "qwen spec serving",
 "per-request ISO is not preserved (autotune-harness STEP A')", "cycle-level K/W switch on batch occupancy"},
{"PMAX",                KNOB_INT,   "32",  "1..",           KNOB_NUMERIC, "qwen serving prefill chunk",
 "golden, batch==single", "max prefill chunk length in the serving benches/daemon"},
// ---- Serving ---------------------------------------------------------------------------------------------------------
{"STOP_IDS",            KNOB_INT_LIST, "", "comma-separated token ids", KNOB_FIXED, "online daemons",
 "-", "end-of-generation token ids (model generation_config)"},
{"REP_WINDOW",          KNOB_INT,   "128", "0..",           KNOB_FIXED, "online daemon sampling",
 "-", "repetition-penalty window (tokens)"},
{"DGX_LOG_INPUT_IDS",   KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "online daemon", "-", "log request input ids to stderr"},
{"DGX_KNOBS_FILE",      KNOB_STRING,"",    "path",          KNOB_FIXED, "all binaries", "-", "recipe knob file (NAME=VALUE lines), below env in priority"},
// ---- Diagnostics -----------------------------------------------------------------------------------------------------
{"G4_CUDA_GRAPH_STATS", KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "gemma4 graph", "-", "print graph replay/update counters"},
{"G4_KV_STATS",         KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "gemma4 16-bit KV store", "-", "count non-finite / overflowing KV stores, print at exit"},
{"G4_ATTN_DIAG_REVERSE",KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "gemma4 per-query-head decode attention", "-", "reverse output accumulation order (quality noise baseline)"},
{"DBG0",                KNOB_STRING,"",    "directory",     KNOB_DIAGNOSTIC, "qwen layer-0 GDN", "-", "dump layer-0 intermediates"},
// ---- Bench / test harness --------------------------------------------------------------------------------------------
{"BENCH_G4_PREFILL_MIN_SEC", KNOB_FLOAT, "", "0..",        KNOB_DIAGNOSTIC, "bench_gemma4", "-", "repeat prefill until this many seconds (> 0)"},
{"BENCH_G4_DECODE_INPUT",    KNOB_ENUM,  "fixed", "fixed|greedy", KNOB_DIAGNOSTIC, "bench_gemma4", "-", "single-decode input tokens"},
{"BENCH_G4_DEPTH",           KNOB_INT,   "0",  "0..",       KNOB_DIAGNOSTIC, "bench_gemma4", "-", "starting n_past for single decode"},
{"BENCH_G4_SINGLE_UQ",       KNOB_ENUM,  "0",  "0|1|2",     KNOB_DIAGNOSTIC, "bench_gemma4", "-", "single-decode precision (0 = both)"},
{"BENCH_G4_DECODE_INTERVAL", KNOB_ENUM,  "0",  "0|1",       KNOB_DIAGNOSTIC, "bench_gemma4", "-", "print decode interval epochs"},
{"BENCH_G4_DUMP_LOGITS",     KNOB_STRING,"",   "path prefix", KNOB_DIAGNOSTIC, "bench_gemma4", "-", "dump single-decode logits"},
{"BENCH_G4_AMORT",           KNOB_ENUM,  "1",  "0|1",       KNOB_DIAGNOSTIC, "bench_gemma4", "-", "amortized prefill matmul in the prefill bench"},
{"SKIP1TOK",            KNOB_PRESENCE, "", "set",           KNOB_DIAGNOSTIC, "bench_serve", "-", "skip the one-token-per-step serving mode"},
{"SKIPSTAT",            KNOB_PRESENCE, "", "set",           KNOB_DIAGNOSTIC, "bench_serve", "-", "skip the static-batch mode"},
{"BONLY",               KNOB_INT,   "0",   "0..",           KNOB_DIAGNOSTIC, "bench_batch", "-", "run only this batch size (0 = all)"},
{"REPS",                KNOB_INT,   "1",   "1..",           KNOB_DIAGNOSTIC, "bench_batch", "-", "timed loop repetitions (min taken)"},
{"SHORTP",              KNOB_FLOAT, "0.7", "0..1",          KNOB_DIAGNOSTIC, "qwen spec bench/daemon workload", "-", "short-request ratio of the synthetic trace"},
{"TIME",                KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "qwen spec bench/daemon, test_batchspec2", "-", "per-phase timing"},
{"CYC",                 KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "qwen spec bench/daemon", "-", "spec cycle decomposition"},
{"ISO",                 KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "qwen spec bench/daemon", "-", "batched==solo token check"},
{"TRIALS",              KNOB_ENUM,  "0",   "0|1",           KNOB_DIAGNOSTIC, "qwen spec bench/daemon", "-", "warm autotune server (stdin trials)"},
{"SOLO",                KNOB_PRESENCE, "", "set",           KNOB_DIAGNOSTIC, "serve_daemon_qwen_spec_online", "-", "canonical solo spec gate mode"},
{"ADAPT",               KNOB_ENUM,  "0",   "0|1|2",         KNOB_DIAGNOSTIC, "test_gemma4_spec_e2e", "-", "adaptive draft length experiment"},
{"KMIN",                KNOB_INT,   "1",   "..",            KNOB_DIAGNOSTIC, "test_gemma4_spec_e2e", "-", "adaptive draft minimum K (clamped to 1..KMAX)"},
{"KMAX",                KNOB_INT,   "8",   "..",            KNOB_DIAGNOSTIC, "test_gemma4_spec_e2e", "-", "adaptive draft maximum K (clamped to 8)"},
{"TAUS",                KNOB_FLOAT_LIST, "", "comma-separated floats", KNOB_DIAGNOSTIC, "test_gemma4_spec_e2e", "-", "adaptive draft margin thresholds"},
{"CONF_LOG",            KNOB_STRING,"",    "path",          KNOB_DIAGNOSTIC, "test_gemma4_spec_e2e", "-", "draft confidence CSV"},
{"DUMP_PLAIN",          KNOB_STRING,"",    "directory",     KNOB_DIAGNOSTIC, "test_gemma4_spec_e2e", "-", "dump plain trajectories"},
{"SKIP_SPEC",           KNOB_PRESENCE, "", "set",           KNOB_DIAGNOSTIC, "test_gemma4_spec_e2e", "-", "plain only"},
{"DUMP",                KNOB_STRING,"",    "directory",     KNOB_DIAGNOSTIC, "test_27b", "-", "dump intermediates"},
{"V27",                 KNOB_PRESENCE, "", "set",           KNOB_DIAGNOSTIC, "test_27b", "-", "per-layer verbose"},
{"V35",                 KNOB_PRESENCE, "", "set",           KNOB_DIAGNOSTIC, "test_35b", "-", "per-layer verbose"},
{"FULLONLY",            KNOB_PRESENCE, "", "set",           KNOB_DIAGNOSTIC, "probe_chunked_prefill", "-", "only the full-prompt chunk"},
};
// clang-format on

// Env names under these prefixes must be registered knobs (typo guard; unknown ones only warn).
static const char *const g_knob_prefixes[] = {"G4_", "BENCH_G4_", "GEMV_", "MMQ", "DRAFT_", "MTP_", "MOE_", "DGX_"};

namespace knob_detail {
inline const KnobDef *find(const char *name) {
    for (const KnobDef &k : g_knobs) if (!strcmp(k.name, name)) return &k;
    return nullptr;
}
inline const char *class_name(KnobClass c) {
    return c == KNOB_TUNABLE ? "tunable" : c == KNOB_NUMERIC ? "numeric" : c == KNOB_DIAGNOSTIC ? "diagnostic" : "fixed";
}
inline const char *type_name(KnobType t) {
    switch (t) {
        case KNOB_ENUM: return "enum"; case KNOB_INT: return "int"; case KNOB_FLOAT: return "float";
        case KNOB_STRING: return "string"; case KNOB_INT_LIST: return "int_list"; case KNOB_FLOAT_LIST: return "float_list";
        default: return "presence";
    }
}
inline bool parse_long(const char *s, long &out) {
    char *end = nullptr; errno = 0; out = strtol(s, &end, 10);
    return errno == 0 && end != s && *end == '\0';
}
inline bool parse_double(const char *s, double &out) {
    char *end = nullptr; errno = 0; out = strtod(s, &end);
    return errno == 0 && end != s && *end == '\0' && std::isfinite(out);
}
inline bool in_bounds(const char *range, double v) {   // "min..max", either side may be empty
    const char *dots = strstr(range, "..");
    if (!dots) return true;
    std::string lo(range, dots - range), hi(dots + 2);
    if (!lo.empty() && v < atof(lo.c_str())) return false;
    if (!hi.empty() && v > atof(hi.c_str())) return false;
    return true;
}
inline bool valid(const KnobDef &k, const char *v) {
    switch (k.type) {
        case KNOB_ENUM: {
            std::string r(k.range); size_t p = 0;
            while (p <= r.size()) { size_t q = r.find('|', p); if (q == std::string::npos) q = r.size();
                if (r.compare(p, q - p, v) == 0 && strlen(v) == q - p) return true; p = q + 1; }
            return false;
        }
        case KNOB_INT: { long x; return parse_long(v, x) && in_bounds(k.range, (double)x); }
        case KNOB_FLOAT: { double x; return parse_double(v, x) && in_bounds(k.range, x); }
        case KNOB_INT_LIST: case KNOB_FLOAT_LIST: {
            std::string s(v); size_t p = 0;
            while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
                std::string t = s.substr(p, q - p);
                if (!t.empty()) { long a; double b;
                    if (k.type == KNOB_INT_LIST ? !parse_long(t.c_str(), a) : !parse_double(t.c_str(), b)) return false; }
                p = q + 1; }
            return true;
        }
        default: return true;
    }
}
inline size_t edit_distance(const std::string &a, const std::string &b) {
    std::vector<size_t> d(b.size() + 1);
    for (size_t j = 0; j <= b.size(); j++) d[j] = j;
    for (size_t i = 1; i <= a.size(); i++) {
        size_t prev = d[0]; d[0] = i;
        for (size_t j = 1; j <= b.size(); j++) {
            size_t cur = d[j];
            d[j] = std::min(std::min(d[j] + 1, d[j - 1] + 1), prev + (a[i - 1] == b[j - 1] ? 0 : 1));
            prev = cur;
        }
    }
    return d[b.size()];
}

struct State {
    std::map<std::string, std::string> recipe;      // values from DGX_KNOBS_FILE
    std::string recipe_path;
};

inline void fail(const char *msg) { fprintf(stderr, "knobs: %s\n", msg); exit(2); }

inline State load() {
    State s;
    if (const char *path = getenv("DGX_KNOBS_FILE")) {
        s.recipe_path = path;
        std::ifstream in(path);
        if (!in) { std::string m = std::string("cannot read DGX_KNOBS_FILE=") + path; fail(m.c_str()); }
        std::string line; int n = 0;
        while (std::getline(in, line)) {
            n++;
            size_t h = line.find('#'); if (h != std::string::npos) line.resize(h);
            size_t a = line.find_first_not_of(" \t\r"); if (a == std::string::npos) continue;
            size_t b = line.find_last_not_of(" \t\r"); line = line.substr(a, b - a + 1);
            size_t eq = line.find('=');
            if (eq == std::string::npos || eq == 0) {
                std::string m = std::string(path) + ":" + std::to_string(n) + ": expected NAME=VALUE"; fail(m.c_str()); }
            auto trim = [](std::string t) { size_t a = t.find_first_not_of(" \t"), b = t.find_last_not_of(" \t");
                                            return a == std::string::npos ? std::string() : t.substr(a, b - a + 1); };
            std::string key = trim(line.substr(0, eq)), val = trim(line.substr(eq + 1));
            if (!find(key.c_str())) { std::string m = std::string(path) + ": unknown knob " + key; fail(m.c_str()); }
            s.recipe[key] = val;
        }
    }
    // Validate every set value and warn about unregistered names under a registry prefix.
    for (const KnobDef &k : g_knobs) {
        const char *v = getenv(k.name); const char *src = "env";
        auto it = s.recipe.find(k.name);
        if (!v && it != s.recipe.end()) { v = it->second.c_str(); src = "recipe"; }
        if (v && !valid(k, v)) {
            std::string m = std::string("invalid ") + k.name + "=" + v + " (" + src + "); allowed: " + k.range;
            fail(m.c_str());
        }
    }
    for (char **e = environ; e && *e; e++) {
        const char *eq = strchr(*e, '='); if (!eq) continue;
        std::string name(*e, eq - *e);
        bool prefixed = false;
        for (const char *p : g_knob_prefixes) if (!name.compare(0, strlen(p), p)) prefixed = true;
        if (!prefixed || find(name.c_str())) continue;
        const KnobDef *best = nullptr; size_t bd = 99;
        for (const KnobDef &k : g_knobs) { size_t d = edit_distance(name, k.name); if (d < bd) { bd = d; best = &k; } }
        if (best && bd <= 3) fprintf(stderr, "knobs: warning: unknown knob env %s (did you mean %s?)\n", name.c_str(), best->name);
        else fprintf(stderr, "knobs: warning: unknown knob env %s\n", name.c_str());
    }
    return s;
}
inline const State &state() { static const State s = load(); return s; }
inline void json_str(std::string &o, const char *s) {
    o += '"';
    for (; *s; s++) { if (*s == '"' || *s == '\\') o += '\\'; if ((unsigned char)*s < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", *s); o += b; } else o += *s; }
    o += '"';
}
}  // namespace knob_detail

// Value as set by env or the recipe file, nullptr when unset (the caller applies the default listed above).
// Reading an unregistered name is a programming error.
inline const char *knob_raw(const char *name) {
    const knob_detail::State &s = knob_detail::state();
    if (!knob_detail::find(name)) { fprintf(stderr, "knobs: %s is not registered in runtime/knobs.h\n", name); abort(); }
    if (const char *v = getenv(name)) return v;
    auto it = s.recipe.find(name);
    return it == s.recipe.end() ? nullptr : it->second.c_str();
}
inline const char *knob_source(const char *name) {
    if (getenv(name)) return "env";
    const knob_detail::State &s = knob_detail::state();
    return s.recipe.count(name) ? "recipe" : "default";
}

// Validate, warn, and print the effective configuration once: KNOBS {"tool":...,"knobs":{name:{value,source}}}.
inline void knobs_init(const char *tool) {
    static std::once_flag once;
    std::call_once(once, [&]() {
        const knob_detail::State &s = knob_detail::state();
        std::string o = "{\"tool\":"; knob_detail::json_str(o, tool);
        o += ",\"recipe\":"; knob_detail::json_str(o, s.recipe_path.c_str());
        o += ",\"knobs\":{";
        bool first = true;
        for (const KnobDef &k : g_knobs) {
            if (k.cls == KNOB_DIAGNOSTIC && !knob_raw(k.name)) continue;   // unset harness knobs add only noise
            const char *v = knob_raw(k.name);
            if (!first) o += ','; first = false;
            knob_detail::json_str(o, k.name);
            o += ":{\"value\":"; knob_detail::json_str(o, v ? v : k.dflt);
            o += ",\"source\":"; knob_detail::json_str(o, knob_source(k.name));
            o += ",\"class\":"; knob_detail::json_str(o, knob_detail::class_name(k.cls)); o += '}';
        }
        o += "}}";
        fprintf(stderr, "KNOBS %s\n", o.c_str());
    });
}
