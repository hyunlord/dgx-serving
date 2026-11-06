#!/usr/bin/env python3
"""Knob registry gate (no GPU): defaults, env/recipe priority, range rejection and typo warnings.

Runs tools/knobs_dump --effective, which goes through the same knobs_init() path as the bench,
daemon and ppl mains, under controlled environments.
"""
import json, os, subprocess, sys, tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.join(REPO, "tools", "knobs_dump")
PREFIXES = ("G4_", "BENCH_G4_", "GEMV_", "MMQ", "DRAFT_", "MTP_", "MOE_", "DGX_")


def run(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIXES)}
    env.update(extra)
    r = subprocess.run([BIN, "--effective"], env=env, capture_output=True, text=True)
    knobs = None
    for line in r.stderr.splitlines():
        if line.startswith("KNOBS "):
            knobs = json.loads(line[6:])["knobs"]
    return r.returncode, r.stderr, knobs


fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + detail))
    fails += 0 if ok else 1


rc, err, k = run({})
check("default resolves", rc == 0 and k["G4_KV_DTYPE"] == {"value": "f16", "source": "default", "class": "numeric"}, err)
check("default graph mode", k is not None and k["G4_CUDA_GRAPH"]["value"] == "2")
# Given no overrides, the R25 rollout is opt-in and grouped diagnostics use G8.
check("raster default off", k is not None and k.get("G4_MMQ_RASTER", {}).get("value") == "0")
check("raster group default eight", k is not None and k.get("G4_MMQ_RASTER_GROUP", {}).get("value") == "8")
# Given registered raster/group choices, effective configuration preserves each choice.
for raster in ("0", "1", "2"):
    for group in ("4", "8", "16"):
        rc, err, k = run({"G4_MMQ_RASTER": raster, "G4_MMQ_RASTER_GROUP": group})
        check(f"raster {raster} group {group} accepted", rc == 0 and k is not None
              and k.get("G4_MMQ_RASTER", {}).get("value") == raster
              and k.get("G4_MMQ_RASTER_GROUP", {}).get("value") == group, err)
# Given unsupported values, registry initialization must reject rather than silently route.
for name, value in (("G4_MMQ_RASTER", "-1"), ("G4_MMQ_RASTER", "3"),
                    ("G4_MMQ_RASTER_GROUP", "0"), ("G4_MMQ_RASTER_GROUP", "3"),
                    ("G4_MMQ_RASTER_GROUP", "32")):
    rc, err, k = run({name: value})
    check(f"{name}={value} rejected", rc == 2 and f"{name}={value}" in err, err)
rc, err, k = run({"G4_KV_DTYPE": "bf16"})
check("env value", rc == 0 and k["G4_KV_DTYPE"]["value"] == "bf16" and k["G4_KV_DTYPE"]["source"] == "env", err)
rc, err, k = run({"G4_KV_DTYPE": "fp8"})
check("enum out of range rejected", rc == 2 and "allowed: fp32|f16|bf16" in err, err)
rc, err, k = run({"G4_MMQ_MIN_S": "0"})
check("int below range rejected", rc == 2 and "allowed: 1.." in err, err)
rc, err, k = run({"G4_ATTN_CHUNK": "abc"})
check("non-numeric rejected", rc == 2 and "G4_ATTN_CHUNK=abc" in err, err)
rc, err, k = run({"G4_ATTN_CHUNKS": "64"})
check("typo warns with suggestion", rc == 0 and "unknown knob env G4_ATTN_CHUNKS (did you mean G4_ATTN_CHUNK?)" in err, err)
rc, err, k = run({"GEMV_D1": "2"})
check("unregistered prefixed env warns", rc == 0 and "unknown knob env GEMV_D1" in err, err)
with tempfile.NamedTemporaryFile("w", suffix=".knobs", delete=False) as f:
    f.write("# recipe\nG4_ATTN_CHUNK=128\nG4_KV_DTYPE = bf16\n")
    recipe = f.name
rc, err, k = run({"DGX_KNOBS_FILE": recipe})
check("recipe value", rc == 0 and k["G4_ATTN_CHUNK"] == {"value": "128", "source": "recipe", "class": "numeric"}, err)
rc, err, k = run({"DGX_KNOBS_FILE": recipe, "G4_ATTN_CHUNK": "32"})
check("env overrides recipe", rc == 0 and k["G4_ATTN_CHUNK"]["value"] == "32" and k["G4_ATTN_CHUNK"]["source"] == "env", err)
with open(recipe, "w") as f:
    f.write("G4_MMQ_RASTER=2\nG4_MMQ_RASTER_GROUP=16\n")
rc, err, k = run({"DGX_KNOBS_FILE": recipe})
check("raster recipe values", rc == 0 and k is not None
      and k["G4_MMQ_RASTER"]["value"] == "2"
      and k["G4_MMQ_RASTER_GROUP"]["value"] == "16", err)
rc, err, k = run({"DGX_KNOBS_FILE": recipe, "G4_MMQ_RASTER": "1"})
check("raster env overrides recipe", rc == 0 and k is not None
      and k["G4_MMQ_RASTER"]["value"] == "1"
      and k["G4_MMQ_RASTER"]["source"] == "env", err)
with open(recipe, "w") as f:
    f.write("G4_ATTN_CHUNK=100\n")
rc, err, k = run({"DGX_KNOBS_FILE": recipe})
check("recipe out of range rejected", rc == 2 and "G4_ATTN_CHUNK=100 (recipe)" in err, err)
with open(recipe, "w") as f:
    f.write("G4_NOT_A_KNOB=1\n")
rc, err, k = run({"DGX_KNOBS_FILE": recipe})
check("recipe unknown knob rejected", rc == 2 and "unknown knob G4_NOT_A_KNOB" in err, err)
os.unlink(recipe)
print("[test_knobs] " + ("PASS" if not fails else f"FAIL ({fails})"))
sys.exit(1 if fails else 0)
