#!/usr/bin/env python3
"""M2 Probe — 장비 지문 자동 측정기: device profile JSON을 실측으로 생성.

측정 항목:
  ① STREAM 대역폭(tools/bw_stream.cu 재사용 — copy R+W, DRAM cold 4GB)
  ② cudaDeviceProp(이름·SM 수·arch·총 메모리·integrated=unified 감지)
  ③ 시스템 메모리(/proc/meminfo — MemAvailable 방식은 기존 가드와 동일)
  ④ toolkit caps 컴파일 시험: cp.async / mma.s8 / FP4(mxf4) — 각 미니 커널을 nvcc로 실제 컴파일해 판정
     (FP4는 GB10+CUDA 13.0에서 ptxas 실패가 ground truth — toolkit-blocked 재확인)

검증 원리(Phase 1 방식): 자동 생성 프로파일을 수동 프로파일과 대조(--compare) — 자동화가 수동을 재현하는지.

사용:
  python tools/autotune/probe.py --out /tmp/gb10-auto.json [--compare tools/autotune/profiles/gb10.json]
"""
import argparse, json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ARCH = "sm_120"  # GB10(sm_121)은 sm_120 바이너리 호환


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe_bandwidth():
    """STREAM copy(R+W) 실측 — 기존 bw_stream.cu 빌드·실행."""
    binp = os.path.join(REPO, "tools", "bw_stream")
    if not os.path.exists(binp):
        r = sh(["nvcc", f"-arch={ARCH}", "-O2", os.path.join(REPO, "tools", "bw_stream.cu"), "-o", binp])
        if r.returncode:
            return None, f"bw_stream 빌드 실패: {r.stderr[-200:]}"
    r = sh([binp])
    mc = re.search(r"copy\(R\+W\).*-> ([\d.]+) GB/s", r.stdout)
    mr = re.search(r"read-only.*-> ([\d.]+) GB/s", r.stdout)
    bw = {"copy": float(mc.group(1)) if mc else None,
          "read": float(mr.group(1)) if mr else None}
    return (bw if (mc or mr) else None), r.stdout.strip()


DEV_PROBE_CU = r"""
#include <cstdio>
#include <cuda_runtime.h>
int main(){ cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("{\"name\":\"%s\",\"sm_count\":%d,\"arch\":\"sm_%d%d\",\"total_mem_gb\":%.1f,\"integrated\":%d,"
         "\"managed\":%d,\"pageable_access\":%d}\n",
         p.name, p.multiProcessorCount, p.major, p.minor, p.totalGlobalMem/1e9,
         p.integrated, p.managedMemory, p.pageableMemoryAccess);
  return 0; }
"""

CAP_TESTS = {
    # 각 cap을 실제 nvcc 컴파일로 판정(문서 아닌 measured — FP4는 실패가 GB10/13.0 ground truth)
    "cp_async": r"""
__global__ void k(float* g){ __shared__ float s[32];
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "l"((unsigned long long)(__cvta_generic_to_shared(s))), "l"(g));
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n"); }
int main(){return 0;}
""",
    "mma_int8": r"""
__global__ void k(int* d){ int a=1,b=1,c0=0,c1=0,r0,r1;
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
               : "=r"(r0),"=r"(r1) : "r"(a),"r"(b),"r"(c0),"r"(c1)); d[0]=r0+r1; }
int main(){return 0;}
""",
    "fp4_mma": r"""
__global__ void k(float* d){ unsigned a=0,b=0; float c0=0,c1=0,c2=0,c3=0,r0,r1,r2,r3;
  asm volatile("mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32.block_scale "
               "{%0,%1,%2,%3}, {%4}, {%5}, {%6,%7,%8,%9};\n"
               : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3)
               : "r"(a),"r"(b),"f"(c0),"f"(c1),"f"(c2),"f"(c3)); d[0]=r0; }
int main(){return 0;}
""",
}


def probe_device():
    with tempfile.TemporaryDirectory() as td:
        src, binp = os.path.join(td, "d.cu"), os.path.join(td, "d")
        open(src, "w").write(DEV_PROBE_CU)
        r = sh(["nvcc", f"-arch={ARCH}", src, "-o", binp])
        if r.returncode:
            return None, r.stderr[-300:]
        r = sh([binp])
        return json.loads(r.stdout.strip()), None


def probe_caps():
    caps, notes = {}, {}
    with tempfile.TemporaryDirectory() as td:
        for name, code in CAP_TESTS.items():
            src, binp = os.path.join(td, f"{name}.cu"), os.path.join(td, name)
            open(src, "w").write(code)
            r = sh(["nvcc", f"-arch={ARCH}", src, "-o", binp])
            caps[name] = (r.returncode == 0)
            if r.returncode:
                notes[name] = (r.stderr.strip().splitlines() or ["?"])[-1][:160]
    return caps, notes


def probe_sysmem():
    tot = avail = None
    for line in open("/proc/meminfo"):
        m = re.match(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
        if m:
            v = int(m.group(2)) * 1024 / 1e9
            tot, avail = (v, avail) if m.group(1) == "MemTotal" else (tot, v)
    return tot, avail


def nvcc_version():
    r = sh(["nvcc", "--version"])
    m = re.search(r"release ([\d.]+)", r.stdout)
    return m.group(1) if m else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="수동 프로파일과 대조(ground truth 검증)")
    ap.add_argument("--bw-tol-pct", type=float, default=10.0)
    args = ap.parse_args()

    print("[probe] cudaDeviceProp...", flush=True)
    dev, err = probe_device()
    if dev is None:
        sys.exit(f"[probe] device probe 실패: {err}")
    print(f"  {dev}", flush=True)
    print("[probe] STREAM 대역폭(4GB copy R+W, ~30s)...", flush=True)
    bw, bw_raw = probe_bandwidth()
    print(f"  {bw_raw}", flush=True)
    print("[probe] toolkit caps 컴파일 시험...", flush=True)
    caps, cap_notes = probe_caps()
    print(f"  {caps}" + (f" (실패 사유: {cap_notes})" if cap_notes else ""), flush=True)
    mem_tot, mem_avail = probe_sysmem()
    unified = bool(dev["integrated"]) or (dev["managed"] and dev["pageable_access"]
                                          and abs(dev["total_mem_gb"] - (mem_tot or 0)) < 0.3 * (mem_tot or 1))

    profile = {
        "name": dev["name"],
        "arch": dev["arch"] + ("(=sm_121 호환)" if dev["arch"] == "sm_121" else ""),
        "compile_arch": ARCH,
        "sm_count": dev["sm_count"],
        # decode(GEMV)는 read-지배 — 달성 가능 대역폭은 STREAM read 실측으로 기록(스펙 피크 아님을 명시)
        "bandwidth_gbps": round(bw["read"]) if bw and bw.get("read") else None,
        "bandwidth_copy_gbps": round(bw["copy"]) if bw and bw.get("copy") else None,
        "bandwidth_note": "STREAM read-only 실측(달성 가능 상한) — 수동본의 spec peak와 다른 의미",
        "unified_memory": unified,
        "mem_total_gb": round(mem_tot) if mem_tot else round(dev["total_mem_gb"]),
        "mem_probe": "max(cudaMemGetInfo.free, /proc/meminfo MemAvailable)",
        "mem_guard_margin": 0.7,
        "toolkit": {"cuda_version": nvcc_version(),
                    "cp_async": caps["cp_async"], "mma_int8": caps["mma_int8"],
                    "fp4_mma": caps["fp4_mma"],
                    **({"fp4_blocked_reason": cap_notes.get("fp4_mma", "")} if not caps["fp4_mma"] else {})},
        "probe": {"generated_by": "probe.py", "stream_raw": bw_raw.splitlines()[0] if bw_raw else None},
    }
    json.dump(profile, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[probe] 프로파일 생성: {args.out}")

    if args.compare:
        manual = json.load(open(args.compare))
        diffs = []
        bw_auto = profile["bandwidth_gbps"]
        if bw_auto and manual.get("bandwidth_gbps"):
            # 수동본 값은 spec peak — 실측(read)이 peak의 (100-tol)% 이내면 정합(recipe tolerance와 동일 기준)
            pct = abs(bw_auto - manual["bandwidth_gbps"]) / manual["bandwidth_gbps"] * 100
            status = "OK" if pct <= args.bw_tol_pct else "오차 초과"
            diffs.append(f"bandwidth: auto(read 실측) {bw_auto} vs manual(peak) {manual['bandwidth_gbps']} "
                         f"({pct:.1f}%, 효율 {bw_auto/manual['bandwidth_gbps']*100:.0f}% — {status})")
        if abs(profile["mem_total_gb"] - manual.get("mem_total_gb", 0)) > 2:
            diffs.append(f"mem_total: auto {profile['mem_total_gb']} vs manual {manual['mem_total_gb']} — 불일치!")
        else:
            diffs.append(f"mem_total: auto {profile['mem_total_gb']} vs manual {manual.get('mem_total_gb')} (±2GB 내 OK)")
        if profile["unified_memory"] != manual.get("unified_memory"):
            diffs.append(f"unified: auto {profile['unified_memory']} vs manual {manual['unified_memory']} — 불일치!")
        for cap in ("cp_async", "mma_int8", "fp4_mma"):
            a, m = profile["toolkit"][cap], manual.get("toolkit", {}).get(cap)
            diffs.append(f"cap {cap}: auto {a} vs manual {m}" + ("" if a == m else " — 불일치!"))
        print("[compare] ground truth 대조:")
        for d in diffs:
            print(f"  {d}")
        bad = [d for d in diffs if "불일치!" in d or "오차 초과" in d]
        print(f"[compare] {'PASS — 자동화가 수동본 재현' if not bad else 'FAIL: ' + '; '.join(bad)}")
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
