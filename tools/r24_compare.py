#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run: PYTHONPATH=/opt/nvidia/nsight-compute/2025.3.1/extras/python \
#   python3 tools/r24_compare.py RUN_DIR FRESH_OUTPUT_DIR
"""Reproduce R24 CSV exports, source attribution and comparison using CPU only."""
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path

import ncu_report  # Supplied by installed Nsight Compute; set PYTHONPATH above.
import r24_ncu_summary as s

if len(sys.argv) != 3:
    raise SystemExit("Usage: r24_compare.py RUN_DIR FRESH_OUTPUT_DIR")
base = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
if output.exists():
    raise SystemExit(f"Output already exists: {output}")
output.mkdir(parents=True)
scriptdir = Path(__file__).resolve().parent
for reportpath in sorted(base.glob("ncu_*.ncu-rep")):
 stem=reportpath.stem
 if not (base/f"{stem}.csv").exists():continue
 d=output/"exports"/stem
 if not (d/f"{stem}_summary.json").exists():
  subprocess.run(["bash",str(scriptdir/"r24_source_export.sh"),str(reportpath),str(d)],check=True)
 if (d/f"{stem}_source_attribution.json").exists():continue
 lines=s.parse_source((d/f"{stem}_source.csv").read_text()); report=ncu_report.load_report(str(reportpath)); a=report.range_by_idx(0).action_by_idx(0); contents=a.source_files()
 total=sum(x.samples or 0 for x in lines); groups=collections.defaultdict(lambda:[0,0,0]); results=[]
 candidates={x.address for x in sorted(lines,key=lambda x:x.samples or 0,reverse=True)[:10]}
 for x in lines:
  if not x.shared_excessive and x.address not in candidates:continue
  info=a.source_info(int(x.address,16)); path=info.file_name() if info else None; line=info.line() if info else None
  text=contents[path] if path and path in contents else ""
  if not text and path and Path(path).exists():text=Path(path).read_text()
  snippet=text.splitlines()[line-1] if line and len(text.splitlines())>=line else None
  if x.shared_excessive:
   key=(x.category,path,line); groups[key][0]+=x.shared_excessive;groups[key][1]+=x.samples or 0;groups[key][2]+=1
  results.append(dict(pc=x.address,sass=x.instruction,file=path,line=line,source=snippet,excessive_wavefronts=x.shared_excessive,stall_samples=x.samples,stall_sample_pct=100*(x.samples or 0)/total if total else None))
 grouped=[dict(category=k[0],file=k[1],line=k[2],excessive_wavefronts=v[0],stall_samples=v[1],pc_count=v[2]) for k,v in groups.items()]
 (d/f"{stem}_source_attribution.json").write_text(json.dumps(dict(total_stall_samples=total,groups=grouped,pcs=results),indent=2)+"\n")
 print(stem,"PCs",len(lines),"reasons",max(x.sample_reason_count for x in lines),"available",min(x.available_sample_reason_count for x in lines),"samples",total,"groups",grouped)

base=output
records={}
for directory in sorted((base/"exports").glob("ncu_*")):
 stem=directory.name
 raw=directory/f"{stem}_raw.csv"; source=directory/f"{stem}_source.csv"
 if not raw.exists() or not source.exists(): continue
 launch=s.parse_raw(raw.read_text()); lines=s.parse_source(source.read_text())
 records[stem]=(launch,lines,directory)
metrics=(scriptdir/"r24_metrics.txt").read_text().splitlines()
metrics=[m for m in metrics if not m.startswith("smsp__average_warps_issue_stalled_") and not m.startswith("smsp__warp_issue_stalled_")]
metrics+=["launch__registers_per_thread","launch__shared_mem_per_block","launch__shared_mem_per_block_static","launch__shared_mem_per_block_dynamic","launch__occupancy_limit_registers","launch__occupancy_limit_shared_mem","launch__occupancy_limit_warps","launch__waves_per_multiprocessor","sm__maximum_warps_per_active_cycle_pct"]
out=["# R24 NCU comparison (CPU exports)", "", "All L2 byte counts below are **근사(L2 miss 기반)**, not measured DRAM bytes. Calibration is recorded separately in the run calibration evidence (±10% acceptance). NCU duration is replay diagnostic only, not absolute performance. PC percentages use the 16 exported non-selected-state stall reasons; not_selected is included. Stall cycles/issue and percentages are not wall-clock fractions.",""]
def instruction_meaning(instruction: str) -> str:
 """Name instruction mechanics without guessing source-level purpose."""
 opcode=next((token.split(".")[0] for token in instruction.split() if not token.startswith("@")), "")
 meanings={
  "LEA":"effective-address calculation; exact buffer unresolved without source",
  "IADD3":"integer add/address or loop arithmetic; exact purpose unresolved without source",
  "IMAD":"integer multiply-add/address arithmetic; exact purpose unresolved without source",
  "BRA":"control-flow branch; loop/barrier role requires source context",
  "HADD2":"half arithmetic/conversion; scale role unconfirmed without source",
  "LDS":"shared-memory load; weight/activation/scale identity requires source",
  "LDSM":"matrix-fragment shared-memory load; operand identity requires source",
  "STS":"shared-memory store; target buffer unresolved without source",
  "LDG":"global-memory load; target array unresolved without source",
  "LDGSTS":"asynchronous global-to-shared copy",
  "IMMA":"integer tensor-core matrix multiply-accumulate",
  "FFMA":"floating-point fused multiply-add; accumulation role requires source",
  "FMUL":"floating-point multiply; scaling role requires source",
  "LOP3":"three-input bitwise logic; unpack versus address masking requires source",
  "ISETP":"integer comparison producing predicate; branch/bounds role requires source",
  "UMOV":"uniform-register move; exact control/address purpose unresolved without source",
  "BAR":"thread-block barrier synchronization",
  "DEPBAR":"dependency barrier wait",
 }
 return meanings.get(opcode, opcode+": instruction semantics not classified")


def fmt(x: float | None) -> str:
 return "UNAVAILABLE" if x is None else f"{x:,.6g}"
for m in (1024,2048):
 for shape,k,n in [("gate",3840,15360),("down",15360,3840),("o",4096,3840)]:
  names=[f"ncu_main_{shape}_M{m}",f"ncu_llama_{shape}_M{m}_r2"]
  available=[records.get(name) for name in names]
  if not any(available):continue
  out += [f"## {shape}, M={m}, K={k}, N={n}","","| Metric | main | llama |","|---|---:|---:|"]
  for metric in metrics:
   values=[]
   for record in available:
    item=next((x for x in record[0].metrics if x.name==metric),None) if record else None
    values.append(fmt(item.value)+(" "+item.unit if item and item.unit else "") if item else "NOT_RUN/UNAVAILABLE")
   out.append("| "+metric+" | "+" | ".join(values)+" |")
  w=n*k*9/16; a=m*k*9/8; y=4*m*n
  out+=["",f"Logical unique W={w:,.0f} B; A={a:,.0f} B; Y={y:,.0f} B. W+A={w+a:,.0f} B; W+A+Y={w+a+y:,.0f} B. Both formats have same logical unique byte totals. Output Y is not a read-byte denominator; it is supplied only as the requested total-footprint comparison.","","| 근사(L2 miss 기반) | main | llama |","|---|---:|---:|"]
  proxies=[r[0].value(s.MISS)*32 if r and r[0].value(s.MISS) is not None else None for r in available]
  for label,values in [("read-miss requested bytes",proxies),("proxy/(W+A)",[v/(w+a) if v is not None else None for v in proxies]),("proxy/(W+A+Y)",[v/(w+a+y) if v is not None else None for v in proxies]),("implied weight read multiplier (proxy-A)/W",[ (v-a)/w if v is not None else None for v in proxies])]:
   out.append("| "+label+" | "+" | ".join(fmt(v) for v in values)+" |")
  out+=["","Implied weight multiplier assumes A read once and attributes all other misses to W: activation rereads, cache effects, scale traffic already counted in W/A, spill/instruction/other traffic make this an estimate, not an identified weight counter.","","| Warp state | main cycles/issue | main active-warp-cycle % | llama cycles/issue | llama active-warp-cycle % |","|---|---:|---:|---:|---:|"]
  reasons=sorted({x.name[len(s.STALL):-len("_per_issue_active.ratio")] for r in available if r for x in r[0].metrics if x.name.startswith(s.STALL) and x.name.endswith("_per_issue_active.ratio")})
  for reason in reasons:
   values=[]
   for r in available:
    values += [fmt(r[0].value(s.STALL+reason+"_per_issue_active.ratio")) if r else "NOT_RUN",fmt(r[0].value("smsp__warp_issue_stalled_"+reason+"_per_warp_active.pct")) if r else "NOT_RUN"]
   label=reason+(" (overlapping detail)" if reason.startswith(("long_scoreboard_pipe_","mio_throttle_pipe_")) else "")+(" (not stall)" if reason=="selected" else "")
   out.append("| "+label+" | "+" | ".join(values)+" |")
  for name,r in zip(names,available):
   if not r:continue
   launch,lines,d=r;total=sum(x.samples or 0 for x in lines)
   annotation_path=d/f"{name}_source_attribution.json"
   annotation=json.loads(annotation_path.read_text()) if annotation_path.exists() else {"pcs":[],"groups":[]}
   bypc={x["pc"]:x for x in annotation["pcs"]}
   out += ["",f"### {name}: sampled SASS top10","",f"All-stall samples={total:,.0f}; exported reason count={max(x.sample_reason_count for x in lines)}; minimum available={min(x.available_sample_reason_count for x in lines)}.","","| PC | SASS | Samples | All-stall sample % | Source / meaning |","|---|---|---:|---:|---|"]
   for x in sorted(lines,key=lambda x:x.samples or 0,reverse=True)[:10]:
    detail=bypc.get(x.address,{})
    location=(str(detail.get("file"))+":"+str(detail.get("line"))) if detail.get("file") else "NO_LINEINFO"
    meaning=instruction_meaning(x.instruction)+"; "+location+"; "+(detail.get("source") or "source-level role unresolved")
    out.append("| "+x.address+" | "+x.instruction.replace("|","/")+" | "+fmt(x.samples)+" | "+fmt(100*(x.samples or 0)/total if total else None)+" | "+meaning.replace("|","/")+" |")
   out+=["","### Shared excessive wavefront source attribution","","| Instruction/source group | Excessive wavefronts | Samples at conflicting PCs | All-stall sample % |","|---|---:|---:|---:|"]
   for g in annotation["groups"]:
    label=g["category"]+" "+str(g["file"])+":"+str(g["line"])
    out.append("| "+label+" | "+fmt(g["excessive_wavefronts"])+" | "+fmt(g["stall_samples"])+" | "+fmt(100*g["stall_samples"]/total if total else None)+" |")
   out+=["","LDSM and scalar LDS are read-side; LDGSTS async-copy is separate store-side traffic. Source ideal/excess counters and hardware bank-conflict events are different measurements. PC shares do not identify causal stall duration: stalled consumers can appear after their producing loads.",""]
path=base/("comparison.md" if len(records)==12 else f"comparison.partial{len(records)}.md")
if path.exists():print("ALREADY_EXISTS",path)
else:path.write_text("\n".join(out)+"\n");print("WROTE",path,"reports",len(records))

audit=[]
for stem,(launch,lines,directory) in sorted(records.items()):
 summary=json.loads(s.summarize(launch,lines,None))
 states=[x for x in summary["stalls"] if not x["overlapping_subreason"]]
 audit.append(dict(report=stem,nonoverlapping_warp_state_pct_sum=sum(x["active_warp_cycle_pct"] for x in states),reasons=len(states),source_stall_reasons=summary["source"]["sample_reason_count"],dram_bytes=summary["dram_bytes"]))
(base/"comparison_numeric_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
reasons={"barrier":"barrier","branch_resolving":"branch_resolving","dispatch":"dispatch_stall","drain":"drain","lg":"lg_throttle","long_sb":"long_scoreboard","math":"math_pipe_throttle","membar":"membar","mio":"mio_throttle","misc":"misc","no_inst":"no_instructions","not_selected":"not_selected","short_sb":"short_scoreboard","sleep":"sleeping","tex":"tex_throttle","wait":"wait"}
partitions=[]
for stem,(launch,lines,directory) in sorted(records.items()):
 with (directory/f"{stem}_source.csv").open() as stream:rows=list(csv.reader(stream))
 index=next(i for i,row in enumerate(rows) if "Address" in row and "Source" in row);header=rows[index]
 totals={reason:0 for reason in reasons.values()}
 for row in rows[index+1:]:
  if len(row)!=len(header) or not row[header.index("Address")].startswith("0x"):continue
  record=dict(zip(header,row))
  for alias,reason in reasons.items():
   name="stall_"+alias
   if name not in record:name="smsp__pcsamp_warps_issue_stalled_"+reason
   totals[reason]+=int(record[name].replace(",",""))
 total=sum(totals.values())
 if total!=sum(x.samples or 0 for x in lines):raise SystemExit(f"Sample denominator mismatch: {stem}")
 partitions.append(dict(report=stem,family="smsp__pcsamp_warps_issue_stalled_<reason> (all samples; excludes selected; includes not_selected)",denominator=total,reason_counts=totals,reason_sample_percent={key:100*value/total for key,value in totals.items()}))
(base/"pc_sample_reason_totals.json").write_text(json.dumps(partitions,indent=2)+"\n")
with path.open("a") as stream:
 sums=[entry["nonoverlapping_warp_state_pct_sum"] for entry in audit]
 stream.write(f"\n## Numeric consistency limitation\n\nThe 17 non-overlapping warp-state percentage metrics (including selected, excluding two nested pipe subreasons) sum to {min(sums):.2f}–{max(sums):.2f}%. Treat these as individual NCU replay diagnostics, not a normalized wall-clock partition. The reason is not established; replay variation is a possibility, not a verified explanation. Details: comparison_numeric_audit.json.\n")
 stream.write("\n## Complete PC-sampled stall-reason partition\n\nAll-samples PC sampling summed across every SASS PC; selected excluded, not_selected included. The 16 reasons share one denominator within each kernel. Counts sum exactly to the denominator, shares to 100% apart from rounding. These are sample shares, not time allocation. Raw warp-state percentages above are not renormalized. Sampling intervals can differ between kernels.\n")
 for row in partitions:
  stream.write("\n### "+row["report"]+"\n\nDenominator: "+str(row["denominator"])+" all-stall samples.\n\n| Reason | Samples | Share of all-stall samples % |\n|---|---:|---:|\n")
  for reason in reasons.values():stream.write(f"| {reason} | {row['reason_counts'][reason]} | {row['reason_sample_percent'][reason]:.6f} |\n")
print("REPRODUCED",len(records),"reports",path)


# Keep hardware events and source excess wavefront denominators distinct.
discrepancies=[]
for directory in sorted((base/"exports").glob("ncu_main_*")):
 stem=directory.name
 summary=json.loads((directory/f"{stem}_summary.json").read_text())
 values={item["name"]:item["value"] for item in summary["metrics"]}
 attribution=json.loads((directory/f"{stem}_source_attribution.json").read_text())
 hardware=values["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]
 wavefronts=values["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"]
 a_excess=sum(g["excessive_wavefronts"] for g in attribution["groups"] if g["category"]=="A_ldmatrix")
 b_excess=sum(g["excessive_wavefronts"] for g in attribution["groups"] if g["category"]=="shared_load")
 source_excess=a_excess+b_excess
 discrepancies.append(dict(report=stem,hardware_load_conflicts=hardware,hardware_load_wavefronts=wavefronts,hardware_conflict_wavefront_percent=100*hardware/wavefronts,source_a_excessive=a_excess,source_b_excessive=b_excess,source_load_excessive=source_excess,hardware_minus_source=hardware-source_excess,gap_percent_of_hardware=100*(hardware-source_excess)/hardware,source_a_percent_of_source_load_excessive=100*a_excess/source_excess))
(base/"shared_counter_discrepancy.json").write_text(json.dumps(discrepancies,indent=2)+"\n")
with path.open("a") as stream:
 stream.write("\n## Hardware conflicts versus source excessive wavefronts\n\nHardware load-conflict events / load wavefronts is distinct from source excessive shared-load wavefronts (actual minus ideal). A=85.714286% applies only to source excessive loads, not an exact allocation of the hardware ~59% ratio. Async-copy store excess is excluded. The discrepancy is measured; its cause is not established.\n\n| Case | Hardware load conflicts | Source A+B excess | Hardware minus source | Gap / hardware % | A / source excess % |\n|---|---:|---:|---:|---:|---:|\n")
 for row in discrepancies:stream.write(f"| {row['report']} | {row['hardware_load_conflicts']:.0f} | {row['source_load_excessive']:.0f} | {row['hardware_minus_source']:.0f} | {row['gap_percent_of_hardware']:.6f} | {row['source_a_percent_of_source_load_excessive']:.6f} |\n")
