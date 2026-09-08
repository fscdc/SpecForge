#!/bin/bash
# Cluster-wide PBS status.
#
# `qstat` only ever shows your own jobs here -- the server sets
# `query_other_jobs = False`, so `qstat -a`, `qstat -u '*'` and `qselect` all
# come back empty for everyone else's work. Everything below therefore comes
# from the two sources that are NOT gated: `qstat -B/-Q` aggregate counters,
# and `pbsnodes`, whose per-node `jobs =` field names every job id occupying a
# core slot regardless of who owns it. That field is what makes a real
# running-job census possible; only the job *details* (owner, name, walltime)
# stay private.
#
# Usage: bash scripts/cluster_status.sh [-n]     (-n also lists busy nodes)
set -uo pipefail

echo "═══════════ 服务器 ═══════════"
qstat -Bs 2>/dev/null

echo
echo "═══════════ 队列（排队/运行，积压比）═══════════"
qstat -Q 2>/dev/null | awk '
  NR>2 && ($6+$7+$8)>0 {
    ratio = ($7 > 0) ? sprintf("%.1fx", $6/$7) : ($6 > 0 ? "  --" : " 0.0x")
    printf "  %-14s 排队 %4d  运行 %4d  暂挂 %4d   积压 %s\n", $1, $6, $7, $8, ratio
  }'

echo
echo "═══════════ GPU 资源 ═══════════"
pbsnodes -a 2>/dev/null | awk '
  /^[^ \t]/                          { n=$1; order[++cnt]=n }
  /^     state = /                   { st[n]=$3 }
  /resources_available.gpu_model/    { gm[n]=$3 }
  /resources_available.ngpus/        { tot[n]=$3 }
  /resources_assigned.ngpus/         { use[n]=$3 }
  END {
    for (i=1; i<=cnt; i++) {
      n=order[i]; if (tot[n]+0 == 0) continue
      T+=tot[n]; U+=use[n]; bt[gm[n]]+=tot[n]; bu[gm[n]]+=use[n]; nodes[gm[n]]++
      if (st[n] !~ /free|job-busy/) bad[st[n]]++
    }
    printf "  总计 %d 张   已分配 %d 张 (%.0f%%)   空闲 %d 张\n\n", T, U, U/T*100, T-U
    for (x in bt)
      printf "  %-6s %2d 节点   %3d/%-3d 已用 (%.0f%%)   空闲 %d\n",
             x, nodes[x], bu[x], bt[x], bu[x]/bt[x]*100, bt[x]-bu[x]
    for (x in bad) printf "\n  异常节点 [%s]: %d 个\n", x, bad[x]
  }'

echo
echo "═══════════ 在跑的作业（全用户，来自 pbsnodes）═══════════"
pbsnodes -a 2>/dev/null | awk '
  /^[^ \t]/                       { n=$1; order[++cnt]=n }
  /resources_available.gpu_model/ { gm[n]=$3 }
  # `jobs =` is printed BEFORE gpu_model, so stash the raw line and resolve the
  # model in END rather than reading gm[] mid-stream (it is not populated yet).
  /^     jobs = /                 { line=$0; sub(/^     jobs = /,"",line); raw[n]=line }
  END {
    for (i=1; i<=cnt; i++) {
      n=order[i]; if (raw[n]=="") continue
      split(raw[n], slots, ", ")
      for (s in slots) {
        split(slots[s], p, "/"); j=p[1]
        cores[j]++
        model[j]=gm[n]
        if (index(" " where[j] " ", " " n " ") == 0) { where[j]=where[j] n " "; nn[j]++ }
      }
    }
    for (j in cores) { total++; bym[model[j]]++; if (nn[j]>1) multi++ }
    printf "  GPU 节点上共 %d 个作业", total
    printf "   （跨多节点 %d 个）\n\n", multi
    for (x in bym) printf "  %-6s %d 个作业\n", x, bym[x]
  }'

if [ "${1:-}" = "-n" ]; then
    echo
    echo "═══════════ 各节点占用 ═══════════"
    pbsnodes -a 2>/dev/null | awk '
      /^[^ \t]/                       { n=$1; order[++cnt]=n }
      /^     state = /                { st[n]=$3 }
      /resources_available.gpu_model/ { gm[n]=$3 }
      /resources_available.ngpus/     { tot[n]=$3 }
      /resources_assigned.ngpus/      { use[n]=$3 }
      /^     jobs = /                 { line=$0; gsub(/[^,]/,"",line); nj[n]=length(line)+1 }
      END {
        for (i=1; i<=cnt; i++) {
          n=order[i]; if (tot[n]+0 == 0) continue
          printf "  %-12s %-5s %d/%d GPU  %3d 核槽  %s\n",
                 n, gm[n], use[n], tot[n], nj[n], (use[n]<tot[n] ? "← 有空闲" : st[n])
        }
      }' | sort -k2
fi
