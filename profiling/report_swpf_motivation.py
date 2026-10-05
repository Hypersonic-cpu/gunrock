#!/usr/bin/env python3
"""Render the four-control experiment as paper-oriented tables and figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collect_swpf_motivation import COUNTERS, KEY, STATES
from diagnose_swpf import write_csv


MODES = ("original", "prefetch_only", "schedule_prefetch", "schedule_only")
LABELS = ("Original", "Hints only", "Schedule + hints", "Schedule only")


def read(path):
    return list(csv.DictReader(path.open()))


def markdown_table(rows, headers):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Diagnostics directory")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "motivation-controls"
    rows = read(output / "warp-cycle-breakdown.csv")
    performance = read(output / "kernel-performance.csv")
    lookup = {(tuple(r[k] for k in KEY), r["mode"]): r for r in rows}
    perf = {(tuple(r[k] for k in KEY), r["mode"]): r for r in performance}
    points = [tuple(r[k] for k in KEY) for r in rows if r["mode"] == "original"]
    names = [f"{p[0].upper()} / {p[1]}\n{p[3]} #{p[4]}" for p in points]
    speedup_rows, best_rows, contribution = [], [], []
    csv_best, bottlenecks = [], []
    for point in points:
        values = [float(perf[point, mode]["speedup"]) for mode in MODES]
        speedup_rows.append([f"{point[0]} / {point[1]} / {point[3]} #{point[4]}",
                             *[f"{v:.3f}×" for v in values[1:]]])
        mode = max(MODES[1:], key=lambda m: float(perf[point, m]["speedup"]))
        row = lookup[point, mode]
        speedup = float(perf[point, mode]["speedup"])
        csv_best.append({**row, "speedup": speedup, "positive": speedup > 1})
        gbps = float(row["dram_bytes"]) / float(row["duration_ns"])
        leading_state = max(("long_scoreboard_pct", "short_scoreboard_pct", "sync_pct"),
                            key=lambda k: float(row[k]))
        bottlenecks.append(dict(zip(KEY, point), mode=mode, speedup=speedup,
                                leading_of_long_short_sync=leading_state,
                                **{k: row[k] for k in ("long_scoreboard_pct", "short_scoreboard_pct",
                                                      "sync_pct", "eligible_warps_per_scheduler",
                                                      "eligible_active_warp_pct", "issue_slots_used_pct",
                                                      "no_issue_slot_pct", "registers_per_thread", "occupancy_pct")},
                                dram_GB_per_second=gbps, pct_vendor_peak_696_GBps=100 * gbps / 696))
        best_rows.append([f"{point[0]} / {point[1]} / {point[3]} #{point[4]}", mode,
                          f"{speedup:.3f}×", *[f"{float(row[k]):.2f}%" for k in
                          ("long_scoreboard_pct", "sync_pct", "selected_pct", "issue_slots_used_pct")]])
        times = {m: float(lookup[point, m]["duration_ns"]) for m in MODES}
        contribution.append(dict(zip(KEY, point),
                                 schedule_only_speedup=times["original"] / times["schedule_only"],
                                 hints_on_original_speedup=times["original"] / times["prefetch_only"],
                                 hints_on_schedule_speedup=times["schedule_only"] / times["schedule_prefetch"],
                                 combined_speedup=times["original"] / times["schedule_prefetch"]))
    write_csv(output / "best-measured-combinations.csv", csv_best)
    write_csv(output / "residual-bottlenecks.csv", bottlenecks)
    write_csv(output / "contribution-summary.csv", contribution)
    definitions = []
    for label, metric in COUNTERS.items():
        denominator = ("active warp-cycles" if label.endswith("_pct") and label[:-4] in STATES
                       else "scheduler active cycles" if label in
                       ("active_warps_per_scheduler", "eligible_warps_per_scheduler",
                        "issued_warps_per_scheduler", "issue_slots_used_pct") else "NCU metric definition")
        definitions.append(dict(column=label, source=metric, denominator=denominator,
                                measurement="direct NCU raw metric"))
    for label, formula, denominator in (
            ("sync_pct", "barrier_pct + membar_pct", "active warp-cycles"),
            ("eligible_active_warp_pct", "100 * eligible_warps_per_scheduler / active_warps_per_scheduler", "active warp-cycles"),
            ("no_issue_slot_pct", "100 - issue_slots_used_pct", "scheduler active cycles")):
        definitions.append(dict(column=label, source=formula, denominator=denominator, measurement="derived"))
    write_csv(output / "metric-definitions.csv", definitions)
    phases = read(output / "phase-performance.csv")
    phase_best = {}
    for phase in phases:
        key = phase["app"], phase["graph"], phase["phase"]
        if key not in phase_best or float(phase["phase_speedup"]) > float(phase_best[key]["phase_speedup"]):
            phase_best[key] = phase
    write_csv(output / "best-phase-candidates.csv", list(phase_best.values()))

    x = np.arange(len(points))
    fig, ax = plt.subplots(figsize=(15, 4.8))
    colors = ("#8c8c8c", "#e69f00", "#0072b2", "#009e73")
    for i, mode in enumerate(MODES[1:]):
        ax.bar(x + (i - 1) * .25, [float(perf[p, mode]["speedup"]) for p in points],
               width=.24, label=LABELS[i + 1], color=colors[i + 1])
    ax.axhline(1, color="black", linewidth=.8)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Exact-launch NCU speedup vs original")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("four-control-speedups." + suffix), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), sharey=True)
    categories = ("long_scoreboard_pct", "short_scoreboard_pct", "sync_pct",
                  "selected_pct", "not_selected_pct", "other_pct")
    catlabels = ("Long scoreboard", "Short scoreboard", "Barrier + membar",
                 "Selected", "Not selected", "Other")
    palette = ("#d55e00", "#e69f00", "#cc79a7", "#009e73", "#56b4e9", "#bdbdbd")
    for ax, point, name in zip(axes.flat, points, names):
        bottom = np.zeros(4)
        for category, label, color in zip(categories, catlabels, palette):
            vals = []
            for mode in MODES:
                row = lookup[point, mode]
                value = (100 - sum(float(row[k]) for k in categories[:-1])
                         if category == "other_pct" else float(row[category]))
                vals.append(value)
            ax.bar(np.arange(4), vals, bottom=bottom, color=color, label=label)
            bottom += vals
        ax.set_title(name.replace("\n", " / "), fontsize=9)
        ax.set_xticks(np.arange(4), ("Orig", "PF", "Sched+PF", "Sched"), fontsize=9)
        ax.set_ylim(0, 100)
    axes[0, 0].set_ylabel("Active warp-cycles (%)")
    axes[1, 0].set_ylabel("Active warp-cycles (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("active-warp-breakdown." + suffix), dpi=220)
    plt.close(fig)

    positive = [r for r in csv_best if r["positive"]]
    ranges = {k: (min(float(r[k]) for r in positive), max(float(r[k]) for r in positive))
              for k in ("long_scoreboard_pct", "sync_pct", "selected_pct", "issue_slots_used_pct")}
    state_sum_min = min(float(r["state_sum_pct"]) for r in rows)
    state_sum_max = max(float(r["state_sum_pct"]) for r in rows)
    speedup_table = markdown_table(speedup_rows, ("Kernel", "仅 PF", "调度 + PF", "仅调度"))
    best_table = markdown_table(best_rows, ("Kernel", "最好实测组合", "Speedup", "Long scoreboard",
                                           "Sync", "Selected", "Issue slots used"))
    report = f"""# 软件预取、线程内部调度与硬件机会

## 实验范围与四个对照

- Original：正式程序 `--swpf none`，原始 Merge Path 串行边循环。
- Prefetch only：隔离编译的 `motivation-lookahead/prefetch-only-build/bin/{{bfs,sssp,bc}}`；保留原始串行循环、线程划分、当前边的需求加载位置和操作顺序，不缓存边记录。插入本 CSR 行内 `edge + distance` 的索引/权重提示，额外读取未来目的顶点来提前发出状态提示；边界检查复用已有 shared-memory 行终点，不增加 CSR 行指针加载。
- Schedule + prefetch：正式 GP/SPP，包含实际边分组/流水线、真实 CSR 提前加载和 PTX 提示。
- Schedule only：已有 no-hint 隔离版本，保留 GP/SPP 的实际加载提前、分组与流水线，仅去掉 PTX 提示。

“调度”指线程内部访存/处理组织；所有版本仍使用 Merge Path 分区与线程工作划分，不是修改 GPU 硬件 warp scheduler。

仅 PF 需要额外的未来索引加载来发现间接地址；它是明确、可复现的软件 lookahead 方案，不是能将提示和地址发现成本完全正交拆开的神奇对照。Schedule only 也包含真实软件 load-ahead，不能称为“完全没有软件预取”。上述因素相互作用，收益不能线性相加。另保留原始循环内即时状态提示的更短 lead-time 对照（prefetch_immediate），见 `all-prefetch-controls.csv`。

八个定位过的热点、四个主对照，共 32 个新 NCU 报告，加上 8 个即时提示对照，共 40 个；统一 metric 集合。另补 14 个仅 PF 的 NSYS trace（7 lookahead + 7 immediate）。两个 PF 版本各有 42 个小图 + 7 个完整图 CPU-reference 验证，均通过。source=13331（大图），num_runs=1，A40/SM86，application replay，strict grid match，cache-control=none，clock-control=none；每个报告核对精确 mangled function、1-based invocation 和 grid。

GP/SPP 参数来自先前 native sweep 的候选，保持同一热点、target 与 distance；不是独立调优后的全部软件方法最优解。仅 PF 的 `--swpf gp|spp` 在此隔离版本中只用于选择同名模板/参数，二者都执行原始串行边循环。

`best-phase-candidates.csv` 从全部已有 GP/SPP/no-hint phase trace 和此次 PF trace 中选取最快候选；某些 GP no-hint 的 phase 总时间比下面选定的 SPP no-hint 更好。下面的“最好实测组合”仅表示该热点、该 GP/SPP 参数下三种主优化对照的最好值，不声称所有软件配置中的最优。

## 1. 预取和调度贡献

以下都是 NCU kernel 时间比 `original / variant`，大于 1 表示加速；不是全程序加速。

{speedup_table}

详细分解见 `contribution-summary.csv`：

- `schedule_only_speedup`：Original / Schedule only。
- `hints_on_original_speedup`：Original / Prefetch only。
- `hints_on_schedule_speedup`：Schedule only / Schedule + prefetch；直接衡量在同一 GP/SPP 组织下添加提示的净收益。
- `combined_speedup`：Original / Schedule + prefetch。

`phase-performance.csv` 保存 NSYS 的所有 Merge Path kernel 时间之和（按 BC 前后向分别统计）；其中原始、GP/SPP 和 no-hint 使用前次 trace，PF 使用此次 trace，属于单次测量，不是 NCU 时间或 native 10 次中位数。

## 2. 优化后的 active warp-cycle breakdown

每个点选择此次三种优化中实测最快的组合，保留原始对照；详细四种组合数据没有筛除负收益。

{best_table}

正收益的最好实测组合中，long scoreboard 范围 {ranges['long_scoreboard_pct'][0]:.2f}–{ranges['long_scoreboard_pct'][1]:.2f}%，sync {ranges['sync_pct'][0]:.2f}–{ranges['sync_pct'][1]:.2f}%，selected {ranges['selected_pct'][0]:.2f}–{ranges['selected_pct'][1]:.2f}%，issue-slot 利用率 {ranges['issue_slots_used_pct'][0]:.2f}–{ranges['issue_slots_used_pct'][1]:.2f}%。

### 分母与解释

- `long_scoreboard_pct`、`barrier_pct`、`membar_pct`、`selected_pct` 等直接读取 NCU `smsp__warp_issue_stalled_*_per_warp_active.pct`，是 active warp-cycle 的状态比例。它们不是 kernel wall-time 比例。此次 18 种状态之和为 {state_sum_min:.2f}–{state_sum_max:.2f}%，反映多 pass 下计数器/分母与原子竞争导致的重放变化；CSV 保留 raw 值，不强行归一化。堆叠图的 Other 包含剩余状态与这项重放残差，仅用于展示；精确数值以 CSV 为准。
- `sync_pct = barrier_pct + membar_pct`；`wait_pct`（固定延迟等待）、short scoreboard 和 atomic 依赖没有混进 sync。
- `selected_pct`：active warp-cycle 中，该 warp 被 scheduler 选中发射的比例；`not_selected_pct`：已经 eligible、但未被选中的比例。
- `eligible_warps_per_scheduler`：每个 scheduler 的每个 active cycle 平均有多少个 ready warp。
- `eligible_active_warp_pct = eligible / active × 100`：每个 cycle active warp 中平均可发射的比例。
- `issue_slots_used_pct`：scheduler active cycle 中实际发射的比例；`no_issue_slot_pct = 100 - issue_slots_used_pct`。这比 selected 更直接回答“每个 cycle 能不能发射”。这些统计平均值有多 pass 的四舍五入/重放差异，不应要求两个 independently measured ratios 完全恒等。
- Long scoreboard 是 L1TEX 路径的长依赖等待，可能涉及 global/local/texture 数据与相关依赖，不能直接等同于 DRAM miss、纯带宽瓶颈或可由硬件预取消除的时间。
- 一个版本可以更快、但 long scoreboard 的百分比更高：其他指令/等待减少后，残余访存依赖在 warp-cycle 分母里占比更大。因此不得仅凭一个 stall 百分比给性能排序，也不能直接计算 `1 / (1 - stall_pct)` 当作硬件加速上限。

NCU 定义依据：[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)；本机 NCU 2024.3.2 的 metric query 和 raw CSV 是本实验具体指标的依据。

## 3. 残余瓶颈与可用于论文的 observations

### Observation 1：软件调度可以提速，但仍未充分隐藏访存依赖

SSSP 等热点的正收益说明 workload 并非无法优化；在优化后仍应联合报告 long scoreboard、eligible warps 和 issue-slot 空洞，而不能只展示原始程序的 stall。CSR edge → destination → state/atomic 的间接地址链使后续地址晚于当前需求加载才可用，原始循环的即时提示难以获得足够提前量。

硬件机会：从 CSR 索引流提前解析目的顶点，使用脱离 SM 执行路径的请求队列建立更长 lead time / 更多 memory-level parallelism。硬件还需要明确如何获得 frontier、行范围与 phase 信息；普通 stride prefetcher 不能自动解决所有间接访问。

### Observation 2：在 SM 上实现长提前量有指令与寄存器成本

GP/SPP 必须显式维护边记录、buffer 和流水线状态；warp_instructions、registers_per_thread、occupancy_pct 的对照可量化成本。仅 PF 不改变原始循环但提前量短；长提前量方案可能因寄存器压力损失 latency hiding，收益随图和 phase 变化。

硬件机会：将地址生成、缓存提示和预取记录存储移到专用 engine，减少占用 SM issue slots 与线程寄存器的成本；不能据此假设面积/能耗免费，也不能把 issue 空洞当作可实现的速度上限。

### Observation 3：准确性与带宽控制比盲目增加提示更重要

BC backward 的 next-level 条件让很多提前读取的 sigma/delta 最终不被使用。前次结构统计：Indochina level17 的有效 next-level 边仅 1.312%，Orkut level4 为 9.817%；前次一致 metric 的直接 DRAM 采样中，Orkut backward 提示版 / 同调度 no-hint 版产生约 4.48× DRAM bytes。此次 `dram_bytes`、`l2_sectors` 与运行时间可独立复核方向。

硬件机会：先解析 labels/phase 条件，再决定是否继续追踪 sigma/delta；同 cache-line 请求合并、重复抑制、按需求流压力节流，并选择适合原子目标的 L2 路径。这里只证明当前无条件提示的浪费，不证明任何硬件能无成本预测应用语义。

### Observation 4：prefetch 不是所有残余瓶颈的解法

Barrier、shared-memory/short-scoreboard 依赖、原子更新冲突、线程分歧、窄 frontier、grid 尾部和 host-side CUDA 空洞，不能简单靠预取消除。尤其 roadNet 有数百个很短的 iteration，不能将一个热点的 kernel 加速套到整体程序。优化后 sync 若仍高，需要另行研究分区、CTA 协作或同步；GPU 数据不能证明 CPU/API gap 能由 prefetch engine 修复。

`residual-bottlenecks.csv` 同时报告 long/short/sync、issue 空洞和直接 DRAM bytes / kernel duration 算得的 GB/s，并与 [A40 标称 696 GB/s](https://www.nvidia.com/en-us/data-center/a40/) 比较；标称比例不是本机实测可达带宽比例或严格 roofline。若带宽已很忙，增加预取请求可能恶化拥塞，engine 需要减少重复流量/改善合并，而不能只增加 outstanding requests。Barrier 等待也可能部分反映各 warp 的访存进度差异，但这里没有证据将全部同步等待归因于 cache misses。

## 可以与不可以声称的结论

可以声称：**在已评估的软件调度/预取方案下，多个热点虽然加速，仍存在显著访存依赖和发射空洞；显式预取还伴随提前量、SM 资源和带宽准确性之间的权衡。这提供了研究解耦、间接、按需节流的硬件预取 engine 的动机。**

不能声称：软件优化没有效果、所有软件方法都不足、硬件必然更快/最佳、所有 long scoreboard 都可以消除，或将本实验替代硬件方案验证。要证明硬件更好，需要 engine 原型/仿真与优化后的 Schedule only 基线比较，报告 IPC/时间、准确性/覆盖率/及时性、额外流量、面积/能耗，并保留仍有同步和原子开销的上限。

建议 motivation 句子：**“Software restructuring improves graph traversal, yet optimized kernels retain substantial memory-dependency stalls and idle issue slots. Explicit software prefetching trades lookahead against instruction/register overhead and redundant traffic, motivating a decoupled, dependency-aware hardware engine.”**

## CSV 和图

- `kernel-performance.csv`：32 行，四对照的 kernel 时间、speedup、traffic 与 instruction 比例。
- `contribution-summary.csv`：8 行，调度贡献及两种上下文中的 hint 净收益。
- `warp-cycle-breakdown.csv`：32 行，完整 18 状态、eligible/active/issued warps、issue-slot、occupancy、registers、DRAM/L2 和 report 路径。
- `all-prefetch-controls.csv`：40 行，包括上面的主对照与额外即时提示对照。
- `positive-combinations.csv`：所有此次正收益的非 Original 组合（单次采样判断，没有显著性检验）。
- `best-measured-combinations.csv`：8 行，每点的最好实测软件组合及残余瓶颈。
- `residual-bottlenecks.csv`：8 行，主要状态、issue 空洞及有效 DRAM 流量速率；供区分延迟、资源与同步限制。
- `phase-performance.csv`：NSYS phase 汇总，包含前次所有 GP/SPP 候选的对照；不是所有全程序时间。
- `best-phase-candidates.csv`：每个 app/graph/phase 的最快已有 phase 候选，避免将代表性 SPP 热点误称为全局最佳软件基线。
- `metric-definitions.csv`：每列的 NCU metric 或公式，以及分母。
- `four-control-speedups.pdf/png`、`active-warp-breakdown.pdf/png`：论文可导出的独立图。

限制：每个 application replay process 的 num_runs=1；计数器可能多 pass；原子竞争导致 frontier 内顺序/缓存历史可变化。未锁定时钟，导出实际时钟供审查。8 个代表性热点不是所有 iteration 加权统计；未对 PF-only 做全参数 sweep，不应将选定参数称为全局最优。现有大图二进制权重全部为1，SSSP 的性能结论限定这些输入。
"""
    (output / "analysis-zh.md").write_text(report)
    extra = root / "motivation-lookahead"
    manifest = dict(main_ncu_reports=len(rows), total_new_ncu_reports=len(read(output / "all-prefetch-controls.csv")),
                    nsys_traces=len(list((output / "nsys").rglob("*.nsys-rep"))) + len(list((extra / "nsys").rglob("*.nsys-rep"))),
                    validation_cases=len(read(output / "validation-summary.csv")) + len(read(extra / "validation-summary.csv")),
                    warp_state_sum_min=min(float(r["state_sum_pct"]) for r in rows),
                    warp_state_sum_max=max(float(r["state_sum_pct"]) for r in rows),
                    clock_hz_min=min(float(r["sm_clock_hz"]) for r in rows),
                    clock_hz_max=max(float(r["sm_clock_hz"]) for r in rows),
                    replay="application", num_runs=1, source=13331,
                    report_sha256=hashlib.sha256(report.encode()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
