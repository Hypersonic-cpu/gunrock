| 指标 | 我建议的名字 | 用途 |
|---|---|---|
| **Long Scoreboard** | `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | ⭐ global/L1TEX load latency |
| **Short Scoreboard** | `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` | shared/MIO dependency |
| **LG Throttle** | `smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct` | LSU/global memory issue queue 压力 |
| **Barrier** | `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | CTA 内负载不均 / sync |
| **Wait** | `smsp__warp_issue_stalled_wait_per_warp_active.pct` | execution dependency |
| **Not Selected** | `smsp__warp_issue_stalled_not_selected_per_warp_active.pct` | ready 但另一个 warp 被选中 |
| **Selected** | `smsp__warp_issue_stalled_selected_per_warp_active.pct` | 真正 issue |

| 指标 | 解释 | 重要性 |
|---|---|---|
| **Active Warps / Scheduler** | 平均每个 scheduler resident 的 active warps | 判断 latency hiding capacity |
| **Eligible Warps / Scheduler** | 平均每周期 ready 的 warps | ⭐ 判断是否真的 starvation |
| **Issued Warps / Scheduler** | 平均每周期真正 issue 的 warps | ⭐ 实际 scheduler utilization |
