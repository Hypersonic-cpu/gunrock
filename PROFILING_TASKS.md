# GPU profiling methodology

This document defines the measurement, invocation-selection, Nsight Compute,
metric, validation, and reporting methodology shared by a Gunrock profiling
campaign. It intentionally does not define the application matrix, graph
choices, source vertices, or per-application command-line recipes. Those
belong in the target implementation's application guide or campaign manifest.

## 1. Methodology scope

The method applies to repeated CUDA graph kernels and to single-invocation
kernels. It distinguishes three levels of evidence:

1. native application timing, which is the authoritative end-to-end runtime;
2. Nsight Systems timing and launch composition, which establishes whole-
   application structure and maps every selected kernel invocation;
3. Nsight Compute counters, which describe one selected kernel invocation at a
   time.

Do not use an NCU kernel duration as the application runtime, and do not
generalize one NCU launch to an entire iterative execution.

The application case matrix, binaries, input paths, and exact command
parameters belong in the application guide or campaign manifest. This method
is version-agnostic: it can be applied to either side of a version comparison
or to a standalone profiling campaign. A campaign's completion status belongs
in its result-root manifest, not in this methodology document.

## 2. Experimental invariants

For a paired implementation comparison, both binaries must receive the same
original physical input path and the same semantic workload. Do not rewrite an
input to make a loader match another loader. Preserve the source file's
declared directed/symmetric semantics and record any implementation-imposed
internal behavior separately.

The application-specific guide must define the per-case command. The profiling
method must then use the identical application arguments and repeated-run
count for:

```text
normal run
Nsight Systems run
each Nsight Compute point
```

Record the exact command beside every artifact. Never infer a command from
memory or substitute a short kernel name when an exact identity is available.

Use the approved tool invocation form:

```bash
sudo $(which ncu)
sudo $(which nsys)
```

The actual toolkit, driver, GPU, compiler, build commit, clock policy, and
profiler versions must be recorded in each case's environment metadata. Keep
the application runtime environment and profiler environment explicit when
they differ.

## 3. Reproducibility record

Every case records:

```text
application and graph labels
input path
source/configuration label when applicable
exact application command
normal-run count and warmup policy
Nsight Systems command and report
Nsight Compute command for each selected point
cache/replay mode
exact mangled kernel identity
NSYS invocation index and selection role
grid and block dimensions
metric names and units
verification result
```

The command and metadata are part of the result, not optional notes. If a
report exists without the command that generated it, do not silently reuse it.

## 4. Native timing methodology

Native execution is the authoritative performance measurement.

Record the warmup and repetition policy explicitly. Use the same policy for
the normal run, Nsys capture, and NCU application replay unless a deliberate
difference is documented. In general, keep the following fixed:

```text
same process where supported
same input, parameters, and source for every run
no Nsys/NCU attached to native timing
no artificial cache flush
```

Prefer the following timing boundary:

```text
cudaDeviceSynchronize()
start host timer
algorithm
cudaDeviceSynchronize()
stop host timer
```

This includes GPU execution and framework launch/synchronization gaps inside
the algorithm while excluding input loading and preprocessing. Store every
measured run and report:

```text
median
mean
standard deviation
minimum
maximum
```

Use the median as the primary runtime statistic and the mean when a
paper-compatible statistic is useful.

Also obtain the sum of GPU kernel durations from Nsys. The diagnostic

```text
application wall time - sum of GPU kernel durations
```

exposes framework-side launch, synchronization, and other gaps. It is not a
replacement for either native timing or the Nsys launch trace.

Before a campaign, inspect `nvidia-smi`, keep the GPU otherwise idle when
possible, and record any manually locked clocks. Do not mix clock-control
policies within a comparison.

## 5. Nsight Systems collection

Nsys establishes:

```text
which kernels run
how often each exact kernel runs
which exact kernel identity dominates aggregate GPU time
CPU launch and synchronization gaps
H2D/D2H transfers
cudaMalloc/cudaFree activity
whole-application timing structure
```

The generic collection template is:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
sudo $(which nsys) profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --stats=true \
  --force-overwrite=true \
  -o "$OUT/nsys/$CASE/report" \
  $CMD
```

Export a SQLite database and useful summaries with the installed Nsys version:

```bash
sudo $(which nsys) export --type=sqlite \
  --output="$OUT/nsys/$CASE/report.sqlite" \
  --force-overwrite=true "$OUT/nsys/$CASE/report.nsys-rep"

sudo $(which nsys) stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  "$OUT/nsys/$CASE/report.nsys-rep" > "$OUT/nsys/$CASE/stats.txt"
```

If report names differ, inspect `sudo $(which nsys) stats --help-reports` and
record the selected report names.

Capture every launch in the configured repeated application run. Do not
collapse repeated launches into an average before selection.

### 5.1 Exact kernel identity

CUPTI short names are not unique. Different kernels may share a displayed name
such as `Kernel`, and Nsys and NCU can render demangled namespaces differently.
Group and filter on the exact CUDA mangled symbol:

```text
NSYS: exact mangledName
NCU:  --kernel-name-base mangled --kernel-name <exact-symbol>
```

Retain the short and demangled names for readability, but never use either as
the identity when exact mangled data is available.

Rank exact identities by aggregate GPU time. Select the highest aggregate-time
repeated identity as the dominant kernel. A true dominant repeated identity
must have at least five invocations; if it does not, recapture with a larger
repeated application run count rather than substituting a lower-ranked kernel.

### 5.2 Invocation table

Write the complete launch table to:

```text
invocations/kernel-rankings.csv
invocations/dominant-invocations.csv
invocations/selection.tsv
invocations/selection.csv
invocations/analysis.json
invocations/analysis.txt
```

Each dominant-invocation row contains at least:

```text
zero-based index among launches of the exact mangled identity
NSYS start and end timestamps
duration in ns and ms
grid and block dimensions
device, context, stream, and grid ID when available
short, demangled, and exact mangled names
aggregate selected-function GPU time
per-invocation GPU-time contribution
share of all kernel GPU time
trace position
work-size proxy and proxy kind
near-empty threshold
meaningful/not-meaningful flag
iteration/frontier size when an existing activity or log exposes it
```

Number invocations in CUDA grid-ID launch order, with start time as a fallback.
This is the index namespace that maps to NCU `--launch-skip`, including when
streams overlap. Do not number by duration or completion order.

## 6. Invocation-aware five-point selection

A repeated graph kernel changes its frontier, edge work, cache state, and
scheduler behavior over time. The first launch may be initialization or a tiny
frontier; the longest launch alone does not characterize the execution.
Therefore, never profile launch 0 by default and never use one maximum as the
whole-application representative.

For each repeated dominant exact kernel, select exactly five distinct
invocations:

1. `p10`: small-work / low-percentile meaningful invocation;
2. `p50`: representative-medium invocation;
3. `p90`: representative-heavy invocation;
4. `peak`: maximum-work invocation, with duration breaking work-size ties;
5. `tail`: small-work invocation from the final execution phase when available.

Use the most meaningful non-intrusive work signal:

```text
frontier or edge-work size when already available
grid size when it varies with work
duration when neither work nor grid size is available
```

Record the chosen proxy explicitly. The current selector uses grid blocks when
the p10-to-p90 grid range varies by more than 20%; otherwise it uses duration.
The selector applies a 5% of the p90 proxy cutoff to mark near-empty launches
before computing p10/p50/p90. This cutoff is recorded in `analysis.json`; it
must not be changed silently.

The tail point is taken from the final 20% of matching launches and is kept as
a separate late-phase role even if its work is below the percentile cutoff. If
fewer than five meaningful distinct invocations remain, stop and recapture or
resolve the selection. Do not silently lower the cutoff or replace the
dominant kernel.

For every matching invocation calculate:

```text
invocation contribution (%) =
  invocation duration / aggregate duration of the selected exact function

heavy-regime GPU-time fraction (%) =
  duration of the top decile of meaningful invocations by work proxy
  / aggregate duration of every invocation of the selected exact function
```

The complete distribution and the heavy-regime fraction must be reported.
NCU metrics are five separate per-invocation observations. Do not calculate an
unweighted mean across the five roles. If an aggregate estimate is needed,
weight observations by the corresponding Nsys GPU time and state the weighting.

## 7. NCU collection

Before collection, verify that the installed NCU supports the requested
sections:

```bash
sudo $(which ncu) --list-sections
```

Every selected point uses:

```text
exact mangled kernel filter
matching Nsys invocation index
--launch-count=1
SpeedOfLight
MemoryWorkloadAnalysis
WarpStateStats
SchedulerStats
LaunchStats
```

The required natural-cache command is:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
sudo $(which ncu) \
  --target-processes application-only \
  --replay-mode application \
  --app-replay-match grid \
  --app-replay-mode relaxed \
  --cache-control none \
  --clock-control base \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SchedulerStats \
  --section LaunchStats \
  --rename-kernels 0 \
  --print-kernel-base mangled \
  --kernel-name-base mangled \
  --kernel-name "<EXACT_MANGLED_KERNEL>" \
  --launch-skip="<NSYS_INVOCATION_INDEX>" \
  --launch-count=1 \
  --force-overwrite \
  -o "$OUT/<GROUP>/<CASE>/ncu/natural/<ROLE>/report" \
  $CMD
```

Natural/application replay reruns the complete application for profiler passes
and preserves the preceding kernel history as closely as the profiler allows.
It is the primary microarchitecture observation for the selected point.

The required cold-cache control is:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
sudo $(which ncu) \
  --target-processes application-only \
  --replay-mode kernel \
  --cache-control all \
  --clock-control base \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SchedulerStats \
  --section LaunchStats \
  --rename-kernels 0 \
  --print-kernel-base mangled \
  --kernel-name-base mangled \
  --kernel-name "<EXACT_MANGLED_KERNEL>" \
  --launch-skip="<NSYS_INVOCATION_INDEX>" \
  --launch-count=1 \
  --force-overwrite \
  -o "$OUT/<GROUP>/<CASE>/ncu/cold/<ROLE>/report" \
  $CMD
```

Cold/kernel replay intentionally removes cache history. Report natural and
cold rows separately and, when useful, report their L2-hit delta. A difference
is evidence of cache-history sensitivity, not proof of an optimization benefit.

Use the same executable arguments and repeated-run count for normal, Nsys, and
NCU runs. Do not fall back to a broad short-name filter or to launch 0 if
application replay matching fails. Stop, inspect the NCU log and selected Nsys
row, correct the matching configuration, and rerun the validation gate.

After every capture, import raw metrics and write:

```bash
sudo $(which ncu) --import "$REPORT" --page raw --csv \
  --print-units base --print-kernel-base mangled > metrics-raw.csv
```

The verifier must confirm:

```text
selection index equals --launch-skip
--launch-count=1
exact mangled filter and mangled printed kernel name
required sections are present
required metric values are present
NCU grid/block equals the selected Nsys grid/block
report and source Nsys report exist
```

A generated `verification.json` is required beside each report.

### Launch-ID-driven NCU runner

`profiling/profile_suite.py --mode ncu` profiles only explicitly selected
CUDA kernel launches. First capture an Nsight Systems trace using the same
suite YAML and application arguments. Its `kernel-launches.csv` assigns a
1-based `kernel_launch_id` in GPU start-time order and records each launch's
exact mangled name plus its 0-based occurrence among launches with that exact
name. The NCU runner maps the requested trace ID to the exact-name occurrence
and uses `--launch-count=1` with application replay. Small setup kernels do
not change the selected ID.

```bash
python3 profiling/profile_suite.py \
  --mode nsys-trace \
  --suite-config profiling/bfs_roadnet_merge_path_v2.yaml \
  --output-dir /data-8/Profiling-Results/bfs-roadnet-mpv2-trace

python3 profiling/profile_suite.py \
  --mode ncu \
  --suite-config profiling/bfs_roadnet_merge_path_v2.yaml \
  --output-dir /data-8/Profiling-Results/bfs-roadnet-mpv2-ncu \
  --only bfs/roadNet-CA \
  --kernel-launch-table /data-8/Profiling-Results/bfs-roadnet-mpv2-trace/bfs/roadNet-CA/src-0/nsys-trace/kernel-launches.csv \
  --kernel-launch-id 123 --kernel-launch-id 124 --kernel-launch-id 125
```

Pass launch IDs taken from the trace table after confirming their demangled
names and grid sizes. The selected launch must have the same suite YAML,
executable, and arguments as the application replay; the runner rejects a
table copied from a different suite config. Each launch gets a separate
`.ncu-rep`, raw metrics CSV, details report, and `verification.json`.

The runner requests the six report sections listed above plus explicit A40
counters for kernel duration/cycles, SM/L2/DRAM elapsed-cycle frequency, L1
sector and L2 request hit/miss counts, L2 sectors/throughput, DRAM bytes and
throughput, occupancy, eligible warps, and the NCU-exposed warp stall reasons.
The `.per_second` elapsed-cycle counters report effective clock frequency in
Hz. On the A40, device memory is GDDR DRAM; results are labeled DRAM, not HBM.
L2 bandwidth can be derived as `lts__d_sectors.sum * 32 / duration_seconds`,
and device-memory bandwidth as `dram__bytes.sum / duration_seconds`; the raw
counter values and units remain in the CSV.

## 8. Counter and metric definitions

Keep raw NCU values, interpreted values, units, and quality flags separate.

### 8.1 L1/TEX and L2

For each selected invocation report:

```text
L1/TEX sector hit and miss counts
L1/TEX hit and miss rates
L2 request hit and miss counts
L2 hit and miss rates
L2 data sectors/traffic
L2 throughput percentage when exposed
```

Use the NCU-reported rates when their quality is valid. If an interpreted miss
rate is derived from a valid hit rate, record the formula and keep the
reported value separately. Do not reconstruct a rate from counters collected on
different NCU replay passes when the report flags the result as unreliable.

### 8.2 Effective bandwidth

Record raw bytes/sectors and the NCU unit. The current summary estimates:

```text
L2 effective bandwidth =
  L2 data sectors × 32 bytes / kernel duration seconds

device-memory effective bandwidth =
  dram__bytes.sum / kernel duration seconds
```

Also report read/write bytes separately when exposed and preserve NCU's
device-memory throughput percentage. These are per-invocation estimates, not
application averages.

Use the device-memory quantity appropriate to the GPU (for example DRAM/GDDR
or HBM) and label it according to the hardware and NCU metric. Do not call a
DRAM result HBM bandwidth, or infer HBM traffic when the device exposes only a
DRAM counter.

The reference collection requests these raw counters when supported by the
installed NCU version:

```text
l1tex__t_sectors.sum
l1tex__t_sectors_lookup_hit.sum
l1tex__t_sectors_lookup_miss.sum
l1tex__t_requests.sum
l1tex__m_xbar2l1tex_read_sectors.sum
lts__t_requests.sum
lts__t_requests_aperture_device_lookup_hit.sum
lts__t_requests_aperture_device_lookup_miss.sum
lts__d_sectors.sum
lts__d_sectors_fill_device.sum
dram__bytes.sum
dram__bytes_read.sum
dram__bytes_write.sum
```

If a metric is unavailable or renamed, record the substitute and its unit in
the case metadata; do not silently map an unrelated counter to the requested
quantity.

### 8.3 Scheduler, occupancy, and efficiency

Extract:

```text
active warps per scheduler
eligible warps per scheduler
issued warps or issue utilization
theoretical and achieved occupancy
grid and block size
registers per thread
shared memory per block
waves per SM
SM throughput and instruction/execution efficiency
```

### 8.4 Stall breakdown

Retain the complete stall-reason breakdown exposed by the installed NCU
version, including at least:

```text
long scoreboard
short scoreboard
wait
barrier
not selected
branch resolving
MIO/memory throttle when exposed
drain, membar, math-pipe, sleep, texture, and other exposed reasons
```

Stall fields ending in `.ratio` are not automatically percentages. Report the
value and the unit shown by NCU. In particular, do not call a long-scoreboard
ratio a percent without checking the exported unit.

### 8.5 Short and invalid rate metrics

Multi-pass hit-rate counters can be inconsistent or outside their expected
range for short non-steady-state launches. The current reporting rule is:

```text
duration < 20 us:
  retain raw value; mark under_20us_multipass_sensitive;
  exclude interpreted hit-rate, bandwidth-saturation, and
  stall-correlation classifications

reported rate < 0 or > 100:
  retain raw value; mark out_of_range;
  exclude interpreted rate from trends/classifications
```

Do not clamp an invalid value and do not silently infer a valid percentage.
This is particularly important for workloads with thousands of very short
launches. Long-lived selected points remain eligible when their own duration
meets the 20-us guidance.

## 9. Interpretation rules

Use metric combinations, not one counter.

A latency-leaning signature is:

```text
low or moderate device-memory bandwidth
high long-scoreboard behavior
low eligible-warp availability
resident/active warps that are not ready to issue
```

A bandwidth/queueing signature is:

```text
high device-memory utilization
high memory-related stalls
many outstanding accesses
eligible warps still available
```

A poor-parallelism/frontier signature is:

```text
low active and eligible warps
low bandwidth
small grids or frontiers
short kernels
```

A load-imbalance signature can include long tails, low average SM use, and
different behavior between high-degree and mesh-like graphs. Treat these as
hypotheses to validate against the full Nsys distribution.

Report separately:

```text
per-invocation NCU observations
Nsys aggregate GPU-time and heavy-regime contribution
whole-application native timing
```

Never extend a peak or tail observation to the whole application without the
normal-run timing and Nsys kernel composition.

## 10. Validation gate

Before a full suite, use the validation cases defined in the target
application guide or campaign manifest. The gate must exercise both a workload
with a few heavy iterations and a workload with hundreds or thousands of short
iterations. It
must also include an exact-kernel-name collision regression when applicable.

For every validation case require:

```text
complete Nsys per-invocation table
five distinct selected indices
p10/p50/p90/peak/tail labels in that order
five successful natural-cache NCU reports
dynamic --launch-skip matching each selected index
--launch-count=1
exact mangled kernel filter
NCU kernel identity/grid/block matching Nsys
raw and interpreted metric quality flags
heavy-regime GPU-time fraction
```

Explicitly verify that the old launch-0 failure is gone: the five selected
points must be distinct and selection must be driven by Nsys work/timing
metadata, not a hard-coded skip value. A full-suite script must stop if this
gate fails.

The campaign validation marker is stored under its result root:

```text
$RESULTS_ROOT/<campaign>/validation/validation-pass.json
$RESULTS_ROOT/<campaign>/validation/validation-pass.txt
```

## 11. Output directory structure

The result root is supplied by the campaign. A reusable layout is:

```text
<results-root>/<campaign>/
├── validation/
│   ├── <implementation>/<case>/
│   │   ├── native/run.log
│   │   ├── nsys/{report.nsys-rep,report.sqlite}
│   │   ├── invocations/{dominant-invocations.csv,selection.tsv,analysis.json}
│   │   └── ncu/natural/<role>/{report.ncu-rep,metrics-raw.csv,verification.json}
│   └── validation-pass.{json,txt}
└── full-suite/
    ├── <implementation>/<case>/
    │   ├── native/run.log
    │   ├── nsys/{report.nsys-rep,report.sqlite}
    │   ├── invocations/
    │   │   ├── kernel-rankings.csv
    │   │   ├── dominant-invocations.csv
    │   │   ├── selection.tsv
    │   │   └── analysis.json
    │   ├── ncu/{natural,cold}/<role>/
    │   │   ├── report.ncu-rep
    │   │   ├── metrics-raw.csv
    │   │   ├── command.txt
    │   │   └── verification.json
    │   └── summary/
    │       ├── ncu-five-point.csv
    │       ├── ncu-memory.csv
    │       ├── ncu-scheduler.csv
    │       ├── ncu-stalls.csv
    │       ├── ncu-launch.csv
    │       └── ncu-five-point.txt
    ├── summary/
    │   ├── architecture-case-summary.csv
    │   └── architecture-per-invocation.csv
    ├── architecture-findings.md
    ├── suite-complete.json
    └── status.txt
```

The application guide or campaign manifest defines the names represented by
`<implementation>` and `<case>`; the methodology does not assume a particular
application, graph, branch, or implementation version. Keep the Nsys report,
selection metadata, NCU reports, raw imports, commands, and verification files
together under the same case directory.

## 12. Reusable scripts

The profiling entry points in this repository are:

```text
profiling/profile_suite.py
profiling/profile_one.sh
profiling/analyze_nsys_invocations.py
profiling/verify_ncu_invocation.py
profiling/summarize_invocation_profiles.py
profiling/summarize_architecture.py
```

The YAML-driven suite implements `nsys-trace` and `nsys-metrics`; `ncu` is
reserved for later implementation. For example, to capture BFS on roadNet-CA
with device-wide GPU metrics:

```bash
python3 profiling/profile_suite.py \
  --mode nsys-metrics \
  --suite-config profiling/profile_suite.yaml \
  --output-dir "$RESULTS_ROOT/bfs-roadNet-metrics" \
  --data-root "$DATA_ROOT" \
  --build-bin "$REPO_ROOT/build/bin" \
  --device "$GPU_VISIBLE" \
  --only bfs/roadNet-CA
```

The suite YAML is the source for selected applications, graphs, expected graph
sizes, application arguments, metric set, and candidate sampling rates.
`nsys-metrics` retains each candidate report and selects the highest rate with
regular samples and no detected overflow warning. GPU metrics are device-wide
and include other activity on the selected GPU. `profile_one.sh` is only the
process/log/result-file wrapper called by `profile_suite.py`.

The existing invocation-selection and NCU verification utilities remain
available for future profiling modes. They must not hard-code
`--launch-skip 0`; each launch index must come from the saved Nsys selection
table and use `--launch-count=1`.

## 13. Required summaries

For each case, produce:

```text
invocations/kernel-rankings.csv
invocations/dominant-invocations.csv
invocations/selection.tsv
invocations/analysis.json
summary/ncu-five-point.csv
summary/ncu-memory.csv
summary/ncu-scheduler.csv
summary/ncu-stalls.csv
summary/ncu-launch.csv
summary/ncu-five-point.txt
```

The five-point table has one row per role and cache mode. It includes Nsys
duration and contribution, NCU duration, work proxy, L1/L2 counts and rates,
L2 bandwidth, device-memory bandwidth, occupancy, eligible warps, execution
efficiency, long scoreboard, and the full stall summary where available.

The architecture summary must state:

```text
which phases are latency-leaning
which selected points approach device-memory saturation
where bandwidth headroom exists
how L2 hit rate changes by regime
whether scoreboard and eligible-warp observations are correlated
how peak differs from tail
which prefetch hypotheses are promising, poor, or inconclusive
```

These are evidence-qualified classifications. Short or invalid rate points,
setup/reset-dominated selected functions, and whole applications whose dominant
kernel is only a small share of Nsys time must not be overgeneralized.

## 14. Campaign completion record

A campaign is complete only when its result-root verifier reports every planned
case and every case contains the normal-run log, Nsys report/export, complete
invocation metadata, five selected roles, and all requested NCU cache-mode
reports. Store a machine-readable completion marker and a short human-readable
status file under the result root. The application guide or campaign manifest
may separately record which cases and implementations were completed; that
status does not belong in this methodology document.

## 15. Final checklist

Before declaring any profiling campaign complete, confirm:

```text
original input path and semantics are recorded
loader V/E/interpretation gate passed
normal, Nsys, and NCU commands are saved
Nsys exact mangled identity is used
complete invocation distribution is saved
near-empty cutoff and work proxy are recorded
five distinct p10/p50/p90/peak/tail points exist
launch skip matches each Nsys invocation index
launch count is one
all required sections and counters are present
raw values and units are retained
short/out-of-range metric quality is flagged
natural and cold modes are separate
no unweighted cross-iteration NCU average is reported
native timing, Nsys timing, and NCU observations are clearly separated
full-suite and validation markers pass
```

## References

- NVIDIA Nsight Compute Profiling Guide and CLI documentation.
- NVIDIA Nsight Systems User Guide.
- Gunrock source and application documentation; application-specific choices are
  maintained in the target implementation's application guide.
