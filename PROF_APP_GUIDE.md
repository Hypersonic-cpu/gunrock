# Gunrock v2 profiling application guide

This file defines the v2-side application, input, validation, and command
policy for the methodology in [`PROFILING_TASKS.md`](PROFILING_TASKS.md). The
methodology document controls measurement, invocation selection, Nsys/Ncu
quality checks, and reporting; this file records the repository-specific
workload choices.

Concrete checkout, dataset, result-root, branch, GPU, and profiler-host values
are machine-local. Read `AGENTS.local.md` and set the names documented there
before running the commands below; this guide intentionally does not embed
those absolute paths.

## 1. Comparison scope

| Item | v2 comparison value |
| --- | --- |
| Repository | `REPO_ROOT` from `AGENTS.local.md` |
| Branch / revision | `PROFILE_BRANCH` / `PROFILE_COMMIT` from `AGENTS.local.md` |
| v1.2 reference checkout | `REFERENCE_REPO` from `AGENTS.local.md` (read-only) |
| v1.2 reference results | `REFERENCE_RESULTS_ROOT` from `AGENTS.local.md` (read-only) |
| v2 result root | `RESULTS_ROOT` from `AGENTS.local.md` |
| GPU | `GPU_VISIBLE` and host hardware from `AGENTS.local.md` |
| Input root | `DATA_ROOT` from `AGENTS.local.md` |
| Source vertex | `0` for BFS and SSSP |
| v2 advance load balance | `block_mapped` (the v2 counterpart of v1.2 `--advance-mode=LB`) |

Only files under the v2 checkout and the new v2 result root may be written by
this study. The v1.2 checkout and result root are inputs only.

## 2. Selected applications and matrices

The current comparison contains the 13 cases in the campaign manifest
`profiling/profile_config.json`:

| Application | Matrix files | Source/configuration | Validation |
| --- | --- | --- | --- |
| BFS | `soc-orkut`, `indochina-2004`, `roadNet-CA` | source `0`; `indochina-2004` uses 2 in-process profile repetitions | required; CPU reference |
| SSSP | `soc-orkut`, `indochina-2004`, `roadNet-CA` | source `0`; `indochina-2004` uses 2 in-process profile repetitions | required; CPU reference |
| BC | `soc-orkut`, `indochina-2004`, `roadNet-CA` | single source `0`; 10 profile repetitions | explicitly skipped by request |
| PageRank | `soc-orkut`, `indochina-2004`, `roadNet-CA` | default v2 PR convergence; 10 profile repetitions | explicitly skipped by request |
| Triangle Counting | `hollywood-2009` | `--reduce true`; 10 profile repetitions | explicitly skipped by request |

The same original files are passed directly to both versions; no generated,
canonicalized, deduplicated, or symmetrized input is permitted. BC, PR, and TC
are profiled despite the absence of a v2 CPU-reference validation path because
the user explicitly authorized those cases with validation recorded as
`SKIPPED`. They must not be reported as functionally validated. GraphSAGE is
not in this v2 manifest because there is no corresponding v2 executable in the
selected build.

The `soc-orkut`, `roadNet-CA`, and `hollywood-2009` headers are symmetric
pattern matrices; `indochina-2004` is a general pattern matrix. The v2 loader
keeps each header's interpretation: symmetric inputs expand off-diagonal
entries into directed CSR, while the general input remains as stored. Expected
loaded graph checks are:

| Matrix | Vertices | Initial Matrix Market entries | Expected loaded CSR entries |
| --- | ---: | ---: | ---: |
| `soc-orkut` | 2,997,166 | 106,349,209 | 212,698,418 |
| `indochina-2004` | 7,414,866 | 194,109,311 | 194,109,311 |
| `roadNet-CA` | 1,971,281 | 2,766,607 | 5,533,214 |
| `hollywood-2009` | 1,139,905 | 57,515,616 | 115,031,232 |

The application logs and exported metrics must confirm the vertex and loaded
edge counts. The source header and loader interpretation are recorded together;
the comparison must not silently treat these files as already-directed.

TC has an additional result-quality limitation in this batch: with the direct
v2 `tc` command and `--reduce true`, the native logs report `Total Graph
Traingles : 0`, and the Nsys kernel table contains only initialization and
reduction kernels rather than a meaningful triangle-intersection workload. TC
is therefore timing/profile data with validation skipped, not a functionally
validated v1.2-versus-v2 performance claim. The result must be revalidated or
the TC input/driver precondition corrected before interpreting its apparent
speedup.

## 3. Build and validation commands

The selected targets are rebuilt from the current branch with the Release
build type and CUDA architecture configured in `AGENTS.local.md`:

```bash
cmake --build build --target bfs sssp bc pr tc bfs_bench sssp_bench bc_bench pr_bench tc_bench -j"$(nproc)"
```

The reproducible entry points are kept in this checkout under `profiling/`:

```bash
profiling/profile_one.sh bfs soc-orkut 0 --stage all --modes natural,cold
profiling/profile_selected.sh --stage all --modes natural,cold
```

`profile_one.sh` resumes an existing case, validates it when required by the
manifest, collects native and Nsys data, selects exact-mangled invocation
points, runs both Ncu replay modes, and verifies each report.
`profile_selected.sh` is the one-script full-suite entry point: it reads all 13
cases from the one JSON manifest and writes the suite markers. Both scripts
copy `profiling/profile_config.json` into the result root and each case's
metadata; the result tree therefore contains the resolved study configuration
used to produce it. Use `--only app/graph` to resume or reproduce a subset
without changing the manifest.

The summary stage is reproducible from the recorded artifacts without
recapturing the profilers:

```bash
python3 profiling/summarize_invocation_profiles.py \
  --case-dir "$RESULTS_ROOT/full-suite/gunrock-v2/bfs/soc-orkut/src-0"

python3 profiling/summarize_architecture.py \
  --root "$RESULTS_ROOT" \
  --group full-suite \
  --reference-root "$REFERENCE_RESULTS_ROOT"
```

`profile_one.sh` invokes the per-case summarizer after both replay modes
finish, and `profile_selected.sh` invokes the architecture/comparison
summarizer when all 13 cases have summaries. The generated per-case files
are `summary/ncu-five-point.csv`, `ncu-memory.csv`, `ncu-scheduler.csv`,
`ncu-stalls.csv`, `ncu-launch.csv`, and `ncu-five-point.txt`. The suite files
are `full-suite/summary/architecture-case-summary.csv`,
`architecture-per-invocation.csv`, `comparison-summary.csv`, and
`full-suite/architecture-findings.md`.

Each required-validation case uses the example executable, because it contains
the v2 CPU reference check. The JSON export is retained with the validation log
so that the graph-size gate is independently auditable:

```bash
CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" \
"$REPO_ROOT/build/bin/bfs" \
  --market "$DATA_ROOT/<graph>.mtx" \
  --src 0 --num_runs 1 --advance_load_balance block_mapped \
  --validate --export_metrics \
  --json_dir "$RESULTS_ROOT/validation/gunrock-v2/bfs/<graph>/src-0/native" \
  --json_file validation.json

CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" \
"$REPO_ROOT/build/bin/sssp" \
  --market "$DATA_ROOT/<graph>.mtx" \
  --src 0 --num_runs 1 --advance_load_balance block_mapped \
  --validate --export_metrics \
  --json_dir "$RESULTS_ROOT/validation/gunrock-v2/sssp/<graph>/src-0/native" \
  --json_file validation.json
```

The BFS and SSSP validation logs must contain `Number of errors : 0`; a
missing validator, nonzero error count, graph-count mismatch, or changed input
interpretation invalidates the case and prevents profiling it. BC, PR, and TC
instead write `validation-skipped.json` and `validation=SKIPPED` with the
manifest reason. Those cases are accepted for the requested profiling batch,
but are excluded from any claim of functional validation.

For paired native timing, the same example command is run with
`--num_runs 10`, without `--validate`, and with an exported metrics file. The
reported per-run GPU times are summarized as median, mean, standard deviation,
minimum, and maximum. The v1.2 reference values are read from its read-only
result summaries generated from its native logs; toolchain/runtime differences are reported as comparison
confounders rather than hidden.

## 4. Profiling commands

The v2 example and v2 NVBench driver use the same matrix, source, load-balance
policy, and algorithm implementation. The benchmark driver is used for Nsys
and Ncu because it supports NVBench's required `--profile` mode. BC's
profiling driver uses the single-source API with explicit `--src=0`; PR and TC
use their corresponding native algorithm configuration. The configured
`--profile-runs` value is used inside the profiled process so the repeated
kernel identity has enough launches. Set `GUNROCK_PROFILE_NVTX=1` to retain the
v2 `algorithm` NVTX range; the range is closed after a device synchronize.

Nsys commands are host-native and must resolve the installed executable through
`which`:

```bash
CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" GUNROCK_PROFILE_NVTX=1 \
sudo "$(which nsys)" profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --stats=true \
  --force-overwrite=true \
  -o "$RESULTS_ROOT/full-suite/gunrock-v2/<app>/<graph>/nsys/report" \
  env CUDA_VISIBLE_DEVICES="$GPU_VISIBLE" GUNROCK_PROFILE_NVTX=1 "$REPO_ROOT/build/bin/<app>_bench" \
  --market "$DATA_ROOT/<graph>.mtx" \
  --devices 0 --profile [--src=0] [--profile-runs N] [--reduce true]
```

The bracketed arguments are applied by the manifest: `--src=0` is used for BC,
`--profile-runs 10` for BC/PR/TC, `--profile-runs 2` for BFS/SSSP on
`indochina-2004`, and `--reduce true` for TC.

The `env` wrapper is intentional: this host's sudo policy does not permit
preserving custom variables, so the wrapper supplies the GPU mask and NVTX
switch to the profiled child while only the approved Nsys binary runs through
sudo.

The Nsys trace is the source for the exact mangled kernel identity, complete
launch distribution, algorithm-range time, and the five invocation points
(`p10`, `p50`, `p90`, `peak`, and `tail`). No launch-zero default is allowed;
each selected repeated kernel must have at least five invocations.

For each selected exact mangled kernel and invocation point, run both natural
application replay and cold-kernel replay with Ncu. The concrete commands are
recorded in each case's `ncu/commands/` directory and use the required host
native form:

```bash
sudo "$(which ncu)" --target-processes application-only \
  --kernel-name-base mangled --kernel-name '<exact-mangled-name>' \
  --launch-skip <selected-launch-index> --launch-count 1 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section WarpStateStats --section SchedulerStats --section LaunchStats \
  --export "$RESULTS_ROOT/full-suite/gunrock-v2/<app>/<graph>/ncu/natural/<role>/report" \
  "$REPO_ROOT/build/bin/<app>_bench" --market "$DATA_ROOT/<graph>.mtx" \
  --devices 0 --profile [--src=0] [--profile-runs N] [--reduce true]
```

If the NVBench child-process layout prevents application-only attachment, the
case records the failed attempt and repeats with
`--target-processes all`; that deviation is never hidden. Ncu output is
accepted only when the report contains the requested exact kernel and one
launch. Raw values and units are retained, with the methodology quality flags
for sub-20-us durations and out-of-range derived rates.

## 5. Required result layout and comparison report

The v2 result root contains, for every selected case:

```text
validation/validation-pass.{json,txt}
full-suite/suite-complete.{json,txt}
full-suite/gunrock-v2/<app>/<graph>/
  metadata/
    environment.json
  native/
  nsys/
  invocations/
  ncu/
  summary/
```

`invocations/` retains the complete launch table, kernel rankings, dominant
kernel selections, and `selection.tsv`. `summary/` retains raw Nsys/Ncu
summaries, methodology quality flags, and per-case analysis. The final v2
report compares the selected v2 native timing with the corresponding v1.2
case, then reports algorithm-range and exact-kernel measurements separately;
kernel-sum values are not substituted for end-to-end timing.

The comparison uses the v1.2 recorded native average versus the v2 native
median as the primary statistic. The v2 native JSON retains all ten samples
and the mean, population standard deviation, minimum, and maximum. The Nsys
algorithm range is available for v2 because its capture enables the
`algorithm` NVTX range; the v1.2 reference trace does not contain that marker,
so its algorithm-range field remains unavailable rather than being inferred.
