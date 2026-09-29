# Gunrock v2 profiling application guide

`profiling/profile_suite.py` is the suite entry point. It reads the
application, graph, expected graph sizes, and application arguments from
[`profiling/profile_suite.yaml`](profiling/profile_suite.yaml), which is based
on the native timing suite with `runs: 1` for one algorithm iteration per
profiled application.

The current implementation supports `nsys-trace` and `nsys-metrics`. The `ncu`
mode is reserved for later implementation. Trace mode records CUDA activity
and exports a kernel launch table. Metrics mode adds device-wide GPU metric
sampling to the same `.nsys-rep`, exports the samples to CSV, and tries the
configured rates from 10 kHz through 200 kHz, selecting the highest rate with
regular samples and no detected overflow warning. On the current A40,
`ga10x-gfxt` includes GPU memory bandwidth, SM/warp occupancy, and L1/L2 cache
metrics. These hardware samples cover the whole device, so concurrent GPU work
will also affect them.

## Run a case or the configured suite

Run from the repository root. Use a new, empty output directory:

```bash
python3 profiling/profile_suite.py \
  --mode nsys-trace \
  --suite-config profiling/profile_suite.yaml \
  --output-dir /data-8/Profiling-Results/0929-profile-suite \
  --data-root /data-8/Graph-Datasets/mtx \
  --build-bin build/bin \
  --device 0
```

For device-wide GPU metrics, use `--mode nsys-metrics` with a new output
directory. The metric set and rate progression are read from the `gpu_metrics`
section of the suite YAML:

```bash
python3 profiling/profile_suite.py \
  --mode nsys-metrics \
  --suite-config profiling/profile_suite.yaml \
  --output-dir /data-8/Profiling-Results/0929-profile-metrics \
  --data-root /data-8/Graph-Datasets/mtx \
  --build-bin build/bin \
  --device 0 \
  --only bfs/roadNet-CA
```

By default, the runner schedules only applications present in the YAML. To
profile one graph case from that suite, add:

```bash
--only bfs/roadNet-CA
```

The build directory can also be selected with `--build-bin` or the
`GUNROCK_BUILD_BIN` environment variable; absent either, the runner uses
`<repository>/build/bin`. Nsys is resolved from `PATH` and invoked using
`sudo -n`, as required by the host profiling policy in `AGENTS.local.md`.

## Output

Each case is written under `<output-dir>/<application>/<graph>/src-<source>/`:

```text
application-output/native.json       # when the app's YAML args request it
nsys-trace/command.txt               # exact Nsys and application command
nsys-trace/run.log                   # raw combined Nsys/application log
nsys-trace/report.nsys-rep           # Nsight Systems trace
nsys-trace/cuda-gpu-trace.csv        # raw GPU activities, including memory copies
nsys-trace/kernel-launches.csv       # kernel-only start/end/duration/grid table
nsys-trace/stats-command.txt
nsys-trace/status.txt
nsys-metrics/attempts.csv            # status and observed rate for each attempt
nsys-metrics/selection.json          # selected rate and available metric names
nsys-metrics/attempts/<rate>Hz/
  report.nsys-rep                    # CUDA trace and sampled GPU metrics
  report.sqlite                      # exported Nsight Systems data
  gpu-metrics.csv                    # timestamped metric series
  gpu-metrics-summary.csv            # per-metric min/mean/max and units
  run.log
```

Nsight Systems documents GPU metric sampling from 10 Hz through 200 kHz. This
suite tests 10, 50, 100, and 200 kHz; the report summary records the observed
cadence and gaps for each attempt. The `Throughput %` metrics are normalized to
peak throughput; the CSV retains each metric's unit in its column name.

`profile_one.sh` is the small execution helper called by the Python suite. It
only launches the supplied command, saves its combined output, returns its
exit code, and requires the expected result file to be non-empty.
