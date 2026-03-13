# Password Hashing Benchmark & Crack-Time Estimator

This script project provides a comprehensive benchmarking suite for measuring the performance of various password hashing algorithms on CPU and optionally GPU (OpenCL / Vulkan).
It also includes a rudimentary brute-force crack-time estimator based on measured throughput.

The tool measures:
- MD5, SHA-256
- bcrypt (multiple cost factors)
- scrypt (configurable N, r, p)
- Argon2 (configurable memory/time cost)
- Parallel CPU performance across multiple worker processes
- Optional GPU benchmarks (OpenCL or Vulkan)

All results can be exported as JSON for later analysis.
Plotting results also available.

---

## Features

- Synchronous and parallel CPU benchmarking
- Automatic detection of available GPU backends
- OpenCL MD5-round benchmark (real kernel execution)
- Vulkan compute benchmark (full 64-round MD5 execution)
- Automatic keyspace estimation for common character sets
- Crack time estimation for multiple attacker models
- Configurable rounds, runs, algorithms, GPU paths, plotting, and output

---

## Installation

### Dependencies

All being done in a python virtual environment.

```bash
pip install bcrypt argon2-cffi psutil vulkan pyopencl scipy pandas matplotlib seaborn
```

Optional:
- A working OpenCL runtime (GPU or CPU)
- Vulkan SDK or drivers for Vulkan backend
- A dedicated GPU for Vulkan benchmark
- Compiled Vulkand MD5 shader (included)
---

## Usage

### Basic Run
```bash
python main_hashBenchmarker.py
```

### With custom rounds
```bash
python main_pwdEncrypter.py --rounds 50
```
### With custom script runs
```bash
python main_pwdEncrypter.py --runs 10
``` 

### Enable GPU benchmarks
```bash
python main_pwdEncrypter.py --opencl
python main_pwdEncrypter.py --vulkan
```

### Combined example
```bash
python main_pwdEncrypter.py --rounds 30 --runs 5 --opencl --vulkan
```

---

## Command-Line Flags

| Flag | Type | Description |
|------|------|-------------|
| `--rounds N` | int | Number of iterations per sequential worker. Default: 30 |
| `--opencl` | bool flag | Enables OpenCL GPU MD5 microbenchmark |
| `--vulkan` | bool flag | Enables Vulkan GPU benchmark (initialization-based) |
| `--runs N`  | int | Number of automated script runs. Default: 5 |

---

## Automated Plotting and table generating

If the number of runs exceed 1, after all the script runs, the program generates comparison plots automatically.

List of type of plots being generated:
- Individually combined sequential and parallel performance comparison by algorithm.
- Average hash/s bargraph across all algorithms with a 95% Confidence interval.
- Hash/s boxplot by algorithms across all N runs x M rounds.
- Parallel heatmap Wall time related to assigned worker number.
- Parallel speedup lineplot.
- Parallel throughput boxplot.
- Execution stability through script runs.

Also, a summary of sequential and parallel runs per algorithm gets generated via .csv export.

## Output Structure

Results are written into the `results/` directory as:

```
hash_benchmark_<timestamp>.json
```

Generated plots are created into the `results/figures/` directory.
Summary tables can be located at `results/csv/` directory.

The JSON includes:
- System metadata
- All sequential and parallel metrics
- GPU benchmark results
- Aggregated peak hash-rate per algorithm
- Crack-time estimates for:
  - multiple attacker models
  - common charset configurations

---

## Algorithms Benchmarked

### CPU Hashes
- MD5
- SHA-256
- bcrypt (cost=10, 12)
- scrypt with multiple (N, r, p) sets
- Argon2 with configurable:
  - time_cost
  - memory_cost
  - parallelism

### GPU Benchmarks
**OpenCL:**
- Runs an actual MD5 round kernel across millions of work-items.

**Vulkan:**
- Runs a full 64-round MD5 hash via compiled shader.

---

## Crack-Time Estimation

Using the measured best hash-rate per algorithm, the estimator computes brute-force times across:

### Character Sets
- Lowercase (a–z)
- Lowercase + digits
- Alphanumeric
- Printable ASCII (95 chars)

### Default Attacker Models
- Local CPU
- Single GPU estimate
- 8-GPU cluster estimate
- Large botnet simulation

Outputs include:
- Keyspace size
- Hashes per second
- Duration in seconds
- Human-readable duration

---

## Output Directory

All benchmark output is stored under:
```
results/
```
All plot outputs are stored under:
```
results/figures/
```
All summarizing tables are located at:
```
results/csv/
```

All directories are auto-created if missing.

---

## Notes

- Parallel execution is automatically skipped for memory-hard algorithms (scrypt, Argon2) due to if RAM constraints are present.
- OpenCL benchmark validates device memory and work-group limits before running.
- Vulkan path requires a dedicated GPU and re-compiled shader file.

---

## Made by Balázs Magyar

---

