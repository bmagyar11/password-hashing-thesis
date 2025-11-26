# Password Hashing Benchmark & Crack-Time Estimator prototype

This prototype project provides a comprehensive benchmarking suite for measuring the performance of various password hashing algorithms on CPU and optionally GPU (OpenCL / Vulkan).
It also includes a rudimentary brute-force crack-time estimator based on measured throughput.

The tool measures:
- MD5, SHA-256
- bcrypt (multiple cost factors)
- scrypt (configurable N, r, p)
- Argon2 (configurable memory/time cost)
- Parallel CPU performance across multiple worker processes
- Optional GPU micro-benchmarks (OpenCL or Vulkan)

All results can be exported as JSON for later analysis.

---

## Features

- Synchronous and parallel CPU benchmarking
- Automatic detection of available GPU backends
- OpenCL MD5-round microbenchmark (real kernel execution)
- Minimal Vulkan compute benchmark (initialization-only timing)
- Automatic keyspace estimation for common character sets
- Crack time estimation for multiple attacker models
- Configurable rounds, algorithms, GPU paths, and output

---

## Installation

### Dependencies

All being done in a python virtual environment.

```bash
pip install bcrypt argon2-cffi psutil vulkan pyopencl
```

Optional:
- A working OpenCL runtime (GPU or CPU)
- Vulkan SDK or drivers for Vulkan backend

---

## Usage

### Basic Run
```bash
python main_pwdEncrypter.py
```

### With custom rounds
```bash
python main_pwdEncrypter.py --rounds 50
```

### Enable GPU benchmarks
```bash
python main_pwdEncrypter.py --opencl
python main_pwdEncrypter.py --vulkan
```

### Combined example
```bash
python main_pwdEncrypter.py --rounds 30 --opencl --vulkan
```

---

## Command-Line Flags

| Flag | Type | Description |
|------|------|-------------|
| `--rounds N` | int | Number of iterations per sequential worker. Default: 30 |
| `--opencl` | bool flag | Enables OpenCL GPU MD5 microbenchmark |
| `--vulkan` | bool flag | Enables Vulkan GPU benchmark (initialization-based) |
| `--no-save` | bool flag | Skip saving JSON results into `results/` |

---

## Output Structure

Results are written into the `results/` directory as:

```
cpuPass_benchmark_<timestamp>.json
```

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
- Initializes a compute queue and simulates a workload.
- (No real shader dispatch unless SPIR-V is supplied and extended.)

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
This directory is auto-created if missing.

---

## Notes

- Parallel execution is automatically skipped for memory-hard algorithms (scrypt, Argon2) due to if RAM constraints are present.
- OpenCL benchmark validates device memory and work-group limits before running.
- Vulkan path requires a functioning Vulkan loader and GPU driver.

---

## Made by Balázs Magyar

---

