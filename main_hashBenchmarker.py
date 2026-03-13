#!/usr/bin/env python3
"""
Run:
  python main_pwdEncrypter.py --rounds 30 --run 5 --opencl --vulkan

Dependencies:
  pip install bcrypt argon2-cffi psutil vulkan pyopencl scipy pandas matplotlib seaborn

"""
import os
import time
import json
import argparse
import statistics
import hashlib
import bcrypt
from argon2 import PasswordHasher
import psutil
from datetime import datetime
import multiprocessing
import math
import scipy.stats as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --- Importing GPU libraries with fallback ---
HAVE_OPENCL = True
try:
    import pyopencl as cl
    import numpy as np
except Exception:
    HAVE_OPENCL = False

HAVE_VULKAN = True
try:
    # lightweight wrapper name is 'vulkan' (may differ). If not present, skip.
    import vulkan as vk
    import ctypes
except Exception:
    HAVE_VULKAN = False

# -------------------------
# Configuration (adjustable)
# -------------------------
OUTDIR = "results"
ROUNDS = 30 ## base round number, how many task runs via one worker(thread)
PASSWORDS = [
    "password",
    "P@ssw0rd!",
    "hunter2",
    "LetMeIn1234",
    "S0m3VeryL0ngPassphrase!"
]
BCRYPT_COSTS = [10, 12]
SCRYPT_PARAMS = [
    (2**14, 8, 1),
    (2**15, 8, 1)
]
ARGON2_PARAMS = [
    {"time_cost":1, "memory_cost":65536, "parallelism":1},   # 64 MB
    {"time_cost":2, "memory_cost":131072, "parallelism":1}   # 128 MB
]
WORKER_COUNTS = [1, max(1, multiprocessing.cpu_count()//2), multiprocessing.cpu_count()]

# default attacker models for brute force simulation
ATTACKER_MODELS = [
    {"name": "same_machine_cpu", "source": "measured", "multiplier": 1.0, "desc": "a mert hash rate a jelen gepen"},
    {"name": "single_gpu_est", "source": "measured", "multiplier": 50.0, "desc": "nyers GPU gyorsitas ~ durva becsles"},
    {"name": "multi_gpu_8_est", "source": "measured", "multiplier": 400.0, "desc": "8× gyorsitas (durva)"},
    {"name": "botnet_1000_est", "source": "measured", "multiplier": 1000.0, "desc": "nagyszabasu, tobb gepes (szimbolikus)"},
]

CHARSETS = {
    "lower": "abcdefghijklmnopqrstuvwxyz",
    "lowernum": "abcdefghijklmnopqrstuvwxyz0123456789",
    "alphanum": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "printable95": ''.join(chr(i) for i in range(32, 127))  # 95 printable ASCII
}

# -------------------------
# Helpers
# -------------------------
def ts_utc():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "csv"), exist_ok=True)

def mem_rss_bytes():
    return psutil.Process(os.getpid()).memory_info().rss

def human_dur(seconds):
    s = int(round(seconds)) if seconds is not None else 0
    if seconds is None or seconds == float("inf"):
        return "inf"
    if s < 0:
        return "0s"
    years, s = divmod(s, 365*24*3600)
    days, s = divmod(s, 24*3600)
    hours, s = divmod(s, 3600)
    minutes, seconds = divmod(s, 60)
    parts=[]
    if years: parts.append(f"{years}y")
    if days: parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    if seconds or not parts: parts.append(f"{seconds}s")
    return " ".join(parts)

def keyspace_from_charset_maxlen(charset, maxlen, minlen=1):
    k = len(charset)
    if k <= 0 or maxlen < minlen:
        return 0
    total = 0
    for l in range(minlen, maxlen+1):
        total += k**l
    return total

# -------------------------
# measuring synchronously
# -------------------------
def measure_sync(func, inputs, rounds):
    times = []
    mem_before = mem_rss_bytes()
    for i in range(rounds):
        pw = inputs[i % len(inputs)].encode()
        t0 = time.perf_counter()
        func(pw)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mem_after = mem_rss_bytes()

    n = len(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) # empirical std
    stderr = stdev / (n**0.5)
    ci95 = sp.t.interval(0.95, df=n-1, loc=mean, scale=stderr) # confidence interval 95%

    return {
        "samples": n,
        "mean_s": mean,
        "median_s": statistics.median(times),
        "stdev_s": stdev,
        "stderr_s": stderr,
        "min_s": min(times),
        "max_s": max(times),
        "cv_percent": (stdev / mean) * 100,
        "ci95_low_s": ci95[0],
        "ci95_high_s": ci95[1],
        "rss_bytes_delta": mem_after - mem_before,
        "raw_times_s": times[:]
    }

def hash_md5(pw_bytes): hashlib.md5(pw_bytes).digest()
def hash_sha256(pw_bytes): hashlib.sha256(pw_bytes).digest()
def make_bcrypt(cost):
    def f(pw_bytes):
        bcrypt.hashpw(pw_bytes, bcrypt.gensalt(rounds=cost))
    return f

## making an estimate of how much memory is usable for scrypt
def estimate_scrypt_mem_bytes(n,r):
    return 128 * n * r

def make_scrypt(n, r, p):
    expected = estimate_scrypt_mem_bytes(n, r)
    def f(pw_bytes):
        avail = psutil.virtual_memory().available
        if avail < expected:
            raise ValueError(f"scrypt skip: need ~{expected // 1024 // 1024}MB, available~{avail // 1024 // 1024}MB")
        try:
            hashlib.scrypt(password=pw_bytes, salt=b'somesalt', n=n, r=r, p=p, dklen=64, maxmem=expected * 2)
        except ValueError:
            raise
    return f

def make_argon2(time_cost, memory_cost, parallelism):
    ph = PasswordHasher(time_cost=time_cost, memory_cost=memory_cost, parallelism=parallelism)
    def f(pw_bytes):
        ph.hash(pw_bytes.decode(errors="ignore"))
    return f

## defining multithread worker
def mp_worker(descriptor, inputs, rounds_per_worker):
    kind = descriptor["kind"]
    p = descriptor.get("params", {})

    match kind:
        case "md5":
            func = hash_md5
        case "sha256":
            func = hash_sha256
        case "bcrypt":
            func = make_bcrypt(p["cost"])
        case "scrypt":
            func = make_scrypt(p["n"], p["r"], p["p"])
        case "argon2":
            func = make_argon2(p["time_cost"], p["memory_cost"], p["parallelism"])
        case _:
            raise RuntimeError("Unknown kind")

    t0 = time.perf_counter()
    for i in range(rounds_per_worker):
        func(inputs[i % len(inputs)].encode())
    t1 = time.perf_counter()
    return t1 - t0

## measurement with multiple worker/nummber of tasks
def measure_parallel(descriptor, inputs, rounds, workers):
    if descriptor["kind"] in ["scrypt", "argon2"] and workers > 1:
        raise ValueError("Parallel run for memory-hard algorithm skipped")
    rounds_per_worker = max(1, rounds // workers)
    pool = multiprocessing.Pool(processes=workers)
    args = [(descriptor, inputs, rounds_per_worker) for _ in range(workers)]
    timings = pool.starmap(mp_worker, args)
    pool.close()
    pool.join()
    wall_time = max(timings)
    total_hashes = rounds_per_worker * workers
    throughput = total_hashes / wall_time if wall_time > 0 else 0.0

    # CI calculation from worker timings
    n = len(timings)
    mean_t = statistics.mean(timings)
    stdev_t = statistics.stdev(timings) if n > 1 else 0.0
    stderr_t = stdev_t / (n ** 0.5) if n > 0 else 0.0
    ci95 = sp.t.interval(0.95, df=n-1, loc=mean_t, scale=stderr_t) if n > 1 else (mean_t, mean_t)

    # throughput CI: inverse ratio because throughput = hashes/time
    total_hashes_f = float(total_hashes)
    ci95_throughput_low = total_hashes_f / ci95[1] if ci95[1] > 0 else 0.0
    ci95_throughput_high = total_hashes_f / ci95[0] if ci95[0] > 0 else 0.0

    return {
        "workers": workers,
        "rounds_per_worker": rounds_per_worker,
        "total_hashes": total_hashes,
        "wall_time_s_est": wall_time,
        "throughput_hash_per_s_est": throughput,
        "worker_times": timings,
        "mean_worker_time_s": mean_t,
        "stdev_worker_time_s": stdev_t,
        "ci95_throughput_low": ci95_throughput_low,
        "ci95_throughput_high": ci95_throughput_high,
    }
# -------------------------
# GPU: OpenCL benchmark (MD5-full benchmark)
# -------------------------
OPENCL_MD5_KERNEL = r"""
__constant uint T[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

__constant uint S[64] = {
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
    5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};

uint rotl(uint x, uint n){ return (x << n) | (x >> (32u - n)); }

__kernel void md5_round_kernel(__global const uint *input, __global uint *output) {
    int gid = get_global_id(0);

    // each work item gets 16 uint32 words (one 64-byte MD5 block)
    uint M[16];
    for (int i = 0; i < 16; i++) {
        M[i] = input[gid * 16 + i];
    }

    uint a = 0x67452301u;
    uint b = 0xefcdab89u;
    uint c = 0x98badcfeu;
    uint d = 0x10325476u;

    for (int i = 0; i < 64; i++) {
        uint F_val; uint g;
        if (i < 16) {
            F_val = (b & c) | (~b & d);
            g = i;
        } else if (i < 32) {
            F_val = (d & b) | (~d & c);
            g = (5*i + 1) % 16;
        } else if (i < 48) {
            F_val = b ^ c ^ d;
            g = (3*i + 5) % 16;
        } else {
            F_val = c ^ (b | ~d);
            g = (7*i) % 16;
        }
        uint temp = d;
        d = c;
        c = b;
        b = b + rotl(a + F_val + M[g] + T[i], S[i]);
        a = temp;
    }

    output[gid] = a + 0x67452301u;
}
"""


def benchmark_opencl_md5(n_items=2_000_000, preferred_device_type=cl.device_type.GPU):
    if not HAVE_OPENCL:
        raise RuntimeError("pyopencl not available on this system.")

    # choosing context: preferring GPU, if not available, using CPU
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found on this machine.")

    # pick a device matching preference if possible
    chosen_dev = None
    chosen_plat = None
    for plat in platforms:
        for dev in plat.get_devices():
            # prefer GPU but accept any
            if dev.type & preferred_device_type:
                chosen_dev = dev
                chosen_plat = plat
                break
        if chosen_dev:
            break
    if not chosen_dev:
        # fallback: pick first device
        chosen_plat = platforms[0]
        chosen_dev = chosen_plat.get_devices()[0]

    # Print / return device info for debugging
    dev_name = chosen_dev.name.strip()
    global_mem = int(chosen_dev.global_mem_size)
    max_alloc = int(chosen_dev.max_mem_alloc_size)
    max_work_group = int(chosen_dev.max_work_group_size)
    max_work_item_sizes = tuple(chosen_dev.max_work_item_sizes)

    # Basic safety checks
    needed_bytes = n_items * 16 * np.dtype(np.uint32).itemsize + n_items * np.dtype(np.uint32).itemsize
    if needed_bytes > global_mem:
        raise RuntimeError(f"Requested buffers (~{needed_bytes//1024//1024}MB) exceed device global memory ({global_mem//1024//1024}MB). Reduce n_items.")

    if n_items * np.dtype(np.uint32).itemsize > max_alloc:
        raise RuntimeError(f"Single buffer size ({(n_items*np.dtype(np.uint32).itemsize)//1024//1024}MB) exceeds device max alloc ({max_alloc//1024//1024}MB). Reduce n_items.")

    # Create context and queue for the chosen device
    ctx = cl.Context(devices=[chosen_dev])
    queue = cl.CommandQueue(ctx, device=chosen_dev)

    # choose a sensible local_size (power-of-two, <= max_work_group)
    preferred_local = 256
    local_size = min(preferred_local, max_work_group)
    # make sure local_size divides global by rounding up
    def round_up(divisor, value):
        return int(((value + divisor - 1) // divisor) * divisor)
    global_size = round_up(local_size, n_items)

    # Use new Generator.integers to avoid int32 'high' problems
    rng = np.random.default_rng()
    host_in = rng.integers(low=0, high=2**32, size=global_size * 16, dtype=np.uint32)
    host_out = np.zeros(global_size, dtype=np.uint32)

    mf = cl.mem_flags
    buf_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_in)
    buf_out = cl.Buffer(ctx, mf.WRITE_ONLY, host_out.nbytes)

    # build program (use existing OPENCL_MD5_KERNEL from your script)
    program = cl.Program(ctx, OPENCL_MD5_KERNEL).build(options=[])
    kernel = program.md5_round_kernel

    # Warm-up: small dispatch
    warm_local = min(local_size, 64)
    warm_global = round_up(warm_local, min(1024, global_size))
    kernel(queue, (warm_global,), (warm_local,), buf_in, buf_out)
    queue.finish()

    # Timed runs: iterate to amortize small jitter
    repeats = 3
    total_items_processed = 0
    total_time = 0.0
    for rep in range(repeats):
        start = time.perf_counter()
        # dispatch exactly global_size work-items (driver may ignore extra tails, kernel uses gid < n_items if you guard)
        kernel(queue, (global_size,), (local_size,), buf_in, buf_out)
        queue.finish()
        end = time.perf_counter()
        dur = end - start
        total_time += dur
        total_items_processed += global_size

    # readback a few elements to ensure completion and correctness (optional)
    read_back = np.empty(16, dtype=np.uint32)
    cl.enqueue_copy(queue, read_back, buf_out, src_offset=0, is_blocking=True)

    items_effective = n_items  # we interpret throughput for actual requested items
    # compute throughput normalized to requested n_items

    hash_per_sec = total_items_processed / total_time

    notes = {
        "device": dev_name,
        "global_mem_MB": global_mem//1024//1024,
        "max_alloc_MB": max_alloc//1024//1024,
        "max_work_group_size": max_work_group,
        "max_work_item_sizes": max_work_item_sizes,
        "global_size_used": global_size,
        "local_size_used": local_size,
        "repeats": repeats
    }

    return {"backend":"opencl", "device": dev_name, "items_requested": n_items, "items_dispatched": global_size,
            "seconds": total_time, "hash_per_sec": hash_per_sec, "notes": notes}


# -------------------------
# GPU: Vulkan compute benchmark, only works with a dedicated GPU
# -------------------------

def benchmark_vulkan_md5(spv_path="md5_vulkan.spv", n_items=2_000_000):
    """
    Real Vulkan compute MD5 benchmark.
    Dispatches a SPIR-V compute shader that performs actual MD5 rounds on the GPU.
    Requires: compiled md5_vulkan.spv (glslc md5_vulkan.comp -o md5_vulkan.spv)
    """
    if not HAVE_VULKAN:
        raise RuntimeError("Vulkan python wrapper is not installed.")

    if spv_path is None or not os.path.exists(spv_path):
        raise RuntimeError(
            f"SPIR-V shader not found at '{spv_path}'. "
            "Compile it with: glslc md5_vulkan.comp -o md5_vulkan.spv"
        )

    def VK_NAME_VERSION(major, minor, patch):
        return (major << 22) | (minor << 12) | patch

    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName=b"VulkanMD5Bench",
        applicationVersion=VK_NAME_VERSION(1, 0, 0),
        pEngineName=b"NoEngine",
        engineVersion=VK_NAME_VERSION(1, 0, 0),
        apiVersion=VK_NAME_VERSION(1, 0, 0),
    )
    instance = vk.vkCreateInstance(
        vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        ), None
    )


    phys_devs = vk.vkEnumeratePhysicalDevices(instance)
    if not phys_devs:
        raise RuntimeError("No Vulkan-compatible GPU found.")
    phys = phys_devs[0]
    props = vk.vkGetPhysicalDeviceProperties(phys)
    device_name = props.deviceName if isinstance(props.deviceName, str) else props.deviceName.decode("utf-8", errors="ignore")

    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(phys)


    qf_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(phys)
    compute_family = next(
        (i for i, q in enumerate(qf_props) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT),
        None
    )
    if compute_family is None:
        raise RuntimeError("No compute queue family found.")


    queue_priority = (ctypes.c_float * 1)(1.0)
    device = vk.vkCreateDevice(phys, vk.VkDeviceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=compute_family,
            queueCount=1,
            pQueuePriorities=ctypes.pointer(queue_priority)
        )
    ), None)
    queue = vk.vkGetDeviceQueue(device, compute_family, 0)


    def find_memory_type(type_filter, props_required):
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and \
               (mem_props.memoryTypes[i].propertyFlags & props_required) == props_required:
                return i
        raise RuntimeError("No suitable memory type found.")


    def make_buffer(size_bytes, usage):
        buf = vk.vkCreateBuffer(device, vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        ), None)
        mem_req = vk.vkGetBufferMemoryRequirements(device, buf)
        mem = vk.vkAllocateMemory(device, vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_req.size,
            memoryTypeIndex=find_memory_type(
                mem_req.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
        ), None)
        vk.vkBindBufferMemory(device, buf, mem, 0)
        return buf, mem, mem_req.size



    item_count   = n_items
    in_bytes     = item_count * 16 * 4   # uint32 per word, 16 words per block
    out_bytes    = item_count * 4        # one uint32 result per item

    buf_in,  mem_in,  _  = make_buffer(in_bytes,  vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    buf_out, mem_out, _  = make_buffer(out_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)


    ptr = vk.vkMapMemory(device, mem_in, 0, in_bytes, 0)
    import numpy as np
    rng = np.random.default_rng()
    host_in = rng.integers(0, 2**32, size=item_count * 16, dtype=np.uint32)
    ctypes.memmove(ptr, host_in.ctypes.data, in_bytes)
    vk.vkUnmapMemory(device, mem_in)


    with open(spv_path, "rb") as f:
        spv_bytes = f.read()
    spv_words = np.frombuffer(spv_bytes, dtype=np.uint32)
    shader_module = vk.vkCreateShaderModule(device, vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(spv_bytes),
        pCode=spv_words.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    ), None)


    bindings = [
        vk.VkDescriptorSetLayoutBinding(
            binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
        ) for b in (0, 1)
    ]
    ds_layout = vk.vkCreateDescriptorSetLayout(device, vk.VkDescriptorSetLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        bindingCount=2, pBindings=bindings
    ), None)


    pipeline_layout = vk.vkCreatePipelineLayout(device, vk.VkPipelineLayoutCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        setLayoutCount=1, pSetLayouts=[ds_layout]
    ), None)

    pipeline = vk.vkCreateComputePipelines(device, None, 1, [
        vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=shader_module,
                pName=b"main"
            ),
            layout=pipeline_layout
        )
    ], None)[0]


    desc_pool = vk.vkCreateDescriptorPool(device, vk.VkDescriptorPoolCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        maxSets=1,
        poolSizeCount=1,
        pPoolSizes=[vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=2
        )]
    ), None)

    desc_set = vk.vkAllocateDescriptorSets(device, vk.VkDescriptorSetAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool=desc_pool,
        descriptorSetCount=1,
        pSetLayouts=[ds_layout]
    ))[0]

    vk.vkUpdateDescriptorSets(device, 2, [
        vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=desc_set, dstBinding=b,
            descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[vk.VkDescriptorBufferInfo(
                buffer=buf_in if b == 0 else buf_out,
                offset=0,
                range=in_bytes if b == 0 else out_bytes
            )]
        ) for b in (0, 1)
    ], 0, None)


    cmd_pool = vk.vkCreateCommandPool(device, vk.VkCommandPoolCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex=compute_family
    ), None)

    cmd_buf = vk.vkAllocateCommandBuffers(device, vk.VkCommandBufferAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool=cmd_pool,
        level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1
    ))[0]

    vk.vkBeginCommandBuffer(cmd_buf, vk.VkCommandBufferBeginInfo(
        sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    ))
    vk.vkCmdBindPipeline(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vk.vkCmdBindDescriptorSets(
        cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout, 0, 1, [desc_set], 0, None
    )
    local_size = 64
    vk.vkCmdDispatch(cmd_buf, (item_count + local_size - 1) // local_size, 1, 1)
    vk.vkEndCommandBuffer(cmd_buf)


    fence = vk.vkCreateFence(device, vk.VkFenceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=0
    ), None)

    def submit_and_wait():
        vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1, pCommandBuffers=[cmd_buf]
        )], fence)
        vk.vkWaitForFences(device, 1, [fence], vk.VK_TRUE, int(10_000_000_000))  # 10s timeout
        vk.vkResetFences(device, 1, [fence])

    submit_and_wait()  # warmup


    repeats = 3
    total_time = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        submit_and_wait()
        total_time += time.perf_counter() - t0

    hash_per_sec = (item_count * repeats) / total_time


    vk.vkDestroyFence(device, fence, None)
    vk.vkDestroyCommandPool(device, cmd_pool, None)
    vk.vkDestroyDescriptorPool(device, desc_pool, None)
    vk.vkDestroyPipeline(device, pipeline, None)
    vk.vkDestroyPipelineLayout(device, pipeline_layout, None)
    vk.vkDestroyDescriptorSetLayout(device, ds_layout, None)
    vk.vkDestroyShaderModule(device, shader_module, None)
    vk.vkFreeMemory(device, mem_out, None)
    vk.vkFreeMemory(device, mem_in, None)
    vk.vkDestroyBuffer(device, buf_out, None)
    vk.vkDestroyBuffer(device, buf_in, None)
    vk.vkDestroyDevice(device, None)
    vk.vkDestroyInstance(instance, None)

    return {
        "backend": "vulkan",
        "device": device_name,
        "items_requested": item_count,
        "seconds": total_time,
        "hash_per_sec": hash_per_sec,
        "notes": {
            "shader": spv_path,
            "repeats": repeats,
            "local_size": local_size,
            "info": "Vulkan compute MD5 — full 64-round shader dispatch."
        }
    }

# -------------------------
# BUILD TESTS
# -------------------------
def build_tests():
    tests = []
    tests.append({"name":"md5", "kind":"md5", "params":{}})
    tests.append({"name":"sha256", "kind":"sha256", "params":{}})
    for c in BCRYPT_COSTS:
        tests.append({"name":f"bcrypt_cost{c}", "kind":"bcrypt", "params":{"cost":c}})
    for (n,r,p) in SCRYPT_PARAMS:
        tests.append({"name":f"scrypt_n{n}_r{r}_p{p}", "kind":"scrypt", "params":{"n":n,"r":r,"p":p}})
    for ap in ARGON2_PARAMS:
        nm = f"argon2_t{ap['time_cost']}_m{ap['memory_cost']//1024}MB_p{ap['parallelism']}"
        tests.append({"name":nm, "kind":"argon2", "params":ap})
    return tests

# -------------------------
# Crack-time calculator
# -------------------------
def compute_estimates_for_test(best_throughput, custom_attack_models=None, charset_specs=None):
    try:
        best_throughput = float(best_throughput)
    except (TypeError, ValueError):
        return {}

    if best_throughput <= 0 or math.isinf(best_throughput) or math.isnan(best_throughput):
        return {}

    models = custom_attack_models if custom_attack_models is not None else ATTACKER_MODELS
    cspecs = charset_specs if charset_specs is not None else [
        {"name":"6_lower", "charset_name":"lower", "min_length":1, "max_length":6},
        {"name":"6_lowernum", "charset_name":"lowernum", "min_length":1, "max_length":6},
        {"name":"8_alphanum", "charset_name":"alphanum", "min_length":1, "max_length":8},
        {"name":"8_printable", "charset_name":"printable95", "min_length":1, "max_length":8},
    ]

    out = {"attack_models":[], "charset_specs": cspecs, "estimates": {}}
    for m in models:
        if m.get("source") == "measured":
            hps = best_throughput * m.get("multiplier", 1.0)
        elif m.get("source") == "fixed":
            hps = m.get("hash_s", 0.0)
        else:
            hps = best_throughput * m.get("multiplier", 1.0)
        out["attack_models"].append({"name": m["name"], "hash_s": hps, "desc": m.get("desc", "")})

    for cs in cspecs:
        charset_str = CHARSETS.get(cs["charset_name"], cs.get("charset", ""))
        if not charset_str:
            continue
        K = keyspace_from_charset_maxlen(charset_str, cs["max_length"], cs.get("min_length",1))
        out["estimates"][cs["name"]] = {"keyspace": K, "per_model": {}}
        for am in out["attack_models"]:
            hps = am["hash_s"]
            if hps <= 0:
                t_seconds = None
                human = "inf"
            else:
                t_seconds = K / hps
                human = human_dur(t_seconds)
            out["estimates"][cs["name"]]["per_model"][am["name"]] = {
                "hash_s": hps,
                "time_seconds": t_seconds,
                "time_human": human
            }
    return out

# -------------------------
# PLOT RESULTS
# -------------------------
def plot_results(all_runs):
    ensure_outdir()
    # gather data in a DataFrame
    rows = []
    for run_idx, run in enumerate(all_runs):
        for r in run["results"]:
            if r["mode"] != "sequential":
                continue
            if "mean_s" not in r["metrics"]:
                continue
            rows.append({
                "run": run_idx+1,
                "algorithm": r["test"]["name"],
                "mean_s": r["metrics"]["mean_s"],
                "stdev_s": r["metrics"]["stdev_s"],
                "cv_percent": r["metrics"]["cv_percent"],
                "ci95_low": r["metrics"]["ci95_low_s"],
                "ci95_high": r["metrics"]["ci95_high_s"],
                "hash_per_s": 1.0 / r["metrics"]["mean_s"] if r["metrics"]["mean_s"] > 0 else 0,
            })
    df = pd.DataFrame(rows)

    # 1. Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="algorithm", y="hash_per_s", ax=ax)
    ax.set_yscale("log") # log scale because of the differences between MD5 and bcrypt/Argon2
    ax.set_title("Hash/s eloszlás algoritmusonként (összes futás × összes kör mérés)")
    ax.set_xlabel("Algoritmus")
    ax.set_ylabel("Hash/s (log skála)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/figures/boxplot.png", dpi=150)
    plt.close()
    print("Saved: results/figures/boxplot.png")

    # 2. Bargraph with 95% CI error lanes
    summary = df.groupby("algorithm").agg(
        grand_mean=("hash_per_s", "mean"),
    ).reset_index()

    ci_rows = []
    for algo in summary["algorithm"]:
        vals = df[df["algorithm"] == algo]["hash_per_s"].values
        n = len(vals)
        m = vals.mean()
        s = vals.std(ddof=1)
        stderr = s / (n ** 0.5)
        ci = sp.t.interval(0.95, df=n - 1, loc=m, scale=stderr)
        ci_rows.append({
            "algorithm": algo,
            "err_low": max(0, m - ci[0]),
            "err_high": max(0, ci[1] - m)
        })

    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on="algorithm")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary["algorithm"], summary["grand_mean"],
           yerr=[summary["err_low"], summary["err_high"]],
           capsize=5, color="steelblue", edgecolor="black")
    ax.set_yscale("log")
    ax.set_title("Átlagos Hash/s 95%-os CI intervallummal")
    ax.set_xlabel("Algoritmus")
    ax.set_ylabel("Hash/s (log skála)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/figures/bargraph_ci95.png", dpi=150)
    plt.close()
    print("Saved: results/figures/bargraph_ci95.png")

    # 3. stability of runs
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        ax.plot(subset["run"], subset["hash_per_s"], marker="o", label=algo)
    ax.set_yscale("log")
    ax.set_title("Hash/s stabilitás futásonként")
    ax.set_xlabel("Futás száma")
    ax.set_ylabel("Hash/s (log skála)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/stability.png", dpi=150)
    plt.close()
    print("Saved: results/figures/stability.png")

    #4. CV% table exported to .csv too
    cv_summary = df.groupby("algorithm").agg(
        grand_mean_hash_s=("hash_per_s", "mean"),
        stdev_between_runs=("hash_per_s", "std"),
        cv_percent=("cv_percent", "mean"),
    ).reset_index()

    cv_summary.to_csv("results/csv/summary_table.csv", index=False)
    print("Saved: results/csv/summary_table.csv")
    print(cv_summary.to_string(index=False))

# -------------------------
# PLOT PARALLEL RUNS
# -------------------------
def plot_parallel_results(all_runs):
    ensure_outdir()
    # Gather data
    rows = []
    for run_idx, run in enumerate(all_runs):
        for r in run["results"]:
            if r["mode"] != "parallel":
                continue
            if "throughput_hash_per_s_est" not in r["metrics"]:
                continue
            rows.append({
                "run": run_idx+1,
                "algorithm": r["test"]["name"],
                "workers": r["metrics"]["workers"],
                "throughput_hash_per_s": r["metrics"]["throughput_hash_per_s_est"],
                "wall_time_s": r["metrics"]["wall_time_s_est"],
                "total_hashes": r["metrics"]["total_hashes"],
            })
    df = pd.DataFrame(rows)

    # 1. Throughput vs Workers bargraph per algorithms
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=df,
        x="algorithm",
        y="throughput_hash_per_s",
        hue="workers", # diff colors for every worker nums
        errorbar="sd",
        ax=ax
    )
    ax.set_yscale("log")
    ax.set_title("Többszálas Hash/s - algoritmus és worker szám alapján")
    ax.set_xlabel("Algoritmus")
    ax.set_ylabel("Throughput Hash/s (log skála)")
    ax.legend(title="Workerek", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/figures/parallel_throughput.png", dpi=150)
    plt.close()
    print("Saved: results/figures/parallel_throughput.png")

    # 2. Scalability - 1 worker vs N worker (speedup)
    # shows how much it speeds up by parallelism
    baseline = df[df["workers"] == 1][["algorithm", "throughput_hash_per_s"]].copy()
    baseline = baseline.groupby("algorithm")["throughput_hash_per_s"].mean().reset_index()
    baseline.columns = ["algorithm", "baseline_hps"]

    df_speedup = df.merge(baseline, on="algorithm")
    df_speedup["speedup"] = df_speedup["throughput_hash_per_s"] / df_speedup["baseline_hps"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=df_speedup,
        x="workers",
        y="speedup",
        hue="algorithm",
        marker="o",
        ax=ax
    )
    # reference line for ideal linear scaling
    max_workers = df["workers"].max()
    ax.plot(
        [1, max_workers], [1, max_workers],
        "k--", linewidth=1, label="Ideális lineáris skálázódás"
    )
    ax.set_title("Többszálas gyorsulás egy szálhoz viszonyítva")
    ax.set_xlabel("Worker szám")
    ax.set_ylabel("Gyorsulás (x)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/parallel_speedup.png", dpi=150)
    plt.close()
    print("Saved: results/figures/parallel_speedup.png")

    # 3. Wall time heatmap
    # algorithm x worker matrix, with wall time values
    pivot = df.groupby(["algorithm", "workers"])["wall_time_s"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True, # values printed out in cells
        fmt=".6f",
        cmap="RdYlGn_r", # red == slow, green == fast
        ax=ax
    )
    ax.set_title("Wall time (s) - algoritmus x worker szám")
    ax.set_xlabel("Worker szám")
    ax.set_ylabel("Algoritmus")
    plt.tight_layout()
    plt.savefig("results/figures/parallel_heatmap.png", dpi=150)
    plt.close()
    print("Saved: results/figures/parallel_heatmap.png")

    # 4. CSV export
    summary = df.groupby(["algorithm", "workers"]).agg(
        mean_throughput=("throughput_hash_per_s", "mean"),
        stdev_throughput=("throughput_hash_per_s", "std"),
        mean_wall_time=("wall_time_s", "mean")
    ).reset_index()

    summary.to_csv("results/csv/parallel_summary.csv", index=False)
    print("Saved: results/csv/parallel_summary.csv")
    print(summary.to_string(index=False))


# -------------------------
# PLOT PER ALGORITHM
# -------------------------
def plot_per_algorithm(all_runs):
    ensure_outdir()
    # Gather data
    seq_rows = []
    par_rows = []

    for run_idx, run in enumerate(all_runs):
        for r in run["results"]:
            name = r["test"]["name"]
            metrics = r["metrics"]

            if r["mode"] == "sequential" and "mean_s" in metrics:
                seq_rows.append({
                    "run": run_idx + 1,
                    "algorithm": name,
                    "hash_per_s": 1.0 / metrics["mean_s"],
                    "ci95_low": 1.0 / metrics["ci95_low"] if metrics.get("ci95_low") else None,
                    "ci95_high": 1.0 / metrics["ci95_high"] if metrics.get("ci95_high") else None,
                })
            elif r["mode"] == "parallel" and "throughput_hash_per_s_est" in metrics:
                par_rows.append({
                    "run": run_idx + 1,
                    "algorithm": name,
                    "workers": metrics["workers"],
                    "hash_per_s": metrics["throughput_hash_per_s_est"],
                    "wall_time_s": metrics["wall_time_s_est"],
                    "ci95_throughput_low": metrics.get("ci95_throughput_low", 0),
                    "ci95_throughput_high": metrics.get("ci95_throughput_high", 0)
                })
    df_seq = pd.DataFrame(seq_rows)
    df_par = pd.DataFrame(par_rows)

    algorithms = df_seq["algorithm"].unique()

    for algo in algorithms:
        seq = df_seq[df_seq["algorithm"] == algo]
        par = df_par[df_par["algorithm"] == algo]

        # for every algorithm, 1 figure and 2 subplot next to each other
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{algo} - Teljesítmény összehasonlítás", fontsize=14, fontweight="bold")

        # left side: one thread boxplot based on 5 runs
        ax = axes[0]
        ax.boxplot(seq["hash_per_s"], tick_labels=["Egyszálas"])
        grand_mean = seq["hash_per_s"].mean()
        ax.axhline(grand_mean, color="red", linestyle="--", linewidth=1, label=f"Átlag: {grand_mean:.0f}")
        ax.set_title("Egyszálas Hash/s (összes futás)")
        ax.set_ylabel("Hash/s")
        ax.legend()

        # right side: parallel throughput vs workers
        ax = axes[1]
        if not par.empty:
            par_summary = par.groupby("workers").agg(
                mean=("hash_per_s", "mean"),
                std=("hash_per_s", "std"),
                ci95_low=("ci95_throughput_low", "mean"),
                ci95_high=("ci95_throughput_high", "mean")
            ).reset_index()
            # CI errorbars
            err_low = (par_summary["mean"] - par_summary["ci95_low"]).clip(lower=0)
            err_high = (par_summary["ci95_high"] - par_summary["mean"]).clip(lower=0)

            ax.errorbar(
                par_summary["workers"],
                par_summary["mean"],
                yerr=[err_low, err_high],
                marker="o",
                capsize=5,
                color="steelblue",
                label="Mért throughput ± 95% CI"
            )

            #ideal linear scaling
            baseline = par_summary[par_summary["workers"] == 1]["mean"].values
            if len(baseline) > 0:
                ideal=[baseline[0] * w for w in par_summary["workers"]]
                ax.plot(par_summary["workers"], ideal, "r--", linewidth=1, label="Ideális lineáris")
            ax.set_title("Többszálas throughput")
            ax.set_xlabel("Worker szám")
            ax.set_ylabel("Hash/s")
            ax.legend()
        else:
            ax.text(0.5,0.5, "Többszálas futás\nnem elérhető\n(memory-hard algoritmus)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title("Többszálas throughput")

        plt.tight_layout()
        plt.savefig(f"results/figures/{algo}_combined.png", dpi=150)
        plt.close()
        print(f"Saved: results/figures/{algo}_combined.png")



# -------------------------
# RUN ALL
# -------------------------
def run_all(rounds=ROUNDS, do_opencl=False, do_vulkan=False, save_json=True):
    ensure_outdir()
    meta = {
        "timestamp": ts_utc(),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "cpu_count": multiprocessing.cpu_count(),
        "have_opencl": HAVE_OPENCL,
        "have_vulkan": HAVE_VULKAN
    }
    results = []
    tests = build_tests()

    # CPU sequential
    for t in tests:
        kind = t["kind"]
        p = t["params"]
        if kind == "md5":
            func = hash_md5
        elif kind == "sha256":
            func = hash_sha256
        elif kind == "bcrypt":
            func = make_bcrypt(p["cost"])
        elif kind == "scrypt":
            func = make_scrypt(p["n"], p["r"], p["p"])
        elif kind == "argon2":
            func = make_argon2(p["time_cost"], p["memory_cost"], p["parallelism"])
        else:
            continue
        print(f"[SEQ] {t['name']}")
        try:
            sync_metrics = measure_sync(func, PASSWORDS, rounds)
        except Exception as e:
            sync_metrics = {"error": str(e)}
            print("Sequential measurement error for", t["name"], e)
        results.append({"mode":"sequential", "test":t, "metrics":sync_metrics})

    # CPU parallel
    for t in tests:
        for w in WORKER_COUNTS:
            if t["kind"] in ["scrypt", "argon2"] and w > 1:
                print(f"[PAR] Skipping {t['name']} workers={w} (memory-heavy)")
                continue
            print(f"[PAR] {t['name']} workers={w}")
            try:
                pmetrics = measure_parallel(t, PASSWORDS, rounds, w)
            except Exception as e:
                pmetrics = {"error": str(e)}
                print("Parallel measurement skipped due to:", e)
            results.append({"mode":"parallel", "test":t, "metrics":pmetrics})

    # GPU OpenCL benchmark (optional)
    if do_opencl:
        print("[GPU-OPENCL] Running OpenCL benchmarks...")
        try:
            oc_res = benchmark_opencl_md5(n_items=2_000_000)
            print("[GPU-OPENCL] done:", oc_res)
            results.append({"mode":"gpu_opencl", "test":{"name":"md5_opencl_round", "kind":"opencl_md5"}, "metrics":oc_res})
        except Exception as e:
            print("[GPU-OPENCL] failed:", e)
            results.append({"mode":"gpu_opencl", "test":{"name":"md5_opencl_round", "kind":"opencl_md5"}, "metrics":{"error":str(e)}})

    # GPU Vulkan benchmark (optional) — requires vulkan pip library and SDK
    if do_vulkan:
        print("[GPU-VULKAN] Running Vulkan benchmarks...")
        try:
            vk_res = benchmark_vulkan_md5(n_items=2_000_000)
            print("[GPU-VULKAN] done:", vk_res)
            results.append({"mode":"gpu_vulkan", "test":{"name":"md5_vulkan_round", "kind":"vulkan_md5"}, "metrics":vk_res})
        except Exception as e:
            print("[GPU-VULKAN] failed:", e)
            results.append({"mode":"gpu_vulkan", "test":{"name":"md5_vulkan_round", "kind":"vulkan_md5"}, "metrics":{"error":str(e)}})

    # --- compute best throughput per test name ---
    aggregated = []
    for t in tests:
        name = t["name"]
        par_entries = [r for r in results if r["test"]["name"]==name and r["mode"]=="parallel"]
        best_hps = 0.0
        if par_entries:
            for e in par_entries:
                h = e["metrics"].get("throughput_hash_per_s_est", 0.0)
                if h and h > best_hps:
                    best_hps = h
        seq_entries = [r for r in results if r["test"]["name"]==name and r["mode"]=="sequential"]
        if best_hps == 0.0 and seq_entries:
            med = seq_entries[0]["metrics"].get("median_s", None)
            if med and med > 0:
                best_hps = 1.0 / med
        aggregated.append({"test_name":name, "best_hash_s": best_hps})

    # also consider GPU opencl/vulkan results to fill aggregated (simple heuristic)
    # if opencl produced hash_per_sec, attach it as "md5_opencl" aggregated entry
    for r in results:
        if r["mode"]=="gpu_opencl" and "hash_per_sec" in r["metrics"]:
            aggregated.append({"test_name":"md5_opencl_round", "best_hash_s": r["metrics"]["hash_per_sec"]})
        if r["mode"]=="gpu_vulkan" and "hash_per_sec" in r["metrics"]:
            aggregated.append({"test_name":"md5_vulkan_round", "best_hash_s": r["metrics"]["hash_per_sec"]})

    estimates_per_test = {}
    for a in aggregated:
        estimates_per_test[a["test_name"]] = compute_estimates_for_test(a["best_hash_s"])

    out = {"meta":meta, "results": results, "aggregated": aggregated, "crack_estimates": estimates_per_test}
    if save_json:
        outpath = os.path.join(OUTDIR, f"hash_benchmark_{ts_utc()}.json")
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print("Saved results to", outpath)
    return out

## Run the script multiple times
def run_multiple(rounds=ROUNDS, n_runs=5, do_opencl=False, do_vulkan=False):
    all_runs = []
    for i in range(n_runs):
        print(f"\n{'='*50}")
        print(f"=== Running {i+1}/{n_runs} ===")
        print(f"{'='*50}")
        result = run_all(
            rounds=rounds,
            do_opencl=do_opencl,
            do_vulkan=do_vulkan,
            save_json=True
        )
        all_runs.append(result)
    return all_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--runs", type=int, default=5, help="Number of script executions")
    parser.add_argument("--opencl", action="store_true", help="Run OpenCL GPU benchmark (requires pyopencl)")
    parser.add_argument("--vulkan", action="store_true", help="Attempt Vulkan GPU benchmark (requires Vulkan env)")
    args = parser.parse_args()

    if args.runs > 1:
        all_runs = run_multiple(
            rounds=args.rounds,
            n_runs=args.runs,
            do_opencl=args.opencl,
            do_vulkan=args.vulkan
        )
        print("\n=== Generating plots... ===")
        plot_results(all_runs)
        plot_parallel_results(all_runs)
        plot_per_algorithm(all_runs)
        print("\n=== Done! Results are in results/figures/ folder ===")
    else:
        run_all(rounds=args.rounds, do_opencl=args.opencl, do_vulkan=args.vulkan)
