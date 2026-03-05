#!/usr/bin/env python3
"""
Run:
  python main_pwdEncrypter.py --rounds 30 --opencl --vulkan

Dependencies:
  pip install bcrypt argon2-cffi psutil vulkan pyopencl

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
    return {
        "samples": rounds,
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
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
    return {
        "workers": workers,
        "rounds_per_worker": rounds_per_worker,
        "total_hashes": total_hashes,
        "wall_time_s_est": wall_time,
        "throughput_hash_per_s_est": throughput
    }

# -------------------------
# GPU: OpenCL benchmark (MD5-round microbenchmark)
# -------------------------
OPENCL_MD5_KERNEL = r"""
__constant uint A = 0x67452301;
__constant uint B = 0xefcdab89;
__constant uint C = 0x98badcfe;
__constant uint D = 0x10325476;

uint F(uint x, uint y, uint z){ return (x & y) | (~x & z); }
uint rotl(uint x, uint n){ return (x << n) | (x >> (32-n)); }

__kernel void md5_round_kernel(__global const uint *input, __global uint *output) {
    int gid = get_global_id(0);
    uint a = A, b = B, c = C, d = D;
    uint m = input[gid];
    // single simplified MD5 round for benchmark
    a = b + rotl(a + F(b,c,d) + m + 0xd76aa478, 7);
    output[gid] = a;
}
"""


def benchmark_opencl_md5(n_items=2_000_000, preferred_device_type=cl.device_type.GPU):
    """
    Robust OpenCL micro-benchmark:
      - uses numpy.random.Generator.integers (for big high values),
      - asking for machine limits, checking memory,
      - rounding the global_size to the multiple of local_size,
      - warmup + timed run + readback.
    return value: dict { backend, device_name, items, seconds, hash_per_sec, notes }
    """
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
    needed_bytes = n_items * np.dtype(np.uint32).itemsize * 2  # in + out
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
    host_in = rng.integers(low=0, high=2**32, size=global_size, dtype=np.uint32)  # note: global_size >= n_items
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
    # scale: (n_items / global_size) * (total_items_processed / total_time)
    scale = n_items / float(global_size)
    hash_per_sec = (total_items_processed / total_time) * scale

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
            "info": "Real Vulkan compute MD5 — full 64-round shader dispatch."
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
                print(f"[PAR] Skipping {t['name']} workers={w} (memory-heavy, uses internal parallelism)")
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
        outpath = os.path.join(OUTDIR, f"cpuPass_benchmark_{ts_utc()}.json")
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print("Saved results to", outpath)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--opencl", action="store_true", help="Run OpenCL GPU benchmark (requires pyopencl)")
    parser.add_argument("--vulkan", action="store_true", help="Attempt Vulkan GPU benchmark (requires Vulkan env)")
    args = parser.parse_args()
    run_all(rounds=args.rounds, do_opencl=args.opencl, do_vulkan=args.vulkan)
