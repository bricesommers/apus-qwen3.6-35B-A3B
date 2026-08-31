/*
 * c/backend_metal.mm — Metal backend for the dense BF16 compute of
 * Qwen3.6-35B-A3B (M10). Objective-C++ (clang++); the only file in the
 * project that needs the Metal/Foundation frameworks. Built only into
 * bin/apus-qwen-metal and the tests/m10 binaries (make metal /
 * test-m10); the plain CPU binary never links it and is behaviorally
 * untouched, and the Linux build never sees it (the weak stubs in the
 * c/bf16.h TU serve there).
 *
 * What runs on the GPU (all FP32 shader math, fast-math DISABLED — no
 * contraction, no reassociation):
 *   - bf16_gemv / bf16_gemm: the c/bf16.h M9b ILP NEON kernels' per-row
 *     rounding sequence EXACTLY — four float4 accumulators per output
 *     row; per ascending 32-wide chunk, the eight single-rounded product
 *     vectors (exact u16<<16 widening, one mul rounding each) accumulate
 *     acc[q & 3] += p_q (q ascending: i=0,8,16,24 -> q=0,2,4,6); fixed
 *     combine ((a0+a1)+(a2+a3)) then ((s.x+s.y)+(s.z+s.w)); the K tail
 *     (< 32) appended with scalar ascending adds (mul + add, two
 *     roundings, never fma); one RNE bf16 narrow at the end (the integer
 *     bit-trick of apus_bf16_bits). BITWISE identical to
 *     apus_bf16_gemv_hot / apus_bf16_gemm_hot on ARM (tests/m10 asserts
 *     it on every shape — this is a bitwise offload, not a reorder).
 *   - bf16_matvec_f32: the c/moe.h fp32-out matvec — staged
 *     single-rounded products per 32-wide chunk, then the 32 strictly
 *     ascending scalar adds (no fp32 out narrowing). BITWISE identical
 *     to apus_moe_matvec_f32_hot. (Kept generic machinery: the Qwen
 *     router scores through the bf16 gemv hook — apus_moe_route — so
 *     this kernel has no model caller; pinned at kernel level anyway.)
 * Everything else (the GDN recurrence, gated-GQA attention, norms,
 * RoPE, expert FFN in tiered mode, FFN at prefill, sampling) stays on
 * the CPU.
 *
 * Buffers: resident dense weights (st.h shard slabs, model arrays) are
 * wrapped as MTLBuffers on first use, keyed by CPU pointer — zero-copy
 * newBufferWithBytesNoCopy over the vm_region covering the page-rounded
 * range (unified memory) whenever possible, one upload copy otherwise.
 * APUS_METAL_DENSE_MB (default 8192) caps wrapped+uploaded weight bytes;
 * past the cap hooks return "unsupported" and the op falls back to the
 * CPU kernel (per-op fail-soft). Activations/outputs use small grow-on-
 * demand shared staging buffers. One command buffer per op, synchronous
 * waitUntilCompleted.
 *
 * Weight-pointer invariant: the cache assumes weight pointers are STABLE
 * and their contents immutable for the lifetime of the backend (true for
 * resident dense weights; TIERED expert slabs are transient — those call
 * sites use apus_bf16_gemv_hot_cpu in c/cache.h and never reach the
 * cache). apus_metal_disable() drops the cache.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>

#include <pthread.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unordered_map>

#include "backend_metal.h"

/* ========================================================================
 * Shader source (compiled at apus_metal_enable with fastMathEnabled = NO /
 * MTLMathModeSafe). Every kernel mirrors its dispatched CPU kernel's
 * rounding sequence exactly (c/bf16.h M9b ILP, c/moe.h fp32-out anchor).
 * ====================================================================== */

static NSString *const kShaderSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;

/* apus_bf16_f32: exact widen (u16 << 16). */
inline float bf16w(ushort c) {
    return as_type<float>((uint)c << 16);
}

/* apus_bf16_bits: RNE narrow (NaN passes through as the high 16 bits). */
inline ushort bf16n(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)(u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

/* One output row of the M9b ILP GEMV (c/bf16.h apus_bf16_dot_ilp_neon):
 * four float4 accumulators; per ascending 32-wide chunk the eight
 * single-rounded product vectors accumulate acc[q & 3] += p_q
 * (i = 0,8,16,24 -> q = 0,2,4,6); fixed combine; scalar ascending tail.
 * One GPU thread per output row. */
inline float dot_ilp(device const ushort *w, ulong wb,
                     device const ushort *x, ulong xb, uint K) {
    float4 a0 = float4(0.0f), a1 = float4(0.0f);
    float4 a2 = float4(0.0f), a3 = float4(0.0f);
    uint k = 0u;
    for (; k + 32u <= K; k += 32u) {
        for (uint i = 0u; i < 32u; i += 8u) {
            uint q = i / 4u;
            float4 w0 = float4(bf16w(w[wb+k+i]),      bf16w(w[wb+k+i+1u]),
                               bf16w(w[wb+k+i+2u]),   bf16w(w[wb+k+i+3u]));
            float4 w1 = float4(bf16w(w[wb+k+i+4u]),   bf16w(w[wb+k+i+5u]),
                               bf16w(w[wb+k+i+6u]),   bf16w(w[wb+k+i+7u]));
            float4 x0 = float4(bf16w(x[xb+k+i]),      bf16w(x[xb+k+i+1u]),
                               bf16w(x[xb+k+i+2u]),   bf16w(x[xb+k+i+3u]));
            float4 x1 = float4(bf16w(x[xb+k+i+4u]),   bf16w(x[xb+k+i+5u]),
                               bf16w(x[xb+k+i+6u]),   bf16w(x[xb+k+i+7u]));
            if (q & 2u) {
                a2 += w0 * x0;
                a3 += w1 * x1;
            } else {
                a0 += w0 * x0;
                a1 += w1 * x1;
            }
        }
    }
    float4 s = (a0 + a1) + (a2 + a3);
    float total = (s.x + s.y) + (s.z + s.w);
    for (; k < K; k++)
        total += bf16w(w[wb + k]) * bf16w(x[xb + k]);
    return total;
}

kernel void bf16_gemv_ilp(device const uchar *w8 [[buffer(0)]],
                          device const ushort *x [[buffer(1)]],
                          device ushort *y [[buffer(2)]],
                          constant uint &K [[buffer(3)]],
                          constant ulong &woff [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    device const ushort *w = (device const ushort *)(w8 + woff);
    y[gid] = bf16n(dot_ilp(w, (ulong)gid * K, x, 0, K));
}

kernel void bf16_gemm_ilp(device const uchar *w8 [[buffer(0)]],
                          device const ushort *x [[buffer(1)]],
                          device ushort *y [[buffer(2)]],
                          constant uint &O [[buffer(3)]],
                          constant uint &K [[buffer(4)]],
                          constant ulong &woff [[buffer(5)]],
                          uint gid [[thread_position_in_grid]]) {
    device const ushort *w = (device const ushort *)(w8 + woff);
    uint o = gid % O;
    uint m = gid / O;
    y[(ulong)m * O + o] = bf16n(
        dot_ilp(w, (ulong)o * K, x, (ulong)m * K, K));
}

/* The c/moe.h fp32-out router matvec (the NEON anchor): staged single-
 * rounded products per 32-wide chunk, then the 32 strictly ascending
 * scalar adds; fp32 out, no narrowing. One GPU thread per output row. */
kernel void bf16_matvec_f32(device const uchar *w8 [[buffer(0)]],
                            device const ushort *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant uint &K [[buffer(3)]],
                            constant ulong &woff [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    device const ushort *w = (device const ushort *)(w8 + woff);
    ulong wb = (ulong)gid * K;
    float acc = 0.0f;
    uint k = 0u;
    for (; k + 32u <= K; k += 32u) {
        for (uint i = 0u; i < 32u; i += 8u) {
            float4 w0 = float4(bf16w(w[wb+k+i]),      bf16w(w[wb+k+i+1u]),
                               bf16w(w[wb+k+i+2u]),   bf16w(w[wb+k+i+3u]));
            float4 w1 = float4(bf16w(w[wb+k+i+4u]),   bf16w(w[wb+k+i+5u]),
                               bf16w(w[wb+k+i+6u]),   bf16w(w[wb+k+i+7u]));
            float4 x0 = float4(bf16w(x[k+i]),         bf16w(x[k+i+1u]),
                               bf16w(x[k+i+2u]),      bf16w(x[k+i+3u]));
            float4 x1 = float4(bf16w(x[k+i+4u]),      bf16w(x[k+i+5u]),
                               bf16w(x[k+i+6u]),      bf16w(x[k+i+7u]));
            float4 p0 = w0 * x0;
            float4 p1 = w1 * x1;
            acc += p0.x; acc += p0.y; acc += p0.z; acc += p0.w;
            acc += p1.x; acc += p1.y; acc += p1.z; acc += p1.w;
        }
    }
    for (; k < K; k++)
        acc += bf16w(w[wb + k]) * bf16w(x[k]);
    y[gid] = acc;
}
)MSL";

/* ========================================================================
 * Context
 * ====================================================================== */

struct BufEnt {
    id<MTLBuffer> buf;
    size_t bytes;
    size_t off;     /* tensor's byte offset inside buf (page-rounded wraps) */
    int nocopy;
};

struct MtlCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLComputePipelineState> pso_gemv = nil;
    id<MTLComputePipelineState> pso_gemm = nil;
    id<MTLComputePipelineState> pso_matvec = nil;
    std::unordered_map<const void *, BufEnt> wcache;
    pthread_mutex_t opmu;   /* serializes whole ops: the tiered pilot's
                             * prediction thread calls the same hooked
                             * entry points CONCURRENTLY with the forward
                             * thread — the shared staging buffers must
                             * not be raced (M10 bug, caught by the
                             * 512-token tiered A/B text diff) */
    uint64_t budget = 0;
    uint64_t bytes_wrapped = 0;   /* zero-copy weight views */
    uint64_t bytes_uploaded = 0;  /* copied weight buffers */
    uint64_t dispatches = 0;
    /* staging (grown on demand) */
    id<MTLBuffer> xbuf = nil;   size_t xcap = 0;    /* u16 activations */
    id<MTLBuffer> obuf = nil;   size_t ocap = 0;    /* u16 / f32 outputs */
};

static MtlCtx *g_ctx = NULL;

static id<MTLBuffer> stage_buf(MtlCtx *c, id<MTLBuffer> &b, size_t &cap,
                               size_t need) {
    if (b && cap >= need) return b;
    b = [c->dev newBufferWithLength:need options:MTLResourceStorageModeShared];
    cap = b ? need : 0;
    return b;
}

/* Zero-copy wrap of an existing CPU allocation: verify the page-rounded
 * range [lo, hi) is covered by (possibly several) contiguous mapped
 * vm_regions — large mallocs are split into region chunks by the allocator
 * — then wrap it (page-aligned base, page-multiple length, no deallocator:
 * the engine owns the memory for the process lifetime). The tensor starts
 * *off bytes into the returned buffer. nil when not possible. */
static id<MTLBuffer> wrap_nocopy(MtlCtx *c, const void *ptr, size_t len,
                                 size_t *off) {
    uintptr_t lo = (uintptr_t)ptr & ~(uintptr_t)4095;
    uintptr_t hi = ((uintptr_t)ptr + len + 4095) & ~(uintptr_t)4095;
    if (hi <= lo) return nil;
    uintptr_t cur = lo;
    while (cur < hi) {
        vm_address_t q = (vm_address_t)cur;
        vm_size_t rsize = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t obj = MACH_PORT_NULL;
        if (vm_region_64(mach_task_self(), &q, &rsize,
                         VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info,
                         &count, &obj) != KERN_SUCCESS)
            return nil;
        if ((uintptr_t)q > cur || (uintptr_t)q + rsize <= cur)
            return nil;   /* gap / unexpected layout */
        if (!(info.protection & VM_PROT_READ)) return nil;
        cur = (uintptr_t)q + rsize;
    }
    *off = (uintptr_t)ptr - lo;
    return [c->dev newBufferWithBytesNoCopy:(void *)lo
                                     length:(NSUInteger)(hi - lo)
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

/* Weight buffer for CPU range [ptr, ptr+len): cached; zero-copy when
 * possible (tensor then sits *off_out bytes into the buffer), upload copy
 * otherwise (offset 0); nil when the budget is exhausted. */
static id<MTLBuffer> weight_buf(MtlCtx *c, const void *ptr, size_t len,
                                size_t *off_out) {
    auto it = c->wcache.find(ptr);
    if (it != c->wcache.end()) {
        *off_out = it->second.off;
        return it->second.buf;
    }
    if (c->bytes_wrapped + c->bytes_uploaded + len > c->budget) return nil;
    static int force_upload = getenv("APUS_METAL_NO_NOCOPY") != NULL;
    size_t off = 0;
    id<MTLBuffer> b = force_upload ? nil : wrap_nocopy(c, ptr, len, &off);
    BufEnt e;
    e.buf = b;
    e.bytes = len;
    e.off = off;
    if (b) {
        e.nocopy = 1;
        c->bytes_wrapped += len;
    } else {
        b = [c->dev newBufferWithLength:len
                                options:MTLResourceStorageModeShared];
        if (!b) return nil;
        memcpy([b contents], ptr, len);
        e.buf = b;
        e.off = 0;
        e.nocopy = 0;
        c->bytes_uploaded += len;
    }
    c->wcache.emplace(ptr, e);
    *off_out = e.off;
    return b;
}

/* One synchronous dispatch: encode with `encode`, commit, wait. Returns 0
 * on success, 1 on GPU error (fail-soft: caller falls back to CPU). */
typedef void (^EncodeFn)(id<MTLComputeCommandEncoder> enc);
static int run_op(MtlCtx *c, EncodeFn encode) {
    id<MTLCommandBuffer> cb = [c->q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    c->dispatches++;
    return [cb status] == MTLCommandBufferStatusError ? 1 : 0;
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc, NSUInteger total,
                        NSUInteger tg) {
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

/* ========================================================================
 * Ops (all return 0 = done, 1 = unsupported -> CPU fallback)
 * ====================================================================== */

static int mtl_gemv(MtlCtx *c, const uint16_t *w, const uint16_t *x,
                    uint16_t *y, size_t O, size_t K) {
    if (!c || !O || !K) return 1;
    pthread_mutex_lock(&c->opmu);
    size_t woff = 0;
    id<MTLBuffer> wb = weight_buf(c, w, O * K * sizeof(uint16_t), &woff);
    id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap, K * sizeof(uint16_t));
    id<MTLBuffer> ob = stage_buf(c, c->obuf, c->ocap, O * sizeof(uint16_t));
    if (!wb || !xb || !ob) { pthread_mutex_unlock(&c->opmu); return 1; }
    memcpy([xb contents], x, K * sizeof(uint16_t));
    uint Ku = (uint)K;
    uint64_t woff64 = woff;
    int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:c->pso_gemv];
        [enc setBuffer:wb offset:0 atIndex:0];
        [enc setBuffer:xb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&Ku length:4 atIndex:3];
        [enc setBytes:&woff64 length:8 atIndex:4];
        dispatch_1d(enc, (NSUInteger)O, 128);
    });
    if (rc) { pthread_mutex_unlock(&c->opmu); return 1; }
    memcpy(y, [ob contents], O * sizeof(uint16_t));
    pthread_mutex_unlock(&c->opmu);
    return 0;
}

static int mtl_gemm(MtlCtx *c, const uint16_t *w, const uint16_t *x,
                    uint16_t *y, size_t M, size_t O, size_t K) {
    if (!c || !M || !O || !K) return 1;
    pthread_mutex_lock(&c->opmu);
    size_t woff = 0;
    id<MTLBuffer> wb = weight_buf(c, w, O * K * sizeof(uint16_t), &woff);
    id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap,
                                 M * K * sizeof(uint16_t));
    id<MTLBuffer> ob = stage_buf(c, c->obuf, c->ocap,
                                 M * O * sizeof(uint16_t));
    if (!wb || !xb || !ob) { pthread_mutex_unlock(&c->opmu); return 1; }
    memcpy([xb contents], x, M * K * sizeof(uint16_t));
    uint Ou = (uint)O, Ku = (uint)K;
    uint64_t woff64 = woff;
    int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:c->pso_gemm];
        [enc setBuffer:wb offset:0 atIndex:0];
        [enc setBuffer:xb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&Ou length:4 atIndex:3];
        [enc setBytes:&Ku length:4 atIndex:4];
        [enc setBytes:&woff64 length:8 atIndex:5];
        dispatch_1d(enc, (NSUInteger)(M * O), 128);
    });
    if (rc) { pthread_mutex_unlock(&c->opmu); return 1; }
    memcpy(y, [ob contents], M * O * sizeof(uint16_t));
    pthread_mutex_unlock(&c->opmu);
    return 0;
}

static int mtl_matvec(MtlCtx *c, const uint16_t *w, const uint16_t *x,
                      float *y, size_t O, size_t K) {
    if (!c || !O || !K) return 1;
    pthread_mutex_lock(&c->opmu);
    size_t woff = 0;
    id<MTLBuffer> wb = weight_buf(c, w, O * K * sizeof(uint16_t), &woff);
    id<MTLBuffer> xb = stage_buf(c, c->xbuf, c->xcap, K * sizeof(uint16_t));
    id<MTLBuffer> ob = stage_buf(c, c->obuf, c->ocap, O * sizeof(float));
    if (!wb || !xb || !ob) { pthread_mutex_unlock(&c->opmu); return 1; }
    memcpy([xb contents], x, K * sizeof(uint16_t));
    uint Ku = (uint)K;
    uint64_t woff64 = woff;
    int rc = run_op(c, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:c->pso_matvec];
        [enc setBuffer:wb offset:0 atIndex:0];
        [enc setBuffer:xb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBytes:&Ku length:4 atIndex:3];
        [enc setBytes:&woff64 length:8 atIndex:4];
        dispatch_1d(enc, (NSUInteger)O, 128);
    });
    if (rc) { pthread_mutex_unlock(&c->opmu); return 1; }
    memcpy(y, [ob contents], O * sizeof(float));
    pthread_mutex_unlock(&c->opmu);
    return 0;
}

/* ========================================================================
 * Public C API (strong definitions; weak stubs live in the bf16.h TU)
 * ====================================================================== */

static int mtl_init(char *err, size_t errcap) {
    if (g_ctx) return 0;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            snprintf(err, errcap, "no Metal device");
            return -1;
        }
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        /* FP32 IEEE semantics: no fast-math contraction/reassociation.
         * (mathMode supersedes fastMathEnabled on macOS 15+.) */
        if ([opts respondsToSelector:@selector(setMathMode:)])
            opts.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *nserr = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kShaderSrc
                                               options:opts
                                                 error:&nserr];
        if (!lib) {
            snprintf(err, errcap, "shader compile: %s",
                     nserr ? [[nserr localizedDescription] UTF8String]
                           : "unknown");
            return -1;
        }
        MtlCtx *c = new MtlCtx();
        pthread_mutex_init(&c->opmu, NULL);
        c->dev = dev;
        c->q = [dev newCommandQueue];
        if (!c->q) { snprintf(err, errcap, "no command queue"); delete c;
                     return -1; }
        struct { const char *name; id<MTLComputePipelineState> *slot; }
            t[] = {
            { "bf16_gemv_ilp", &c->pso_gemv },
            { "bf16_gemm_ilp", &c->pso_gemm },
            { "bf16_matvec_f32", &c->pso_matvec },
        };
        for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:t[i].name]];
            *t[i].slot = fn ? [dev newComputePipelineStateWithFunction:fn
                                                                 error:&nserr]
                            : nil;
            if (!*t[i].slot) {
                snprintf(err, errcap, "pipeline %s failed", t[i].name);
                delete c;
                return -1;
            }
        }
        long mb = 8192;
        const char *e = getenv("APUS_METAL_DENSE_MB");
        if (e && *e) mb = atol(e);
        if (mb < 0) mb = 0;
        c->budget = (uint64_t)mb << 20;
        g_ctx = c;
    }
    return 0;
}

/* --- hook trampolines --- */
static int hook_gemv(const uint16_t *w, const uint16_t *x, uint16_t *y,
                     size_t O, size_t K) {
    return mtl_gemv(g_ctx, w, x, y, O, K);
}
static int hook_gemm(const uint16_t *w, const uint16_t *x, uint16_t *y,
                     size_t M, size_t O, size_t K) {
    return mtl_gemm(g_ctx, w, x, y, M, O, K);
}
static int hook_matvec(const uint16_t *w, const uint16_t *x, float *y,
                       size_t O, size_t K) {
    return mtl_matvec(g_ctx, w, x, y, O, K);
}

int apus_metal_enable(char *err, size_t errcap) {
    char e2[256];
    if (!err) { err = e2; errcap = sizeof e2; }
    if (mtl_init(err, errcap)) return -1;
    apus_backend_hooks.bf16_gemv = getenv("APUS_METAL_NO_GEMV") ? NULL
        : hook_gemv;
    apus_backend_hooks.bf16_gemm = getenv("APUS_METAL_NO_GEMM") ? NULL
        : hook_gemm;
    apus_backend_hooks.bf16_matvec_f32 = getenv("APUS_METAL_NO_MATVEC")
        ? NULL : hook_matvec;
    return 0;
}

void apus_metal_disable(void) {
    memset(&apus_backend_hooks, 0, sizeof apus_backend_hooks);
    MtlCtx *c = g_ctx;
    g_ctx = NULL;
    if (c) delete c;   /* ARC off: ObjC members leak harmlessly at exit */
}

int apus_metal_is_enabled(void) {
    return g_ctx && apus_backend_hooks.bf16_gemv;
}

uint64_t apus_metal_bytes_wrapped(void) {
    return g_ctx ? g_ctx->bytes_wrapped : 0;
}
uint64_t apus_metal_bytes_uploaded(void) {
    return g_ctx ? g_ctx->bytes_uploaded : 0;
}
uint64_t apus_metal_dispatches(void) {
    return g_ctx ? g_ctx->dispatches : 0;
}

/* --- direct entry points (kernel-level tests; lazy init, no hook fill) --- */

int apus_metal_bf16_gemv(const uint16_t *w, const uint16_t *x, uint16_t *y,
                         size_t O, size_t K) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    return mtl_gemv(g_ctx, w, x, y, O, K);
}

int apus_metal_bf16_gemm(const uint16_t *w, const uint16_t *x, uint16_t *y,
                         size_t M, size_t O, size_t K) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    return mtl_gemm(g_ctx, w, x, y, M, O, K);
}

int apus_metal_bf16_matvec_f32(const uint16_t *w, const uint16_t *x,
                               float *y, size_t O, size_t K) {
    char e[128];
    if (mtl_init(e, sizeof e)) return 1;
    return mtl_matvec(g_ctx, w, x, y, O, K);
}
