/*
 * c/backend_metal.h — optional Metal backend for the dense BF16 compute
 * of Qwen3.6-35B-A3B (M10): the bf16 GEMV/GEMM hot paths (GDN and gated-
 * GQA attention projections, the router scoring matvec, eager expert and
 * shared-expert linears, lm_head) as FP32 Metal shaders, zero-copy
 * unified-memory weight buffers. Pure C11 interface (the implementation
 * lives in c/backend_metal.mm, Objective-C++, only in the metal=1
 * build).
 *
 * Backend interface design (the Apus M7b pattern): a FUNCTION-POINTER
 * TABLE, `apus_backend_hooks` (defined once in the APUS_BF16_IMPLEMENTATION
 * TU — bf16.h is linked into every engine binary), all-NULL by default =
 * the existing CPU kernels. apus_metal_enable() (strong definition in
 * c/backend_metal.mm; weak stub in the bf16.h TU so the plain CPU binary
 * links and behaves exactly as before, and the Linux build is unaffected)
 * initializes Metal and fills the table. Call sites (the c/bf16.h hot
 * wrappers, c/moe.h's router matvec) try the hook first and fall back to
 * the CPU kernel when the pointer is NULL or the call returns nonzero
 * (unsupported shape / budget exhausted / no GPU) — per-op fail-soft,
 * never a crash.
 *
 * Numerics contract: FP32 compute only, fast-math DISABLED (no
 * contraction, no reassociation). The shaders replicate the DISPATCHED
 * CPU kernels' rounding sequences EXACTLY — not merely their tolerance
 * class:
 *   - bf16_gemv / bf16_gemm: the c/bf16.h M9b ILP NEON kernel's per-row
 *     order — four float4 accumulators per output row, per ascending
 *     32-wide chunk the eight single-rounded product vectors accumulate
 *     acc[q & 3] += p_q (q ascending), fixed combine
 *     ((a0+a1)+(a2+a3)) then ((s.x+s.y)+(s.z+s.w)), K tail appended with
 *     scalar ascending adds, one RNE bf16 narrow at the end. Widening is
 *     the exact u16<<16 shift. BITWISE identical to apus_bf16_gemv_hot /
 *     apus_bf16_gemm_hot on ARM (tests/m10 asserts it on every shape).
 *     This covers the Qwen router too: apus_moe_route scores through
 *     apus_bf16_gemv_hot (256x2048, bf16 logits -> fp32 softmax).
 *   - bf16_matvec_f32: the c/moe.h fp32-out matvec — staged
 *     single-rounded products per 32-wide chunk, strictly ascending
 *     scalar adds per output, fp32 out unrounded. BITWISE identical to
 *     apus_moe_matvec_f32_hot. (Kept generic machinery — the Qwen model
 *     has no fp32-out router caller; the hook is pinned at kernel level
 *     in tests/m10 regardless.)
 * Weight residency: the pointer-keyed zero-copy cache assumes weight
 * pointers are STABLE and contents immutable for the backend's lifetime
 * (true for resident dense weights: shard slabs, model arrays). TIERED
 * expert slabs are transient (LRU) — the store's per-expert call sites
 * (c/cache.h apus_store_moe) deliberately use the no-hook
 * apus_bf16_gemv_hot_cpu entry point, so expert compute stays on the
 * CPU in tiered mode; everything resident is offloaded.
 *
 * APUS_METAL_DENSE_MB (default 8192) caps wrapped+uploaded weight bytes;
 * past the cap the hook returns "unsupported" and the op runs on the
 * CPU. apus_metal_disable() drops the cache (call before freeing model
 * weights).
 */
#ifndef APUS_BACKEND_METAL_H
#define APUS_BACKEND_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backend hook table. All-NULL = CPU kernels (default). A hook returns 0
 * when it produced the result, nonzero = unsupported (caller runs the
 * CPU kernel). Shapes match the CPU entry points they replace:
 *  - bf16_gemv:       c/bf16.h apus_bf16_gemv_hot (w [O,K], x [K] bf16,
 *                     y [O] bf16) — every dense decode GEMV: GDN/gated-GQA
 *                     projections, router scoring (256x2048), eager
 *                     expert + shared-expert linears, lm_head
 *                     (248320x2048)
 *  - bf16_gemm:       c/bf16.h apus_bf16_gemm_hot (w [O,K], x [M,K],
 *                     y [M,O] bf16) — the M9 batched prefill phase-A
 *                     projections and phase-B expert batches (M=T)
 *  - bf16_matvec_f32: c/moe.h apus_moe_matvec_f32_hot (w [O,K], x [K]
 *                     bf16, y [O] FP32 unrounded) — kept generic
 *                     machinery, no model caller in Qwen */
typedef struct {
    int (*bf16_gemv)(const uint16_t *w, const uint16_t *x, uint16_t *y,
                     size_t O, size_t K);
    int (*bf16_gemm)(const uint16_t *w, const uint16_t *x, uint16_t *y,
                     size_t M, size_t O, size_t K);
    int (*bf16_matvec_f32)(const uint16_t *w, const uint16_t *x, float *y,
                           size_t O, size_t K);
} ApusBackendHooks;

extern ApusBackendHooks apus_backend_hooks;

/* Enable the Metal backend: initialize the device/queue/pipelines and
 * fill apus_backend_hooks. Returns 0 on success; nonzero (err filled)
 * when Metal is unavailable or the backend was not compiled in — the
 * hooks stay NULL and the engine runs on the CPU kernels (fail-soft).
 * Idempotent. */
int  apus_metal_enable(char *err, size_t errcap);
/* Detach the hooks (CPU kernels again) and release backend state. */
void apus_metal_disable(void);
int  apus_metal_is_enabled(void);

/* Instrumentation: resident weight bytes wrapped zero-copy / uploaded by
 * copy, GPU dispatches submitted. */
uint64_t apus_metal_bytes_wrapped(void);
uint64_t apus_metal_bytes_uploaded(void);
uint64_t apus_metal_dispatches(void);

/* Direct entry points (kernel-level tests; same semantics as the hooks,
 * no hook-table indirection, Metal initialized lazily). Return 0 on
 * success, 1 when Metal/unsupported. */
int apus_metal_bf16_gemv(const uint16_t *w, const uint16_t *x, uint16_t *y,
                         size_t O, size_t K);
int apus_metal_bf16_gemm(const uint16_t *w, const uint16_t *x, uint16_t *y,
                         size_t M, size_t O, size_t K);
int apus_metal_bf16_matvec_f32(const uint16_t *w, const uint16_t *x,
                               float *y, size_t O, size_t K);

#ifdef __cplusplus
}
#endif

#endif /* APUS_BACKEND_METAL_H */
