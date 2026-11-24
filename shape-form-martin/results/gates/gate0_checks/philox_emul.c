/* Gate 0 helper: CPU re-implementation of the random stream of torch.rand(..., device="cuda", generator=g)
 * for float32 tensors (aten/src/ATen/native/cuda/DistributionTemplates.h, uniform_and_transform with
 * curand_uniform4 on a Philox4x32-10 state per CUDA thread).
 *
 * PyTorch launches `grid` blocks of 256 threads, grid = min(SM_count * (maxThreadsPerSM / 256), ceil(numel/256)).
 * Thread idx (0 .. T-1, T = 256*grid) is initialised with curand_init(seed, subsequence=idx, offset) and its n-th
 * call of curand_uniform4 fills the elements  li = idx + T*(4n + ii), ii = 0..3.  The generator's offset then advances
 * by ((numel-1)/(4T) + 1)*4.  Hence the stream of a large tensor depends on the GPU model through T.
 *
 * philox_fill(seed, offset, T, numel, out): out[li] for li < numel, exactly as that kernel would with T threads.
 * Build: gcc -O2 -fopenmp -shared -fPIC -ffp-contract=off philox_emul.c -o philox_emul.so
 */
#include <stdint.h>

static inline void philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1, uint32_t *o)
{
    for (int r = 0; r < 10; r++) {
        uint64_t p0 = (uint64_t)0xD2511F53u * c0;
        uint64_t p1 = (uint64_t)0xCD9E8D57u * c2;
        uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
        uint32_t n1 = (uint32_t)p1;
        uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
        uint32_t n3 = (uint32_t)p0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    o[0] = c0; o[1] = c1; o[2] = c2; o[3] = c3;
}

static inline float to_uniform(uint32_t x)
{
    /* _curand_uniform: x * 2^-32 + 2^-33 in float32, range (0, 1]; PyTorch then maps 1.0 -> 0.0 */
    float y = (float)x * 2.3283064e-10f + (2.3283064e-10f / 2.0f);
    return y == 1.0f ? 0.0f : y;
}

void philox_fill(uint64_t seed, uint64_t offset, int64_t T, int64_t numel, float *out)
{
    const uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
    const uint64_t base = offset / 4;                    /* offsets are multiples of 4 (asserted by PyTorch) */
    const int64_t n_iter = (numel - 1) / (4 * T) + 1;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int64_t n = 0; n < n_iter; n++) {
        for (int64_t idx = 0; idx < T; idx++) {
            uint64_t ctr = base + (uint64_t)n;
            uint32_t r[4];
            philox4x32_10((uint32_t)ctr, (uint32_t)(ctr >> 32), (uint32_t)idx, (uint32_t)((uint64_t)idx >> 32), k0, k1, r);
            for (int ii = 0; ii < 4; ii++) {
                int64_t li = idx + T * (4 * n + ii);
                if (li < numel) out[li] = to_uniform(r[ii]);
            }
        }
    }
}
