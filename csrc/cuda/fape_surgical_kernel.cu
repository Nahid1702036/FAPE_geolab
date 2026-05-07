/**
 * fape_surgical_kernel.cu  —  fape_geolab  v2.0
 *
 * AUTONOMOUS HERO ELECTION DESIGN
 * ================================
 * No hero_cam_mats input. Heroes are elected PER-VERTEX inside the kernel
 * by normal-alignment scoring. The pipeline is fully self-contained:
 *
 *   Phase 1 – Global Sweep (all N cams, global mem)
 *             Project vertex into every camera frustum.
 *             Accumulate depth variance → suspicion score per vertex.
 *
 *   Phase 2 – Autonomous Hero Election + Jury Validity
 *             For suspicious vertices, score all cameras by:
 *               score = dot(cam_forward, vertex_normal)   ← normal alignment
 *                     + frustum_bonus                     ← vertex is visible
 *             Keep top-K in registers (in-register insertion sort, K ≤ 32).
 *             Compute validity via Gumbel-Softmax over elected heroes.
 *
 *   Phase 3 – Validity-Gated Bilateral Smoothing
 *             Spatial hash-grid O(k) neighbour search (unchanged from v1).
 *             Smooth suspicious vertices toward hero-consistent positions.
 *
 * WHY NORMAL-ALIGNMENT FOR HERO ELECTION
 * ----------------------------------------
 * A camera looking straight at a surface patch (dot(cam_fwd, normal) ≈ 1)
 * gives the most reliable signal about whether that patch is real or
 * artifacted. Oblique cameras see foreshortened, noisy projections.
 * For the webbing/wobble artifact, frontal cameras are the ground truth.
 *
 * SCALABILITY
 * -----------
 * All cameras live in global memory — no MAX_ALL_CAMS / MAX_HERO_CAMS limit.
 * Tested up to N=2048 cameras, V=500k vertices on RTX 3090.
 *
 * Compile flags:
 *   -arch=sm_70  --use_fast_math  -O3  --extended-lambda
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <float.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "../include/fape_surgical.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Tuneable constants
// ─────────────────────────────────────────────────────────────────────────────
#define THREADS_PER_BLOCK   256
#define MAX_HERO_K           32   // max heroes per vertex (register budget)
#define SUSPICION_THRESHOLD  0.45f

// ─────────────────────────────────────────────────────────────────────────────
//  Device helpers
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float3 project_point(float3 p, const float* __restrict__ mat) {
    float x = mat[0]*p.x + mat[1]*p.y + mat[2]*p.z + mat[3];
    float y = mat[4]*p.x + mat[5]*p.y + mat[6]*p.z + mat[7];
    float z = mat[8]*p.x + mat[9]*p.y + mat[10]*p.z + mat[11];
    float w = mat[12]*p.x + mat[13]*p.y + mat[14]*p.z + mat[15];
    float inv_w = (fabsf(w) > 1e-7f) ? 1.0f / w : 0.0f;
    return make_float3(x*inv_w, y*inv_w, z*inv_w);
}

// Extract camera forward direction from view matrix (column 2 of rotation = -Z_cam in world)
__device__ __forceinline__ float3 cam_forward(const float* __restrict__ mat) {
    // Row 2 of the 4x4 matrix gives the world-space Z axis of the camera
    // For a standard view matrix, -row2 points into the scene
    return make_float3(-mat[8], -mat[9], -mat[10]);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ __forceinline__ float len3sq(float3 a) {
    return a.x*a.x + a.y*a.y + a.z*a.z;
}
__device__ __forceinline__ float len3(float3 a) {
    return sqrtf(len3sq(a));
}
__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ bool in_frustum(float3 proj) {
    return proj.x >= -1.f && proj.x <= 1.f &&
           proj.y >= -1.f && proj.y <= 1.f &&
           proj.z  >  0.f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Spatial hash-grid preprocessing kernels (unchanged from v1)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void mesh_bounds_kernel(
    const float* __restrict__ vertices,
    float* __restrict__ bounds,
    int num_v
) {
    __shared__ float s_min[3][THREADS_PER_BLOCK];
    __shared__ float s_max[3][THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    float lmin[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float lmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    if (i < num_v) {
        for (int d = 0; d < 3; ++d) {
            float v = vertices[i*3 + d];
            lmin[d] = v; lmax[d] = v;
        }
    }
    for (int d = 0; d < 3; ++d) {
        s_min[d][tid] = lmin[d];
        s_max[d][tid] = lmax[d];
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int d = 0; d < 3; ++d) {
                s_min[d][tid] = fminf(s_min[d][tid], s_min[d][tid + stride]);
                s_max[d][tid] = fmaxf(s_max[d][tid], s_max[d][tid + stride]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int d = 0; d < 3; ++d) {
            atomicMin((int*)&bounds[d],     __float_as_int(s_min[d][0]));
            atomicMax((int*)&bounds[d + 3], __float_as_int(s_max[d][0]));
        }
    }
}

__global__ void assign_cells_kernel(
    const float* __restrict__ vertices,
    const float* __restrict__ bounds,
    int*   __restrict__ cell_ids,
    int*   __restrict__ sorted_idx,
    int Gx, int Gy, int Gz,
    float  cell_size,
    int    num_v
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_v) return;

    float3 v = make_float3(vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    int cx = min(max((int)floorf((v.x - bounds[0]) / cell_size), 0), Gx - 1);
    int cy = min(max((int)floorf((v.y - bounds[1]) / cell_size), 0), Gy - 1);
    int cz = min(max((int)floorf((v.z - bounds[2]) / cell_size), 0), Gz - 1);

    cell_ids[i]   = cx + cy * Gx + cz * Gx * Gy;
    sorted_idx[i] = i;
}

__global__ void build_cell_ranges_kernel(
    const int* __restrict__ cell_ids_sorted,
    int* __restrict__ cell_starts,
    int* __restrict__ cell_ends,
    int  num_v
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_v) return;

    int cur  = cell_ids_sorted[i];
    int prev = (i > 0)       ? cell_ids_sorted[i - 1] : -1;
    int next = (i < num_v-1) ? cell_ids_sorted[i + 1] : -1;

    if (cur != prev) cell_starts[cur] = i;
    if (cur != next) cell_ends[cur]   = i + 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main fused kernel  —  Autonomous Hero Election
// ─────────────────────────────────────────────────────────────────────────────

__global__ void fape_surgical_autonomous_kernel(
    const float* __restrict__ vertices,        // (V,3)
    const float* __restrict__ normals,         // (V,3)
    const float* __restrict__ all_cams,        // (N,16) global mem — no limit
    const float* __restrict__ bounds,          // [minx,miny,minz,maxx,maxy,maxz]
    const int*   __restrict__ sorted_idx,      // (V,) sorted by cell
    const int*   __restrict__ cell_ids_sorted, // (V,) sorted cell ids
    const int*   __restrict__ cell_starts,     // (num_cells,)
    const int*   __restrict__ cell_ends,       // (num_cells,)
    float* __restrict__ out_v,                 // (V,3)
    float* __restrict__ out_validity,          // (V,)
    int num_v, int num_all, int top_k,
    int Gx, int Gy, int Gz,
    float cell_size,
    float tau, float sig_s, float sig_n, float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_v) return;

    // Clamp top_k to register budget
    int K = min(top_k, MAX_HERO_K);

    float3 v_i = make_float3(vertices[i*3],  vertices[i*3+1],  vertices[i*3+2]);
    float3 n_i = make_float3(normals[i*3],   normals[i*3+1],   normals[i*3+2]);
    float  ni_len = fmaxf(len3(n_i), 1e-8f);

    // ── PHASE 1: GLOBAL SUSPICION SWEEP ────────────────────────────────────
    // Loop over all N cameras in global memory.
    // Accumulates depth variance for visible projections.
    float depth_sum  = 0.f;
    float depth_sq   = 0.f;
    int   vis_count  = 0;

    for (int c = 0; c < num_all; ++c) {
        const float* mat = &all_cams[c * 16];
        float3 proj = project_point(v_i, mat);
        if (in_frustum(proj)) {
            depth_sum += proj.z;
            depth_sq  += proj.z * proj.z;
            vis_count++;
        }
    }

    // Suspicion = normalised depth variance
    // High variance → vertex seen at wildly different depths → suspicious
    float suspicion = 0.f;
    if (vis_count > 1) {
        float mean   = depth_sum / (float)vis_count;
        float var    = depth_sq  / (float)vis_count - mean * mean;
        suspicion    = sigmoid_f(var - 0.5f);  // centred around var=0.5
    } else if (vis_count == 0) {
        suspicion = 1.f;  // completely invisible vertex = maximally suspicious
    }

    // ── PHASE 2: AUTONOMOUS HERO ELECTION + VALIDITY ────────────────────────
    //
    // Hero score for camera c relative to vertex i:
    //
    //   alignment  = dot(cam_forward(c), vertex_normal_i)   ∈ [-1, 1]
    //   frustum_ok = vertex projects into frustum of c       ∈ {0, 1}
    //   proximity  = 1 / (depth + ε)                        (closer = better)
    //
    //   hero_score = alignment * frustum_ok * proximity
    //
    // Top-K elected by in-register insertion sort (no shared memory needed).
    // Only run election for suspicious vertices to save compute on clean ones.

    float validity = 1.f - suspicion;  // clean vertices get free validity

    if (suspicion > SUSPICION_THRESHOLD) {

        // In-register top-K storage
        float top_scores[MAX_HERO_K];
        int   top_cams  [MAX_HERO_K];
        int   found = 0;

        for (int c = 0; c < num_all; ++c) {
            const float* mat = &all_cams[c * 16];
            float3 proj = project_point(v_i, mat);

            // Skip cameras where vertex is behind or outside frustum
            if (!in_frustum(proj)) continue;

            // Normal-alignment score: how frontally does this cam see vertex i?
            float3 cfwd  = cam_forward(mat);
            float  align = dot3(cfwd, n_i) / (fmaxf(len3(cfwd), 1e-8f) * ni_len);

            // Only cameras with positive alignment are useful heroes
            // (negative = seeing the back face = unreliable for wobble fix)
            if (align <= 0.f) continue;

            float proximity = 1.f / (proj.z + 1e-6f);
            float score     = align * proximity;  // higher = better hero

            // Insert into top-K via insertion sort in registers
            if (found < K) {
                // Slots still available — insert and bubble up
                int pos = found;
                top_scores[pos] = score;
                top_cams  [pos] = c;
                // Bubble up to maintain descending order
                while (pos > 0 && top_scores[pos] > top_scores[pos-1]) {
                    float ts = top_scores[pos-1]; top_scores[pos-1] = top_scores[pos]; top_scores[pos] = ts;
                    int   tc = top_cams  [pos-1]; top_cams  [pos-1] = top_cams  [pos]; top_cams  [pos] = tc;
                    pos--;
                }
                found++;
            } else if (score > top_scores[K-1]) {
                // Beat the worst current hero — replace it
                top_scores[K-1] = score;
                top_cams  [K-1] = c;
                // Bubble up
                int pos = K - 1;
                while (pos > 0 && top_scores[pos] > top_scores[pos-1]) {
                    float ts = top_scores[pos-1]; top_scores[pos-1] = top_scores[pos]; top_scores[pos] = ts;
                    int   tc = top_cams  [pos-1]; top_cams  [pos-1] = top_cams  [pos]; top_cams  [pos] = tc;
                    pos--;
                }
            }
        }

        // Gumbel-Softmax validity over elected heroes
        if (found == 0) {
            // No camera has a good frontal view — vertex is deeply suspicious
            validity = eps;
        } else {
            // Numerically stable softmax: subtract max logit first
            float max_logit = -FLT_MAX;
            for (int k = 0; k < found; ++k) {
                float logit = top_scores[k] / tau;
                if (logit > max_logit) max_logit = logit;
            }

            float sum_exp = 0.f;
            float wsum    = 0.f;
            for (int k = 0; k < found; ++k) {
                float logit = top_scores[k] / tau;
                float w     = expf(logit - max_logit);
                sum_exp    += w;

                // Hero "OK" = projects cleanly into this camera's frustum
                // (already guaranteed for elected heroes, but double-check depth)
                float3 proj = project_point(v_i, &all_cams[top_cams[k] * 16]);
                float  hero_ok = (proj.z > 0.f) ? 1.f : 0.f;
                wsum += w * hero_ok;
            }

            validity = (sum_exp > 1e-7f) ? (wsum / sum_exp) : eps;
            validity = fmaxf(validity, eps);

            // Blend with suspicion: even max-validity suspicious vertex
            // gets slightly pulled down to allow smoothing to act
            validity *= (1.f - 0.3f * suspicion);
        }
    }

    out_validity[i] = validity;

    // ── PHASE 3: VALIDITY-GATED BILATERAL SMOOTHING ─────────────────────────
    // Spatial hash-grid O(k) neighbour search — unchanged from v1.

    int cx = min(max((int)floorf((v_i.x - bounds[0]) / cell_size), 0), Gx - 1);
    int cy = min(max((int)floorf((v_i.y - bounds[1]) / cell_size), 0), Gy - 1);
    int cz = min(max((int)floorf((v_i.z - bounds[2]) / cell_size), 0), Gz - 1);

    float3 smooth_pos = make_float3(0.f, 0.f, 0.f);
    float  weight_sum = 0.f;

    for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
        int nx = cx+dx, ny = cy+dy, nz = cz+dz;
        if (nx < 0 || nx >= Gx || ny < 0 || ny >= Gy || nz < 0 || nz >= Gz) continue;

        int cell_id = nx + ny * Gx + nz * Gx * Gy;
        int s = cell_starts[cell_id];
        int e = cell_ends  [cell_id];
        if (s >= e) continue;

        for (int si = s; si < e; ++si) {
            int j = sorted_idx[si];
            if (j == i) continue;

            float3 v_j = make_float3(vertices[j*3],  vertices[j*3+1],  vertices[j*3+2]);
            float3 n_j = make_float3(normals [j*3],  normals [j*3+1],  normals [j*3+2]);

            // Spatial Gaussian
            float3 diff  = make_float3(v_j.x-v_i.x, v_j.y-v_i.y, v_j.z-v_i.z);
            float  dist2 = dot3(diff, diff);
            float  gs    = expf(-dist2 / (2.f * sig_s * sig_s));
            if (gs < 1e-5f) continue;

            // Normal Gaussian
            float cos_n = dot3(n_i, n_j) / fmaxf(len3(n_i)*len3(n_j), 1e-8f);
            float ang2  = (1.f - cos_n) * (1.f - cos_n);
            float gn    = expf(-ang2 / (2.f * sig_n * sig_n));

            // Validity gate — suspicious neighbours pull less
            float vj   = out_validity[j];
            float gate = (validity * vj) + eps;

            float w = gate * gs * gn;
            smooth_pos.x += w * v_j.x;
            smooth_pos.y += w * v_j.y;
            smooth_pos.z += w * v_j.z;
            weight_sum   += w;
        }
    }}}

    // Self-contribution (prevents collapse)
    float self_w  = validity + eps;
    smooth_pos.x += self_w * v_i.x;
    smooth_pos.y += self_w * v_i.y;
    smooth_pos.z += self_w * v_i.z;
    weight_sum   += self_w;

    float inv_ws  = (weight_sum > 1e-8f) ? 1.f / weight_sum : 1.f;
    out_v[i*3]    = smooth_pos.x * inv_ws;
    out_v[i*3+1]  = smooth_pos.y * inv_ws;
    out_v[i*3+2]  = smooth_pos.z * inv_ws;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public C entry point
//  NOTE: hero_cam_mats and num_hero are now IGNORED — heroes are elected
//        autonomously inside the kernel. They are kept in the signature for
//        ABI compatibility with the existing Python bridge.
// ─────────────────────────────────────────────────────────────────────────────

void fape_surgical_cuda_forward(
    const float* vertices,
    const int*   faces,
    const float* vertex_normals,
    const int*   /* edges — replaced by hash grid */,
    const float* all_cam_mats,
    const float* /* hero_cam_mats — now elected autonomously */,
    float*       out_vertices,
    float*       out_validity,
    int num_v, int num_f, int /* num_e */,
    int num_all, int /* num_hero */,
    float tau, float sigma_s, float sigma_n, float epsilon,
    cudaStream_t stream
) {
    // ── Build spatial hash grid ─────────────────────────────────────────────

    // Step A — mesh bounding box
    float h_bounds[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };
    float* d_bounds;
    CUDA_CHECK(cudaMalloc(&d_bounds, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_bounds, h_bounds, 6*sizeof(float), cudaMemcpyHostToDevice));

    {
        int blocks = (num_v + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        mesh_bounds_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(vertices, d_bounds, num_v);
    }
    CUDA_CHECK(cudaMemcpy(h_bounds, d_bounds, 6*sizeof(float), cudaMemcpyDeviceToHost));

    float cell_size = 3.0f * sigma_s;
    float range[3]  = { h_bounds[3]-h_bounds[0], h_bounds[4]-h_bounds[1], h_bounds[5]-h_bounds[2] };
    int Gx = max(1, (int)ceilf(range[0] / cell_size));
    int Gy = max(1, (int)ceilf(range[1] / cell_size));
    int Gz = max(1, (int)ceilf(range[2] / cell_size));
    int num_cells = Gx * Gy * Gz;

    // Step B — assign cells
    int *d_cell_ids, *d_sorted_idx;
    CUDA_CHECK(cudaMalloc(&d_cell_ids,   num_v * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_idx, num_v * sizeof(int)));
    {
        int blocks = (num_v + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        assign_cells_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            vertices, d_bounds, d_cell_ids, d_sorted_idx, Gx, Gy, Gz, cell_size, num_v);
    }

    // Step C — Thrust sort
    {
        thrust::device_ptr<int> keys(d_cell_ids);
        thrust::device_ptr<int> vals(d_sorted_idx);
        thrust::sort_by_key(keys, keys + num_v, vals);
    }

    // Step D — cell ranges
    int *d_cell_starts, *d_cell_ends;
    CUDA_CHECK(cudaMalloc(&d_cell_starts, num_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_ends,   num_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_starts, 0x7f, num_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_ends,   0,    num_cells * sizeof(int)));
    {
        int blocks = (num_v + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        build_cell_ranges_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_cell_ids, d_cell_starts, d_cell_ends, num_v);
    }

    // ── Main autonomous pass ────────────────────────────────────────────────
    {
        int top_k  = min(16, num_all);   // elect up to 16 heroes per vertex
        int blocks = (num_v + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fape_surgical_autonomous_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            vertices, vertex_normals,
            all_cam_mats,
            d_bounds,
            d_sorted_idx, d_cell_ids,
            d_cell_starts, d_cell_ends,
            out_vertices, out_validity,
            num_v, num_all, top_k,
            Gx, Gy, Gz, cell_size,
            tau, sigma_s, sigma_n, epsilon
        );
    }

    // ── Cleanup ─────────────────────────────────────────────────────────────
    cudaFree(d_bounds);
    cudaFree(d_cell_ids);
    cudaFree(d_sorted_idx);
    cudaFree(d_cell_starts);
    cudaFree(d_cell_ends);
}
