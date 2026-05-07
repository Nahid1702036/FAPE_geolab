#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            throw std::runtime_error(                                         \
                std::string("CUDA error: ") + cudaGetErrorString(_e) +       \
                " at " __FILE__ ":" + std::to_string(__LINE__));              \
        }                                                                     \
    } while (0)

void fape_surgical_cuda_forward(
    const float* vertices,
    const int*   faces,
    const float* vertex_normals,
    const int*   edges,
    const float* all_cam_mats,
    const float* hero_cam_mats,
    float*       out_vertices,
    float*       out_validity,
    int num_v, int num_f, int num_e,
    int num_all, int num_hero,
    float tau, float sigma_s, float sigma_n, float epsilon,
    cudaStream_t stream
);
