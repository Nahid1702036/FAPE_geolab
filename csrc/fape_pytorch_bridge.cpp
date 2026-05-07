/**
 * fape_pytorch_bridge.cpp
 * PyTorch pybind11 C++ extension — part of fape_surgical_torch package.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "include/fape_surgical.h"

#define CHECK_CUDA(x)    TORCH_CHECK((x).device().is_cuda(),    #x " must be a CUDA tensor")
#define CHECK_CONTIG(x)  TORCH_CHECK((x).is_contiguous(),       #x " must be contiguous")
#define CHECK_F32(x)     TORCH_CHECK((x).dtype()==torch::kFloat32, #x " must be float32")
#define CHECK_I32(x)     TORCH_CHECK((x).dtype()==torch::kInt32,   #x " must be int32")
#define CHK_F(x) do { CHECK_CUDA(x); CHECK_CONTIG(x); CHECK_F32(x); } while(0)
#define CHK_I(x) do { CHECK_CUDA(x); CHECK_CONTIG(x); CHECK_I32(x); } while(0)

std::vector<torch::Tensor> run_surgical_geometry_pass(
    torch::Tensor vertices,        // (V,3)  float32 CUDA
    torch::Tensor faces,           // (F,3)  int32   CUDA
    torch::Tensor vertex_normals,  // (V,3)  float32 CUDA
    torch::Tensor edges,           // (E,2)  int32   CUDA  [ignored — kept for ABI compat]
    torch::Tensor all_cam_mats,    // (N,4,4)float32 CUDA
    torch::Tensor hero_cam_mats,   // (K,4,4)float32 CUDA
    float tau     = 0.1f,
    float sigma_s = 0.05f,
    float sigma_n = 0.15f,
    float epsilon = 1e-4f
) {
    CHK_F(vertices);      CHK_I(faces);
    CHK_F(vertex_normals);CHK_I(edges);
    CHK_F(all_cam_mats);  CHK_F(hero_cam_mats);

    TORCH_CHECK(vertices.dim()==2 && vertices.size(1)==3,       "vertices: (V,3)");
    TORCH_CHECK(vertex_normals.sizes()==vertices.sizes(),        "normals: (V,3)");
    TORCH_CHECK(faces.dim()==2 && faces.size(1)==3,             "faces: (F,3)");
    TORCH_CHECK(edges.dim()==2 && edges.size(1)==2,             "edges: (E,2)");
    TORCH_CHECK(all_cam_mats.dim()==3 &&
                all_cam_mats.size(1)==4 && all_cam_mats.size(2)==4, "all_cams: (N,4,4)");
    TORCH_CHECK(hero_cam_mats.dim()==3 &&
                hero_cam_mats.size(1)==4 && hero_cam_mats.size(2)==4,"hero_cams:(K,4,4)");

    const int num_v    = vertices.size(0);
    const int num_f    = faces.size(0);
    const int num_e    = edges.size(0);
    const int num_all  = all_cam_mats.size(0);
    const int num_hero = hero_cam_mats.size(0);

    auto opts_f = vertices.options();
    torch::Tensor out_vertices = torch::zeros({num_v, 3}, opts_f);
    torch::Tensor out_validity = torch::zeros({num_v},    opts_f);

    fape_surgical_cuda_forward(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        vertex_normals.data_ptr<float>(),
        edges.data_ptr<int>(),
        all_cam_mats.reshape({-1}).data_ptr<float>(),
        hero_cam_mats.reshape({-1}).data_ptr<float>(),
        out_vertices.data_ptr<float>(),
        out_validity.data_ptr<float>(),
        num_v, num_f, num_e, num_all, num_hero,
        tau, sigma_s, sigma_n, epsilon,
        at::cuda::getCurrentCUDAStream()
    );

    return {out_vertices, out_validity};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FAPE Surgical Geometry — PyTorch CUDA Extension";
    m.def("run_surgical_geometry_pass", &run_surgical_geometry_pass,
          "Fused surgical geometry pass (CUDA)",
          py::arg("vertices"), py::arg("faces"),
          py::arg("vertex_normals"), py::arg("edges"),
          py::arg("all_cam_mats"), py::arg("hero_cam_mats"),
          py::arg("tau")=0.1f, py::arg("sigma_s")=0.05f,
          py::arg("sigma_n")=0.15f, py::arg("epsilon")=1e-4f);
}
