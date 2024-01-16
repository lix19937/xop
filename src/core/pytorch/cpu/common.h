
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"

#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st) \
  CHECK_TH_CUDA(x);        \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)
#define PRINT_TENSOR(x) std::cout << #x << ":\n" << x << std::endl
#define PRINT_TENSOR_SIZE(x) std::cout << "size of " << #x << ": " << x.sizes() << std::endl

namespace torch_ext {

template <typename T>
inline T* get_ptr(torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

} // namespace torch_ext
