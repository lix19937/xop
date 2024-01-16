/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#ifndef ONNXRUNTIME_GRIDSAMPLE_H
#define ONNXRUNTIME_GRIDSAMPLE_H

#include "onnxruntime_name_domain_ep.h"
#include <onnxruntime_cxx_api.h>

struct GridSampleKernel {
    GridSampleKernel(OrtApi api, const OrtKernelInfo* info): api_(api), ort_(api_), info_(info)
    {
        align_corners_ = ort_.KernelInfoGetAttribute<int64_t>(info, "align_corners");
        interpolation_mode_ = ort_.KernelInfoGetAttribute<int64_t>(info, "interpolation_mode");
        padding_mode_ = ort_.KernelInfoGetAttribute<int64_t>(info, "padding_mode");

        // create allocator if need
        allocator_ = Ort::AllocatorWithDefaultOptions();
    }

    void Compute(OrtKernelContext* context);

protected:
    OrtApi api_;
    Ort::CustomOpApi ort_;
    const OrtKernelInfo* info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // attri
    int64_t align_corners_;
    int64_t interpolation_mode_;
    int64_t padding_mode_;
};

struct GridSampleCustomOp: Ort::CustomOpBase<GridSampleCustomOp, GridSampleKernel> {
    void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const
    {
        return new GridSampleKernel(api, info);
    }

    const char* GetName() const
    {
        return "grid_sampler";
    }

    size_t GetInputTypeCount() const
    {
        return 2;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    const char* GetExecutionProviderType() const
    {
        return ort_static_val::kCPU;
    }
};
#endif
