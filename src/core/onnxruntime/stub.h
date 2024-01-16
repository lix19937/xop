/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#ifndef ONNXRUNTIME_STUB_H
#define ONNXRUNTIME_STUB_H

#include "onnxruntime_name_domain_ep.h"
#include <onnxruntime_cxx_api.h>

struct StubKernel {
public:
    StubKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info): ort_(ort)
    {
        offset_ = ort_.KernelInfoGetAttribute<int64_t>(info, "offset");
        scale_ = ort_.KernelInfoGetAttribute<float>(info, "scale");
    }

    void Compute(OrtKernelContext* context);

private:
    Ort::CustomOpApi ort_;

    int64_t offset_;
    float scale_;
};

struct StubCustomOp: Ort::CustomOpBase<StubCustomOp, StubKernel> {
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const
    {
        return new StubKernel(api, info);
    }

    const char* GetName() const
    {
        return "stub_i1o2fi";
    }

    size_t GetInputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const
    {
        return 2;
    }
    ONNXTensorElementDataType GetOutputType(size_t index) const
    {
        return index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    const char* GetExecutionProviderType() const
    {
        return ort_static_val::kCPU;
    }
};

#endif  // ONNXRUNTIME_STUB_H
