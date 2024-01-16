/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#ifndef ONNXRUNTIME_CORNER_POOL_H
#define ONNXRUNTIME_CORNER_POOL_H

#include <onnxruntime_cxx_api.h>
#include "onnxruntime_name_domain_ep.h"

struct MMCVCornerPoolKernel {
    MMCVCornerPoolKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info): ort_(ort)
    {
        mode_ = ort_.KernelInfoGetAttribute<int64_t>(info, "mode");
    }

    void Compute(OrtKernelContext* context);

private:
    Ort::CustomOpApi ort_;
    int64_t mode_;
};

struct MMCVCornerPoolCustomOp: Ort::CustomOpBase<MMCVCornerPoolCustomOp, MMCVCornerPoolKernel> {
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const
    {
        return new MMCVCornerPoolKernel(api, info);
    }

    const char* GetName() const
    {
        return "MMCVCornerPool";
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
#endif  // ONNXRUNTIME_CORNER_POOL_H
