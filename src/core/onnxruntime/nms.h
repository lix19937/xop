/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#ifndef ONNXRUNTIME_NMS_H
#define ONNXRUNTIME_NMS_H

#include "onnxruntime_name_domain_ep.h"
#include <onnxruntime_cxx_api.h>

struct NmsKernel {
    NmsKernel(OrtApi api, const OrtKernelInfo* info): api_(api), ort_(api_), info_(info)
    {
        iou_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "iou_threshold");
        offset_ = ort_.KernelInfoGetAttribute<int64_t>(info, "offset");

        // create allocator if need
        allocator_ = Ort::AllocatorWithDefaultOptions();
    }

    void Compute(OrtKernelContext* context);

protected:
    OrtApi api_;
    Ort::CustomOpApi ort_;
    const OrtKernelInfo* info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    float iou_threshold_;
    int64_t offset_;
};

struct NmsCustomOp: Ort::CustomOpBase<NmsCustomOp, NmsKernel> {
    void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const
    {
        return new NmsKernel(api, info);
    }

    const char* GetName() const
    {
        return "NonMaxSuppression";
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
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }

    const char* GetExecutionProviderType() const
    {
        return ort_static_val::kCPU;
    }
};

#endif
