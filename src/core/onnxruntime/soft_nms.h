/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#ifndef ONNXRUNTIME_SOFT_NMS_H
#define ONNXRUNTIME_SOFT_NMS_H

#include <onnxruntime_cxx_api.h>
#include "onnxruntime_name_domain_ep.h"

struct SoftNmsKernel {
    SoftNmsKernel(OrtApi api, const OrtKernelInfo* info): api_(api), ort_(api_), info_(info)
    {
        iou_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "iou_threshold");
        sigma_ = ort_.KernelInfoGetAttribute<float>(info, "sigma");
        min_score_ = ort_.KernelInfoGetAttribute<float>(info, "min_score");
        method_ = ort_.KernelInfoGetAttribute<int64_t>(info, "method");
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
    float sigma_;
    float min_score_;
    int64_t method_;
    int64_t offset_;
};

struct SoftNmsCustomOp: Ort::CustomOpBase<SoftNmsCustomOp, SoftNmsKernel> {
    void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const
    {
        return new SoftNmsKernel(api, info);
    }

    const char* GetName() const
    {
        return "SoftNonMaxSuppression";
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
        return 2;
    }
    ONNXTensorElementDataType GetOutputType(size_t index) const
    {
        if (index == 1) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    const char* GetExecutionProviderType() const
    {
        return ort_static_val::kCPU;
    }
};
#endif  // ONNXRUNTIME_SOFT_NMS_H
