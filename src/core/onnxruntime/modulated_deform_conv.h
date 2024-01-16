/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#ifndef ONNXRUNTIME_MODULATED_DEFORM_CONV_H
#define ONNXRUNTIME_MODULATED_DEFORM_CONV_H

#include "onnxruntime_name_domain_ep.h"
#include <onnxruntime_cxx_api.h>

struct MMCVModulatedDeformConvKernel {
    MMCVModulatedDeformConvKernel(OrtApi api, const OrtKernelInfo* info): api_(api), ort_(api_), info_(info)
    {
        std::vector<int64_t> stride = ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "stride");
        stride_height_ = stride[0];
        stride_width_ = stride[1];
        std::vector<int64_t> padding = ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "padding");
        padding_height_ = padding[0];
        padding_width_ = padding[1];
        std::vector<int64_t> dilation = ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "dilation");
        dilation_height_ = dilation[0];
        dilation_width_ = dilation[1];
        deformable_group_ = ort_.KernelInfoGetAttribute<int64_t>(info, "deform_groups");
        group_ = ort_.KernelInfoGetAttribute<int64_t>(info, "groups");

        // create allocator if need
        allocator_ = Ort::AllocatorWithDefaultOptions();
    }

    void Compute(OrtKernelContext* context);

protected:
    OrtApi api_;
    Ort::CustomOpApi ort_;
    const OrtKernelInfo* info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    int64_t stride_height_;
    int64_t stride_width_;
    int64_t padding_height_;
    int64_t padding_width_;
    int64_t dilation_height_;
    int64_t dilation_width_;
    int64_t deformable_group_;
    int64_t group_;
};

struct MMCVModulatedDeformConvCustomOp:
    Ort::CustomOpBase<MMCVModulatedDeformConvCustomOp, MMCVModulatedDeformConvKernel> {
    void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const
    {
        return new MMCVModulatedDeformConvKernel(api, info);
    }

    const char* GetName() const
    {
        return "MMCVModulatedDeformConv2d";
    }

    size_t GetInputTypeCount() const
    {
        return 5;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const
    {
        // The last input (index == 4) is optional, which is bias
        if (index == 4)
            return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
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
