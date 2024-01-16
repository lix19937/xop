/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#ifndef ONNXRUNTIME_SVT_H
#define ONNXRUNTIME_SVT_H

#include "onnxruntime_name_domain_ep.h"
#include <onnxruntime_cxx_api.h>

struct SvTransformerKernel {
public:
    SvTransformerKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info): ort_(ort)
    {
        num_cam_ = ort_.KernelInfoGetAttribute<int64_t>(info, "num_cam");
        num_classes_ = ort_.KernelInfoGetAttribute<int64_t>(info, "num_classes");
        num_reg_points_ = ort_.KernelInfoGetAttribute<int64_t>(info, "num_reg_points");
        seq_len_ = ort_.KernelInfoGetAttribute<int64_t>(info, "seq_len");
        max_seq_len_ = ort_.KernelInfoGetAttribute<int64_t>(info, "max_seq_len");
        max_batch_ = ort_.KernelInfoGetAttribute<int64_t>(info, "max_batch");

        auto img_shape_t = ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "img_shape");
        auto pc_range_t = ort_.KernelInfoGetAttribute<std::vector<float>>(info, "pc_range");

        printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");

        printf("num_cam_:%ld\n", num_cam_);
        printf("num_classes_:%ld\n", num_classes_);
        printf("num_reg_points_:%ld\n", num_reg_points_);
        printf("seq_len_:%ld\n", seq_len_);
        printf("max_seq_len_:%ld\n", max_seq_len_);
        printf("max_batch_:%ld\n", max_batch_);
        printf("img_shape_t num:%d\n", int(img_shape_t.size()));
        for (int i = 0; i < int(img_shape_t.size()); ++i) {
            printf("\t%ld", img_shape_t[i]);
        }
        printf("\n");

        printf("pc_range_t num:%d\n", int(pc_range_t.size()));
        for (int i = 0; i < int(pc_range_t.size()); ++i) {
            pc_range_[i] = pc_range_t[i];
            printf("\t%f", pc_range_[i]);
        }
        printf("\n");

        printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n");
    }

    void Compute(OrtKernelContext* context);

private:
    Ort::CustomOpApi ort_;
    /// if you implement detail logic, need allocator_
    // Ort::AllocatorWithDefaultOptions allocator_;

    int64_t num_cam_;
    int64_t num_classes_;
    int64_t num_reg_points_;
    int64_t seq_len_;
    int64_t max_seq_len_;
    int64_t max_batch_;
    float pc_range_[6];
};

struct SvTransformerCustomOp: Ort::CustomOpBase<SvTransformerCustomOp, SvTransformerKernel> {
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const
    {
        return new SvTransformerKernel(api, info);
    }

    const char* GetName() const
    {
        return "SvTransformerDecoder";
    }

    size_t GetInputTypeCount() const
    {
        return 4;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const
    {
        return 2;
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
