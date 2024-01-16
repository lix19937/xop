/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#include "../svtransformer.h"
#include "../helper_macros.h"
#include "../ort_mmcv_utils.h"

void SvTransformerKernel::Compute(OrtKernelContext* context)
{
    APP_PRINTF("Do Compute ...\n");
    // Get input data from ctx, (4 input)
    // const auto input_features_0 = ort_.KernelContext_GetInput(context, 0);
    // const float* input_data_0 = reinterpret_cast<const float*>(ort_.GetTensorData<float>(input_features_0));
    //

    // Setup output_0, (2 output_0)  here we assume out shape eq input shape
    std::vector<int64_t> out_dimensions_0{max_batch_, seq_len_, num_reg_points_};
    auto output_0 = ort_.KernelContext_GetOutput(context, 0, out_dimensions_0.data(), out_dimensions_0.size());
    auto output_data_0 = ort_.GetTensorMutableData<float>(output_0);

    std::vector<int64_t> out_dimensions_1{max_batch_, seq_len_, num_classes_};
    auto output_1 = ort_.KernelContext_GetOutput(context, 1, out_dimensions_1.data(), out_dimensions_1.size());
    auto output_data_1 = ort_.GetTensorMutableData<float>(output_1);

    // Do forward
    int64_t ndims = out_dimensions_0.size();
    int64_t output_size = out_dimensions_0.data()[0];
    for (int64_t i = 1; i < ndims; ++i) {
        output_size *= out_dimensions_0.data()[i];
    }

    for (int64_t i = 0; i < output_size; ++i) {
        output_data_0[i] = .5f;
    }

    // Another out
    ndims = out_dimensions_1.size();
    output_size = out_dimensions_1.data()[0];
    for (int64_t i = 1; i < ndims; ++i) {
        output_size *= out_dimensions_1.data()[i];
    }
    for (int64_t i = 0; i < output_size; ++i) {
        output_data_1[i] = .1f;
    }

    APP_PRINTF("Do Compute Done\n");
}
