/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#include "../stub.h"
#include "../helper_macros.h"
#include "../ort_mmcv_utils.h"

void StubKernel::Compute(OrtKernelContext* context)
{
    APP_PRINTF("Do Compute ...\n");

    // Get input data from ctx, (1 input)
    auto input_features = ort_.KernelContext_GetInput(context, 0);
    auto input_data = reinterpret_cast<const float*>(ort_.GetTensorData<float>(input_features));

    // Setup output, (2 output); here we assume out shape eq input shape
    OrtTensorDimensions out_dimensions(ort_, input_features);
    auto output = ort_.KernelContext_GetOutput(context, 0, out_dimensions.data(), out_dimensions.size());
    auto output_data = ort_.GetTensorMutableData<float>(output);

    auto indices = ort_.KernelContext_GetOutput(context, 1, out_dimensions.data(), out_dimensions.size());
    auto indices_data = ort_.GetTensorMutableData<int64_t>(indices);

    // Do forward
    const int64_t ndims = out_dimensions.size();
    int64_t output_size = out_dimensions.data()[0];
    for (int64_t i = 1; i < ndims; ++i) {
        output_size *= out_dimensions.data()[i];
    }

    float tmp;
    for (int64_t i = 0; i < output_size; ++i) {
        tmp = input_data[i] * scale_ + offset_;
        output_data[i] = tmp;
        indices_data[i] = tmp > 1.f ? 1 : 0;
    }
    APP_PRINTF("Do Compute Done\n");
}
