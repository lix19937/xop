/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cmath>
#include <mutex>
#include <vector>


static const char* c_OpDomain = "ai.onnx.contrib";

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr =
    std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain,
                                            const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(
      domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct KernelSvTransformerDecoder {
  KernelSvTransformerDecoder(OrtApi api) : api_(api), ort_(api_) {}

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X0 = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_X1 = ort_.KernelContext_GetInput(context, 1);
    const OrtValue* input_X2 = ort_.KernelContext_GetInput(context, 2);
    const OrtValue* input_X3 = ort_.KernelContext_GetInput(context, 3);

    const OrtValue* input_X4 = ort_.KernelContext_GetInput(context, 4);
    const OrtValue* input_X5 = ort_.KernelContext_GetInput(context, 5);

    // const float* X = ort_.GetTensorData<float>(input_X0);

    // OrtTensorDimensions rp_dimensions(ort_, input_X0);
    int32_t num_query = 512;

    // Setup output
    // OrtTensorDimensions dimensions(ort_, input_X);
    std::vector<int64_t> dimensions = {num_query, 1, 256};

    OrtValue* output = ort_.KernelContext_GetOutput(
        context, 0, dimensions.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = (float)(0.5);
    }

    std::vector<int64_t> dimensions1 = {1, num_query, 3};

    OrtValue* output1 = ort_.KernelContext_GetOutput(
        context, 1, dimensions1.data(), dimensions1.size());
    float* out1 = ort_.GetTensorMutableData<float>(output1);

    OrtTensorTypeAndShapeInfo* output_info1 =
        ort_.GetTensorTypeAndShape(output1);
    int64_t size1 = ort_.GetTensorShapeElementCount(output_info1);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info1);

    // Do computation
    for (int64_t i = 0; i < size1; i++) {
      out[i] = (float)(0.5);
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct FeatureSampling
    : Ort::CustomOpBase<FeatureSampling, KernelSvTransformerDecoder> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
    return new KernelSvTransformerDecoder(api);
  };

  const char* GetName() const { return "SvTransformerDecoder"; };

  size_t GetInputTypeCount() const { return 6; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 5) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
  };

  size_t GetOutputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

} c_FeatureSampling;

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                          const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_FeatureSampling)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

/*

g++   ./src/ort/fake-op/svtransformerdecoder.cc  \
-I./third_party/onnxruntime-linux-x64-1.8.1/include \
-L./third_party/onnxruntime-linux-x64-1.8.1/lib -lonnxruntime \
-fPIC -shared -o libsvtransformerdecoder_x86.so

*/
