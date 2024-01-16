/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#include "../onnxruntime_register.h"
#include "../corner_pool.h"
#include "../deform_conv.h"
#include "../grid_sample.h"
#include "../modulated_deform_conv.h"
#include "../nms.h"
#include "../ort_mmcv_utils.h"
#include "../reduce_ops.h"
#include "../roi_align.h"
#include "../roi_align_rotated.h"
#include "../rotated_feature_align.h"
#include "../soft_nms.h"
#include "../stub.h"
#include "../svtransformer.h"

/// map to ModelProto, model properties imports
static const char* c_MMCVOpDomain = "ai.onnx.contrib";  // user define, do not use office domain

#undef INSTANTIATE_OP
#define INSTANTIATE_OP(type) type c_##type

#undef INSTANTIATE_NAME
#define INSTANTIATE_NAME(type) c_##type

#undef REGISTER_NODE_DOMAIN
#define REGISTER_NODE_DOMAIN(CT)                                                                                       \
    do {                                                                                                               \
        if (auto status = ortApi->CustomOpDomain_Add(domain, &CT)) {                                                   \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

INSTANTIATE_OP(SoftNmsCustomOp);
INSTANTIATE_OP(NmsCustomOp);
INSTANTIATE_OP(MMCVRoiAlignCustomOp);
INSTANTIATE_OP(MMCVRoIAlignRotatedCustomOp);
INSTANTIATE_OP(MMCVRotatedFeatureAlignCustomOp);
INSTANTIATE_OP(GridSampleCustomOp);
INSTANTIATE_OP(MMCVCumMaxCustomOp);
INSTANTIATE_OP(MMCVCumMinCustomOp);
INSTANTIATE_OP(MMCVCornerPoolCustomOp);
INSTANTIATE_OP(MMCVModulatedDeformConvCustomOp);
INSTANTIATE_OP(MMCVDeformConvCustomOp);
INSTANTIATE_OP(StubCustomOp);
INSTANTIATE_OP(SvTransformerCustomOp);

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
{
    /// map to NodeProto
    OrtCustomOpDomain* domain = nullptr;
    const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

    if (auto status = ortApi->CreateCustomOpDomain(c_MMCVOpDomain, &domain)) {
        return status;
    }

    ////////////////////////
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(SoftNmsCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(NmsCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVRoiAlignCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVRoIAlignRotatedCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVRotatedFeatureAlignCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(GridSampleCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVCumMaxCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVCumMinCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVCornerPoolCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVModulatedDeformConvCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(MMCVDeformConvCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(StubCustomOp));
    REGISTER_NODE_DOMAIN(INSTANTIATE_NAME(SvTransformerCustomOp));

    ///////////////////////////
    return ortApi->AddCustomOpDomain(options, domain);
}
