/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#include "svtransformerdecoder.h"

namespace th = torch;

namespace torch_ext {

template class SvTransformerDecoderFunc<float>;
template class SvTransformerDecoderFunc<half>;

SvTransformerDecoderClass::SvTransformerDecoderClass(
    std::vector<th::Tensor> w, int64_t max_batch, int64_t img_size,
    int64_t patch_size, int64_t in_chans, int64_t embed_dim, int64_t num_heads,
    int64_t inter_size, int64_t layer_num, int64_t with_cls_token)
    : st_(w[0].scalar_type()), weights_(w) {
  for (int i = 0; i < weights_.size(); i++) {
    CHECK_INPUT(weights_[i], st_);
  }

  switch (st_) {
    case at::ScalarType::Float:
      printf("use ScalarType::Float\n");

      vit_func_ = new SvTransformerDecoderFunc<float>(
          max_batch, img_size, patch_size, in_chans, embed_dim, num_heads,
          inter_size, layer_num, 1.0f, with_cls_token, weights_);
      break;
    case at::ScalarType::Half:
      printf("use ScalarType::Half\n");

      vit_func_ = new SvTransformerDecoderFunc<half>(
          max_batch, img_size, patch_size, in_chans, embed_dim, num_heads,
          inter_size, layer_num, 1.0f, with_cls_token, weights_);
      break;
    default:
      throw std::runtime_error("Wrong th::Tensor type.");
  }
  info_int_ = torch::empty({9}, torch::dtype(torch::kInt64));
  info_int_[0] = max_batch;
  info_int_[1] = img_size;
  info_int_[2] = patch_size;
  info_int_[3] = in_chans;
  info_int_[4] = embed_dim;
  info_int_[5] = num_heads;
  info_int_[6] = inter_size;
  info_int_[7] = layer_num;
  info_int_[8] = with_cls_token;
}

std::vector<th::Tensor> SvTransformerDecoderClass::get_pickle_info() const {
  std::vector<th::Tensor> tmp(weights_);
  tmp.push_back(info_int_);
  return tmp;
}

SvTransformerDecoderClass::~SvTransformerDecoderClass() { delete vit_func_; }

th::Tensor SvTransformerDecoderClass::forward(th::Tensor input) {
  CHECK_INPUT(input, st_);
  int batch_size = input.size(0);
  auto output = torch::empty(
      {batch_size, output_seq_len_, output_emb_dim_},
      torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
  vit_func_->forward(batch_size, input, output);
  return output;
}

}  // namespace torch_ext

static auto visionTransformerTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::SvTransformerDecoderClass>(
        "SvTransformerDecoderClass")
#else
    torch::jit::class_<torch_ext::SvTransformerDecoderClass>("SvTransformerDecoder",
                                                          "Class")
#endif
        .def(torch::jit::init<std::vector<th::Tensor>, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t>())
        .def("forward", &torch_ext::SvTransformerDecoderClass::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::SvTransformerDecoderClass>&
                   self) -> std::vector<th::Tensor> {
              return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state)
                -> c10::intrusive_ptr<torch_ext::SvTransformerDecoderClass> {
              int state_size = state.size();
              std::vector<th::Tensor>::const_iterator first = state.begin();
              std::vector<th::Tensor>::const_iterator last =
                  state.begin() + (state_size - 1);
              std::vector<th::Tensor> weights(first, last);
              int idx = state.size() - 1;
              int i = 0;
              int64_t max_batch = state[idx][i++].item().to<int>();
              int64_t img_size = state[idx][i++].item().to<int>();
              int64_t patch_size = state[idx][i++].item().to<int>();
              int64_t in_chans = state[idx][i++].item().to<int>();
              int64_t embed_dim = state[idx][i++].item().to<int>();
              int64_t num_heads = state[idx][i++].item().to<int>();
              int64_t inter_size = state[idx][i++].item().to<int>();
              int64_t layer_num = state[idx][i++].item().to<int>();
              int64_t with_cls_token = state[idx][i++].item().to<int>();
              return c10::make_intrusive<torch_ext::SvTransformerDecoderClass>(
                  weights, max_batch, img_size, patch_size, in_chans, embed_dim,
                  num_heads, inter_size, layer_num, with_cls_token);
            });
