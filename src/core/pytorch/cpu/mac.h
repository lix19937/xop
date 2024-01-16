/**************************************************************
 * @Copyright: 2021-2022 Copyright SAIC
 * @Author: lijinwen
 * @Date: 2022-08-12 16:05:43
 * @Last Modified by: lijinwen
 * @Last Modified time: 2022-08-15 09:57:15
 **************************************************************/

#include "th_utils.h"

namespace th = torch;

namespace torch_ext {

class IViTFunc {
 public:
  virtual ~IViTFunc() {}
  virtual void forward(int batch_size, th::Tensor& input,
                       th::Tensor& output) = 0;
};

template <typename T>
class SvTransformerDecoderFunc : public IViTFunc {
 public:
  int sm_;
  int max_batch_;
  int img_size_;
  int patch_size_;
  int in_chans_;
  int embed_dim_;
  int num_heads_;
  int head_dim_;
  int inter_size_;
  int layer_num_;
  bool sparse_;
  float q_scaling_;
  bool with_cls_token_;

  SvTransformerDecoderFunc(const int max_batch, const int img_size,
                           const int patch_size, const int in_chans,
                           const int embed_dim, const int num_heads,
                           const int inter_size, const int layer_num,
                           const float q_scaling, const bool with_cls_token,
                           const std::vector<th::Tensor>& w)
      : weights_(w),
        max_batch_(max_batch),
        img_size_(img_size),
        patch_size_(patch_size),
        in_chans_(in_chans),
        embed_dim_(embed_dim),
        num_heads_(num_heads),
        head_dim_(embed_dim / num_heads),
        inter_size_(inter_size),
        layer_num_(layer_num),
        q_scaling_(q_scaling),
        with_cls_token_(with_cls_token) {
    // do some init
  }

  ~SvTransformerDecoderFunc() override {
    // do some delete
  }

  void forward(int batch_size, th::Tensor& input, th::Tensor& output) override {
    // do pipeline
  }

 private:
  std::vector<th::Tensor> weights_;
  cublasLtHandle_t cublaslt_handle_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
};

class SvTransformerDecoderClass : public torch::jit::CustomClassHolder {
 public:
  SvTransformerDecoderClass(std::vector<th::Tensor> w, int64_t max_batch,
                            int64_t img_size, int64_t patch_size,
                            int64_t in_chans, int64_t embed_dim,
                            int64_t num_heads, int64_t inter_size,
                            int64_t layer_num, int64_t with_cls_token);

  ~SvTransformerDecoderClass();

  th::Tensor forward(th::Tensor input);
  std::vector<th::Tensor> get_pickle_info() const;

 private:
  const at::ScalarType st_;
  IViTFunc* vit_func_;
  std::vector<th::Tensor> weights_;
  th::Tensor info_int_;
  int output_seq_len_;
  int output_emb_dim_;
};

}  // namespace torch_ext
