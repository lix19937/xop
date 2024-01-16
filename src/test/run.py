# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-08-12 16:05:43
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-08-15 09:57:15
#  **************************************************************/

#
#
#  pip3 install onnxruntime==1.8.1     
#  pip3 install loguru                  
#
#

import os
import numpy as np
import onnxruntime as ort
from loguru import logger
import argparse

def main(ort_custom_op_path, onnx_file, name_shapes_type):
  ## load so 
  logger.info('start config ... {}'.format(ort_custom_op_path))
  assert os.path.exists(ort_custom_op_path)
  session_options = ort.SessionOptions()
  session_options.register_custom_ops_library(ort_custom_op_path)

  ## parse input tensor
  name_shapes_type = name_shapes_type.replace(' ', '')
  name_shapes_type = name_shapes_type.split(",")
  
  in_tensors_dict = {}
  for it in name_shapes_type:
    nst = it.split(":")
    logger.info(nst[1])
    shape = nst[1].split('x')
    b = [int(i) for i in shape]
    logger.info(tuple(b))

    ## onnx attri
    t = np.random.uniform(size = tuple(b))
    if nst[2] == 'float32':
      in_tensors_dict[nst[0]] = t.astype(np.float32)
    elif nst[2] == 'float64':
      in_tensors_dict[nst[0]] = t.astype(np.float64)
    elif nst[2] == 'int32':
      in_tensors_dict[nst[0]] =t.astype(np.int32) 
    elif nst[2] == 'int64':
      in_tensors_dict[nst[0]] =t.astype(np.int64)   

  ## infer on ort (RTLD_LAZY)
  logger.info('start ort ...')
  sess = ort.InferenceSession(onnx_file, session_options)
  onnx_results = sess.run(None, in_tensors_dict)
  logger.info("out tensor num:{}".format(len(onnx_results)))

  for it in onnx_results:
    logger.info("out shape:{}".format(it.shape))
  logger.info('ort done')

  ## simplify on onnxsim (RTLD_LAZY)
  # logger.info('start simplify ...')
  # import onnxsim
  # model_simp, check = onnxsim.simplify(onnx_file, custom_lib=ort_custom_op_path)
  # onnx.save(model_simp, "compression_test_6cams_op11_stub.onnx")
  # logger.info('simplify done')


### python3  comp.py  --a 1.txt  --b 2.txt  --p 5
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Small tool for compare')
    parser.add_argument('--so', default='', type=str, required=True, help='ort_custom_op_path')
    parser.add_argument('--onnx', default='', type=str, required=True, help='onnx_file')
    #  dtype float32 float64 int32 int64 bool  
    # "input1:1x3x64x64:float32"
    parser.add_argument('--shapes', default='', type=str, required=True, help='input tensor type and shape')
 
    args = parser.parse_args()
    main(args.so, args.onnx, args.shapes)
    logger.info(args.shapes)
    