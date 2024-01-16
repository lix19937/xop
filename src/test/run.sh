
LIB_PATH=../../build
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIB_PATH}

python3 run.py --so ${LIB_PATH}/libxop.so --onnx data/stub.onnx  --shapes  input:1x3x64x64:float32

# python3 run.py --so ${LIB_PATH}/libxop.so --onnx data/gridsampler.onnx  \
# --shapes  input:6x256x36x60:float32,grid:6x900x1x2:float32


# python3 run.py --so ${LIB_PATH}/libxop.so --onnx data/dcnv2.onnx \
#  --shapes  data:6x256x20x50:float32,offset_1:6x18x20x50:mask_1:6x9x20x50:float32,filter_1:256x256x3x3:float32


# python3 run.py --so ${LIB_PATH}/libxop.so --onnx data/sv_tf_decoder_nc12.onnx \
#  --shapes  value_0:12x256x72x184:float32,value_1:12x256x36x92:float32,value_2:12x256x18x46:float32,lidar2img:12x4x4:float32

python3 run.py --so ${LIB_PATH}/libxop.so --onnx data/sv_tf_decoder_nc6.onnx \
 --shapes  value_0:6x256x72x184:float32,value_1:6x256x36x92:float32,value_2:6x256x18x46:float32,lidar2img:6x4x4:float32