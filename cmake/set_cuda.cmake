# /**************************************************************
#  * @Author: lijinwen 
#  * @Date: 2021-08-29 10:14:11  
#  * @Last Modified by: lijinwen 
#  * @Last Modified time: 2021-09-09 13:50:15 
#  **************************************************************/

## Fused by multi modules
macro(set_cuda)
  if (USE_DOCKER)
    if (${RUN_PLATFORM} STREQUAL "aarch64")
      set(AARCH_SDK_ROOT        /opt/ros/orin_env)
      set(CUDA_CUDART_LIBRARY   ${AARCH_SDK_ROOT}/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so)
      set(CUDA_TOOLKIT_INCLUDE  ${AARCH_SDK_ROOT}/usr/local/cuda-11.4/targets/aarch64-linux/include)
      set(CUDA_TOOLKIT_ROOT_DIR ${AARCH_SDK_ROOT}/usr/lib/aarch64-linux-gnu)
    else()
      find_package(CUDA REQUIRED) 
    endif()
  else()
    if (${RUN_PLATFORM} STREQUAL "aarch64")
      set(AARCH_SDK_ROOT        ${PROJECT_SOURCE_DIR}/third_party/nvidia_sdk)
      set(CUDA_CUDART_LIBRARY   ${AARCH_SDK_ROOT}/drive-linux/targetfs/usr/local/cuda-11.4/lib64/libcudart.so)
      set(CUDA_TOOLKIT_INCLUDE  ${AARCH_SDK_ROOT}/drive-linux/targetfs/usr/local/cuda-11.4/include)
      set(CUDA_TOOLKIT_ROOT_DIR ${AARCH_SDK_ROOT}/drive-linux/targetfs/usr/local/cuda-11.4)
    else()
      find_package(CUDA REQUIRED) 
    endif()
  endif()

endmacro()
