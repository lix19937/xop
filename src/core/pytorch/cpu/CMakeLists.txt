# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC 
#  * @Author: lijinwen 
#  * @Date: 2021-08-29 10:14:11 
#  * @Last Modified by: lijinwen 
#  * @Last Modified time: 2021-09-09 14:34:02
#  **************************************************************/

if (NOT BUILD_PYTORCH_OP)
  return()
endif()

if (${RUN_PLATFORM} STREQUAL "aarch64")
  return()
endif()
