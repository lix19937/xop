# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC 
#  * @Author: lijinwen 
#  * @Date: 2021-08-29 10:14:11 
#  * @Last Modified by: lijinwen 
#  * @Last Modified time: 2021-09-09 14:34:02
#  **************************************************************/

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(xop 
  LANGUAGES CXX CUDA
  DESCRIPTION "xop, which strives to include custom op of all backend inference engines, such as ONNXRuntime and libtorch."
  HOMEPAGE_URL "https://gitlab.com/xop")

set(CMAKE_VERBOSE_MAKEFILE OFF)
include(cmake/set_ifndef.cmake)
set_ifndef(CMAKE_BUILD_TYPE Release) 
set_ifndef(LIB_NAME         xop) 
set_ifndef(RUN_PLATFORM     x86_64) 
set_ifndef(GPU_ARCHS        86)
set_ifndef(CUDA_VERSION     11.4) 

include(cmake/basic.cmake)
include(cmake/options.cmake)
include(cmake/set_version.cmake)
include(cmake/set_cuda.cmake)    

user_option(USE_DOCKER      "Build project in docker env"    OFF)   
user_option(USE_CUDA_1104   "Build project with cuda-11.4"   ON)
user_option(BUILD_ORT_OP    "Build ort op"                   ON)
user_option(BUILD_TORCH_OP  "Build torch op"                 OFF)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH    ${PROJECT_SOURCE_DIR}/build)

set_cuda()
set_version(${PROJECT_SOURCE_DIR}/VERSION LIB_VERSION LIB_SOVERSION)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})

if (USE_DOCKER) 
  set(CUDA_INCLUDE        /opt/ros/orin_env/usr/local/cuda-11.4/include)
  if (${RUN_PLATFORM} STREQUAL "x86_64")
    set(CUDA_LIB_PATH     /usr/local/cuda-11.4/lib64) 
    set(SYSTEM_LIB_PATH   /usr/lib/x86_64-linux-gnu)
    
  elseif (${RUN_PLATFORM} STREQUAL "aarch64")
    message(FATAL_ERROR "ONLY SUPPORT x86_64")
  endif()  
endif()

message(STATUS "GPU_ARCHS:              ${GPU_ARCHS}")
message(STATUS "LIB_NAME:               ${LIB_NAME}")
message(STATUS "RUN_PLATFORM:           ${RUN_PLATFORM}")
message(STATUS "CMAKE_BUILD_TYPE:       ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "LIB_VERSION:            ${LIB_VERSION}")
message(STATUS "LIB_SOVERSION:          ${LIB_SOVERSION}")
message(STATUS "CMAKE_VERBOSE_MAKEFILE: ${CMAKE_VERBOSE_MAKEFILE}")
message(STATUS "USE_DOCKER:             ${USE_DOCKER}")
message(STATUS "USE_CUDA_1104:          ${USE_CUDA_1104}")

add_subdirectory(src)
