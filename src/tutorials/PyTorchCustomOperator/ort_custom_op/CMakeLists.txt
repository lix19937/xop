cmake_minimum_required(VERSION 3.10)
project (customop)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

set(TEST_SOURCE custom_op_test.cc)
set(HEADER custom_op.h)
set(SOURCE custom_op.h)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories(/home/igs/workspace/cc-dev/xop/src/tutorials/PyTorchCustomOperator/eigen-340)
include_directories(/home/igs/workspace/cc-dev/xop/third_party/onnxruntime-linux-x64-1.8.1/include)

#Include path to header files for Custom Op Test
include_directories(/home/igs/workspace/cc-dev/xop/src/tutorials/PyTorchCustomOperator/ort_custom_op)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/igs/workspace/cc-dev/xop/third_party/onnxruntime-linux-x64-1.8.1/lib)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
