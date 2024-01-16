# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-01-10 13:49:25
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-01-10 13:53:09
#  **************************************************************/

set(ORT_SOURCES)
if (BUILD_ORT_OP)
  add_subdirectory(onnxruntime/cpu)
endif()

set(TORCH_SOURCES)
if (BUILD_TORCH_OP)
  add_subdirectory(pytorch/cpu)
endif()

set(ORT_INCLUDE  ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.8.1/include)
set(ORT_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.8.1/lib)

## we just wait pytorch v2.0 impl
# set(TORCH_INCLUDE  ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.8.1/include)
# set(TORCH_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.8.1/lib)

add_library(${LIB_NAME} SHARED
  ${ORT_SOURCES}
  ${TORCH_SOURCES}
)
set_target_properties(${LIB_NAME} PROPERTIES VERSION ${LIB_VERSION} SOVERSION ${LIB_SOVERSION})

target_link_directories(${LIB_NAME} PUBLIC
  ${ORT_LIB_PATH}
) 

target_link_libraries(
  ${LIB_NAME}  
  onnxruntime
)

target_include_directories(
  ${LIB_NAME} PUBLIC
  ${ORT_INCLUDE}
)

install(TARGETS ${LIB_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
  