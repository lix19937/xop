# /**************************************************************
#  * @Author: lijinwen 
#  * @Date: 2021-08-29 10:14:11  
#  * @Last Modified by: lijinwen 
#  * @Last Modified time: 2021-09-09 15:56:32 
#  **************************************************************/

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.7 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.7)
  message(STATUS "C++14 activated.")
  message(STATUS "Build type:${CMAKE_BUILD_TYPE}")
  message(STATUS "Build cxx flags:${CMAKE_CXX_FLAGS}")
  message(STATUS "Debug configuration:${CMAKE_CXX_FLAGS_DEBUG}")
  message(STATUS "Release configuration:${CMAKE_CXX_FLAGS_RELEASE}")
  message(STATUS "Release configuration with debug info:${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  message(STATUS "Minimal release configuration:${CMAKE_CXX_FLAGS_MINSIZEREL}")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 ")
  set(CMAKE_CXX_FLAGS "-ffast-math -Werror -Wall -Wno-deprecated-declarations -Werror=maybe-uninitialized -Werror=return-type -fopenmp ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(LINE_FEED " ")

  set(DEBUG_POSTFIX _debug CACHE STRING "suffix for debug builds")
  message(${LINE_FEED})

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in debug mode ${DEBUG_POSTFIX}")
  endif()

  if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG") 
  endif()

  message(STATUS ${CMAKE_CXX_FLAGS})
  message(${LINE_FEED})
  
else()
  add_definitions (-ggdb -Wall -D_GNU_SOURCE=1 -D__STDC_LIMIT_MACROS=1)
  message(STATUS "C++98 activated.")
endif()
