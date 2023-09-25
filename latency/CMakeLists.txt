cmake_minimum_required(VERSION 2.8.12)

project("ops")

find_package(CUDA 10.2 REQUIRED)
find_package(DNNL CONFIG REQUIRED)

message(STATUS ${CUDA_INCLUDE_DIRS})
message(STATUS ${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(cudnn_prof cudnn_prof.cpp)
target_link_libraries(cudnn_prof ${CUDA_LIBRARIES} cudnn)

add_executable(dnnl_prof dnnl_prof.cpp)
target_link_libraries(dnnl_prof DNNL::dnnl)
