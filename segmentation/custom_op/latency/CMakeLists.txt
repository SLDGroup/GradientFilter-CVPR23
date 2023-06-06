cmake_minimum_required(VERSION 2.8.12)

project("ops")
# set(CMAKE_FIND_DEBUG_MODE TRUE)
# set(CUDA_INCLUDE_DIR /usr/local/cuda/include)
# set(CUDA_TOOLKIT_PATH /usr/local/cuda)


find_package(CUDA 10.2 REQUIRED)
find_package(DNNL CONFIG REQUIRED)

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()
# print_all_variables()


add_executable(dnnl_conv dnnl_conv.cpp)
target_link_libraries(dnnl_conv DNNL::dnnl)

message(STATUS ${CUDA_INCLUDE_DIRS})
message(STATUS ${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(cudnn_conv cudnn_conv.cpp)
target_link_libraries(cudnn_conv ${CUDA_LIBRARIES} cudnn)

add_executable(cudnn_prof cudnn_prof.cpp)
target_link_libraries(cudnn_prof ${CUDA_LIBRARIES} cudnn)

add_executable(dnnl_prof dnnl_prof.cpp)
target_link_libraries(dnnl_prof DNNL::dnnl)

add_executable(jetson_pwr_logger jetson_pwr_logger.cpp)
