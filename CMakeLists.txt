cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
projeCt(GNNPro LANGUAGES CXX CUDA)

####################################################################################
set($ENV{PATH} "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/openmpi-4.1.1/bin/:$ENV{PATH}")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/")
set(MPI_HOME "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/openmpi-4.1.1/")
set(OPENBLAS_HOME "/usr/local/OpenBLAS/")
set(CUDA_HOME "/usr/local/cuda/")
set(CUDNN_HOME "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cudnn-v8.2")
#set(NVSHMEM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/nvshmem_src_2.11.0-5/build/")
set(NVSHMEM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/nvshmem_src_2.0.3-0/build/")
#message( STATUS "NVSHMEM_HOME iS ${NVSHMEM_HOME}")

####################################################################################
#set(SM_ARCH "sm_80")
#set($ENV{NVCC_GENCODE} ${SM_ARCH}
set($ENV{NVSHMEM_USE_GDRCOPY} "0")

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COMMON_FLAGS "-O0 -g -ggdb ")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(COMMON_FLAGS "-O3 ")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -Wall  -std=c++14 -fopenmp -march=native " )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_FLAGS} --compiler-options '-fopenmp' -Xcompiler -pthread -rdc=true -ccbin g++")
    # -gencode=arch=compute_70, code=sm_70 -gencode=arch=compute_80,code=sm_80"
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
meSSage(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

####################################################################################
link_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${NVSHMEM_HOME}/lib
    ${OPENBLAS_HOME}/lib
    ${CUDNN_HOME}/lib64
    ${MPI_HOME}/lib
    /usr/local/lib
)

####################################################################################
# set(GNNPro_SRC_FILES
#     "main.cpp"
#     "include/graph.cpp"
#     "include/run_config.cpp"
# )


file(GLOB_RECURSE GNNPro_SRC_FILES
    "main.cu"
    ./include/*.cpp
    ./include/cuda/*.cu
)

add_executable(leo ${GNNPro_SRC_FILES} )

target_include_directories(leo
    PRIVATE
    ${NVSHMEM_HOME}/include
    ${OPENBLAS_HOME}/include
    ${CUDA_HOME}/include
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

set_target_properties(leo 
    PROPERTIES 
    CUDA_ARCHITECTURES "70"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(leo
    nvshmem
    cuda
    mpi_cxx
    mpi
    openblas
    cublas
    cudnn
    gomp
    curand
    dl
)

file(GLOB_RECURSE UVM_GCN_SRC_FILES
    "uvm_gcn.cu"
    ./include/*.cpp
    ./include/cuda/*.cu    
)

add_executable(uvm_gcn ${UVM_GCN_SRC_FILES})

target_include_directories(uvm_gcn
    PRIVATE
    ${NVSHMEM_HOME}/include
    ${OPENBLAS_HOME}/include
    ${CUDA_HOME}/include
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

set_target_properties(uvm_gcn 
    PROPERTIES 
    CUDA_ARCHITECTURES "70"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(uvm_gcn
    nvshmem
    cuda
    mpi_cxx
    mpi
    openblas
    cublas
    cudnn
    gomp
    curand
    dl
)

file(GLOB_RECURSE UVM_GIN_SRC_FILES
    "uvm_gin.cu"
    ./include/*.cpp
    ./include/cuda/*.cu    
)

add_executable(uvm_gin ${UVM_GIN_SRC_FILES})

target_include_directories(uvm_gin
    PRIVATE
    ${NVSHMEM_HOME}/include
    ${OPENBLAS_HOME}/include
    ${CUDA_HOME}/include
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

set_target_properties(uvm_gin 
    PROPERTIES 
    CUDA_ARCHITECTURES "70"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(uvm_gin
    nvshmem
    cuda
    mpi_cxx
    mpi
    openblas
    cublas
    cudnn
    gomp
    curand
    dl
)