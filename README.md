# LEO

LEO is a multi-GPU system designed to accelerate full-graph GNN training. 
LEO adopts a hybrid parallelism strategy to reduce inter-GPU communication. 
Furthermore, LEO builds the GNN-tailored GPU kernels to accelerate inference in GNN models.
LEO doesn't rely on any intelligent graph partitioning scheme and the support of high-speed interconnects.
## Hardware Requirements
In this artifact, LEO is evaluated on three platforms:
* Platform A
  - CPU: 2 × AMD EPYC 7742 Processor
  - GPU: 8 × NVIDIA A100 (40GB) GPUs
* Platform B
  - CPU: 2 × Intel Xeon Gold 6258R CPUs
  - GPU: 3 × NVIDIA A100 (80GB) GPUs
* Platform C
  - CPU: 2 × Intel Xeon Gold 6252R CPUs
  - GPU: 4 × NVIDIA V100 (32GB) GPUs
## 1. Setting up the Software Environment
### 1.1 Download datasets.
```
wget https://proj-dat.s3.us-west-1.amazonaws.com/dataset.tar.gz && tar -zxvf dataset.tar.gz && rm dataset.tar.gz
```
### 1.2 Download libraries. 
Download libraries ([CUDA toolkit 11.3](https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run), 
[NVSHMEM 2.0.3-0](https://developer.download.nvidia.com/compute/redist/nvshmem/2.0.3/source/nvshmem_src_2.0.3-0.txz), [Open MPI 4.1.1](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz))
### 1.3 Prepare graph preprocessing tools.
Download and compile the softwares ([Rabbit Order](https://github.com/araij/rabbit_order.git), [Gorder](https://github.com/datourat/Gorder.git))
### 1.4 Compile implementation.
```
mkdir build && cd build && cmake .. && cd ..
./0_leo_build.sh
```
