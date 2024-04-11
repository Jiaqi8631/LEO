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
