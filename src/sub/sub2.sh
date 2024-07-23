#!/bin/sh
#BSUB -q gpuh100
### -- specify that the cores must be on the same host --
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -o single_node_%J.out
#BSUB -e single_node_%J.err

module load cuda/12.2.2
module load gcc/12.3.0-binutils-2.40

# run the different tests
# echo "n,k,num_groups,num_chunks,n_per_group,chunks_per_group,num_threads,marked_chunk,markedState,marked_max_idx,marked_max_val,time\n"

./grover_cuda_v1_2_gpu 4 1 1 3

rm grover_cuda_v1_2_gpu_report.nsys-rep -f
nsys profile -o grover_cuda_v1_2_gpu_report ./grover_cuda_v1_2_gpu 4 1 1 3