#!/bin/sh
#BSUB -q gpuh100
### -- specify that the cores must be on the same host --
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o single_node_%J.out
#BSUB -e single_node_%J.err

module load cuda/12.2.2
module load gcc/12.3.0-binutils-2.40

# run the different tests
echo "n,k,num_groups,num_chunks,n_per_group,chunks_per_group,num_threads,marked_chunk,markedState,marked_max_idx,marked_max_val,time\n"
./grover_cuda_stream 20 3 1024 20
./grover_cuda_stream 20 3 512 19
./grover_cuda_stream 20 3 256 18
./grover_cuda_stream 20 3 128 17
./grover_cuda_stream 20 3 64 16
./grover_cuda_stream 20 3 32 15
./grover_cuda_stream 20 3 16 14
./grover_cuda_stream 20 3 8 13
./grover_cuda_stream 20 3 4 12
./grover_cuda_stream 20 3 2 11
./grover_cuda_stream 20 3 1 10
./grover_cuda_stream 20 3 1 9
./grover_cuda_stream 20 3 1 8
./grover_cuda_stream 20 3 1 7
./grover_cuda_stream 20 3 1 6
./grover_cuda_stream 20 3 1 5
./grover_cuda_stream 20 3 1 4
./grover_cuda_stream 20 3 1 3
./grover_cuda_stream 20 3 1 2
./grover_cuda_stream 20 3 2048 20
./grover_cuda_stream 20 3 1024 19
./grover_cuda_stream 20 3 512 18
./grover_cuda_stream 20 3 256 17
./grover_cuda_stream 20 3 128 16
./grover_cuda_stream 20 3 64 15
./grover_cuda_stream 20 3 32 14
./grover_cuda_stream 20 3 16 13
./grover_cuda_stream 20 3 8 12
./grover_cuda_stream 20 3 4 11
./grover_cuda_stream 20 3 2 10
./grover_cuda_stream 20 3 1 9
./grover_cuda_stream 20 3 1 8
./grover_cuda_stream 20 3 1 7
./grover_cuda_stream 20 3 1 6
./grover_cuda_stream 20 3 1 5
./grover_cuda_stream 20 3 1 4
./grover_cuda_stream 20 3 1 3
./grover_cuda_stream 20 3 1 2
./grover_cuda_stream 20 3 4096 20
./grover_cuda_stream 20 3 2048 19
./grover_cuda_stream 20 3 1024 18
./grover_cuda_stream 20 3 512 17
./grover_cuda_stream 20 3 256 16
./grover_cuda_stream 20 3 128 15
./grover_cuda_stream 20 3 64 14
./grover_cuda_stream 20 3 32 13
./grover_cuda_stream 20 3 16 12
./grover_cuda_stream 20 3 8 11
./grover_cuda_stream 20 3 4 10
./grover_cuda_stream 20 3 2 9
./grover_cuda_stream 20 3 1 8
./grover_cuda_stream 20 3 1 7
./grover_cuda_stream 20 3 1 6
./grover_cuda_stream 20 3 1 5
./grover_cuda_stream 20 3 1 4
./grover_cuda_stream 20 3 1 3
./grover_cuda_stream 20 3 1 2
./grover_cuda_stream 20 3 8192 20
./grover_cuda_stream 20 3 4096 19
./grover_cuda_stream 20 3 2048 18
./grover_cuda_stream 20 3 1024 17
./grover_cuda_stream 20 3 512 16
./grover_cuda_stream 20 3 256 15
./grover_cuda_stream 20 3 128 14
./grover_cuda_stream 20 3 64 13
./grover_cuda_stream 20 3 32 12
./grover_cuda_stream 20 3 16 11
./grover_cuda_stream 20 3 8 10
./grover_cuda_stream 20 3 4 9
./grover_cuda_stream 20 3 2 8
./grover_cuda_stream 20 3 1 7
./grover_cuda_stream 20 3 1 6
./grover_cuda_stream 20 3 1 5
./grover_cuda_stream 20 3 1 4
./grover_cuda_stream 20 3 1 3
./grover_cuda_stream 20 3 1 2