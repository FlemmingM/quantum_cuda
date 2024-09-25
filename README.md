# Quantum Cuda Project

Grover's algorihm which finds a solution for arbitrary N = 2^n

For more information about the project, how this quantum algorithm works, and how it can be simulated on CPUs and GPUs, please have a look at the paper `quantum_cuda_paper.pdf`



# Commands
```
# Modules needed
module load cuda/12.2.2
module load gcc/13.1.0-binutils-2.40
```

# Algorithms

- omp
- cuda baseline
- cuda v1
- cuda v2
- cuda v3 (fastest)
- cuda v1 2 GPU
- cuda v2 2 GPU
- cuda v3 2 GPU (fastest)
- cuda stream (only to show how to test for correct solution)


# Experiments
- Speedup
- Grid search
- profiling: Nsight Systems and Compute
- Multi GPU speedup
- Gracy performance (profiling not available)

# Compile
```
cd src
make clean && make
```
Then you can run the different programs. Currently 2 GPUs are supported but with small modifications it can scale to more GPUs.

# Examples

```
OMP version

// ./grover_omp n markedState
// run for 10 qubits with marked state 1
./grover_omp 10 1



CUDA Base version

// ./grover_cuda_baseline n markedState block_size
// run for 10 qubits with marked state 1 and block size 1024
./grover_omp 10 1 1024


CUDA v1, 2, 3

As an example we can run the programmes of v1, 2, 3 with the same input arguments

// ./grover_cuda_v1 n markedState n_chunks_per_group n_qubits_per_group

// run for 10 qubits with 1 chunk per group and 1 group
// 2^10 fits into 1 block of size 1024
./grover_cuda_v1 10 1 1 10

// run for 10 qubits with 2 chunks per group and 1 group
// 2^10 fits into 2 blocks of size 512
./grover_cuda_v1 10 1 2 10

// run for 10 qubits with 2 chunks per group and 2 groups
// 2^9 fits into 2 blocks of size 256
./grover_cuda_v1 10 1 2 9

// run for 11 qubits with 1 chunks per group and 2 groups
// 2^10 fits into 1 block of size 1024
./grover_cuda_v1 11 1 1 10




CUDA versions with 2 GPUs

These versions are the 2 GPU versions of v1, 2, 3 and divide the work between 2 GPUs running in parallel. Thus, we need at least 2 groups. Let's take the examples from the single GPU versions and adapt them for 2 GPUs:

// ./grover_cuda_v1_2_gpu n markedState n_chunks_per_group n_qubits_per_group
// n_qubits_per_group must be 1 lower than n

// run for 10 qubits with 1 chunk per group and 2 groups
// 2^9 fits into 1 block of size 512 on each device
// The 2 groups are divided between the 2 devices and run in parallel
./grover_cuda_v1_2_gpu 10 1 1 9

// run for 10 qubits with 2 chunks per group and 2 groups
// 2^10 fits into 2 blocks of size 256 on each device
// The 2 groups are divided between the 2 devices and run in parallel
./grover_cuda_v1_2_gpu 10 1 2 9

// run for 10 qubits with 2 chunks per group and 4 groups
// 2^8 fits into 2 blocks of size 128 on each device
// The 4 groups are divided between the 2 devices and 2 groups are run in parallel at a time
./grover_cuda_v1_2_gpu 10 1 2 8

// run for 11 qubits with 1 chunks per group and 2 groups
// 2^10 fits into 1 blocks of size 1024 on each device
// The 2 groups are divided between the 2 devices and run in parallel
./grover_cuda_v1_2_gpu 11 1 1 10
```