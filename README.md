# quantum_cuda

## TODO List:

- Grover's algorihm with solution checking and random oracle --> encode the solution for arbitrary N = 2^n



# Commands
```
#hpcintrogpush
h100sh

ssh gracy

#module load /appl9/nvhpc/2023_2311/modulefiles/nvhpc-nompi/23.11
module load cuda/12.2.2
module load gcc/13.1.0-binutils-2.40
```

# Performance

- omp:
- cuda baseline:
- cuda v1:
- cuda v2:
- cuda v3: similar performance as 2. uses warp shuffles
- cuda v4: same as v2 but new_state is gone now --> much faster without all the updates
- cuda v5:
- cuda v6: 2 GPUs
- cuda v7: pre-compute the old and new indices
<!-- - cuda v2: -->


# Experiments
- speedup
- profiling: memory, cache
- 2 GPUs --> should be 2x fast
- check with theoretical values
- Gracy: how fast?
- new version of algorithm with reshuffling and then coalesced access
- compute diagonilisation for a few qubits and compare memory footprint
--> will increase exponentially
- relate memory access to the algorithm and the "jumps" performed with the indices

# ideas
- define number of blocks as arg
- pre-compute combined gates in the correct order and remove some of the steps in the loop
- compute all old and new indecis up front and use them as shared arrays for the computations
- try to parallelise the diffusion operator as well
- make animation about how Grover's works --> barchart fx or circle
- switch between state / new_state
- make v2 without shared memory --> expected to be slower since we can only let 2 threads cooperate
- check if reduction (v4) gives improvement
- omp version with reduction --> compare with cuda versions
- write update / zeroOut code on 2 GPUs
- can I by using the shared memory get rid of the new_state? --> probably yes! makes it much faster as well!
- do computation on 2 GPUs
    - 1) 2 vals from gate per GPU --> sum up on 1 GPU
    - 2) half of array per GPU, gates are duplicated --> how would I need to compute the indices? --> I could have a peer connection and share the values like a shared_mem on 2 GPUs within the same node and different nodes.
- async computation = possible? --> would it be faster to reshuffle the indices to load the data in different order so that we get better memory access? is there a faster version/algorithm than what I currently have?



- could use async copy and index shuffling for the first computation --> probably not very effective though.... --> maybe we can show that with more qubits this might change by timing compute and memory operations and showing trends