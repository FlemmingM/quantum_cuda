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


# ideas
- switch between state / new_state
- check if reduction (v4) gives improvement
- omp version with reduction --> compare with cuda versions
- write update / zeroOut code on 2 GPUs
- can I by using the shared memory get rid of the new_state? --> probably yes! makes it much faster as well!
- do computation on 2 GPUs
    - 1) 2 vals from gate per GPU --> sum up on 1 GPU
    - 2) half of array per GPU, gates are duplicated --> how would I need to compute the indices? --> I could have a peer connection and share the values like a shared_mem on 2 GPUs within the same node and different nodes.
- async computation = possible? --> would it be faster to reshuffle the indices to load the data in different order so that we get better memory access? is there a faster version/algorithm than what I currently have?
- could use async copy and index shuffling for the first computation --> probably not very effective though....