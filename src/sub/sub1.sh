#!/bin/sh
#BSUB -q gpuh100
### -- specify that the cores must be on the same host --
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o single_node_%J.out
#BSUB -e single_node_%J.err

module load cuda/12.2.2
module load gcc/12.3.0-binutils-2.40

rm warptest_report.nsys-rep -f
nsys profile -o warptest_report ./warptest