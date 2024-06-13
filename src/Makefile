TARGET_OMP	= grover_omp
# TARGET_CUDA1 = grover_cuda1
OBJS	= grover_omp.o utils_omp.o

OPT	= -g -O3
ISA	=
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_90 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

F90C  	= gfortran

CUDA_PATH ?= /appl/cuda/12.2.2
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common

all: $(TARGET_OMP)

$(TARGET_OMP): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lm

.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET_OMP)

# dependencies
grover_omp.o: grover_omp.c utils_omp.h
utils_omp.o: utils_omp.c utils_omp.h
