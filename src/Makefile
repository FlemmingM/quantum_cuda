TARGET_OMP	= grover_omp
TARGET_CUDA1 = grover_cuda_baseline
TARGET_CUDA2 = grover_cuda_opt1
TARGET_CUDA3 = grover_cuda_opt2

OBJS_OMP	= grover_omp.o utils_omp.o
OBJS_CUDA1  = grover_cuda_baseline.o utils_cuda.o
OBJS_CUDA2  = grover_cuda_opt1.o utils_cuda_opt1.o
OBJS_CUDA3  = grover_cuda_opt2.o utils_cuda_opt2.o


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

all: $(TARGET_OMP) $(TARGET_CUDA1) $(TARGET_CUDA2) $(TARGET_CUDA3)

$(TARGET_OMP): $(OBJS_OMP)
	$(CC) $(CFLAGS) -o $@ $(OBJS_OMP) -lm

$(TARGET_CUDA1): $(OBJS_CUDA1)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA1) -lm

$(TARGET_CUDA2): $(OBJS_CUDA2)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA2) -lm

$(TARGET_CUDA3): $(OBJS_CUDA3)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA3) -lm

.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET_OMP)
	@rm -f $(TARGET_CUDA1)
	@rm -f $(TARGET_CUDA2)
	@rm -f $(TARGET_CUDA3)

# dependencies
grover_omp.o: grover_omp.c utils_omp.h
utils_omp.o: utils_omp.c utils_omp.h
grover_cuda_baseline.o: grover_cuda_baseline.cu utils_cuda.h
utils_cuda.o: utils_cuda.cu utils_cuda.h
grover_cuda_opt1.o: grover_cuda_opt1.cu utils_cuda_opt1.h
utils_cuda_opt1.o: utils_cuda_opt1.cu utils_cuda_opt1.h
grover_cuda_opt2.o: grover_cuda_opt2.cu utils_cuda_opt2.h
utils_cuda_opt2.o: utils_cuda_opt2.cu utils_cuda_opt2.h
