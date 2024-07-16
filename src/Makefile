TARGET_STREAM = grover_cuda_stream


OBJS_STREAM = grover_cuda_stream.o utils_cuda.o utils_cuda_stream.o





OPT	= -g -O3
ISA	=
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_90 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPTS)

F90C  	= gfortran

CUDA_PATH ?= /appl/cuda/12.2.2
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common

# all: $(TARGET_OMP) $(TARGET_CUDA1) $(TARGET_CUDA2) $(TARGET_CUDA3) $(TARGET_CUDA4) $(TARGET_CUDA5) $(TARGET_CUDA6) $(TARGET_CUDA8) $(TARGET_CUDA9) $(TARGET_CUDA10) $(TARGET_CUDA11) $(TARGET_CUDA12)
# all: $(TARGET_CUDA5) $(TARGET_CUDA9) $(TARGET_CUDA10)
# all: $(TARGET_CUDA5)

# all: $(TARGET_CUDA5) $(TARGET_CUDA1) $(TARGET_CUDA9)
# all:  $(TARGET_CUDA8) $(TARGET_CUDA10) $(TARGET_CUDA11) $(TARGET_CUDA12)
# all: $(TARGET_CUDA2) $(TARGET_CUDA3) $(TARGET_CUDA4) $(TARGET_CUDA9)
# all: $(TARGET_CUDA13) $(TARGET_CUDA5) $(TARGET_CUDA9)
all: $(TARGET_STREAM)

$(TARGET_STREAM): $(OBJS_STREAM)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_STREAM) -lm




.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET_STREAM)


# dependencies
grover_cuda_stream.o: grover_cuda_stream.cu utils_cuda.h utils_cuda_stream.h
utils_cuda_stream.o: utils_cuda_stream.cu utils_cuda_stream.h
utils_cuda.o: utils_cuda.cu utils_cuda.h