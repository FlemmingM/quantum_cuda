TARGET_CUDA_V1 = grover_cuda_v1
TARGET_CUDA_V1_2_GPU = grover_cuda_v1_2_gpu
TARGET_STREAM = grover_cuda_stream
TARGET_STREAM_2_GPU = grover_cuda_stream_2_gpu

OBJS_CUDA_V1 = grover_cuda_v1.o utils_cuda.o utils_cuda_v1.o
OBJS_CUDA_V1_2_GPU = grover_cuda_v1_2_gpu.o utils_cuda.o utils_cuda_v1_2_gpu.o
OBJS_STREAM = grover_cuda_stream.o utils_cuda.o utils_cuda_stream.o
OBJS_STREAM_2_GPU = grover_cuda_stream_2_gpu.o utils_cuda.o utils_cuda_stream_2_gpu.o




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
# all: $(TARGET_STREAM) $(TARGET_STREAM_2_GPU) $(TARGET_CUDA_V1)
all: $(TARGET_CUDA_V1_2_GPU)

$(TARGET_STREAM): $(OBJS_STREAM)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_STREAM) -lm
$(TARGET_STREAM_2_GPU): $(OBJS_STREAM_2_GPU)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_STREAM_2_GPU) -lm
$(TARGET_CUDA_V1): $(OBJS_CUDA_V1)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA_V1) -lm
$(TARGET_CUDA_V1_2_GPU): $(OBJS_CUDA_V1_2_GPU)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA_V1_2_GPU) -lm



.SUFFIXES: .cu .c
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

.c.o:
	$(CC) -o $*.o -c $*.c $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET_STREAM)
	@rm -f $(TARGET_STREAM_2_GPU)
	@rm -f $(TARGET_CUDA_V1)
	@rm -f $(TARGET_CUDA_V1_2_GPU)


# dependencies
grover_cuda_stream.o: grover_cuda_stream.cu utils_cuda.h utils_cuda_stream.h
utils_cuda_stream.o: utils_cuda_stream.cu utils_cuda_stream.h
grover_cuda_stream2.o: grover_cuda_stream2.cu utils_cuda.h utils_cuda_stream_2_gpu.h
utils_cuda_stream_2_gpu.o: utils_cuda_stream.cu utils_cuda_stream_2_gpu.h


grover_cuda_v1.o: grover_cuda_v1.cu utils_cuda.h utils_cuda_v1.h
utils_cuda_v1.cu: utils_cuda.cu utils_cuda.h

grover_cuda_v1_2_gpu.o: grover_cuda_v1_2_gpu.cu utils_cuda.h utils_cuda_v1_2_gpu.h
utils_cuda_v1_2_gpu.cu: utils_cuda.cu utils_cuda.h

utils_cuda.o: utils_cuda.cu utils_cuda.h