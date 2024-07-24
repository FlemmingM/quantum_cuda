TARGET_CUDA_BASE = grover_cuda_baseline
TARGET_CUDA_V1 = grover_cuda_v1
TARGET_CUDA_V1_2_GPU = grover_cuda_v1_2_gpu
TARGET_CUDA_V2 = grover_cuda_v2
TARGET_CUDA_V2_2_GPU = grover_cuda_v2_2_gpu

TARGET_STREAM = grover_cuda_stream
TARGET_STREAM_2_GPU = grover_cuda_stream_2_gpu

OBJS_CUDA_BASE  = grover_cuda_baseline.o utils_cuda_baseline.o
OBJS_CUDA_V1 = grover_cuda_v1.o utils_cuda.o utils_cuda_v1.o
OBJS_CUDA_V1_2_GPU = grover_cuda_v1_2_gpu.o utils_cuda.o utils_cuda_v1_2_gpu.o
OBJS_CUDA_V2 = grover_cuda_v2.o utils_cuda.o utils_cuda_v2.o
OBJS_CUDA_V2_2_GPU = grover_cuda_v2_2_gpu.o utils_cuda.o utils_cuda_v2_2_gpu.o



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


all: $(TARGET_CUDA_V2_2_GPU) $(TARGET_CUDA_V2)

$(TARGET_STREAM): $(OBJS_STREAM)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_STREAM) -lm
$(TARGET_STREAM_2_GPU): $(OBJS_STREAM_2_GPU)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_STREAM_2_GPU) -lm
$(TARGET_CUDA_V1): $(OBJS_CUDA_V1)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA_V1) -lm
$(TARGET_CUDA_V1_2_GPU): $(OBJS_CUDA_V1_2_GPU)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA_V1_2_GPU) -lm
$(TARGET_CUDA_V2): $(OBJS_CUDA_V2)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA_V2) -lm
$(TARGET_CUDA_V2_2_GPU): $(OBJS_CUDA_V2_2_GPU)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA_V2_2_GPU) -lm
$(TARGET_CUDA_BASE): $(OBJS_CUDA_BASE)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA_BASE) -lm


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

grover_cuda_baseline.o: grover_cuda_baseline.cu utils_cuda_baseline.h
utils_cuda_baseline.o: utils_cuda_baseline.cu utils_cuda_baseline.h

grover_cuda_v1.o: grover_cuda_v1.cu utils_cuda.h utils_cuda_v1.h
utils_cuda_v1.cu: utils_cuda.cu utils_cuda.h

grover_cuda_v1_2_gpu.o: grover_cuda_v1_2_gpu.cu utils_cuda.h utils_cuda_v1_2_gpu.h
utils_cuda_v1_2_gpu.cu: utils_cuda.cu utils_cuda.h

grover_cuda_v2.o: grover_cuda_v2.cu utils_cuda.h utils_cuda_v2.h
utils_cuda_v2.cu: utils_cuda.cu utils_cuda.h

grover_cuda_v2_2_gpu.o: grover_cuda_v2_2_gpu.cu utils_cuda.h utils_cuda_v2_2_gpu.h
utils_cuda_v2_2_gpu.cu: utils_cuda.cu utils_cuda.h

utils_cuda.o: utils_cuda.cu utils_cuda.h