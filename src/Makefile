TARGET_OMP	= grover_omp
TARGET_CUDA1 = grover_cuda_baseline
TARGET_CUDA2 = grover_cuda_opt1
TARGET_CUDA3 = grover_cuda_opt2
TARGET_CUDA4 = grover_cuda_opt3
TARGET_CUDA5 = grover_cuda_opt4
TARGET_CUDA6 = grover_cuda_opt5
# TARGET_CUDA7 = grover_cuda_opt6
TARGET_CUDA8 = grover_cuda_opt7
TARGET_CUDA9 = grover_cuda_opt8
TARGET_CUDA10 = grover_cuda_opt9
TARGET_CUDA11 = grover_cuda_opt10
TARGET_CUDA12 = grover_cuda_opt11


OBJS_OMP	= grover_omp.o utils_omp.o
OBJS_CUDA1  = grover_cuda_baseline.o utils_cuda.o
OBJS_CUDA2  = grover_cuda_opt1.o utils_cuda_opt1.o
OBJS_CUDA3  = grover_cuda_opt2.o utils_cuda_opt2.o
OBJS_CUDA4  = grover_cuda_opt3.o utils_cuda_opt3.o
OBJS_CUDA5  = grover_cuda_opt4.o utils_cuda_opt4.o
OBJS_CUDA6  = grover_cuda_opt5.o utils_cuda_opt5.o
# OBJS_CUDA7  = grover_cuda_opt7.o utils_cuda_opt6.o
OBJS_CUDA8  = grover_cuda_opt7.o utils_cuda_opt7.o
OBJS_CUDA9  = grover_cuda_opt8.o utils_cuda_opt8.o
OBJS_CUDA10  = grover_cuda_opt9.o utils_cuda_opt9.o
OBJS_CUDA11  = grover_cuda_opt10.o utils_cuda_opt10.o
OBJS_CUDA12  = grover_cuda_opt11.o utils_cuda_opt11.o



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

# all: $(TARGET_OMP) $(TARGET_CUDA1) $(TARGET_CUDA2) $(TARGET_CUDA3) $(TARGET_CUDA4) $(TARGET_CUDA5) $(TARGET_CUDA6) $(TARGET_CUDA8) $(TARGET_CUDA9) $(TARGET_CUDA10)
# all: $(TARGET_CUDA5) $(TARGET_CUDA9) $(TARGET_CUDA10)
# all: $(TARGET_CUDA5)

all: $(TARGET_CUDA5) $(TARGET_CUDA1)


$(TARGET_OMP): $(OBJS_OMP)
	$(CC) $(CFLAGS) -o $@ $(OBJS_OMP) -lm

$(TARGET_CUDA1): $(OBJS_CUDA1)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA1) -lm

$(TARGET_CUDA2): $(OBJS_CUDA2)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA2) -lm

$(TARGET_CUDA3): $(OBJS_CUDA3)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA3) -lm

$(TARGET_CUDA4): $(OBJS_CUDA4)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA4) -lm

$(TARGET_CUDA5): $(OBJS_CUDA5)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA5) -lm

$(TARGET_CUDA6): $(OBJS_CUDA6)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA6) -lm

# $(TARGET_CUDA7): $(OBJS_CUDA7)
# 	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA7) -lm

$(TARGET_CUDA8): $(OBJS_CUDA8)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA8) -lm

$(TARGET_CUDA9): $(OBJS_CUDA9)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA9) -lm

$(TARGET_CUDA10): $(OBJS_CUDA10)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA10) -lm

$(TARGET_CUDA11): $(OBJS_CUDA11)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA11) -lm

$(TARGET_CUDA12): $(OBJS_CUDA12)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CUDA12) -lm



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
	@rm -f $(TARGET_CUDA4)
	@rm -f $(TARGET_CUDA5)
	@rm -f $(TARGET_CUDA6)
	@rm -f $(TARGET_CUDA8)
	@rm -f $(TARGET_CUDA9)
	@rm -f $(TARGET_CUDA10)
	@rm -f $(TARGET_CUDA11)
	@rm -f $(TARGET_CUDA12)

# dependencies
grover_omp.o: grover_omp.c utils_omp.h
utils_omp.o: utils_omp.c utils_omp.h
grover_cuda_baseline.o: grover_cuda_baseline.cu utils_cuda.h
utils_cuda.o: utils_cuda.cu utils_cuda.h
grover_cuda_opt1.o: grover_cuda_opt1.cu utils_cuda_opt1.h
utils_cuda_opt1.o: utils_cuda_opt1.cu utils_cuda_opt1.h
grover_cuda_opt2.o: grover_cuda_opt2.cu utils_cuda_opt2.h
utils_cuda_opt2.o: utils_cuda_opt2.cu utils_cuda_opt2.h
grover_cuda_opt3.o: grover_cuda_opt3.cu utils_cuda_opt3.h
utils_cuda_opt3.o: utils_cuda_opt3.cu utils_cuda_opt3.h
grover_cuda_opt4.o: grover_cuda_opt4.cu utils_cuda_opt4.h
utils_cuda_opt4.o: utils_cuda_opt4.cu utils_cuda_opt4.h
grover_cuda_opt5.o: grover_cuda_opt5.cu utils_cuda_opt5.h
utils_cuda_opt5.o: utils_cuda_opt5.cu utils_cuda_opt5.h


grover_cuda_opt7.o: grover_cuda_opt7.cu utils_cuda_opt7.h
utils_cuda_opt7.o: utils_cuda_opt7.cu utils_cuda_opt7.h
grover_cuda_opt8.o: grover_cuda_opt8.cu utils_cuda_opt8.h
utils_cuda_opt8.o: utils_cuda_opt8.cu utils_cuda_opt8.h
grover_cuda_opt9.o: grover_cuda_opt9.cu utils_cuda_opt9.h
utils_cuda_opt9.o: utils_cuda_opt9.cu utils_cuda_opt9.h
grover_cuda_opt10.o: grover_cuda_opt10.cu utils_cuda_opt10.h
utils_cuda_opt10.o: utils_cuda_opt10.cu utils_cuda_opt10.h
grover_cuda_opt11.o: grover_cuda_opt11.cu utils_cuda_opt11.h
utils_cuda_opt11.o: utils_cuda_opt11.cu utils_cuda_opt11.h