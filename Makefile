CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcublas -lcusparse

all: sgemm sparse_sgemm sparselt_sgemm

sgemm: sgemm.cu
	nvcc $(INC) sgemm.cu -o cublas_sgemm $(LIBS) -O2

sparse_sgemm: sparse_sgemm.cu
	nvcc $(INC) sparse_sgemm.cu -o cusparse_sgemm $(LIBS) -O2

sparselt_sgemm: sparselt_sgemm.cu
	nvcc $(INC) -I/usr/include  sparselt_sgemm.cu -o cusparselt_sgemm $(LIBS) -O2 -L/usr/lib64 -lcusparseLt_static -ldl /usr/local/cuda-11.5/targets/x86_64-linux/lib/libnvrtc.so

clean:
	rm -f cublas_sgemm sparse_sgemm
 
test:
	@echo "\n==== cuBLAS_Sgemm Test ====\n"
	./cublas_sgemm
	@echo "\n==== cuSparse_Sgemm Test ====\n"
	./cusparse_sgemm
	@echo "\n=== cuSparseLt_Sgemm Test ====\n"
	./cusparselt_sgemm

.PHONY: clean all test
