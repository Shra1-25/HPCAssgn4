all: jacobi innerProduct
jacobi : main.cu jacobi2D.cu jacobi2D.cuh
	nvcc -std=c++11 main.cu jacobi2D.cu -o jacobi2D-cuda
innerProduct : innerProduct.cu
	nvcc -std=c++11 innerProduct.cu -o innerProduct

clean:
	rm -f *.out
	rm -f jacobi2D-cuda
	rm -f innerProduct