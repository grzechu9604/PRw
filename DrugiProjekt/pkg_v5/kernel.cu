/*
grid wieloblokowy, jeden w¹tek oblicza 2 lub 4 (podzia³ pracy dwuwymiarowy)
s¹siednich elementów macierzy wynikowej, obliczenia przy wykorzystaniu pamiêci wspó³dzielonej bloku w¹tków,
*/
/**
* Matrix multiplication: C = A * B.
* Host code.jk
*
* This sample implements matrix multiplication which makes use of shared memory
* to ensure data reuse, the matrix multiplication is done using tiling approach.
* It has been written for clarity of exposition to illustrate various CUDA programming
* principles, not with the goal of providing the most performant generic kernel for matrix multiplication.

*/

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <conio.h>

template <int BLOCK_SIZE> __global__ void MatrixMulKernel_5(float *A, float *B, float *C, int WIDTH) {
		// Block index
		int bx = blockIdx.x;
		int by = blockIdx.y;

		// Thread index
		int tx = threadIdx.x;
		int ty = threadIdx.y;

		// Index of the first sub-matrix of A processed by the block
		int aBegin = WIDTH * BLOCK_SIZE * by;

		// Index of the last sub-matrix of A processed by the block
		int aEnd = aBegin + WIDTH - 1;

		// Step size used to iterate through the sub-matrices of A
		int aStep = BLOCK_SIZE;

		// Index of the first sub-matrix of B processed by the block
		int bBegin = BLOCK_SIZE * bx;

		// Step size used to iterate through the sub-matrices of B
		int bStep = BLOCK_SIZE * WIDTH;

		// Csub is used to store the element of the block sub-matrix
		// that is computed by the thread
		float Csub = 0;

		// Loop over all the sub-matrices of A and B
		// required to compute the block sub-matrix
		for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep) {
			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			As[ty][tx] = A[a + WIDTH * ty + tx];
			Bs[ty][tx] = B[b + WIDTH * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two matrices together;
			// each thread computes one element
			// of the block sub-matrix
#pragma unroll

			for (int k = 0; k < BLOCK_SIZE; ++k) {
				Csub += As[ty][k] * Bs[k][tx];
			}

			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
		}

		// Write the block sub-matrix to device memory;
		// each thread writes one element
		int c = WIDTH * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		C[c + WIDTH * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
	for (int i = 0; i < size; ++i) {
		data[i] = val;
	}
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int MatrixMultiply(int block_size, const dim3 &dimsA, const dim3 &dimsB) {
	// Allocate host memory for matrices A and B
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));

	// Initialize host memory
	const float valB = 0.01f;
	ConstantInit(h_A, size_A, 1.0f);
	ConstantInit(h_B, size_B, valB);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	// Allocate host matrix C
	dim3 dimsC(dimsA.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));

	if (h_C == NULL) {
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}


	ConstantInit(h_C, dimsC.x * dimsC.y, 1.0f);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

	// Setup execution parameters
	dim3 threads(block_size, 1);
	dim3 grid(1, 1); //DLACZEGO (1,1)?

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");
	// Performs warmup operation using matrixMul CUDA kernel
	MatrixMulKernel_5 <16><<<grid,threads>>>(d_A, d_B, d_C, dimsA.x);
	printf("done\n");
	cudaDeviceSynchronize();

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));
	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));
	// Execute the kernel
	int nIter = 300;
	for (int j = 0; j < nIter; j++) {
		MatrixMulKernel_5 <16><<<grid,threads>>>(d_A, d_B, d_C, dimsA.x);
	}
	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));
	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
		static_cast<double>(dimsA.y) *
		static_cast<double>(dimsB.x);
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
		(msecPerMatrixMul / 1000.0f);
	printf(
		"Performance= %.2f GFlop/s\n Time= %.3f msec\n Size= %.0f Ops\n" \
		" WorkgroupSize= %u threads/block\n\n",
		gigaFlops,
		msecPerMatrixMul,
		flopsPerMatrixMul,
		threads.x * threads.y);

	// Copy result from device to host
	checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
	printf("Checking computed result for correctness: ");
	bool correct = true;
	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	double eps = 1.e-2;  // machine zero
	for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
		double abs_err = fabs(h_C[i] - (dimsA.x * valB));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > eps) {
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
			correct = false;
		}
	}
	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	printf("\nNOTE: The CUDA Samples are not meant for performance"\
		"measurements. Results may vary when GPU Boost is enabled.\n");

	if (correct) {
		return EXIT_SUCCESS;
	}
	else {
		return EXIT_FAILURE;
	}
}


/**
* Program main
*/
int main(int argc, char **argv) {
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	// This will pick the best possible CUDA capable device, otherwise
	// override the device ID based on input provided at the command line
	int dev = findCudaDevice(argc, (const char **)argv);

	int block_size = 16;
	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 2 * block_size, 5 * 2 * block_size, 1);

	if (dimsA.x != dimsB.y) {
		printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
		exit(EXIT_FAILURE);
	}
	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

	int matrix_result = MatrixMultiply(block_size, dimsA, dimsB);

	printf("End of program [matrix_result = %d]\n", matrix_result);
	getch();
	exit(matrix_result);
}