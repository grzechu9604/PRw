/*
grid wieloblokowy, jeden w�tek oblicza 2 lub 4 (podzia� pracy dwuwymiarowy)
s�siednich element�w macierzy wynikowej, obliczenia przy wykorzystaniu pami�ci wsp�dzielonej bloku w�tk�w,
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

template <int BLOCK_SIZE> __global__ void MatrixMulKernel_5(float *Ad, float *Bd, float *Cd, int WIDTH, int aElements) {
	int tx = threadIdx.x * aElements;
	int ty = threadIdx.y * aElements;
	int Row = blockIdx.y * (BLOCK_SIZE*aElements) + ty;
	int Col = blockIdx.x * (BLOCK_SIZE*aElements) + tx;
	float C_local_1 = 0.0;
	float C_local_2 = 0.0;
	float C_local_3 = 0.0;
	float C_local_4 = 0.0;
	const int  newBlockSize = BLOCK_SIZE*aElements;
	__shared__ float Ads[32][32];
	__shared__ float Bds[32][32];

	for (int m = 0; m < WIDTH / (BLOCK_SIZE*aElements); m++) {
		Ads[tx][ty] = Ad[m*(BLOCK_SIZE*aElements) + Row*WIDTH];
		Bds[tx][ty] = Bd[(m*(BLOCK_SIZE*aElements))*WIDTH + Col];
		Ads[tx + 1][ty] = Ad[m*(BLOCK_SIZE*aElements) + Row*WIDTH + 1];
		Bds[tx + 1][ty] = Bd[(m*(BLOCK_SIZE*aElements) + 1)*WIDTH + Col];

		Ads[tx][ty+1] = Ad[m*(BLOCK_SIZE*aElements) + (Row+1)*WIDTH];
		Bds[tx][ty+1] = Bd[(m*(BLOCK_SIZE*aElements))*WIDTH + Col+1];
		Ads[tx+1][ty+1] = Ad[m*(BLOCK_SIZE*aElements) + (Row + 1)*WIDTH];
		Bds[tx+1][ty+1] = Bd[(m*(BLOCK_SIZE*aElements)+1)*WIDTH + Col+1];

		__syncthreads();
		for (int k = 0; k < (BLOCK_SIZE*aElements); k++) {
			C_local_1 += Ads[tx][k] * Bds[k][ty];
			C_local_3 += Ads[tx + 1][k] * Bds[k][ty];

			C_local_2 += Ads[tx][k] * Bds[k][ty+1];
			C_local_4 += Ads[tx+1][k] * Bds[k][ty+1];
		}
		__syncthreads();
	}
	//Cd[Row * WIDTH + Col] = C_local;
	Cd[Row * WIDTH + Col] = C_local_1;
	Cd[Row * WIDTH + 1 + Col] = C_local_3;
	Cd[(Row+1) * WIDTH + Col] = C_local_2;
	Cd[(Row+1) * WIDTH + Col + 1] = C_local_4;
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

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	int aElements = 2;
	dim3 grid(ceil(dimsA.x / block_size)/aElements, ceil(dimsA.x / block_size)/aElements);
	cudaDeviceSynchronize();

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));
	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));
	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));

	// Execute the kernel
	int nIter = grid.x * grid.y;
	MatrixMulKernel_5<16><<<grid,threads>>>(d_A, d_B, d_C, dimsA.x, aElements);

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
	double eps = 1.e-6;  // machine zero
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

	//Results may vary when GPU Boost is enabled?
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