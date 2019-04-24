#pragma comment(lib, "OpenCL.lib")
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <algorithm>
#include <cmath>

int main() {
	std::freopen("input.txt", "r", stdin);
	std::freopen("output.txt", "w", stdout);

	size_t N, M, sqrN, sqrM;
	std::cin >> N >> M;
	sqrN = N * N;
	sqrM = M * M;

	float *A = new float[sqrN], *B = new float[sqrM], * C = new float[sqrN];

	for (size_t i = 0; i < sqrN; i++)
		std::cin >> A[i];

	for (size_t i = 0; i < sqrM; i++)
		std::cin >> B[i];

	for (size_t i = 0; i < N; i++)
		for (size_t j = 0; j < N; j++)
			C[i * N + j] = 0;

	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	std::vector<cl::Kernel> kernels;

	try {
		cl::Platform::get(&platforms);
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
		cl::Context context(devices);
		cl::CommandQueue queue(context, devices[0]);
		std::ifstream cl_file("convolution.cl");
		std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));
		cl::Program program(context, source);

		size_t const BLOCK_SIZE = 16;
		program.build(devices, "-D BLOCK_SIZE=" + BLOCK_SIZE);

		size_t A_buf_size = sizeof(float) * sqrN;
		size_t B_buf_size = sizeof(float) * sqrM;
		size_t C_buf_size = sizeof(float) * sqrN;

		cl::Buffer dev_A(context, CL_MEM_READ_ONLY, A_buf_size);
		cl::Buffer dev_B(context, CL_MEM_READ_ONLY, B_buf_size);
		cl::Buffer dev_C(context, CL_MEM_WRITE_ONLY, C_buf_size);

		// copy from cpu to gpu
		queue.enqueueWriteBuffer(dev_A, CL_TRUE, 0, A_buf_size, A);
		queue.enqueueWriteBuffer(dev_B, CL_TRUE, 0, B_buf_size, B);

		// load named kernel from opencl source
		cl::Kernel convolutionKernel(program, "convolution");
		cl::KernelFunctor convolutionFunctor(convolutionKernel, queue, cl::NullRange, cl::NDRange(N + BLOCK_SIZE - 1, N + BLOCK_SIZE - 1), cl::NDRange(BLOCK_SIZE, BLOCK_SIZE));
		convolutionFunctor(dev_A, dev_B, dev_C, (int)N, (int)(M - 1) / 2);

		queue.enqueueReadBuffer(dev_C, CL_TRUE, 0, C_buf_size, C);

		for (size_t i = 0; i < N; i++) {
			for (size_t j = 0; j < N; j++)
				std::cout << C[i * N + j] << ' ';
			std::cout << std::endl;
		}
	}
	catch (cl::Error &e) {
		std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
	}

	return 0;
}
