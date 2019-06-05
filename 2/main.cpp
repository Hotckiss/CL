#pragma comment(lib, "OpenCL.lib")

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

size_t const BLOCK_SIZE = 256;

void scan_propagation(float *from, int from_size, float *to, int to_size, cl::Context &context, cl::Program &program, cl::CommandQueue &queue);

void recursive_scan(float *arr, int size, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * size);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * size, arr);

    cl::Kernel scan_kernel(program, "scan_hillis_steele_block");
    cl::KernelFunctor scan_functor(scan_kernel, queue, cl::NullRange, 
                cl::NDRange(((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE), cl::NDRange(BLOCK_SIZE));
    scan_functor(size, dev_input, dev_output, cl::__local(sizeof(float) * BLOCK_SIZE), cl::__local(sizeof(float) * BLOCK_SIZE));

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * size, arr);

    if (size <= BLOCK_SIZE)
        return;

    int new_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<float> sums(new_size);

    cl::Buffer dev_blocks(context, CL_MEM_WRITE_ONLY, sizeof(float) * new_size);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * size, arr);

    cl::Kernel sum_blocks_kernel(program, "create_sum_blocks");
    cl::KernelFunctor sum_blocks_functor(sum_blocks_kernel, queue, cl::NullRange, 
                cl::NDRange(((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE), cl::NDRange(BLOCK_SIZE));
    sum_blocks_functor(size, new_size, dev_input, dev_blocks);

    queue.enqueueReadBuffer(dev_blocks, CL_TRUE, 0, sizeof(float) * new_size, sums.data());

    recursive_scan(sums.data(), new_size, context, program, queue);

    scan_propagation(sums.data(), new_size, arr, size, context, program, queue);
}

void scan_propagation(float *from, int from_size, float *to, int to_size, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_blocks(context, CL_MEM_READ_ONLY, sizeof(float) * from_size);
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * to_size);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * to_size);

    queue.enqueueWriteBuffer(dev_blocks, CL_TRUE, 0, sizeof(float) * from_size, from);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * to_size, to);

    cl::Kernel propagation_kernel(program, "scan_propagation");
    cl::KernelFunctor propagation_functor(propagation_kernel, queue, cl::NullRange, 
            cl::NDRange(((to_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE), cl::NDRange(BLOCK_SIZE));
    propagation_functor(to_size, dev_blocks, dev_input, dev_output);

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * to_size, to);
}

int main() {
    std::freopen("input.txt", "r", stdin);
    std::freopen("output.txt", "w", stdout);

    size_t n;
    std::cin >> n;

    std::vector<float> in(n);
    for (size_t i = 0; i < n; i++)
        std::cin >> in[i];

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0]);

        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));
        cl::Program program(context, source);

        try {
            program.build(devices);
        } catch (cl::Error const & e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        recursive_scan(in.data(), n, context, program, queue);
    }
    catch (cl::Error &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    std::cout.precision(3);
    std::cout << std::fixed;
    
    for (size_t i = 0; i < n; i++)
        std::cout << in[i] << " ";

    return 0;
}
