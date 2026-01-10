// test_cuda_basic.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread %d in block %d\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    std::cout << "CUDA Test: Launching kernel..." << std::endl;
    
    // Launch kernel with 1 block and 8 threads
    helloFromGPU<<<1, 8>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    std::cout << "Test completed successfully!" << std::endl;
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    return 0;
}