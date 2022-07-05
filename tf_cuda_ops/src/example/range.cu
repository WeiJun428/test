
__global__ void RangeKernel(int size, int start, int delta,
                            int* output) {

    if (size <=0) {
        return;
    }
    // __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    for (; i < size; i += (blockDim.x * gridDim.x) ){
        output[i] = start + i * delta;
    }
}




void RangeKernelLauncher(int size, int start, int delta,
                           int* output){

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, RangeKernel, 0, size);
    gridSize = (size + blockSize - 1) / blockSize;
    RangeKernel<<<gridSize, blockSize>>>(size, start, delta,
                                                    output);

}
