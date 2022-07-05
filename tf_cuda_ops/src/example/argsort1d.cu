/* Radix Sort 1D Operation
 * GPU implementation of unique operation.
 * Created by Tun Jian TAN
 * All Rights Reserved. Mar., 2022.
 * Tensorflow Custom Operation Development Guid
 * TF1: https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/op.md
 * TF2: https://www.tensorflow.org/guide/create_op#gpu_kernels
*/

#include <cub/cub.cuh> 
    

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

void RadixSortIntAscendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const int * 	d_keys_in,
        int * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(int) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};

void RadixSortIntDescendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const int * 	d_keys_in,
        int * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(int) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};


void RadixSortFloatAscendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const float * 	d_keys_in,
        float * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(float) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};

void RadixSortFloatDescendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const float * 	d_keys_in,
        float * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(float) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};
                            


                            

void RadixSortLongLongAscendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const long long * 	d_keys_in,
        long long * 	d_keys_out,
        const long long * 	d_values_in,
        long long * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(long long) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};

void RadixSortLongLongDescendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const long long * 	d_keys_in,
        long long * 	d_keys_out,
        const long long * 	d_values_in,
        long long * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(long long) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};
                         