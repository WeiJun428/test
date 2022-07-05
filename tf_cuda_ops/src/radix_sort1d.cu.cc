/* Radix Sort 1D Operation
 * GPU implementation of unique operation.
 * Created by Tun Jian TAN
 * All Rights Reserved. Mar., 2022.
 * Tensorflow Custom Operation Development Guide
 * TF1: https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/op.md
 * TF2: https://www.tensorflow.org/guide/create_op#gpu_kernels
 */
// radix_sort1d.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "radix_sort1d.h"
#include <cub/cub.cuh> 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename TKey, typename TIndex>
void RadixAscendingGpuLauncher(
        void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const TKey * 	d_keys_in,
        TKey * 	d_keys_out,
        const TIndex * 	d_values_in,
        TIndex * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(TIndex) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};

template <typename TKey, typename TIndex>
void RadixSortDescendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const TKey * 	d_keys_in,
        TKey * 	d_keys_out,
        const TIndex * 	d_values_in,
        TIndex * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(TIndex) * 8,
        cudaStream_t stream = 0){

        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
                num_items, begin_bit, end_bit, stream);
        

};
         

// Define the GPU implementation that launches the CUDA kernel.
template <typename TKey, typename TIndex>
void RadixSortFunctor<GPUDevice, TKey, TIndex>::operator()(
    const Eigen::GpuDevice& d, 
    void * 	d_temp_storage,
    size_t & 	temp_storage_bytes,
    const TKey * 	d_keys_in,
    TKey * 	d_keys_out,
    const TIndex * 	d_values_in,
    TIndex * 	d_values_out,
    int 	num_items,
    int 	begin_bit,
    int 	end_bit,
    std::string order,
    cudaStream_t stream){
    
    if(order.compare("ASCENDING") == 0){
        RadixAscendingGpuLauncher<TKey, TIndex>(
        d_temp_storage, 
        temp_storage_bytes, 
        d_keys_in, 
        d_keys_out, 
        d_values_in, 
        d_values_out,
        num_items,
        begin_bit,
        end_bit,
        d.stream());

    }else if(order.compare("DESCENDING") == 0){
        RadixSortDescendingGpuLauncher<TKey, TIndex>(
        d_temp_storage, 
        temp_storage_bytes, 
        d_keys_in, 
        d_keys_out, 
        d_values_in, 
        d_values_out,
        num_items,
        begin_bit,
        end_bit,
        d.stream());
    }
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct RadixSortFunctor<GPUDevice, float, int64>;
template struct RadixSortFunctor<GPUDevice, int64, int64>;
template struct RadixSortFunctor<GPUDevice, int32, int32>;
template struct RadixSortFunctor<GPUDevice, float, int32>;

#endif  // GOOGLE_CUDA