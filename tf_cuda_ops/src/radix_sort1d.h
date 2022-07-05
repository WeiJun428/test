/* Radix Sort 1D Operation
 * GPU implementation of unique operation.
 * Created by Tun Jian TAN
 * All Rights Reserved. Mar., 2022.
 * Tensorflow Custom Operation Development Guid
 * TF1: https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/op.md
 * TF2: https://www.tensorflow.org/guide/create_op#gpu_kernels
 */
// radix_sort1d.h
#ifndef RADIX_SORT1D_H_
#define RADIX_SORT1D_H_

#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

template <typename Device, typename TKey, typename TIndex>
struct RadixSortFunctor {
  void operator()(const Device& d, 
    void * 	d_temp_storage,
    size_t & 	temp_storage_bytes,
    const TKey * 	d_keys_in,
    TKey * 	d_keys_out,
    const TIndex * 	d_values_in,
    TIndex * 	d_values_out,
    int 	num_items,
    int 	begin_bit = 0,
    int 	end_bit = sizeof(TIndex) * 8,
    std::string order = "ASCENDING",
    cudaStream_t stream = 0
  );
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename TKey, typename TIndex>
struct RadixSortFunctor<Eigen::GpuDevice, TKey, TIndex> {
  void operator()(const Eigen::GpuDevice& d, 
    void * 	d_temp_storage,
    size_t & 	temp_storage_bytes,
    const TKey * 	d_keys_in,
    TKey * 	d_keys_out,
    const TIndex * 	d_values_in,
    TIndex * 	d_values_out,
    int 	num_items,
    int 	begin_bit = 0,
    int 	end_bit = sizeof(TIndex) * 8,
    std::string order = "ASCENDING",
    cudaStream_t stream = 0
  );
};
#endif

#endif RADIX_SORT1D_H_
