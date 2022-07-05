/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
/******************************************************************************
 * Simple example of DeviceRadixSort::SortPairs().
 *
 * Sorts an array of float keys paired with a corresponding array of int values.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_radix_sort.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/
// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#include <stdio.h>
#include <algorithm>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_radix_sort.cuh>
using namespace cub;

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

/**
 * Main
 */
int main(int argc, char** argv)
{

    // Declare, allocate, and initialize device-accessible pointers for sorting data
    int  num_items;          // e.g., 7
    int key_buf[] = {8, 6, 7, 5, 3, 0, 9};
    // int key_alt_buf[] = {0,0,0,0,0,0,0};
    int value_buf[] = {0, 1, 2, 3, 4, 5, 6};
    // int value_alt_buf[] = {0,0,0,0,0,0,0};
    num_items = 7;

    // int  num_items;          // e.g., 7
    int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_keys_out;        // e.g., [        ...        ]
    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    int  *d_values_out;      // e.g., [        ...        ]

    // Allocate temporary storage
    cudaMalloc(&d_keys_in, num_items*sizeof(int));
    cudaMalloc(&d_keys_out, num_items*sizeof(int));
    cudaMalloc(&d_values_in, num_items*sizeof(int));
    cudaMalloc(&d_values_out, num_items*sizeof(int));

    
    cudaMemcpy(d_keys_in, &key_buf[0], num_items*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_in, &value_buf[0], num_items*sizeof(int), cudaMemcpyHostToDevice);
    
    for(int i = 0; i < num_items; i++){
        printf("elements: %d keys: %d values: %d \n", i, key_buf[i], value_buf[i]);
    }
    printf("\n");

    // // Create a set of DoubleBuffers to wrap pairs of device pointers
    // cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
    // cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // Allocate temporary storage
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // d_keys.Current()      <-- [0, 3, 5, 6, 7, 8, 9]
    // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]

    cudaMemcpy(&key_buf[0], d_keys_out, num_items*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&value_buf[0], d_values_out, num_items*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < num_items; i++){
        printf("elements: %d keys: %d values: %d \n", i, key_buf[i], value_buf[i]);
    }

    if (d_keys_in) CubDebugExit(cudaFree(d_keys_in));
    if (d_keys_out) CubDebugExit(cudaFree(d_keys_out));
    if (d_values_in) CubDebugExit(cudaFree(d_values_in));
    if (d_values_out) CubDebugExit(cudaFree(d_values_out));
    if (d_temp_storage) CubDebugExit(cudaFree(d_temp_storage));


    return 0;
}

