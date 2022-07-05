/* Radix Sort 1D Operation
 * GPU implementation of unique operation.
 * Created by Tun Jian TAN
 * All Rights Reserved. Mar., 2022.
 * Tensorflow Custom Operation Development Guid
 * TF1: https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/op.md
 * TF2: https://www.tensorflow.org/guide/create_op#gpu_kernels
 * How to compile
 * # nvcc src/argsort1d.cu -o build/argsort1d.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 -I ../../../dependencies/thrust # -arch=sm_72 -O3 #
 * # g++ src/argsort1d.cpp build/argsort1d.cu.o -o build/argsort1d.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
 * # echo "Done: argsort1d.cu.o"
 */
// #include <iostream>
// #include <stdio.h>
#include <cuda_runtime.h>

// #include <stdint.h>
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/common_shape_fns.h"
// #include "tensorflow/core/platform/types.h" // namespace se = ::stream_executor;
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h" // gpu_grid_helper
// #include "tensorflow/core/kernels/gpu_prim_helpers.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("RadixSortIntOp")
    .Input("input_array:int32")
    // .Output("sorted_values:int32")
    .Output("sorted_indices:int32")
    .Attr("order: {'ASCENDING', 'DESCENDING'}")
    .SetShapeFn([](InferenceContext* c){
        
        c->set_output(0, c->input(0));
        // ShapeHandle input_coors_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_coors_shape));
        // ShapeHandle input_num_list_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_num_list_shape));

        // DimensionHandle input_point_num = c->Dim(input_coors_shape, 0);
        // ShapeHandle output_idx_shape = c->MakeShape({input_point_num});
        // c->set_output(0, output_idx_shape);
        // c->set_output(1, input_num_list_shape);

        return Status::OK();

    }); // InferenceContext

void RangeKernelLauncher(int size, int start, int delta,
                            int* output);
                         
void RadixSortIntAscendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const int * 	d_keys_in,
        int * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(int) * 8,
        cudaStream_t stream = 0);

void RadixSortIntDescendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const int * 	d_keys_in,
        int * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(int) * 8,
        cudaStream_t stream = 0);

class RadixSortIntOp: public OpKernel {
public:
    explicit RadixSortIntOp(OpKernelConstruction* context): OpKernel(context) {
    // OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution));
    // OP_REQUIRES_OK(context, context->GetAttr("dimension", &dimension));
    // OP_REQUIRES(context,  resolution.size() == 3,
    //             errors::InvalidArgument("Resolution has to be 3-D for Voxel Sample Operation."));
    // OP_REQUIRES(context, dimension.size() == 3,
    //             errors::InvalidArgument("Dimension has to be 3-D for Voxel Sample Operation."));
    // }
        OP_REQUIRES_OK(context, context->GetAttr("order", &order));
        if (num_bits_ == -1) {
            num_bits_ = sizeof(int) * 8;
        }
    }
    void Compute(OpKernelContext* context) override {

        // Get INPUTS
        // input keys
        const Tensor& keys_in = context->input(0);
        auto keys_in_ptr = keys_in.template flat<int>().data();
        int num_elements = keys_in.dim_size(0);

        // Do the computation.
        OP_REQUIRES(context, keys_in.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));
        Tensor indices_in;
        // Allocate temporary input values.
        context->allocate_temp(
            DataTypeToEnum<int>::value, TensorShape({num_elements}),
            &indices_in);
        auto indices_in_ptr = indices_in.template flat<int>().data();

        RangeKernelLauncher(num_elements, 0, 1, indices_in_ptr);
        
        // // Get OUTPUTS
        Tensor* indices_out = nullptr;
        auto indices_out_shape = TensorShape({num_elements});
        OP_REQUIRES_OK(context, context->allocate_output(0, indices_out_shape, &indices_out));
        int* indices_out_ptr = indices_out->template flat<int>().data();

        // RangeKernelLauncher(num_elements, 0, 1, indices_out_ptr);

        // Tensor* keys_out = nullptr;
        // auto keys_out_shape = TensorShape({num_elements});
        // OP_REQUIRES_OK(context, context->allocate_output(0, keys_out_shape, &keys_out));
        // int* keys_out_ptr = keys_out->flat<int>().data();

        Tensor keys_out;
        // // Allocate temporary input values.
        context->allocate_temp(
            DataTypeToEnum<int>::value, TensorShape({num_elements}),
            &keys_out);
        auto keys_out_ptr = keys_out.template flat<int>().data();
        
        // Determine temporary device storage requirements.
        Tensor temp_storage;
        size_t temp_storage_bytes = 0;
            
        if(order.compare("ASCENDING") == 0){
            RadixSortIntAscendingGpuLauncher(
            nullptr, 
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_);

        }else if(order.compare("DESCENDING") == 0){
            RadixSortIntDescendingGpuLauncher(
            nullptr, 
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_);

        }else{
            throw errors::InvalidArgument("order should be {'ASCENDING', 'DESCENDING'}");
        }
        

        // Allocate temporary storage.
        context->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
            &temp_storage);
        // cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // // Sort indices by keys.
        
        if(order.compare("ASCENDING") == 0){
            RadixSortIntAscendingGpuLauncher(
                temp_storage.template flat<int8>().data(), 
                temp_storage_bytes, 
                keys_in_ptr, 
                keys_out_ptr,
                indices_in_ptr, 
                indices_out_ptr, 
                num_elements,
                0,
                num_bits_);
        }else if(order.compare("DESCENDING") == 0){
            RadixSortIntDescendingGpuLauncher(
                temp_storage.template flat<int8>().data(),
                temp_storage_bytes, 
                keys_in_ptr, 
                keys_out_ptr, 
                indices_in_ptr, 
                indices_out_ptr,
                num_elements,
                0,
                num_bits_);
        }else{
            throw errors::InvalidArgument("order should be {'ASCENDING', 'DESCENDING'}");
        }
            
    }
private:
    int num_bits_ = -1;
    string order;
};
REGISTER_KERNEL_BUILDER(Name("RadixSortIntOp").Device(DEVICE_GPU), RadixSortIntOp);



void RadixSortFloatAscendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const float * 	d_keys_in,
        float * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(float) * 8,
        cudaStream_t stream = 0);

void RadixSortFloatDescendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const float * 	d_keys_in,
        float * 	d_keys_out,
        const int * 	d_values_in,
        int * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(float) * 8,
        cudaStream_t stream = 0);

REGISTER_OP("RadixSortFloatOp")
    .Input("input_array:float32")
    // .Output("sorted_values:int32")
    .Output("sorted_indices:int32")
    .Attr("order: {'ASCENDING', 'DESCENDING'}")
    .SetShapeFn([](InferenceContext* c){
        // ShapeHandle input_coors_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_coors_shape));
        // ShapeHandle input_num_list_shape;
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input_num_list_shape));

        // DimensionHandle input_point_num = c->Dim(input_coors_shape, 0);
        // ShapeHandle output_idx_shape = c->MakeShape({input_point_num});
        // c->set_output(0, output_idx_shape);
        // c->set_output(1, input_num_list_shape);
        c->set_output(0, c->input(0));

        return Status::OK();

    }); // InferenceContext

class RadixSortFloatOp: public OpKernel {
public:
    explicit RadixSortFloatOp(OpKernelConstruction* context): OpKernel(context) {
    // OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution));
    // OP_REQUIRES_OK(context, context->GetAttr("dimension", &dimension));
    // OP_REQUIRES(context,  resolution.size() == 3,
    //             errors::InvalidArgument("Resolution has to be 3-D for Voxel Sample Operation."));
    // OP_REQUIRES(context, dimension.size() == 3,
    //             errors::InvalidArgument("Dimension has to be 3-D for Voxel Sample Operation."));
    // }
        OP_REQUIRES_OK(context, context->GetAttr("order", &order));
        if (num_bits_ == -1) {
            num_bits_ = sizeof(float) * 8;
        }
    }
    void Compute(OpKernelContext* context) override {

        // Get INPUTS
        // input keys
        const Tensor& keys_in = context->input(0);
        auto keys_in_ptr = keys_in.template flat<float>().data();
        int num_elements = keys_in.dim_size(0);

        // Do the computation.
        OP_REQUIRES(context, keys_in.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));
        Tensor indices_in;
        // Allocate temporary input values.
        context->allocate_temp(
            DataTypeToEnum<int>::value, TensorShape({num_elements}),
            &indices_in);
        auto indices_in_ptr = indices_in.template flat<int>().data();

        RangeKernelLauncher(num_elements, 0, 1, indices_in_ptr);
        
        // // Get OUTPUTS
        Tensor* indices_out = nullptr;
        auto indices_out_shape = TensorShape({num_elements});
        OP_REQUIRES_OK(context, context->allocate_output(0, indices_out_shape, &indices_out));
        int* indices_out_ptr = indices_out->template flat<int>().data();

        // RangeKernelLauncher(num_elements, 0, 1, indices_out_ptr);

        // Tensor* keys_out = nullptr;
        // auto keys_out_shape = TensorShape({num_elements});
        // OP_REQUIRES_OK(context, context->allocate_output(0, keys_out_shape, &keys_out));
        // int* keys_out_ptr = keys_out->flat<int>().data();

        Tensor keys_out;
        // // Allocate temporary input values.
        context->allocate_temp(
            DataTypeToEnum<float>::value, TensorShape({num_elements}),
            &keys_out);
        auto keys_out_ptr = keys_out.template flat<float>().data();
        
        // Determine temporary device storage requirements.
        Tensor temp_storage;
        size_t temp_storage_bytes = 0;
            
        if(order.compare("ASCENDING") == 0){
            RadixSortFloatAscendingGpuLauncher(
            nullptr, 
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_);

        }else if(order.compare("DESCENDING") == 0){
            RadixSortFloatDescendingGpuLauncher(
            nullptr, 
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_);

        }else{
            throw errors::InvalidArgument("order should be {'ASCENDING', 'DESCENDING'}");
        }
        

        // Allocate temporary storage.
        context->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
            &temp_storage);
        // cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // // Sort indices by keys.
        
        if(order.compare("ASCENDING") == 0){
            RadixSortFloatAscendingGpuLauncher(
                temp_storage.template flat<int8>().data(), 
                temp_storage_bytes, 
                keys_in_ptr, 
                keys_out_ptr,
                indices_in_ptr, 
                indices_out_ptr, 
                num_elements,
                0,
                num_bits_);
        }else if(order.compare("DESCENDING") == 0){
            RadixSortFloatDescendingGpuLauncher(
                temp_storage.template flat<int8>().data(),
                temp_storage_bytes, 
                keys_in_ptr, 
                keys_out_ptr, 
                indices_in_ptr, 
                indices_out_ptr,
                num_elements,
                0,
                num_bits_);
        }else{
            throw errors::InvalidArgument("order should be {'ASCENDING', 'DESCENDING'}");
        }
            
    }
private:
    int num_bits_ = -1;
    string order;
};
REGISTER_KERNEL_BUILDER(Name("RadixSortFloatOp").Device(DEVICE_GPU), RadixSortFloatOp);


void RadixSortLongLongAscendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const long long * 	d_keys_in,
        long long * 	d_keys_out,
        const long long * 	d_values_in,
        long long * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(long long) * 8,
        cudaStream_t stream = 0);

void RadixSortLongLongDescendingGpuLauncher(void * 	d_temp_storage,
        size_t & 	temp_storage_bytes,
        const long long * 	d_keys_in,
        long long * 	d_keys_out,
        const long long * 	d_values_in,
        long long * 	d_values_out,
        int 	num_items,
        int 	begin_bit = 0,
        int 	end_bit = sizeof(long long) * 8,
        cudaStream_t stream = 0);

REGISTER_OP("RadixSortLongLongOp")
    .Input("input_array:int64")
    .Input("input_indices:int64")
    .Output("sorted_values:int64")
    .Output("sorted_indices:int64")
    .Attr("order: {'ASCENDING', 'DESCENDING'}")
    .SetShapeFn([](InferenceContext* c){
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));

        return Status::OK();

    }); // InferenceContext
                         

class RadixSortLongLongOp: public OpKernel {
public:
    explicit RadixSortLongLongOp(OpKernelConstruction* context): OpKernel(context) {
    // OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution));
    // OP_REQUIRES_OK(context, context->GetAttr("dimension", &dimension));
    // OP_REQUIRES(context,  resolution.size() == 3,
    //             errors::InvalidArgument("Resolution has to be 3-D for Voxel Sample Operation."));
    // OP_REQUIRES(context, dimension.size() == 3,
    //             errors::InvalidArgument("Dimension has to be 3-D for Voxel Sample Operation."));
    // }
        OP_REQUIRES_OK(context, context->GetAttr("order", &order));
        if (num_bits_ == -1) {
            num_bits_ = sizeof(long long) * 8;
        }
    }
    void Compute(OpKernelContext* context) override {

        // Get INPUTS
        // input keys
        const Tensor& keys_in = context->input(0);
        auto keys_in_ptr = keys_in.template flat<long long>().data();
        int num_elements = keys_in.dim_size(0);

        // Do the computation.
        OP_REQUIRES(context, keys_in.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));

        
        const Tensor& indices_in = context->input(1);
        auto indices_in_ptr = indices_in.template flat<long long>().data();

        // Tensor indices_in;
        // // Allocate temporary input values.
        // context->allocate_temp(
        //     DataTypeToEnum<int>::value, TensorShape({num_elements}),
        //     &indices_in);
        // auto indices_in_ptr = indices_in.template flat<int>().data();

        // RangeKernelLauncher(num_elements, 0, 1, indices_in_ptr);
        
        // // Get OUTPUTS
        Tensor* indices_out = nullptr;
        auto indices_out_shape = TensorShape({num_elements});
        OP_REQUIRES_OK(context, context->allocate_output(1, indices_out_shape, &indices_out));
        long long* indices_out_ptr = indices_out->template flat<long long>().data();

        // RangeKernelLauncher(num_elements, 0, 1, indices_out_ptr);

        Tensor* keys_out = nullptr;
        auto keys_out_shape = TensorShape({num_elements});
        OP_REQUIRES_OK(context, context->allocate_output(0, keys_out_shape, &keys_out));
        long long* keys_out_ptr = keys_out->template flat<long long>().data();

        // Tensor keys_out;
        // // // Allocate temporary input values.
        // context->allocate_temp(
        //     DataTypeToEnum<int>::value, TensorShape({num_elements}),
        //     &keys_out);
        // auto keys_out_ptr = keys_out.template flat<int>().data();
        
        // Determine temporary device storage requirements.
        Tensor temp_storage;
        size_t temp_storage_bytes = 0;
            
        if(order.compare("ASCENDING") == 0){
            RadixSortLongLongAscendingGpuLauncher(
            nullptr, 
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_);

        }else if(order.compare("DESCENDING") == 0){
            RadixSortLongLongDescendingGpuLauncher(
            nullptr, 
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_);

        }else{
            throw errors::InvalidArgument("order should be {'ASCENDING', 'DESCENDING'}");
        }
        

        // Allocate temporary storage.
        context->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
            &temp_storage);
        // // Sort indices by keys.
        
        if(order.compare("ASCENDING") == 0){
            RadixSortLongLongAscendingGpuLauncher(
                temp_storage.template flat<int8>().data(), 
                temp_storage_bytes, 
                keys_in_ptr, 
                keys_out_ptr,
                indices_in_ptr, 
                indices_out_ptr, 
                num_elements,
                0,
                num_bits_);
        }else if(order.compare("DESCENDING") == 0){
            RadixSortLongLongDescendingGpuLauncher(
                temp_storage.template flat<int8>().data(),
                temp_storage_bytes, 
                keys_in_ptr, 
                keys_out_ptr, 
                indices_in_ptr, 
                indices_out_ptr,
                num_elements,
                0,
                num_bits_);
        }else{
            throw errors::InvalidArgument("order should be {'ASCENDING', 'DESCENDING'}");
        }
            
    }
private:
    int num_bits_ = -1;
    string order;
};
REGISTER_KERNEL_BUILDER(Name("RadixSortLongLongOp").Device(DEVICE_GPU), RadixSortLongLongOp);