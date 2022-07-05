/* Radix Sort 1D Operation
 * GPU implementation of unique operation.
 * Created by Tun Jian TAN
 * All Rights Reserved. Mar., 2022.
 * Tensorflow Custom Operation Development Guid
 * TF1: https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/op.md
 * TF2: https://www.tensorflow.org/guide/create_op#gpu_kernels
 */
#include "radix_sort1d.h"
#include <typeinfo>
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("RadixSort")
    .Attr("TKey: numbertype")
    .Attr("TIndex: numbertype")
    .Input("input_values:TKey")
    .Input("input_indices:TIndex")
    .Output("sorted_values:TKey")
    .Output("sorted_indices:TIndex")
    .Attr("order: {'ASCENDING', 'DESCENDING'}")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle unused_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused_shape));
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));

        return Status::OK();

    }); // InferenceContext
                         
// OpKernel definition.
template <typename Device, typename TKey, typename TIndex>
class RadixSortOp: public OpKernel {
public:
    explicit RadixSortOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("order", &order_));
    }
    void Compute(OpKernelContext* context) override {

        size_t num_bits_ = sizeof(TKey) * 8;

        // Get INPUTS
        // input keys
        const Tensor& keys_in = context->input(0);
        auto keys_in_ptr = keys_in.template flat<TKey>().data();
        size_t num_elements = keys_in.dim_size(0);

        const Tensor& indices_in = context->input(1);
        auto indices_in_ptr = indices_in.template flat<TIndex>().data();
        
        // // Get OUTPUTS
        Tensor* keys_out = nullptr;
        auto keys_out_shape = TensorShape({num_elements});
        OP_REQUIRES_OK(context, context->allocate_output(0, keys_out_shape, &keys_out));
        TKey* keys_out_ptr = keys_out->template flat<TKey>().data();

        Tensor* indices_out = nullptr;
        auto indices_out_shape = TensorShape({num_elements});
        OP_REQUIRES_OK(context, context->allocate_output(1, indices_out_shape, &indices_out));
        TIndex* indices_out_ptr = indices_out->template flat<TIndex>().data();

        // Determine temporary device storage requirements.
        Tensor temp_storage;
        size_t temp_storage_bytes = 0;
            
        RadixSortFunctor<Device, TKey, TIndex>()(
            context->eigen_device<Device>(),
            nullptr, 
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_,
            order_);


        // Allocate temporary storage.
        context->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
            &temp_storage);
        // // Sort indices by keys.
        
        RadixSortFunctor<Device, TKey, TIndex>()(
            context->eigen_device<Device>(),
            temp_storage.template flat<int8>().data(),
            temp_storage_bytes, 
            keys_in_ptr, 
            keys_out_ptr, 
            indices_in_ptr, 
            indices_out_ptr,
            num_elements,
            0,
            num_bits_,
            order_);
            
    }
private:
    std::string order_;
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(TKey, TIndex)                               \
  /* Declare explicit instantiations in radix_sort1d.cu.cc. */ \
  extern template class RadixSortFunctor<GPUDevice, TKey, TIndex>;    \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RadixSort").Device(DEVICE_GPU).TypeConstraint<TKey>("TKey").TypeConstraint<TIndex>("TIndex"), \
      RadixSortOp<GPUDevice, TKey, TIndex>);
REGISTER_GPU(float, int64);
REGISTER_GPU(int64, int64);
REGISTER_GPU(float, int32);
REGISTER_GPU(int32, int32);
#endif  // GOOGLE_CUDA