from os.path import dirname, join, realpath

import tensorflow as tf
from tensorflow.python.framework import ops

from dvdet.models.tf_cuda_ops.others import radix_sort_1d

CURR_DIR = dirname(realpath(__file__))

# =============================================Grid Sampling===============================================

grid_sampling_exe = tf.load_op_library(join(CURR_DIR, "build", "grid_sampling.so"))


def grid_sampling(input_coors, input_num_list, resolution, dimension, offset):
    """
    Grid based sub-sampling

    The grid sub-sampling strategy, aims at taking the place of FPS. This operation intents to yield uniformly distributed
    sampling result. This operation is implemented in stack style, which means the number of points of each input instance
    does not have to be fixed number, and difference instances are differentiated using "input_num_list".

    Args:
        input_coors (tf.Tensor[tf.float32]): 2-D tf.float32 Tensor with shape=[input_npoint, 3].
        input_num_list (tf.Tensor[tf.int32]): 1-D tf.int32 Tensor with shape=[batch_size], indicating how many points within each instance.
        resolution (float): float32, the grid resolution for down sampling.
        dimension (List[float]): 1-D float32 list with shape 3, the maximum in x, y, z orientation of the input coors, this will be used to
                        create the unique voxel ids for each input points
        offset (List[float]): 1-D float32 list with shape 3, the offset on each axis, so that the minimum coors in each axis is > 0.

    Returns:
        A tuple containing:
        - tf.Tensor[tf.float32]: output_coors. 2-D tf.float32 Tensor with shape=[output_npoint, channels], the output coordinates of the sub-sampling.
        - tf.Tensor[tf.int32]: output_num_list. 1-D tf.int32 Tensor with shape=[batch_size], same definition as input_num_list.
        - tf.Tensor[tf.int32]: output_idx. 1-D tf.int32 Tensor with shape=[output_npoint], which indicates the point index in the input_coors. It can
                        be reused to save the computation, if the successive layer share the same grid resolution.
    """
    if type(resolution) is float:
        resolution = [resolution] * 3
    output_idx, output_num_list = grid_sampling_exe.grid_sampling_op(
        input_coors=input_coors + offset, input_num_list=input_num_list, dimension=dimension, resolution=resolution
    )
    output_coors = tf.gather(input_coors, output_idx, axis=0)
    return output_coors, output_num_list, output_idx


ops.NoGradient("GridSamplingOp")


# =============================================Voxel Sampling Idx===============================================

voxel_sampling_idx_exe = tf.load_op_library(join(CURR_DIR, "build", "voxel_sampling_idx.so"))


def voxel_sampling_idx(
    input_coors,
    input_features,
    input_num_list,
    center_coors,
    center_num_list,
    resolution,
    dimension,
    offset,
    grid_buffer_size,
    output_pooling_size,
    with_rpn=False,
):
    """
    Propagate voxel sampling indices through direct indexing

    This operation constructs the point-wise 3D convolution kernels, and fill them with point-idx, so that the actual conv kernels
    can be fulfilled by feature values in voxel_sampling_feature operation. Note that voxel_sampling_idx does not need back-prop
    implementation.

    Args:
        input_coors (tf.Tensor[tf.float32]): 2-D tf.float32 Tensor with shape=[input_npoint, 3].
        input_feaures (tf.Tensor[tf.float32]): 2-D tf.float32 Tensor with shape=[input_npoint, channels]. Point-wise feature vectors.
        input_num_list (tf.Tensor[tf.int32]): 1-D tf.int32 Tensor with shape=[batch_size], indicating how many points within each instance.
        center_coors (tf.Tensor[tf.float32]): 2-D tf.float32 Tensor with shape=[input_ncenter, 3]. The coordinates of kernel centers, which is the output of grid_sampling operaton.
        center_num_list (tf.Tensor[tf.int32]): 1-D tf.int32 Tensor with shape=[batch_size], indicating how many center points within each instance.
        resolution (Union[List[float], float]): a list of length 3, or an single float. Indicating the kernel dimension along x, y, and z axis.
                    if a single float is given, then all the axes will be voxelized with the same resolution.
        dimension (List[float]): a list of length 3, the 3D dimension for the entire voxelization area
        offset (List[float]): a list of length 3, the offset value along x, y and z axis to ensure point coordinates are always above 0.
        grid_buffer_size (int): the maximum number of points to take into consideration, if a grid is occupied by multiple points. Usually dont need to modify.
        output_pooling_size (int):  The maximum number of points in each point-wise convolution, Usually dont need to modify.
        with_rpn (bool): Unused value.

    Returns:
        A tuple containing:
        - tf.Tensor[tf.float32]: output_idx. 3-D tf.float32 Tensor with shape=[input_ncenter, 27, output_pooling_size]. This output stores the point index for the output 3D conv kernel,
                    Note that the kernel resolution are always 3 x 3 x 3 = 27.
        - tf.Tensor[tf.int32]: valid_idx. unused value, ignore.
        - tf.Tensor[tf.float32]: input_features. exactly the input_features for input data, will be used for feature concatenation.
    """
    if type(resolution) is float:
        resolution = [resolution] * 3
    output_idx, valid_idx = voxel_sampling_idx_exe.voxel_sampling_idx_op(
        input_coors=input_coors + offset,
        input_num_list=input_num_list,
        center_coors=center_coors + offset,
        center_num_list=center_num_list,
        dimension=dimension,
        resolution=resolution,
        grid_buffer_size=grid_buffer_size,
        output_pooling_size=output_pooling_size,
        with_rpn=with_rpn,
    )
    return output_idx, valid_idx, input_features


ops.NoGradient("VoxelSamplingIdxOp")

# =============================================Voxel Sampling Idx Binary===============================================

voxel_sampling_idx_binary_exe = tf.load_op_library(join(CURR_DIR, "build", "voxel_sampling_idx_binary.so"))


def voxel_sampling_idx_binary(
    input_coors,
    input_features,
    input_num_list,
    center_coors,
    center_num_list,
    resolution,
    dimension,
    offset,
    grid_buffer_size,
    output_pooling_size,
    with_rpn=False,
):
    """
    Propagate voxel sampling indices using binary search

    This is a memory-saving implementation of voxel_sampling_idx. This operation will use binary search to locate the adjacent points, instead of
    voxelize the entire space in to regular grids. Parameters share the same definition as voxel_sampling_idx.

    Args:
        input_coors (tf.Tensor[tf.float32]): 2-D tf.float32 Tensor with shape=[input_npoint, 3].
        input_feaures (tf.Tensor[tf.float32]): 2-D tf.float32 Tensor with shape=[input_npoint, channels]. Point-wise feature vectors.
        input_num_list (tf.Tensor[tf.int32]): 1-D tf.int32 Tensor with shape=[batch_size], indicating how many points within each instance.
        center_coors (tf.Tensor[tf.float32]): 2-D tf.float32 Tensor with shape=[input_ncenter, 3]. The coordinates of kernel centers, which is the output of grid_sampling operaton.
        center_num_list (tf.Tensor[tf.int32]): 1-D tf.int32 Tensor with shape=[batch_size], indicating how many center points within each instance.
        resolution (Union[List[float], float]): a list of length 3, or an single float. Indicating the kernel dimension along x, y, and z axis.
                    if a single float is given, then all the axes will be voxelized with the same resolution.
        dimention (List[float]): a list of length 3, the 3D dimension for the entire voxelization area
        offset (List[float]): a list of length 3, the offset value along x, y and z axis to ensure point coordinates are always above 0.
        grid_buffer_size (int): the maximum number of points to take into consideration, if a grid is occupied by multiple points. Usually dont need to modify.
        output_pooling_size (int):  The maximum number of points in each point-wise convolution, Usually dont need to modify.
        with_rpn (bool): Unused value.

    Returns:
        A tuple containing:
        - tf.Tensor[tf.float32]: output_idx. 3-D tf.float32 Tensor with shape=[input_ncenter, 27, output_pooling_size]. This output stores the point index for the output 3D conv kernel,
                    Note that the kernel resolution are always 3 x 3 x 3 = 27.
        - tf.Tensor[tf.int32]: valid_idx. unused value, ignore.
        - tf.Tensor[tf.float32]: input_features. exactly the input_features for input data, will be used for feature concatenation.
    """

    if type(resolution) is float:
        resolution = [resolution] * 3
    npoint = tf.shape(input_coors)[0]
    batch_size = tf.shape(input_num_list)[0]
    dim_w = tf.cast(tf.ceil(dimension[0] / resolution[0]), dtype=tf.int64) + 1
    dim_l = tf.cast(tf.ceil(dimension[1] / resolution[1]), dtype=tf.int64) + 1
    dim_h = tf.cast(tf.ceil(dimension[2] / resolution[2]), dtype=tf.int64) + 1
    dim_offset = dim_w * dim_l * dim_h

    point_ids = tf.range(npoint) + 1
    point_ids_array = tf.cast(tf.tile(tf.expand_dims(point_ids, axis=0), [batch_size, 1]), dtype=tf.float32)
    accu_num_list = tf.cast(tf.cumsum(input_num_list), dtype=tf.float32)
    masks = tf.cast(tf.greater(point_ids_array / tf.expand_dims(accu_num_list, axis=-1), 1.0), dtype=tf.int64)
    voxel_offset_masks = tf.reduce_sum(masks, axis=0) * dim_offset

    input_voxel_coors = tf.cast(tf.floor((input_coors + offset) / resolution), dtype=tf.int64)
    input_voxel_ids = (
        input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    )
    input_voxel_ids += voxel_offset_masks

    # assert tf.math.reduce_max(input_voxel_ids) < tf.int32.max
    # assert_op = tf.Assert(tf.less_equal(tf.reduce_max(input_voxel_ids), tf.int32.max), [input_voxel_ids])

    # with tf.control_dependencies([assert_op]):
    sorted_args = radix_sort_1d(input_voxel_ids, order="ASCENDING", indices_dtype=tf.int64)

    # sorted_args = tf.cast(sorted_args, dtype=tf.int64)

    # with tf.device('/cpu:0'): # line 164 ADD THIS LINE
    # sorted_args = tf.argsort(input_voxel_ids)
    sorted_voxel_ids = tf.gather(input_voxel_ids, sorted_args) - voxel_offset_masks
    sorted_coors = tf.gather(input_coors, sorted_args, axis=0)
    sorted_features = tf.gather(input_features, sorted_args, axis=0)
    output_idx, valid_idx = voxel_sampling_idx_binary_exe.voxel_sampling_idx_binary_op(
        input_coors=sorted_coors + offset,
        input_voxel_idx=sorted_voxel_ids,
        input_num_list=input_num_list,
        center_coors=center_coors + offset,
        center_num_list=center_num_list,
        dimension=dimension,
        resolution=resolution,
        grid_buffer_size=grid_buffer_size,
        output_pooling_size=output_pooling_size,
        with_rpn=with_rpn,
    )
    return output_idx, valid_idx, sorted_features


ops.NoGradient("VoxelSamplingIdxBinaryOp")

# =============================================Voxel Sampling Feature===============================================

voxel_sampling_feature_exe = tf.load_op_library(join(CURR_DIR, "build", "voxel_sampling_feature.so"))


def voxel_sampling_feature(input_features, output_idx, padding):
    """
    Aggregate the features based on the output_idx

    This is the operation to fulfill the conv kernels with output_idx, usually users dont need to take much care of this, as no
    hyper parameters are involved.

    Args:
        input_features (tf.Tensor[tf.float32]): 2D tensor of shape [N, C], where N is the number of points
                                                and C is the number of feature channels.
        output_idx (tf.Tensor[tf.int32]): Indices of the input_features of size [N].
        padding (int): Values to pad if there is a void.
    """
    output_features = voxel_sampling_feature_exe.voxel_sampling_feature_op(
        input_features=input_features, output_idx=output_idx, padding_value=padding
    )
    return output_features


@ops.RegisterGradient("VoxelSamplingFeatureOp")
def voxel_sampling_feature_grad(op, grad):
    input_features = op.inputs[0]
    output_idx = op.inputs[1]
    input_features_grad = voxel_sampling_feature_exe.voxel_sampling_feature_grad_op(
        input_features=input_features, output_idx=output_idx, output_features_grad=grad
    )
    return [input_features_grad, None]


def voxel_sampling_feature_grad_test(input_features, output_idx, grad):
    input_features_grad = voxel_sampling_feature_exe.voxel_sampling_feature_grad_op(
        input_features=input_features, output_idx=output_idx, output_features_grad=grad
    )
    return input_features_grad
