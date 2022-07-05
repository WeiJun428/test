from os.path import join, dirname, realpath
import tensorflow as tf
from tensorflow.python.framework import ops

CURR_DIR = dirname(realpath(__file__))

radix_sort1d_kernel_gpu_exe = tf.load_op_library(join(CURR_DIR, "build", "radix_sort1d.so"))


def radix_sort_1d(keys, order="ASCENDING", return_values=False, indices_dtype=tf.int32):
    """
    1D Sort the keys using radix sort.

    It is a custom sort operation that uses cub::DeviceRadixSort::SortPairs and cub::DeviceRadixSort::SortPairsDescending

    Args:
        keys (tf.Tensor): array to be sorted. (,)
        order (str): order of sorting. {ASCENDING, DESCENDING}
        return_values (bool): If True, return the sorted keys's values.
        indices_dtype (tf.dtype): The tensorflow type of the sorted indices.
    Return:
        sorted_values (tf.Tensor): sorted array. (,)
        sorted_indices (tf.Tensor): sorted array's indices/ argsort indices.
    """
    sorted_values, sorted_indices = radix_sort1d_kernel_gpu_exe.radix_sort(
        input_values=keys,
        input_indices=tf.range(tf.cast(tf.shape(keys)[0], indices_dtype), dtype=indices_dtype),
        order=order,
    )

    if return_values:
        return sorted_values, sorted_indices

    return sorted_indices


ops.NoGradient("RadixSort")


# ============================================= NMS ===============================================

iou3d_kernel_gpu_exe = tf.load_op_library(join(CURR_DIR, "build", "nms.so"))


def rotated_nms3d_idx(bbox_attrs, bbox_conf, nms_overlap_thresh, nms_conf_thres):
    """
    rotated nms of the output
    :param boxes: the set of bounding boxes (sorted in decending order based on a score, e.g. confidence)
                  in [M, 7] := [x, y, z, w, l, h, r]
                  where ry = anti-clockwise in Z-up system
    :param nms_overlap_thresh: The boxes that overlaps with a given box more than the threshold will be remove .
                               threshold range [0,1]
    :param nms_overlap_thresh: Only boxes with confidence level greater than this value will be taken into consideration.
    Use case:
    boxes[output_keep_index[:output_num_to_keep]] := gives the list of the valid bounding boxes
    """

    valid_count = tf.reduce_sum(tf.cast(tf.greater(bbox_conf, nms_conf_thres), dtype=tf.int32))

    # with tf.device('/cpu:0'): # line 164 ADD THIS LINE
    # sorted_idx = tf.argsort(bbox_conf, direction="DESCENDING")
    sorted_idx = radix_sort_1d(bbox_conf, order="DESCENDING", return_values=False, indices_dtype=tf.int32)

    sorted_bbox_attrs = tf.gather(bbox_attrs, sorted_idx, axis=0)[:valid_count, :]

    output_keep_index, output_num_to_keep = iou3d_kernel_gpu_exe.rotated_nms3d(
        input_boxes=sorted_bbox_attrs, nms_overlap_thresh=nms_overlap_thresh
    )

    output_idx = tf.gather(sorted_idx, output_keep_index[: output_num_to_keep[0]])

    return output_idx


ops.NoGradient("RotatedNms3d")
