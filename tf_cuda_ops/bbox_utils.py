from os.path import join, dirname, realpath
import tensorflow as tf
from tensorflow.python.framework import ops

CURR_DIR = dirname(realpath(__file__))

# =============================================Get RoI Ground Truth===============================================

get_roi_bbox_exe = tf.load_op_library(join(CURR_DIR, "build", "get_roi_bbox.so"))


def get_roi_bbox(input_coors, bboxes, input_num_list, anchor_size, expand_ratio=0.15, cls_thres=1, det_thres=1):
    """
    Get point-wise RoI ground truth.
    :param input_coors: 2-D Tensor with shape [npoint, 3]
    :param bboxes: 3-D Tensor with shape [batch, nbbox, bbox_attr]
    :param input_num_list: 1-D Tensor with shape [batch]
    :param anchor_size: 1-D list or tensor with shape [4] (w, l, h, z)
    :param expand_ratio: default=0.1
    :param cls_thres: only the points with class_id <= diff_thres will be linked to the classification loss
    :param det_thres: only the points with class_id <= diff_thres will be linked to the detection loss
    :return: output_attrs: 2-D Tensor with shape [npoint, 7]
                           [confidence, w, l, h, offset_x, offset_y, offset_z, rotation angle]
             roi_cls_conf: 1-D Tensor with shape [npoint], foreground points will be assigned with value 1, background points will be 0,
                           if cls_thres are not satisfied, point will have value -1
             roi_det_conf: 1-D Tensor with shape [npoint], foreground points will be assigned with value 1, background points will be 0,
                           if det_thres are not satisfied, point will have value -1
             roi_cls: 1-D Tensor with shape [npoint], indicating the ground_truth class_id. Background and ignored class will have value 0.
    """
    roi_attrs, roi_cls_conf, roi_det_conf, roi_cls = get_roi_bbox_exe.get_roi_bbox_op(
        input_coors=input_coors,
        gt_bbox=bboxes,
        input_num_list=input_num_list,
        anchor_size=anchor_size,
        expand_ratio=expand_ratio,
        cls_thres=cls_thres,
        det_thres=det_thres,
    )
    return roi_attrs, roi_cls_conf, roi_det_conf, roi_cls


ops.NoGradient("GetRoiBboxOp")


# =============================================Roi Logits To Attrs===============================================

roi_logits_to_attrs_exe = tf.load_op_library(join(CURR_DIR, "build", "roi_logits_to_attrs.so"))


def roi_logits_to_attrs(base_coors, input_logits, anchor_size):
    """
    Convert logits to bounding boxes. Note that this function should be used only at the inference stage for speed optimization.

    ** Input **
    base_coors: 2-D Tensor with Shape [npoint, 3], which is the coordinates of the keypoints of the last convolutional layer.
    input_logits: The logits returned from the model.
    anchor_size: a list of length 3, pre-defined anchor size in w, l, h.

    ** output **
    output_attrs: 2-D Tensor with shape [npoint, 7], which is the actual output bounding boxes.
    """
    output_attrs = roi_logits_to_attrs_exe.roi_logits_to_attrs_op(
        base_coors=base_coors, input_logits=input_logits, anchor_size=anchor_size
    )
    return output_attrs


ops.NoGradient("RoiLogitsToAttrsOp")

# =============================================Bbox Logits To Attrs===============================================

# bbox_logits_to_attrs_exe = tf.load_op_library(join(CURR_DIR, 'build', 'bbox_logits_to_attrs.so'))
# def bbox_logits_to_attrs(input_roi_attrs, input_logits):
#     '''
#     This operation converts
#     '''
#     output_attrs = bbox_logits_to_attrs_exe.bbox_logits_to_attrs_op(input_roi_attrs=input_roi_attrs,
#                                                                     input_logits=input_logits)
#     return output_attrs
# ops.NoGradient("BboxLogitsToAttrsOp")


# def get_anchor_attrs(anchor_coors, anchor_param_list):  # [n, 2], [k, f]
#     anchor_param_list = tf.expand_dims(anchor_param_list, axis=0)  # [1, k, f]
#     anchor_param_list = tf.tile(anchor_param_list, [tf.shape(anchor_coors)[0], 1, 1])  # [n, k, f]
#     output_anchor_attrs = []
#     for k in range(anchor_param_list.shape[1]):
#         anchor_param = anchor_param_list[:, k, :] # [n, f] (w, l, h, z, r)
#         anchor_attrs = tf.stack([anchor_param[:, 0],
#                                  anchor_param[:, 1],
#                                  anchor_param[:, 2],
#                                  anchor_coors[:, 0],
#                                  anchor_coors[:, 1],
#                                  anchor_param[:, 3],
#                                  anchor_param[:, 4]], axis=-1) # [n, f]
#         output_anchor_attrs.append(anchor_attrs)

#     return tf.stack(output_anchor_attrs, axis=1) # [n, k, f]


# def logits_to_attrs(anchor_coors, input_logits, anchor_param_list): # [n, k, f]
#     output_attrs = []
#     # anchor_param_list = tf.expand_dims(anchor_param_list, axis=0) # [k, f]
#     for k in range(anchor_param_list.shape[0]):
#         anchor_param = anchor_param_list[k, :] # [f]
#         anchor_diag = tf.sqrt(tf.pow(anchor_param[0], 2.) + tf.pow(anchor_param[1], 2.))
#         w = tf.clip_by_value(tf.exp(input_logits[:, k, 0]) * anchor_param[0], 0., 1e7)
#         l = tf.clip_by_value(tf.exp(input_logits[:, k, 1]) * anchor_param[1], 0., 1e7)
#         h = tf.clip_by_value(tf.exp(input_logits[:, k, 2]) * anchor_param[2], 0., 1e7)
#         x = tf.clip_by_value(input_logits[:, k, 3] * anchor_diag + anchor_coors[:, 0], -1e7, 1e7)
#         y = tf.clip_by_value(input_logits[:, k, 4] * anchor_diag + anchor_coors[:, 1], -1e7, 1e7)
#         z = tf.clip_by_value(input_logits[:, k, 5] * anchor_param[2] + anchor_param[3], -1e7, 1e7)
#         r = tf.clip_by_value((input_logits[:, k, 6] + anchor_param[4]) * np.pi, -1e7, 1e7)
#         output_attrs.append(tf.stack([w, l, h, x, y, z, r], axis=-1)) # [n, f]
#     return tf.stack(output_attrs, axis=1) # [n, k, f]
