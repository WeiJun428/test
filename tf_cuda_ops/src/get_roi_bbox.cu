/* Furthest point sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 * Happy Mid-Autumn Festival! :)
 */
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#define USECPSEC 1000000ULL

__global__ void get_roi_bbox_gpu_kernel(int batch_size, int npoint, int nbbox, int bbox_attr,
                                            int cls_thres, int det_thres, float expand_ratio,
                                            const float* input_coors,
                                            const float* gt_bbox,
                                            const int* input_num_list,
                                            const float* anchor_size,
                                            int* input_accu_list,
                                            float* roi_bbox,
                                            int* roi_cls_conf,
                                            int* roi_det_conf,
                                            int* roi_cls) {
    if (batch_size * nbbox * bbox_attr <=0 || npoint <=0) {
//        printf("Get RoI Logits Op exited unexpectedly.\n");
        return;
    }
//    const float PI = 3.1415927;
    float anchor_diag = sqrtf(anchor_size[0]*anchor_size[0] + anchor_size[1]*anchor_size[1]);
    input_accu_list[0] = 0;
    for (int b=1; b<batch_size; b++) {
        input_accu_list[b] = input_accu_list[b-1] + input_num_list[b-1];
    }
    __syncthreads();
//    printf("%d\n", input_accu_list[5]);
    for (int b=blockIdx.x; b<batch_size; b+=gridDim.x) {
        for (int i=threadIdx.x; i<input_num_list[b]; i+=blockDim.x) {
            roi_bbox[input_accu_list[b]*7 + i*7 + 0] = 0.1;
            roi_bbox[input_accu_list[b]*7 + i*7 + 1] = 0.1;
            roi_bbox[input_accu_list[b]*7 + i*7 + 2] = 0.1;

            float point_x = input_coors[input_accu_list[b]*3 + i*3 + 0];
            float point_y = input_coors[input_accu_list[b]*3 + i*3 + 1];
            float point_z = input_coors[input_accu_list[b]*3 + i*3 + 2];
            roi_cls_conf[input_accu_list[b] + i] = 0;
            roi_det_conf[input_accu_list[b] + i] = 0;
            roi_cls[input_accu_list[b] + i] = 0;
            for (int j=0; j<nbbox; j++) {
            // [w, l, h, x, y, z, r, cls, diff_idx]
            //  0  1  2  3  4  5  6   7      8
                float bbox_w = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 0];
                float bbox_l = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 1];
                float bbox_h = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 2];
                float bbox_x = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 3];
                float bbox_y = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 4];
                float bbox_z = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 5];
                float bbox_r = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 6];
                float bbox_cls = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 7];
                // float bbox_diff = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 8];
                if (bbox_l*bbox_h*bbox_w > 0) {
                    float rel_point_x = point_x - bbox_x;
                    float rel_point_y = point_y - bbox_y;
                    float rel_point_z = point_z - bbox_z;
                    float rot_rel_point_x = rel_point_x*cosf(bbox_r) + rel_point_y*sinf(bbox_r);
                    float rot_rel_point_y = -rel_point_x*sinf(bbox_r) + rel_point_y*cosf(bbox_r);
                    if (abs(rot_rel_point_x)<=bbox_w * (1 + expand_ratio) / 2 &&
                        abs(rot_rel_point_y)<=bbox_l * (1 + expand_ratio) / 2 &&
                        abs(rel_point_z)<=bbox_h * (1 + expand_ratio) / 2) {

                        roi_bbox[input_accu_list[b]*7 + i*7 + 0] = bbox_w;
                        roi_bbox[input_accu_list[b]*7 + i*7 + 1] = bbox_l;
                        roi_bbox[input_accu_list[b]*7 + i*7 + 2] = bbox_h;
                        roi_bbox[input_accu_list[b]*7 + i*7 + 3] = bbox_x;
                        roi_bbox[input_accu_list[b]*7 + i*7 + 4] = bbox_y;
                        roi_bbox[input_accu_list[b]*7 + i*7 + 5] = bbox_z;
                        roi_bbox[input_accu_list[b]*7 + i*7 + 6] = bbox_r;

                        if (bbox_cls <= cls_thres) {
                            roi_cls_conf[input_accu_list[b] + i] = 1;
                            roi_cls[input_accu_list[b] + i] = bbox_cls;
                        } else {
                            roi_cls_conf[input_accu_list[b] + i] = -1;
                            roi_cls[input_accu_list[b] + i] = 0;
                        }

                        if (bbox_cls <= det_thres) {
                            roi_det_conf[input_accu_list[b] + i] = 1;
                        } else {
                            roi_det_conf[input_accu_list[b] + i] = -1;
                        }
                    }
                }
            }
        }
    }
}

long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

void get_roi_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr,
                               int cls_thres, int det_thres, float expand_ratio,
                               const float* input_coors,
                               const float* gt_bbox,
                               const int* input_num_list,
                               const float* anchor_size,
                               int* input_accu_list,
                               float* roi_bbox,
                               int* roi_cls_conf,
                               int* roi_det_conf,
                               int* roi_cls) {
    long long dt = dtime_usec(0);
    get_roi_bbox_gpu_kernel<<<32,512>>>(batch_size, npoint, nbbox, bbox_attr,
                                          cls_thres, det_thres, expand_ratio,
                                          input_coors,
                                          gt_bbox,
                                          input_num_list,
                                          anchor_size,
                                          input_accu_list,
                                          roi_bbox,
                                          roi_cls_conf,
                                          roi_det_conf,
                                          roi_cls);
    dt = dtime_usec(dt);
//	std::cout << "Voxel Sample (forward) CUDA time: " << dt/(float)USECPSEC << "s" << std::endl;
}
