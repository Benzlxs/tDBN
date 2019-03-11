# Copyright 2017 yanyan. All Rights Reserved.
#
# Licensed under the Apache License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""model builder.
"""

from tDBN.protos import tDBN_pb2
from tDBN.builder import losses_builder
from tDBN.models.model import LossNormType, Model


def build(model_cfg: tDBN_pb2.Model, voxel_generator,
          target_assigner) -> Model:
    """build tDBN pytorch instance.
    """
    if not isinstance(model_cfg, tDBN_pb2.Model):
        raise ValueError('model_cfg not of type ' 'tDBN_pb2.Model.')
    vfe_num_filters = list(model_cfg.voxelization.num_filters)
    vfe_with_distance = model_cfg.voxelization.with_distance
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_class = model_cfg.num_class

    num_input_features = model_cfg.num_point_features
    if model_cfg.without_reflectivity:
        num_input_features = 3
    loss_norm_type_dict = {
        0: LossNormType.NormByNumExamples,
        1: LossNormType.NormByNumPositives,
        2: LossNormType.NormByNumPosNeg,
    }
    loss_norm_type = loss_norm_type_dict[model_cfg.loss_norm_type]

    losses = losses_builder.build(model_cfg.loss)
    encode_rad_error_by_sin = model_cfg.encode_rad_error_by_sin
    cls_loss_ftor, loc_loss_ftor, cls_weight, loc_weight, _ = losses
    pos_cls_weight = model_cfg.pos_class_weight
    neg_cls_weight = model_cfg.neg_class_weight
    direction_loss_weight = model_cfg.direction_loss_weight

    net = Model(
        dense_shape,
        num_class=num_class,
        voxelization_name=model_cfg.voxelization.module_class_name,
        vfe_num_filters=vfe_num_filters,
        tdbn_name=model_cfg.tdbnet.module_class_name,
        tdbn_filters_d1=list(
            model_cfg.tdbnet.num_filters_down1),
        tdbn_filters_d2=list(
            model_cfg.tdbnet.num_filters_down2),
        det_net_name=model_cfg.det_net.module_class_name,
        det_net_layer_nums=list(model_cfg.det_net.layer_nums),
        det_net_layer_strides=list(model_cfg.det_net.layer_strides),
        det_net_num_filters=list(model_cfg.det_net.num_filters),
        det_net_upsample_strides=list(model_cfg.det_net.upsample_strides),
        det_net_num_upsample_filters=list(model_cfg.det_net.num_upsample_filters),
        use_norm=True,
        use_rotate_nms=model_cfg.use_rotate_nms,
        multiclass_nms=model_cfg.use_multi_class_nms,
        nms_score_threshold=model_cfg.nms_score_threshold,
        nms_pre_max_size=model_cfg.nms_pre_max_size,
        nms_post_max_size=model_cfg.nms_post_max_size,
        nms_iou_threshold=model_cfg.nms_iou_threshold,
        use_sigmoid_score=model_cfg.use_sigmoid_score,
        encode_background_as_zeros=model_cfg.encode_background_as_zeros,
        use_direction_classifier=model_cfg.use_direction_classifier,
        num_input_features=num_input_features,
        num_groups=model_cfg.det_net.num_groups,
        use_groupnorm=model_cfg.det_net.use_groupnorm,
        with_distance=vfe_with_distance,
        cls_loss_weight=cls_weight,
        loc_loss_weight=loc_weight,
        pos_cls_weight=pos_cls_weight,
        neg_cls_weight=neg_cls_weight,
        direction_loss_weight=direction_loss_weight,
        loss_norm_type=loss_norm_type,
        encode_rad_error_by_sin=encode_rad_error_by_sin,
        loc_loss_ftor=loc_loss_ftor,
        cls_loss_ftor=cls_loss_ftor,
        target_assigner=target_assigner,
    )
    return net
