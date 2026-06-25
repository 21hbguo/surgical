from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from strategies.semi_dycon import DEFAULT_FEATURE_SCALER, DEFAULT_USE_ASPP

from .networks.dformerv2_small import DFormerv2SmallSeg
from .networks.dinov3 import DINOv3UNet_Base, DINOv3UNet_ContrastV1, DINOv3UNet_DepthGuiderV1, DINOv3UNet_DepthGuiderProtoV1, DINOv3UNet_DyCON, DINOv3UNet_Depth, DINOv3UNet_DepthPretrain, DINOv3UNet_RDNet, DINOv3UNet_proto, DINOv3UNet_URPC, DINOv3UNet_W2S
from .networks.resnet import ResNetUNet_Base, ResNetUNet_ContrastV1, ResNetUNet_DepthGuiderV1, ResNetUNet_DepthGuiderV1_2, ResNetUNet_DepthGuiderV2, ResNetUNet_DepthGuiderV3, ResNetUNet_DepthGuiderV4, ResNetUNet_DepthGuiderProtoV1, ResNetUNet_DyCON, ResNetUNet_Depth, ResNetUNet_DepthPretrain, ResNetUNet_RDNet, ResNetUNet_proto, ResNetUNet_URPC, ResNetUNet_W2S, ResNetUNet_GeoRiskSPC, ResNetUNet_DepthGuiderV4_GeoRiskSPC
from .networks.depth import DepthUNet_Base, DepthUNet_ContrastV1, DepthUNet_DepthGuiderV1, DepthUNet_DepthGuiderProtoV1, DepthUNet_DyCON, DepthUNet_Depth, DepthUNet_DepthPretrain, DepthUNet_RDNet, DepthUNet_proto, DepthUNet_URPC, DepthUNet_W2S
from .networks.ternaus import TernausNet16
from .networks.unet import UNet_Base, UNet_ContrastV1, UNet_DepthGuiderV1, UNet_DepthGuiderV1_2, UNet_DepthGuiderV3, UNet_DepthGuiderV4, UNet_DepthGuiderProtoV1, UNet_DyCON, UNet_Depth, UNet_DepthPretrain, UNet_RDNet, UNet_proto, UNet_URPC, UNet_W2S, UNet_GeoRiskSPC, UNet_DepthGuiderV4_GeoRiskSPC


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., Any]
    arg_map: Mapping[str, str]
    static_kwargs: Mapping[str, Any] = field(default_factory=dict)


_UNET_ARG_MAP = {"in_chns": "in_chns", "class_num": "num_classes", "filter_num": "filter_num"}
_RESNET_UNET_ARG_MAP = {"in_chns": "in_chns", "class_num": "num_classes", "filter_num": "filter_num", "variant": "resnet_variant", "pretrain_root": "pretrain_root"}
_DINOV3_UNET_ARG_MAP = {**_RESNET_UNET_ARG_MAP, "dinov3_repo_dir": "dinov3_repo_dir", "dinov3_weights": "dinov3_weights"}
_DYCON_STATIC_KWARGS = {"scale_factor": DEFAULT_FEATURE_SCALER, "use_aspp": DEFAULT_USE_ASPP}
_DFORMERV2_ARG_MAP = {"in_chns": "in_chns", "class_num": "num_classes", "pretrain_path": "dformerv2_pretrain_path"}
_TERNAUS_ARG_MAP = {"in_chns": "in_chns", "class_num": "num_classes", "pretrained": "model_pretrain", "pretrain_root": "pretrain_root"}

def _with_arg_map(base_arg_map, **extra_arg_map):
    merged = dict(base_arg_map)
    merged.update(extra_arg_map)
    return merged


MODEL_REGISTRY = {
    "unet": ModelSpec(builder=UNet_Base, arg_map=_UNET_ARG_MAP),
    "unet_rdnet": ModelSpec(builder=UNet_RDNet, arg_map=_UNET_ARG_MAP),
    "unet_urpc": ModelSpec(builder=UNet_URPC, arg_map=_UNET_ARG_MAP),
    "unet_contrast_v1": ModelSpec(builder=UNet_ContrastV1, arg_map=_with_arg_map(_UNET_ARG_MAP, feature_dim="contrast_feature_dim")),
    "unet_proto": ModelSpec(builder=UNet_proto, arg_map=_with_arg_map(_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "unet_proto_v1": ModelSpec(builder=UNet_proto, arg_map=_with_arg_map(_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "unet_dycon": ModelSpec(builder=UNet_DyCON, arg_map=_UNET_ARG_MAP, static_kwargs=_DYCON_STATIC_KWARGS),
    "unet_w2s": ModelSpec(builder=UNet_W2S, arg_map=_UNET_ARG_MAP),
    "unet_depth_guider_v1": ModelSpec(builder=UNet_DepthGuiderV1, arg_map=_UNET_ARG_MAP),
    "unet_depth_guider_v1_2": ModelSpec(builder=UNet_DepthGuiderV1_2, arg_map=_UNET_ARG_MAP),
    "unet_depth_guider_v3": ModelSpec(builder=UNet_DepthGuiderV3, arg_map=_UNET_ARG_MAP),
    "unet_depth_guider_v4": ModelSpec(builder=UNet_DepthGuiderV4, arg_map=_UNET_ARG_MAP),
    "unet_georisk_spc": ModelSpec(builder=UNet_GeoRiskSPC, arg_map=_with_arg_map(_UNET_ARG_MAP, dropout_rate="risk_dropout_rate", noise_std="risk_noise_std")),
    "unet_georisk_spc_dgv4": ModelSpec(builder=UNet_DepthGuiderV4_GeoRiskSPC, arg_map=_with_arg_map(_UNET_ARG_MAP, dropout_rate="risk_dropout_rate", noise_std="risk_noise_std")),
    "unet_depth_guider_proto_v1": ModelSpec(builder=UNet_DepthGuiderProtoV1, arg_map=_with_arg_map(_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "unet_depth": ModelSpec(builder=UNet_Depth, arg_map=_UNET_ARG_MAP),
    "unet_depth_pretrain": ModelSpec(builder=UNet_DepthPretrain, arg_map=_UNET_ARG_MAP),
    "resnet": ModelSpec(builder=ResNetUNet_Base, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_rdnet": ModelSpec(builder=ResNetUNet_RDNet, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_contrast_v1": ModelSpec(builder=ResNetUNet_ContrastV1, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="contrast_feature_dim")),
    "resnet_urpc": ModelSpec(builder=ResNetUNet_URPC, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_proto": ModelSpec(builder=ResNetUNet_proto, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "resnet_proto_v1": ModelSpec(builder=ResNetUNet_proto, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "resnet_dycon": ModelSpec(builder=ResNetUNet_DyCON, arg_map=_RESNET_UNET_ARG_MAP, static_kwargs=_DYCON_STATIC_KWARGS),
    "resnet_w2s": ModelSpec(builder=ResNetUNet_W2S, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_depth_guider_v1": ModelSpec(builder=ResNetUNet_DepthGuiderV1, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_depth_guider_v1_2": ModelSpec(builder=ResNetUNet_DepthGuiderV1_2, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_depth_guider_v2": ModelSpec(builder=ResNetUNet_DepthGuiderV2, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_depth_guider_v3": ModelSpec(builder=ResNetUNet_DepthGuiderV3, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_depth_guider_v4": ModelSpec(builder=ResNetUNet_DepthGuiderV4, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_georisk_spc": ModelSpec(builder=ResNetUNet_GeoRiskSPC, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, dropout_rate="risk_dropout_rate", noise_std="risk_noise_std")),
    "resnet_georisk_spc_dgv4": ModelSpec(builder=ResNetUNet_DepthGuiderV4_GeoRiskSPC, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, dropout_rate="risk_dropout_rate", noise_std="risk_noise_std")),
    "resnet_depth_guider_proto_v1": ModelSpec(builder=ResNetUNet_DepthGuiderProtoV1, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "resnet_depth": ModelSpec(builder=ResNetUNet_Depth, arg_map=_RESNET_UNET_ARG_MAP),
    "resnet_depth_pretrain": ModelSpec(builder=ResNetUNet_DepthPretrain, arg_map=_RESNET_UNET_ARG_MAP),
    "depth": ModelSpec(builder=DepthUNet_Base, arg_map=_RESNET_UNET_ARG_MAP),
    "depth_rdnet": ModelSpec(builder=DepthUNet_RDNet, arg_map=_RESNET_UNET_ARG_MAP),
    "depth_contrast_v1": ModelSpec(builder=DepthUNet_ContrastV1, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="contrast_feature_dim")),
    "depth_urpc": ModelSpec(builder=DepthUNet_URPC, arg_map=_RESNET_UNET_ARG_MAP),
    "depth_proto": ModelSpec(builder=DepthUNet_proto, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "depth_proto_v1": ModelSpec(builder=DepthUNet_proto, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "depth_dycon": ModelSpec(builder=DepthUNet_DyCON, arg_map=_RESNET_UNET_ARG_MAP, static_kwargs=_DYCON_STATIC_KWARGS),
    "depth_w2s": ModelSpec(builder=DepthUNet_W2S, arg_map=_RESNET_UNET_ARG_MAP),
    "depth_depth_guider_v1": ModelSpec(builder=DepthUNet_DepthGuiderV1, arg_map=_RESNET_UNET_ARG_MAP),
    "depth_depth_guider_proto_v1": ModelSpec(builder=DepthUNet_DepthGuiderProtoV1, arg_map=_with_arg_map(_RESNET_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "depth_depth": ModelSpec(builder=DepthUNet_Depth, arg_map=_RESNET_UNET_ARG_MAP),
    "depth_depth_pretrain": ModelSpec(builder=DepthUNet_DepthPretrain, arg_map=_RESNET_UNET_ARG_MAP),
    "dinov3": ModelSpec(builder=DINOv3UNet_Base, arg_map=_DINOV3_UNET_ARG_MAP),
    "dinov3_rdnet": ModelSpec(builder=DINOv3UNet_RDNet, arg_map=_DINOV3_UNET_ARG_MAP),
    "dinov3_urpc": ModelSpec(builder=DINOv3UNet_URPC, arg_map=_DINOV3_UNET_ARG_MAP),
    "dinov3_proto": ModelSpec(builder=DINOv3UNet_proto, arg_map=_with_arg_map(_DINOV3_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "dinov3_proto_v1": ModelSpec(builder=DINOv3UNet_proto, arg_map=_with_arg_map(_DINOV3_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "dinov3_contrast_v1": ModelSpec(builder=DINOv3UNet_ContrastV1, arg_map=_with_arg_map(_DINOV3_UNET_ARG_MAP, feature_dim="contrast_feature_dim")),
    "dinov3_dycon": ModelSpec(builder=DINOv3UNet_DyCON, arg_map=_DINOV3_UNET_ARG_MAP, static_kwargs=_DYCON_STATIC_KWARGS),
    "dinov3_w2s": ModelSpec(builder=DINOv3UNet_W2S, arg_map=_DINOV3_UNET_ARG_MAP),
    "dinov3_depth_guider_v1": ModelSpec(builder=DINOv3UNet_DepthGuiderV1, arg_map=_DINOV3_UNET_ARG_MAP),
    "dinov3_depth_guider_proto_v1": ModelSpec(builder=DINOv3UNet_DepthGuiderProtoV1, arg_map=_with_arg_map(_DINOV3_UNET_ARG_MAP, feature_dim="proto_feature_dim")),
    "dinov3_depth": ModelSpec(builder=DINOv3UNet_Depth, arg_map=_DINOV3_UNET_ARG_MAP),
    "dinov3_depth_pretrain": ModelSpec(builder=DINOv3UNet_DepthPretrain, arg_map=_DINOV3_UNET_ARG_MAP),
    "dformerv2_small": ModelSpec(builder=DFormerv2SmallSeg, arg_map=_DFORMERV2_ARG_MAP),
    "ternaus16": ModelSpec(builder=TernausNet16, arg_map=_TERNAUS_ARG_MAP, static_kwargs={"num_filters": 32}),
}


def resolve_default_model_name(way, pretrain_mode):
    from strategies.specs import resolve_strategy_default_model_name as _resolve_strategy_default_model_name

    return _resolve_strategy_default_model_name(way, pretrain_mode)


def create_model(args):
    spec = MODEL_REGISTRY[args.model]
    kwargs = {kwarg_name: getattr(args, attr_name) for kwarg_name, attr_name in spec.arg_map.items()}
    kwargs.update(spec.static_kwargs)
    return spec.builder(**kwargs)
