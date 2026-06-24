import json
import os
from dataclasses import dataclass

from .fully_dformerv2 import DFormerv2FullyStrategy
from .fully_contrast_v1 import FullyContrastV1Strategy
from .fully_contrast_v1_1 import FullyContrastV11Strategy
from .fully_depth_pretrain_v1 import FullyDepthPretrainStrategy
from .fully_rgb_masking_depth_v1 import FullyRGBMaskingDepthV1Strategy
from .fully_supervised import FullySupervisedStrategy
from .fully_supervised_depthGAN import FullySupervisedDepthGANStrategy
from .fully_reg import FullyRegSupervisedStrategy
from .fully_ternaus import TernausNet16Strategy
from .semi_depth_guided_mt import DepthGuidedMTStrategy
from .semi_dycon import DyConStrategy
from .semi_mean_teacher import MeanTeacherStrategy
from .semi_mean_teacher_contrast_v1 import MeanTeacherContrastV1Strategy
from .semi_mean_teacher_text_v1 import MeanTeacherTextV1Strategy
from .semi_mt_depth_guider_proto_teacher_v1 import MTDepthGuiderProtoTeacherV1Strategy
from .semi_mt_depth_guider_proto_teacher_v2 import MTDepthGuiderProtoTeacherV2Strategy
from .semi_mt_depth_guider_proto_teacher_v3 import MTDepthGuiderProtoTeacherV3Strategy
from .semi_mt_depth_guider_proto_v1 import MTDepthGuiderProtoV1Strategy
from .semi_mt_depth_guider_v1 import MTDepthGuiderV1Strategy
from .semi_mt_depth_guider_v2 import MTDepthGuiderV2Strategy
from .semi_mt_depth_guider_v3 import MTDepthGuiderV3Strategy
from .semi_mt_depth_teacher_v1 import MTDepthTeacherV1Strategy
from .semi_only_depthInput import OnlyDepthInputStrategy
from .semi_proto_v1 import ProtoV1Strategy
from .semi_georisk_spc import GeoRiskSPCStrategy
from .semi_georisk_spc_v2 import GeoRiskSPCStrategyV2
from .semi_rdnet import RDNetStrategy
from .semi_uncertainty_mt import UncertaintyMTStrategy
from .semi_urpc import URPCStrategy
from .semi_w2s import W2SStrategy
from .semi_segmatch import SegMatchStrategy
from .semi_unimatch import UniMatchStrategy
from .semi_cps import CPSStrategy
from .semi_unimatch_official import UniMatchOfficialStrategy
from .semi_segmatch_official import SegMatchOfficialStrategy
from .semi_mms import MMSStrategy
from .semi_u2pl import U2PLStrategy
from .semi_corrmatch import CorrMatchStrategy
from .semi_cwbass import CWBASStrategy


VALID_DEPTH_CHANNELS = {1, 3, 13}
_PRETRAIN_PREFIXES = {"none": "unet", "resnet": "resnet", "depth": "depth", "dinov3": "dinov3"}
SPECIAL_MODEL_NAMES = {"ternaus16", "dformerv2_small"}
PRETRAINED_UNET_PREFIXES = ("resnetunet", "dinov3unet")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    cls: type
    is_semi: bool
    model_suffix: str | None = None
    fixed_model_name: str | None = None
    model_names: dict[str, str] | None = None
    in_chns: int | str | None = None


def _spec(
    name,
    cls,
    *,
    is_semi,
    model_suffix=None,
    fixed_model_name=None,
    model_names=None,
    in_chns=None,
):
    return StrategySpec(
        name=name,
        cls=cls,
        is_semi=is_semi,
        model_suffix=model_suffix,
        fixed_model_name=fixed_model_name,
        model_names=model_names,
        in_chns=in_chns,
    )


STRATEGY_SPECS = {
    "fully": _spec("fully", FullySupervisedStrategy, is_semi=False, model_suffix=""),
    "fully_supervised_depthgan": _spec("fully_supervised_depthgan", FullySupervisedDepthGANStrategy, is_semi=False, model_suffix="", in_chns="metadata"),
    "fully_reg": _spec("fully_reg", FullyRegSupervisedStrategy, is_semi=False, model_suffix=""),
    "fully_rgb_masking_depth_v1": _spec("fully_rgb_masking_depth_v1", FullyRGBMaskingDepthV1Strategy, is_semi=False, model_suffix="", in_chns="metadata"),
    "fully_depth_pretrain_v1": _spec("fully_depth_pretrain_v1", FullyDepthPretrainStrategy, is_semi=False, model_suffix="depth_pretrain"),
    "fully_depth_pretrain": _spec("fully_depth_pretrain", FullyDepthPretrainStrategy, is_semi=False, model_suffix="depth_pretrain"),
    "fully_contrast_v1": _spec("fully_contrast_v1", FullyContrastV1Strategy, is_semi=False, model_suffix="contrast_v1"),
    "fully_contrast_v1_1": _spec("fully_contrast_v1_1", FullyContrastV11Strategy, is_semi=False, model_suffix="contrast_v1"),
    "mt": _spec("mt", MeanTeacherStrategy, is_semi=True, model_suffix=""),
    "mt_depth_teacher_v1": _spec("mt_depth_teacher_v1", MTDepthTeacherV1Strategy, is_semi=True, model_suffix="", in_chns="metadata"),
    "mt_depth_guider_v1": _spec("mt_depth_guider_v1", MTDepthGuiderV1Strategy, is_semi=True, model_suffix="depth_guider_v1", in_chns="metadata"),
    "mt_depth_guider_v1_2": _spec("mt_depth_guider_v1_2", MTDepthGuiderV1Strategy, is_semi=True, model_names={"none": "unet_depth_guider_v1_2", "resnet": "resnet_depth_guider_v1_2", "depth": "unet_depth_guider_v1_2", "dinov3": "unet_depth_guider_v1_2"}, in_chns="metadata"),
    "mt_depth_guider_v2": _spec("mt_depth_guider_v2", MTDepthGuiderV2Strategy, is_semi=True, model_suffix="depth_guider_v2", in_chns="metadata"),
    "mt_depth_guider_v3": _spec("mt_depth_guider_v3", MTDepthGuiderV3Strategy, is_semi=True, model_suffix="depth_guider_v3", in_chns="metadata"),
    "mt_depth_guider_v4": _spec("mt_depth_guider_v4", MTDepthGuiderV1Strategy, is_semi=True, model_names={"none": "unet_depth_guider_v4", "resnet": "resnet_depth_guider_v4", "depth": "unet_depth_guider_v4", "dinov3": "unet_depth_guider_v4"}, in_chns="metadata"),
    "mt_depth_guider_proto_v1": _spec("mt_depth_guider_proto_v1", MTDepthGuiderProtoV1Strategy, is_semi=True, model_suffix="depth_guider_proto_v1", in_chns="metadata"),
    "mt_depth_guider_proto_teacher_v2": _spec("mt_depth_guider_proto_teacher_v2", MTDepthGuiderProtoTeacherV2Strategy, is_semi=True, model_suffix="depth_guider_proto_v1", in_chns="metadata"),
    "mt_depth_guider_proto_teacher_v3": _spec("mt_depth_guider_proto_teacher_v3", MTDepthGuiderProtoTeacherV3Strategy, is_semi=True, model_suffix="depth_guider_proto_v1", in_chns="metadata"),
    "mt_depth_guider_proto_teacher_v1": _spec("mt_depth_guider_proto_teacher_v1", MTDepthGuiderProtoTeacherV1Strategy, is_semi=True, model_suffix="depth_guider_proto_v1", in_chns="metadata"),
    "semi_mean_teacher_contrast_v1": _spec("semi_mean_teacher_contrast_v1", MeanTeacherContrastV1Strategy, is_semi=True, model_suffix="contrast_v1"),
    "semi_mean_teacher_text_v1": _spec("semi_mean_teacher_text_v1", MeanTeacherTextV1Strategy, is_semi=True, model_suffix="proto_v1"),
    "uamt": _spec("uamt", UncertaintyMTStrategy, is_semi=True, model_suffix=""),
    "urpc": _spec("urpc", URPCStrategy, is_semi=True, model_suffix="urpc"),
    "ternaus": _spec("ternaus", TernausNet16Strategy, is_semi=False, fixed_model_name="ternaus16"),
    "proto_v1": _spec("proto_v1", ProtoV1Strategy, is_semi=True, model_suffix="proto_v1"),
    "proto": _spec("proto", ProtoV1Strategy, is_semi=True, model_suffix="proto_v1"),
    "dycon": _spec("dycon", DyConStrategy, is_semi=True, model_suffix="dycon"),
    "w2s": _spec("w2s", W2SStrategy, is_semi=True, model_suffix="w2s"),
    "depth_mt": _spec("depth_mt", DepthGuidedMTStrategy, is_semi=True, model_suffix="depth"),
    "rdnet": _spec("rdnet", RDNetStrategy, is_semi=True, model_suffix=""),
    "georisk_spc": _spec("georisk_spc", GeoRiskSPCStrategy, is_semi=True, model_suffix="georisk_spc", in_chns=None),
    "georisk_spc_dgv4": _spec("georisk_spc_dgv4", GeoRiskSPCStrategy, is_semi=True, model_names={"none": "unet_georisk_spc_dgv4", "resnet": "resnet_georisk_spc_dgv4", "depth": "unet_georisk_spc_dgv4", "dinov3": "unet_georisk_spc_dgv4"}, in_chns=None),
    "georisk_spc_dgv4_v2": _spec("georisk_spc_dgv4_v2", GeoRiskSPCStrategyV2, is_semi=True, model_names={"none": "unet_georisk_spc_dgv4", "resnet": "resnet_georisk_spc_dgv4", "depth": "unet_georisk_spc_dgv4", "dinov3": "unet_georisk_spc_dgv4"}, in_chns=None),
    "dformerv2_fully": _spec("dformerv2_fully", DFormerv2FullyStrategy, is_semi=False, fixed_model_name="dformerv2_small"),
    "only_depth_input": _spec("only_depth_input", OnlyDepthInputStrategy, is_semi=True, model_suffix="", in_chns=3),
    "segmatch": _spec("segmatch", SegMatchStrategy, is_semi=True, model_suffix=""),
    "unimatch": _spec("unimatch", UniMatchStrategy, is_semi=True, model_suffix=""),
    "cps": _spec("cps", CPSStrategy, is_semi=True, model_suffix=""),
    "unimatch_official": _spec("unimatch_official", UniMatchOfficialStrategy, is_semi=True, model_suffix=""),
    "segmatch_official": _spec("segmatch_official", SegMatchOfficialStrategy, is_semi=True, model_suffix=""),
    "mms": _spec("mms", MMSStrategy, is_semi=True, model_suffix=""),
    "u2pl": _spec("u2pl", U2PLStrategy, is_semi=True, model_suffix=""),
    "corrmatch": _spec("corrmatch", CorrMatchStrategy, is_semi=True, model_suffix=""),
    "cwbass": _spec("cwbass", CWBASStrategy, is_semi=True, model_suffix=""),
}

STRATEGY_REGISTRY = {name: spec.cls for name, spec in STRATEGY_SPECS.items()}
SEMI_STRATEGY_NAMES = {name for name, spec in STRATEGY_SPECS.items() if spec.is_semi}


def _canonicalize_strategy_name(name):
    normalized = str(name or "").strip().lower()
    if normalized in STRATEGY_SPECS:
        return normalized
    basename = os.path.splitext(os.path.basename(normalized))[0]
    if basename in STRATEGY_SPECS:
        return basename
    if basename.startswith("semi_") and basename[5:] in STRATEGY_SPECS:
        return basename[5:]
    raise KeyError(f"Unknown strategy name: {name!r}")


def get_strategy_names():
    return list(STRATEGY_SPECS.keys())


def get_strategy_spec(name):
    return STRATEGY_SPECS[_canonicalize_strategy_name(name)]


def is_semi_strategy(name):
    return get_strategy_spec(name).is_semi


def resolve_strategy_default_model_name(way, pretrain_mode):
    spec = get_strategy_spec(way)
    if spec.model_names is not None and pretrain_mode in spec.model_names:
        return spec.model_names[pretrain_mode]
    if spec.fixed_model_name is not None:
        return spec.fixed_model_name
    prefix = _PRETRAIN_PREFIXES[pretrain_mode]
    suffix = spec.model_suffix or ""
    return f"{prefix}_{suffix}" if suffix else prefix


def _load_metadata_input_channels(root_path, task):
    task_json = os.path.join(root_path, f"task{int(task)}.json")
    with open(task_json, "r", encoding="utf-8") as handle:
        info = json.load(handle)
    value = info.get("input_channels")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid input_channels in {task_json}: {value!r}")
    return value


def resolve_strategy_input_settings(way, root_path, task, use_depth=None):
    metadata_in_chns = _load_metadata_input_channels(root_path, task)
    if use_depth is not None and use_depth not in VALID_DEPTH_CHANNELS:
        raise ValueError(f"Invalid use_depth={use_depth!r}. Expected one of {sorted(VALID_DEPTH_CHANNELS)} or None.")
    normalized_use_depth = use_depth
    depth_in_chns = 1 if normalized_use_depth == 13 else (normalized_use_depth or 0)
    spec = get_strategy_spec(way)

    if spec.in_chns == "metadata":
        resolved_in_chns = metadata_in_chns
    elif spec.in_chns is None:
        resolved_in_chns = metadata_in_chns + depth_in_chns
    else:
        resolved_in_chns = int(spec.in_chns)

    return {
        "metadata_in_chns": metadata_in_chns,
        "use_depth": normalized_use_depth,
        "in_chns": resolved_in_chns,
    }


def create_strategy(name, args, model, optimizer, device, scaler=None):
    strategy_cls = STRATEGY_REGISTRY[_canonicalize_strategy_name(name)]
    return strategy_cls(args, model, optimizer, device, scaler=scaler)


def add_strategy_args(parser, name):
    strategy_cls = STRATEGY_REGISTRY[_canonicalize_strategy_name(name)]
    if hasattr(strategy_cls, "add_args"):
        strategy_cls.add_args(parser)
