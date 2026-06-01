# 策略注册表：统一维护策略名到实现类的映射。

from .base_strategy import BaseTrainingStrategy
from .fully_dformerv2 import DFormerv2FullyStrategy
from .fully_contrast_v1 import FullyContrastV1Strategy
from .fully_contrast_v1_1 import FullyContrastV11Strategy
from .fully_rgb_masking_depth_v1 import FullyRGBMaskingDepthV1Strategy
from .fully_supervised import FullySupervisedStrategy
from .fully_depth_pretrain_v1 import FullyDepthPretrainStrategy
from .fully_ternaus import TernausNet16Strategy
from .semi_depth_guided_mt import DepthGuidedMTStrategy
from .semi_rdnet import RDNetStrategy
from .semi_dycon import DyConStrategy
from .semi_mean_teacher import MeanTeacherStrategy
from .semi_mean_teacher_text_v1 import MeanTeacherTextV1Strategy
from .semi_mt_depth_teacher_v1 import MTDepthTeacherV1Strategy
from .semi_mt_depth_guider_v1 import MTDepthGuiderV1Strategy
from .semi_mt_depth_guider_v2 import MTDepthGuiderV2Strategy
from .semi_mt_depth_guider_v3 import MTDepthGuiderV3Strategy
from .semi_mt_depth_guider_proto_v1 import MTDepthGuiderProtoV1Strategy
from .semi_mt_depth_guider_proto_teacher_v1 import MTDepthGuiderProtoTeacherV1Strategy
from .semi_mt_depth_guider_proto_teacher_v2 import MTDepthGuiderProtoTeacherV2Strategy
from .semi_mt_depth_guider_proto_teacher_v3 import MTDepthGuiderProtoTeacherV3Strategy
from .semi_mean_teacher_contrast_v1 import MeanTeacherContrastV1Strategy
from .semi_proto_v1 import ProtoV1Strategy
from .semi_uncertainty_mt import UncertaintyMTStrategy
from .semi_urpc import URPCStrategy
from .semi_only_depthInput import OnlyDepthInputStrategy
from .semi_w2s import W2SStrategy
from .semi_segmatch import SegMatchStrategy
from .semi_unimatch import UniMatchStrategy
from .semi_cps import CPSStrategy
from .semi_unimatch_official import UniMatchOfficialStrategy
from .semi_segmatch_official import SegMatchOfficialStrategy
from .semi_mms import MMSStrategy
from .specs import STRATEGY_REGISTRY, create_strategy

__all__ = [
    "BaseTrainingStrategy",
    "STRATEGY_REGISTRY",
    "create_strategy",
    "FullySupervisedStrategy",
    "FullyRGBMaskingDepthV1Strategy",
    "FullyDepthPretrainStrategy",
    "FullyContrastV1Strategy",
    "FullyContrastV11Strategy",
    "MeanTeacherTextV1Strategy",
    "MeanTeacherContrastV1Strategy",
    "MTDepthTeacherV1Strategy",
    "MTDepthGuiderV1Strategy",
    "MTDepthGuiderV2Strategy",
    "MTDepthGuiderV3Strategy",
    "MTDepthGuiderProtoV1Strategy",
    "MTDepthGuiderProtoTeacherV1Strategy",
    "MTDepthGuiderProtoTeacherV2Strategy",
    "MTDepthGuiderProtoTeacherV3Strategy",
    "UncertaintyMTStrategy",
    "URPCStrategy",
    "TernausNet16Strategy",
    "ProtoV1Strategy",
    "DyConStrategy",
    "W2SStrategy",
    "DepthGuidedMTStrategy",
    "RDNetStrategy",
    "DFormerv2FullyStrategy",
    "OnlyDepthInputStrategy",
    "SegMatchStrategy",
    "UniMatchStrategy",
    "CPSStrategy",
]
