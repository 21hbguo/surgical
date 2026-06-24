import argparse
import pprint

from models.factory import resolve_default_model_name
from strategies.specs import add_strategy_args, get_strategy_names, is_semi_strategy, resolve_strategy_input_settings
from utils.common import get_n_folds_from_path, get_num_classes_from_path, resolve_runtime_path


def _str2bool(value):
    return value if isinstance(value, bool) else {"1": True, "true": True, "yes": True, "y": True, "on": True, "0": False, "false": False, "no": False, "n": False, "off": False}.get(str(value).strip().lower(), (_ for _ in ()).throw(argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")))


def infer_root_path_from_exp(exp):
    return f"../data/{exp.split('/')[0]}" if exp and "/" in exp else None


def add_normalize_suffix(name, normalize_method):
    suffix_map = {"imagenet": "_imagenet", "minmax": "_minmax", "255": "_255"}
    suffix = suffix_map.get(normalize_method, "")
    if not suffix or name.endswith(suffix):
        return name
    return f"{name.split('/')[0]}{suffix}/{name.split('/', 1)[1]}" if "/" in name else f"{name}{suffix}"


def _finalize_common_args(args):
    args.way = str(args.way).lower()
    raw_exp = args.exp
    inferred_root = infer_root_path_from_exp(raw_exp)
    if not inferred_root:
        raise ValueError(f"Cannot infer dataset root from --exp={raw_exp!r}. Expected format: DATASET/Experiment, e.g. endovis2018ISINet/Fully.")
    args.root_path = inferred_root
    if args.exp is not None:
        args.exp = add_normalize_suffix(args.exp, args.normalize)
    args.num_classes = get_num_classes_from_path(args.root_path, args.task)
    args.num_folds = get_n_folds_from_path(args.root_path, args.task)
    if args.model is None:
        args.model = resolve_default_model_name(args.way, args.pretrain)
    args.pretrain_root = resolve_runtime_path(args.pretrain_root)
    args.depth_pretrain_path = resolve_runtime_path(args.depth_pretrain_path)
    args.dformerv2_pretrain_path = resolve_runtime_path(args.dformerv2_pretrain_path)
    args.dinov3_repo_dir = resolve_runtime_path(args.dinov3_repo_dir)
    if args.pretrain == "depth":
        args.pretrain_root = args.depth_pretrain_path
    input_settings = resolve_strategy_input_settings(way=args.way, root_path=args.root_path, task=args.task, use_depth=args.use_depth)
    args.in_chns = input_settings["in_chns"]
    args.use_depth = input_settings["use_depth"]
    args.is_semi_supervised = is_semi_strategy(args.way)
    return args


def finalize_train_args(args):
    args = _finalize_common_args(args)
    args.train_result_root = resolve_runtime_path(args.result_root)
    args.snapshot_path = resolve_runtime_path(args.snapshot_path)
    if args.pth is not None:
        args.pth = "final" if args.pth == "latest" else args.pth
    return args


def finalize_test_args(args):
    args = _finalize_common_args(args)
    args.predict_result_root = resolve_runtime_path(args.result_root)
    args.train_result_root = resolve_runtime_path(args.train_result_root)
    args.snapshot_path = resolve_runtime_path(args.snapshot_path)
    requested = args.requested_checkpoint_type
    requested = "final" if requested == "latest" else requested
    args.requested_checkpoint_type = args.checkpoint_type = "final" if args.no_val else requested
    args.pth = requested
    return args


def format_args_for_logging(args):
    data = vars(args)
    return pprint.pformat({"common": {k: data[k] for k in ["root_path", "task", "exp", "way", "model", "pretrain", "num_classes", "num_folds", "in_chns", "use_depth", "depth_uint", "normalize", "device", "seed"]}, "train": {k: data[k] for k in ["optimizer", "lr", "max_iterations", "val_iter", "sampling", "snapshot_path", "train_result_root", "amp", "compile", "early_stopping", "retrain"] if k in data}, "test": {k: data[k] for k in ["requested_checkpoint_type", "checkpoint_type", "batch_size", "post_resize", "predict_result_root"] if k in data}}, indent=2, width=100)


class StrategyArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)
        self._strategy_args_added = False

    def _resolve_strategy_from_argv(self, args):
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--way", type=str, default="fully")
        known, _ = probe.parse_known_args(args)
        return str(known.way).lower()

    def _ensure_strategy_args(self, args=None):
        if self._strategy_args_added:
            return
        way = self._resolve_strategy_from_argv(args)
        add_strategy_args(self, way)
        self._strategy_args_added = True

    def parse_args(self, args=None, namespace=None):
        self._ensure_strategy_args(args)
        return super().parse_args(args, namespace)

    def parse_known_args(self, args=None, namespace=None):
        self._ensure_strategy_args(args)
        return super().parse_known_args(args, namespace)


def add_common_args(parser, result_root_default, test_mode=False):
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3], help="task id used for task{n}.json")
    parser.add_argument("--exp", type=str, default="endovis2017/default_exp", help="experiment name")
    parser.add_argument("--model", type=str, default=None, help="model type (auto from strategy if not specified)")
    parser.add_argument("--way", type=str, default="fully", choices=get_strategy_names(), help="training strategy name")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam"], help="optimizer name")
    parser.add_argument("--pretrain", type=str, default="none", choices=["none", "resnet", "depth", "dinov3"], help="strategy model map selector")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--fold", type=str if test_mode else int, nargs="*" if test_mode else None, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snapshot_path", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=result_root_default, help="current mode result root")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--in_chns", type=int, default=None)
    parser.add_argument("--filter_num", type=int, default=16, help="filter number for UNet-family models")
    parser.add_argument("--resize_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--use_depth", type=int, default=None, choices=[1, 3, 13], help="Depth mode: 1=depth1c input, 3=depth3c input, 13=load both depth1c+depth3c while model input keeps depth1c channels.")
    parser.add_argument("--depth_uint", type=int, default=16, choices=[8, 16], help="Depth folder bit-width suffix. Reads depth{c}c_slices_uint{depth_uint}.")
    parser.add_argument("--normalize", type=str, default="255", choices=["minmax", "255", "imagenet"])
    parser.add_argument("--strong", type=str, default="s", nargs="?", const=None, choices=["s", "t"], help="Noise location: t=teacher, s=student, --strong without value=none")
    parser.add_argument("--load_mode", type=str, default="data", choices=["data", "path"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resnet_variant", type=str, default="resnet34")
    parser.add_argument("--pretrain_root", type=str, default="../pre_train_ckp/")
    parser.add_argument("--depth_pretrain_path", type=str, default="../pre_train_ckp/resnet34-depth-pretrain.pth")
    parser.add_argument("--model_pretrain", type=_str2bool, default=True)
    parser.add_argument("--dinov3_repo_dir", type=str, default="../dinov3")
    parser.add_argument("--dinov3_weights", type=str, default="dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    parser.add_argument("--dformerv2_pretrain_path", type=str, default="../pre_train_ckp/DFormerv2_Small_pretrained.pth")
    parser.add_argument("--dformerv2_scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency_rampup", type=int, default=150)
    parser.add_argument("--consistency_rampup_div", type=int, default=200)
    parser.add_argument("--consistency_start_iters", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--data-format", type=str, default="h5", choices=["png", "h5"], help="data storage format.")
    parser.add_argument("--labeled_num", type=float, default=10, help="Labeled percentage. Examples: 0.1=0.1%%, 1=1%%, 10=10%%.")
    parser.add_argument("--labeled_bs", type=int, default=2, help="Number of labeled samples per training batch.")
    parser.add_argument("--unlabeled_bs", type=int, default=2, help="Number of unlabeled samples per training batch.")
    parser.add_argument("--sampling", type=str, default="interval", choices=["none", "interval"], help="Train subset sampling rule when labeled_num selects only part of the train set.")
    parser.add_argument("--max_iterations", type=int, default=30000, help="Total training iterations.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_val", action="store_true", default=False)

def add_train_args(parser):
    parser.add_argument("--val_iter", type=int, default=400, help="Run validation every N iterations.")
    parser.add_argument("--poly_power", type=float, default=0.9, help="Power factor for poly learning-rate decay.")
    parser.add_argument("--lr_scheduler", type=str, default="poly", help="Learning-rate scheduler name.")
    parser.add_argument("--lr_warmup_iters", type=int, default=0, help="Warmup iterations before the main scheduler starts.")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0, help="Initial warmup lr ratio relative to base lr.")
    parser.add_argument("--lr_min_ratio", type=float, default=0, help="Minimum lr ratio relative to base lr.")
    parser.add_argument("--use_checkpoint", action="store_true", default=False, help="Enable model checkpointing features supported by the model implementation.")
    parser.add_argument("--retrain", action="store_true", default=False, help="Retrain even when the target snapshot directory already contains completed checkpoints.")
    parser.add_argument("--freeze", action="store_true", default=False, help="Freeze supported pretrained backbone parameters before training.")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision training")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile for model acceleration")
    parser.add_argument("--debug", action="store_true", default=False, help="Run with reduced train and val subsets for quick debugging.")
    parser.add_argument("--early_stopping", type=float, default=0.3, help="Early stopping patience as fraction of max_iterations (e.g. 0.3). 0=disabled.")
    parser.add_argument("--pth", type=str, default=None, choices=["best", "final", "latest"], help=argparse.SUPPRESS)


def add_test_args(parser):
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint-type", "--pth", "--pth_type", dest="requested_checkpoint_type", type=str, default="best", choices=["best", "final", "latest"])
    parser.add_argument("--rgb", type=int, default=4, choices=[0, 1, 2, 3, 4], help="0=off, 1=pred overlay, 2=label/pred side-by-side, 3=label only, 4=pred mask only")
    parser.add_argument("--distance_metrics", type=int, default=0, choices=[0, 1])
    parser.add_argument("--post_resize", type=str, default="label_to_pred", choices=["label_to_pred", "pred_to_label"])
    parser.add_argument("--train_result_root", type=str, default="../result_train", help="train checkpoint root used by test")


def build_train_parser():
    parser = StrategyArgumentParser()
    add_common_args(parser, result_root_default="../result_train", test_mode=False)
    add_train_args(parser)
    return parser


def build_test_parser():
    parser = StrategyArgumentParser()
    add_common_args(parser, result_root_default="../result_predict", test_mode=True)
    add_test_args(parser)
    return parser


def build_test_feature_parser():
    parser = build_test_parser()
    parser.add_argument("--feat_vis", type=int, default=0, choices=[0, 1], help="enable feature map visualization")
    parser.add_argument("--feat_vis_method", type=str, default="gradcam", choices=["gradcam"], help="feature visualization method")
    parser.add_argument("--feat_vis_layer", type=str, default="", help="name filter for feature/gradcam layers")
    parser.add_argument("--feat_vis_all_layers", type=int, default=1, choices=[0, 1], help="aggregate all matched layers for gradcam")
    parser.add_argument("--feat_vis_max_layers", type=int, default=0, help="0 means no limit; otherwise keep last N matched layers")
    parser.add_argument("--feat_vis_target_class", type=int, default=-1, help="gradcam target class; -1 uses auto non-background class")
    parser.add_argument("--feat_vis_max_cases", type=int, default=10, help="max cases to export per fold")
    parser.add_argument("--feat_vis_alpha", type=float, default=0.45, help="overlay alpha for feature visualization")
    parser.add_argument("--conf_vis", type=int, default=0, choices=[0, 1], help="enable per-class confidence heatmap visualization")
    parser.add_argument("--conf_vis_alpha", type=float, default=0.45, help="overlay alpha for confidence visualization")
    return parser
