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
    args.way = str(getattr(args, "way", "fully")).lower()
    if args.way == "fully" and args.exp and "/" in args.exp:
        exp_way = args.exp.split("/", 1)[1].split("/", 1)[0].strip().lower()
        if exp_way in get_strategy_names():
            args.way = exp_way
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
    if hasattr(args, "pth") and args.pth is not None:
        args.pth = "final" if args.pth == "latest" else args.pth
    return args


def finalize_test_args(args):
    args = _finalize_common_args(args)
    args.predict_result_root = resolve_runtime_path(args.result_root)
    args.train_result_root = resolve_runtime_path(args.train_result_root)
    args.snapshot_path = resolve_runtime_path(args.snapshot_path)
    requested = getattr(args, "requested_checkpoint_type", None) or getattr(args, "pth", None) or "best"
    requested = "final" if requested == "latest" else requested
    args.requested_checkpoint_type = args.checkpoint_type = "final" if args.no_val else requested
    args.pth = requested
    return args


def format_args_for_logging(args):
    return pprint.pformat({"common": {k: getattr(args, k, None) for k in ["root_path", "task", "exp", "way", "model", "pretrain", "num_classes", "num_folds", "in_chns", "use_depth", "normalize", "device", "seed"]}, "train": {k: getattr(args, k, None) for k in ["optimizer", "lr", "max_iterations", "val_iter", "sampling", "snapshot_path", "train_result_root"]}, "test": {k: getattr(args, k, None) for k in ["requested_checkpoint_type", "checkpoint_type", "batch_size", "predict_result_root"]}}, indent=2, width=100)


class StrategyArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)
        self._strategy_args_added = False

    def _resolve_strategy_from_argv(self, args):
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--way", type=str, default="fully")
        probe.add_argument("--exp", type=str, default="endovis2017/default_exp")
        known, _ = probe.parse_known_args(args)
        way = str(getattr(known, "way", "fully")).lower()
        exp = getattr(known, "exp", None)
        if way == "fully" and exp and "/" in exp:
            exp_way = exp.split("/", 1)[1].split("/", 1)[0].strip().lower()
            if exp_way in get_strategy_names():
                way = exp_way
        return way

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
    parser.add_argument("--pretrain", type=str, default="resnet", choices=["none", "resnet", "depth", "dinov3"], help="strategy model map selector")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--fold", type=str if test_mode else int, nargs="*" if test_mode else None, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snapshot_path", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=result_root_default, help="current mode result root")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--in_chns", type=int, default=None)
    parser.add_argument("--filter_num", type=int, default=16, help="filter number for UNet-family models")
    parser.add_argument("--resize_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--use_depth", type=int, default=None, choices=[1, 3, 13], help="Depth mode: 1=depth1c input, 3=depth3c input, 13=load both depth1c+depth3c while model input keeps depth1c channels.")
    parser.add_argument("--normalize", type=str, default="255", choices=["minmax", "255", "imagenet"])
    parser.add_argument("--strong", type=str, default="s", help="Noise location: t=teacher, s=student, empty=none")
    parser.add_argument("--load_mode", type=str, default="data", choices=["data", "path"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
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


def add_train_args(parser):
    parser.add_argument("--labeled_num", type=float, default=10, help="Labeled percentage. Examples: 0.1=0.1%%, 1=1%%, 10=10%%.")
    parser.add_argument("--labeled_bs", type=int, default=2)
    parser.add_argument("--unlabeled_bs", type=int, default=2)
    parser.add_argument("--sampling", type=str, default="none", choices=["none", "interval"])
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--val_iter", type=int, default=300)
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="poly")
    parser.add_argument("--lr_warmup_iters", type=int, default=0)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0)
    parser.add_argument("--lr_min_ratio", type=float, default=0.0)
    parser.add_argument("--use_checkpoint", action="store_true", default=False)
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no_val", action="store_true", default=False)
    parser.add_argument("--pth", type=str, default=None, choices=["best", "final", "latest"], help=argparse.SUPPRESS)


def add_test_args(parser):
    parser.add_argument("--labeled_num", type=float, default=10, help="Labeled percentage. Examples: 0.1=0.1%%, 1=1%%, 10=10%%.")
    parser.add_argument("--sampling", type=str, default="none", choices=["none", "interval"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint-type", "--pth", "--pth_type", dest="requested_checkpoint_type", type=str, default="best", choices=["best", "final", "latest"])
    parser.add_argument("--rgb", type=int, default=2, choices=[0, 1, 2], help="0=off, 1=pred overlay, 2=label/pred side-by-side overlays")
    parser.add_argument("--no_val", action="store_true", default=False)
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
