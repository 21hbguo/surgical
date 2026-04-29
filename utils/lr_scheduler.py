class PolyLRScheduler:
    def __init__(self, optimizer, max_iterations, base_lr, warmup_iters=0, warmup_ratio=0.1, min_lr_ratio=0.001, power=0.9):
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.power = power
        self.warmup_lr_start = base_lr * warmup_ratio

    def step(self, iter_num):
        if iter_num < self.warmup_iters:
            alpha = iter_num / self.warmup_iters
            lr = self.warmup_lr_start + alpha * (self.base_lr - self.warmup_lr_start)
        else:
            progress = (iter_num - self.warmup_iters) / (self.max_iterations - self.warmup_iters)
            lr = max(self.base_lr * ((1 - progress) ** self.power), self.min_lr_ratio * self.base_lr)

        for param_group in self.optimizer.param_groups:
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = lr * scale
        return lr

    def state_dict(self):
        return {
            "max_iterations": self.max_iterations,
            "base_lr": self.base_lr,
            "warmup_iters": self.warmup_iters,
            "warmup_ratio": self.warmup_ratio,
            "min_lr_ratio": self.min_lr_ratio,
            "power": self.power,
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return
        for key in ["max_iterations", "base_lr", "warmup_iters", "warmup_ratio", "min_lr_ratio", "power"]:
            if key in state_dict:
                setattr(self, key, state_dict[key])
        self.warmup_lr_start = self.base_lr * self.warmup_ratio


def build_lr_scheduler(optimizer, args):
    return PolyLRScheduler(
        optimizer,
        args.max_iterations,
        args.lr,
        args.lr_warmup_iters,
        args.lr_warmup_ratio,
        args.lr_min_ratio,
        args.poly_power,
    )
