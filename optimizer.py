import enum
from typing import Tuple

import torch.optim as optim


class OptimizerConfig(enum.Enum):
    """A simple wrapper for pytorch Adam and SGD optimizers."""

    Adam = optim.Adam
    AdamW = optim.AdamW
    SGD = optim.SGD

    def __init__(self, *args, **kwargs) -> None:
        self._opts = {}

    def get_optimizer(self, params) -> optim.Optimizer:
        return self.value(params, **self._opts)

    def set_lr(self, lr: float) -> None:
        self._opts["lr"] = lr

    def set_decay(self, decay: float) -> None:
        self._opts["weight_decay"] = decay

    def set_betas(self, betas: Tuple[float, float]) -> None:
        if self.name != "Adam":
            raise KeyError(f"Invalid argument for {self.name}: betas")
        self._opts["betas"] = betas

    def set_eps(self, eps: float) -> None:
        if self.name != "Adam":
            raise KeyError(f"Invalid argument for {self.name}: eps")
        self._opts["eps"] = eps

    def set_amsgrad(self, amsgrad: bool) -> None:
        if self.name != "Adam":
            raise KeyError(f"Invalid argument for {self.name}: amsgrad")
        self._opts["amsgrad"] = amsgrad

    def set_momentum(self, momentum: float) -> None:
        if self.name != "SGD":
            raise KeyError(f"Invalid argument for {self.name}: momentum")
        self._opts["momentum"] = momentum

    def set_dampening(self, dampening: float) -> None:
        if self.name != "SGD":
            raise KeyError(f"Invalid argument for {self.name}: dampening")
        self._opts["dampening"] = dampening

    def set_nesterov(self, nesterov: bool) -> None:
        if self.name != "SGD":
            raise KeyError(f"Invalid argument for {self.name}: nesterov")
        self._opts["nesterov"] = nesterov
