from abc import ABC, abstractmethod

import torch


class Connections(torch.nn.Module, ABC):
    """Abstract base class for connection strategies."""
    def __init__(
            self,
            lut_rank=2,
            device=None,
            init_method="random",
            **kwargs
        ):
        super().__init__()
        self.lut_rank = lut_rank
        self.device = device
        self.init_method = init_method

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _init_connections(self):
        pass

    def update_temperature(self, temperature: float):
        pass
