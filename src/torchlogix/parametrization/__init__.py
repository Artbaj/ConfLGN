"""LUT parametrization strategies for logic neural networks.
This module provides different ways to parametrize Look-Up Tables (LUTs)
in logic neural networks, using either raw truth table logits,
Walsh-Hadamard basis coefficients, or the indicator-polynomial ('light') basis.
"""

from .base import LUTParametrization
from .raw import RawLUTParametrization
from .warp import WarpLUTParametrization
from .light import LightLUTParametrization


def setup_parametrization(parametrization: str, lut_rank: int, **parametrization_kwargs):
    param_dict = {
        "raw": RawLUTParametrization,
        "warp": WarpLUTParametrization,
        "light": LightLUTParametrization
    }
    if parametrization not in param_dict:
        raise ValueError(
            f"Unsupported parametrization: {parametrization}. "
            f"Choose from {list(param_dict.keys())}."
        )
    param_cls = param_dict[parametrization]
    return param_cls(lut_rank, **parametrization_kwargs)


__all__ = [
    "LUTParametrization",
    "LightLUTParametrization",
    "RawLUTParametrization",
    "WarpLUTParametrization",
    "setup_parametrization",
]
