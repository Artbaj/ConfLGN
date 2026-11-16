from .base import Connections
from .fixed_connections import FixedConvConnections, FixedDenseConnections, get_random_unique_connections
from .learnable_connections import LearnableConnectionFunction, LearnableDenseConnections


def setup_connections(
    connections: str,
    structure: str,
    lut_rank: int,
    device: str = None,
    **connections_kwargs
):

    """Factory method to create connection modules."""
    if structure == "dense":
        if connections == "fixed":
            return FixedDenseConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        elif connections == "learnable":
            return LearnableDenseConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        else:
            raise ValueError(f"Unknown connections method: {connections}")
    elif structure == "conv":
        if connections == "fixed":
            return FixedConvConnections(
                lut_rank=lut_rank,
                device=device,
                **connections_kwargs
            )
        else:
            raise ValueError(f"Unknown connections method: {connections}")
    else:
        raise ValueError(f"Unknown structure method: {structure}")


__all__ = [
    "Connections",
    "FixedConvConnections",
    "FixedDenseConnections",
    "LearnableConnectionFunction",
    "LearnableDenseConnections",
    "get_random_unique_connections",
    "setup_connections",
]
