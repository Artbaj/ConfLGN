"""Base class for LUT parametrization strategies."""
from abc import ABC, abstractmethod

import torch


class LUTParametrization(ABC):
    """Base class for LUT parametrization strategies.

    A parametrization defines how to represent Boolean functions (LUTs)
    as learnable parameters, how to initialize them, and how to compute
    outputs from inputs.
    """

    def __init__(
        self, 
        lut_rank: int, 
        forward_sampling: str = "soft",
        temperature: float = 1.0,
        weight_init: str = "residual",
        residual_probability: float = 0.951,
        materialize_basis: bool = False
    ):
        """Initialize parametrization.

        Args:
            lut_rank: Number of inputs per logic gate (arity of the LUT).
            forward_sampling: Sampling strategy during forward pass. One of:
                - "soft": Continuous relaxation via softmax/sigmoid
                - "hard": Straight-through hard selection
                - "gumbel_soft": Gumbel-Softmax/Sigmoid continuous relaxation
                - "gumbel_hard": Gumbel-Softmax/Sigmoid straight-through
            temperature: Temperature parameter for sampling operations.
            weight_init: Initialization strategy for weights ("residual" or "random").
            residual_probability: Probability controlling residual initialization.
            materialize_basis: Whether to materialize the 2^lut_rank-dimensional basis
                and calculate the sum via the scalar product with weights (strongly discouraged).
                If False, the weighted sum is calculated in-place, saving memory.
        """
        self.lut_rank = lut_rank
        self.lut_entries = 1 << lut_rank

        # Validate and store sampling configuration
        valid_modes = ["soft", "hard", "gumbel_soft", "gumbel_hard"]
        if forward_sampling not in valid_modes:
            raise ValueError(
                f"forward_sampling must be one of {valid_modes}, got {forward_sampling}"
            )
        self.forward_sampling = forward_sampling
        self.temperature = temperature
        self.weight_init = weight_init
        self.residual_probability = residual_probability
        self.materialize_basis = materialize_basis

    @abstractmethod
    def init_weights(
        self,
        num_neurons: int,
        device: str
    ) -> torch.Tensor:
        """Initialize weights for this parametrization.

        Args:
            num_neurons: Number of neurons/kernels.
            weight_init: Initialization strategy ("residual" or "random").
            residual_probability: Probability controlling residual initialization.
            device: Device to allocate weights on.

        Returns:
            Initialized weight tensor.
        """
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        training: bool,
        contraction: str
    ) -> torch.Tensor:
        """Compute forward pass (layer-agnostic).

        Args:
            x: Extracted inputs with lut_rank at dimension 1, shape (batch, lut_rank, ...)
            weight: Weight parameters
            training: Whether in training mode
            contraction: Einsum pattern specifying how to combine basis/ops with weights
                        (e.g., 'bnk,nk->bn' for dense, 'bcsfk,fck->bcsf' for conv)

        Returns:
            Output tensor with lut_rank dimension reduced
        """
        pass

    @abstractmethod
    def get_luts(self, weight: torch.Tensor) -> torch.Tensor:
        """Extract LUT truth tables from weights.

        Args:
            weight: Weight parameters.

        Returns:
                - luts: Boolean tensor of truth tables
        """
        pass

    @abstractmethod
    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract LUT truth tables and IDs from weights.

        Args:
            weight: Weight parameters.

        Returns:
            Tuple of (luts, ids) where:
                - luts: Boolean tensor of truth tables
                - ids: Integer tensor of LUT IDs (or None if not applicable)
        """
        pass

    def update_temperature(self, temperature: float):
        self.temperature = temperature
    

