"""Raw truth-table-logit LUT parametrization."""
import math

import torch
import torch.nn.functional as F

from .base import LUTParametrization
from ..functional import compute_all_logic_ops_vectorized, softmax, weighted_raw_basis_sum


class RawLUTParametrization(LUTParametrization):
    """Raw LUT parametrization using direct truth table logits.

    This parametrization directly assigns logits to each possible truth table
    entry, then samples via softmax. Each LUT of rank n has 2^(2^n) possible
    Boolean functions.
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
        super().__init__(
            lut_rank,
            forward_sampling,
            temperature,
            weight_init,
            residual_probability,
            materialize_basis
        )
        if lut_rank != 2:
            raise ValueError("Raw parametrization currently only supports lut_rank=2")
        # Number of possible Boolean functions (not just truth table entries)
        self.num_functions = 1 << self.lut_entries

    @torch.compiler.disable
    def init_weights(
        self,
        num_neurons: int,
        device: str
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if self.weight_init == "residual":
            # all weights to 0 except for weight number (1 << (1 << (self.lut_rank - 1))) - 1, 
            # which is set to
            value = math.log((2**lut_entries - 1) * self.residual_probability 
                             - 2**(lut_entries - 1) + 1) - math.log(1 - self.residual_probability)
            weights = torch.zeros((num_neurons, 1 << lut_entries), device=device)
            weights[:, (1 << (1 << (self.lut_rank - 1))) - 1] = value * self.temperature
            return weights
        elif self.weight_init == "random" or self.weight_init=="gaussian":
            return torch.randn(num_neurons, 1 << lut_entries, device=device)
        raise ValueError(f"Unknown weight_init: {self.weight_init}")


    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        training: bool,
        contraction: str
    ) -> torch.Tensor:
        """Layer-agnostic forward pass.

        Args:
            x: Extracted inputs, shape (batch, lut_rank, ...)
            weight: Weight parameters
            training: Whether in training mode
            contraction: Einsum pattern for combining ops with weights

        Returns:
            Output with lut_rank dimension reduced
        """
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)

        # Extract inputs
        a, b = x[:, 0], x[:, 1]

        # Sample weights (merged from SoftmaxSampler)
        if training:
            w = self._sample_train(weight)
        else:
            w = self._sample_eval(weight)

        # Handle 1D weight case (single neuron) for backward compatibility
        if w.ndim == 1:
            w = w.unsqueeze(0)

        if self.materialize_basis:
            # add 'k' dimension for basis entries (e.g. 'n,bn->bn' becomes 'nk,bnk->bn')
            contraction_with_basis_dim = contraction.replace(',', 'k,').replace('->', 'k->')
            ops = compute_all_logic_ops_vectorized(a, b)  # Shape: (..., 16)
            return torch.einsum(contraction_with_basis_dim, w, ops)

        return weighted_raw_basis_sum(a, b, w, contraction)
    

    def _sample_train(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply softmax/gumbel_softmax based on forward_sampling mode."""
        if self.forward_sampling == "soft":
            return softmax(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "hard":
            return softmax(weights, tau=self.temperature, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return F.gumbel_softmax(weights, tau=self.temperature, hard=False, dim=-1)
        elif self.forward_sampling == "gumbel_hard":
            return F.gumbel_softmax(weights, tau=self.temperature, hard=True, dim=-1)

    def _sample_eval(self, weights: torch.Tensor) -> torch.Tensor:
        """Select most probable LUT via argmax + one-hot."""
        return F.one_hot(
            weights.argmax(-1), self.num_functions
        ).to(torch.float32)

    def get_luts(self, weight: torch.Tensor) -> torch.Tensor:
        ids = weight.argmax(axis=1)
        luts = ((ids.unsqueeze(-1) >> torch.arange(self.lut_entries, device=ids.device)) & 1).flip(1)
        return luts

    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = weight.argmax(axis=1)
        luts = ((ids.unsqueeze(-1) >> torch.arange(self.lut_entries, device=ids.device)) & 1).flip(1)
        return luts, ids


