"""Indicator-polynomial ('light') basis LUT parametrization."""
import torch

from .base import LUTParametrization
from ..functional import gumbel_sigmoid, light_basis_hard, sigmoid, weighted_light_basis_sum


class LightLUTParametrization(LUTParametrization):
    """Light DGN parametrization using basis coefficients.

    This parametrization represents Boolean functions via positive
    basis coefficients, which are mapped to [0,1] with sigmoid.
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
        if lut_rank not in [2, 4, 6]:
            raise ValueError(
                f"Hard-coded Light basis only supports lut_rank in [2, 4, 6], got {lut_rank}"
            )

    @torch.compiler.disable
    def init_weights(
        self,
        num_neurons: int,
        device: str
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if self.weight_init == "residual":
            weights = torch.randn(num_neurons, lut_entries, device=device)
            weights[:, :lut_entries // 2] -= 3
            weights[:, lut_entries // 2:] += 3
            return weights
        elif self.weight_init == "random":
            return torch.rand(num_neurons, lut_entries, device=device)
        else:
            raise ValueError(self.weight_init)
    

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
            contraction: Einsum pattern for combining basis with weights

        Returns:
            Output with lut_rank dimension reduced
        """
        # Handle 1D weight case (single neuron) for backward compatibility
        if weight.ndim == 1:
            weight = weight.unsqueeze(0)

        if training:
            w = self._sample_train(weight)
        else:
            w = self._sample_eval(weight)

        if self.materialize_basis:
            # add 'k' dimension for basis entries (e.g. 'n,bn->bn' becomes 'nk,bnk->bn')
            contraction_with_basis_dim = contraction.replace(',', 'k,').replace('->', 'k->')
            x = light_basis_hard(x, self.lut_rank)
            x = torch.einsum(contraction_with_basis_dim, w, x)
        else:
            x = weighted_light_basis_sum(x, w, contraction, self.lut_rank)
        return x


    def _sample_train(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid/gumbel_sigmoid based on forward_sampling mode."""
        if self.forward_sampling == "soft":
            return sigmoid(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "hard":
            return sigmoid(weights, tau=self.temperature, hard=True)
        elif self.forward_sampling == "gumbel_soft":
            return gumbel_sigmoid(weights, tau=self.temperature, hard=False)
        elif self.forward_sampling == "gumbel_hard":
            return gumbel_sigmoid(weights, tau=self.temperature, hard=True)

    def _sample_eval(self, weights: torch.Tensor) -> torch.Tensor:
        """Threshold at 0 for discrete output."""
        return (weights > 0).to(dtype=torch.float32)

    def get_luts(self, weight: torch.Tensor) -> torch.Tensor:
        luts = weight > 0
        return luts

    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lut_rank <= 4, "LUT IDs only supported for lut_rank <= 4 due to combinatorial explosion."
        luts = self.get_luts(weight)

        ids = 2 ** torch.arange(self.lut_entries - 1, -1, -1, device=luts.device)
        ids = (luts * ids.unsqueeze(0)).sum(dim=1)

        return luts, ids
