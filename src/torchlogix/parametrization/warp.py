"""Walsh-Hadamard basis ('warp') LUT parametrization."""
import math

import torch

from .base import LUTParametrization
from ..functional import gumbel_sigmoid, sigmoid, walsh_basis_hard, walsh_hadamard_transform, weighted_walsh_basis_sum


class WarpLUTParametrization(LUTParametrization):
    """Walsh-Hadamard parametrization using basis coefficients.

    This parametrization represents Boolean functions via Walsh-Hadamard
    basis coefficients, which provides a smoother optimization landscape
    and exploits structure in Boolean functions (parity, correlations).
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
        if lut_rank not in [1, 2, 4, 6]:
            raise ValueError(
                f"Hard-coded Walsh basis only supports lut_rank in [1, 2, 4, 6], got {lut_rank}"
            )

    @torch.compiler.disable
    def init_weights(
        self,
        num_neurons: int,
        device: str,
    ) -> torch.Tensor:
        lut_entries = 1 << self.lut_rank
        if self.weight_init == "residual":
            weights = torch.empty((num_neurons, lut_entries), device=device)
            # identity representation, corresponds to Boolean function, which maps MSB (last single variable) to itself
            identity = 1 - 2 * torch.cat([torch.zeros(lut_entries // 2), torch.ones(lut_entries - lut_entries // 2)]).to(dtype=torch.int32, device=device)
            transformed_identity = (1 / lut_entries) * walsh_hadamard_transform(identity, self.lut_rank, dtype=torch.int32, device=device)
            # transformation for comparibility with raw parametrization
            c_walsh = self.temperature * (math.log(self.residual_probability) - math.log(1 - self.residual_probability))
            weights[:] = c_walsh * transformed_identity.to(torch.float)
            # add random noise proportional to temperature (breaks the tests but helps optimization by breaking symmetry between neurons)
            # weights += self.temperature * 0.1 * torch.randn(num_neurons, lut_entries, device=device)
            return weights
        elif self.weight_init == "residual-catalog":
            identity_truth_table = 1 - 2 * torch.cat([torch.zeros(lut_entries // 2), torch.ones(lut_entries - lut_entries // 2)]).to(dtype=torch.int32, device=device)
            identity_coefficients = (1 / lut_entries) * walsh_hadamard_transform(identity_truth_table, self.lut_rank, dtype=torch.int32, device=device)
            
            # build weights tensor such that all gates are identity initially
            weights = torch.empty((num_neurons, lut_entries), device=device)
            weights[:] = identity_coefficients

            # set 1 - residual_probability fraction of weights to random different functions
            n_different = round(num_neurons * (1 - self.residual_probability))
            random_truth_tables = torch.randint(0, 2, (n_different, lut_entries), device=device)
            random_truth_tables = 1 - 2 * random_truth_tables.to(dtype=torch.int32)
            random_coefficients = (1 / lut_entries) * walsh_hadamard_transform(random_truth_tables, self.lut_rank, dtype=torch.int32, device=device)

            # randomly pick indices to replace
            indices = torch.randperm(num_neurons, device=device)
            weights[indices[:n_different]] = random_coefficients
            return weights
        elif self.weight_init == "random":
            return torch.randn(num_neurons, lut_entries, device=device)
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

        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)

        # Convert to [-1, +1]
        x = 1 - 2 * x

        if weight.ndim == 1:
            weight = weight.unsqueeze(0)

        if self.materialize_basis:
            # add 'k' dimension for basis entries (e.g. 'n,bn->bn' becomes 'nk,bnk->bn')
            contraction_with_basis_dim = contraction.replace(',', 'k,').replace('->', 'k->')
            x = walsh_basis_hard(x, self.lut_rank)
            x = torch.einsum(contraction_with_basis_dim, weight, x)
        
        else:
            x = weighted_walsh_basis_sum(x, weight, contraction, self.lut_rank)

        if training:
            x = self._sample_train(-x)
        else:
            x = self._sample_eval(-x)

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
        luts = walsh_hadamard_transform(weight, self.lut_rank)
        return luts < 0

    def get_luts_and_ids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.lut_rank <= 4, "LUT IDs only supported for lut_rank <= 4 due to combinatorial explosion."
        luts = self.get_luts(weight)

        ids = 2 ** torch.arange(self.lut_entries - 1, -1, -1, device=luts.device)
        ids = (luts * ids.unsqueeze(0)).sum(dim=1)

        return luts, ids
    

