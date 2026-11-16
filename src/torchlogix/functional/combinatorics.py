"""Combinatorial sampling utilities and weight regularization helpers."""
import math
import random

import torch

from .basis import walsh_hadamard_transform


def get_regularization_loss(weights, regularizer=None):
    if regularizer is None:
        return 0.0
    elif regularizer == "L2":
        return (1 - weights.pow(2).sum(-1)).pow(2)
    elif regularizer == "abs_sum":
        return (1 - weights.sum(-1).abs()).pow(2)
    elif regularizer == "warp":
        # to enforce a growing norm factor (towards infinity)
        norm_factor = weights[:, -1]
        loss = torch.exp(-norm_factor).mean()
        # print(f"WARP regularization loss: {loss.item()}")
        return loss
    else:
        raise ValueError(f"Unknown regularizer: {regularizer}")
    

def rescale_weights(weights, method=None):
    with torch.no_grad():
        if method is None:
            pass
        elif method == "clip":
            weights.clamp_(-1, 1)
        elif method == "abs_sum":
            abs_sum = weights.sum(dim=-1, keepdim=True).abs()
            weights.div_(abs_sum)
        elif method == "L2":
            l2_norm = weights.norm(p=2, dim=-1, keepdim=True)
            weights.div_(l2_norm)
        else:
            raise ValueError(f"Unknown rescale method: {method}")


def build_binom_table(n: int, k: int, device=None) -> torch.Tensor:
    """Build table C[x, i] = binom(x, i) for x in [0..n-1], i in [0..k]."""
    C = torch.zeros(n, k + 1, dtype=torch.long, device=device)
    C[:, 0] = 1
    for x in range(n):
        for i in range(1, k + 1):
            if i > x:
                C[x, i] = 0
            elif i == x:
                C[x, i] = 1
            else:
                # Pascal recurrence: C(x, i) = C(x-1, i-1) + C(x-1, i)
                C[x, i] = C[x - 1, i - 1] + C[x - 1, i]
    return C

def unrank_combinations_batched(n: int, k: int, ranks: torch.Tensor) -> torch.Tensor:
    """Map ranks in [0, C(n,k)) to k-combinations of {0,...,n-1}, batched.

    Args:
        n: Total number of items.
        k: Combination size.
        ranks: Tensor of integer ranks, shape (...,).

    Returns:
        Tensor of combinations of shape (..., k), dtype long.
    """
    device = ranks.device
    ranks = ranks.clone().reshape(-1)           # (B,)
    B = ranks.numel()

    # Precompute binomial coefficients C[x, i]
    C = build_binom_table(n, k, device=device)  # (n, k+1)

    combos = torch.empty(B, k, dtype=torch.long, device=device)

    # Combinadic unranking: for i = k .. 1, find largest x with C(x, i) <= r
    for i in range(k, 0, -1):
        col = C[:, i]                       # (n,)
        # Broadcast compare: (n,B)
        mask = col.unsqueeze(1) <= ranks.unsqueeze(0)
        # monotone in x ⇒ last True index is desired x
        x = mask.long().sum(dim=0) - 1      # (B,)
        combos[:, i - 1] = x
        ranks = ranks - C[x, i]             # C[x, i] gathered by advanced indexing

    return combos.reshape(*ranks.shape, k)

def sample_unique_ranks(total_tuples, sample_size, K, device):
    all_ranks = []
    for _ in range(K):
        r = random.sample(range(total_tuples), sample_size)  # pure Python, no big tensor
        all_ranks.append(r)
    return torch.tensor(all_ranks, dtype=torch.long, device=device)  # (K, sample_size)


def get_combination_indices(n, k, sample_size, num_sets, device):
    """Get unique combination indices for K samples of size sample_size from n items.

    Args:
        n: Total number of items.
        k: Combination size.
        sample_size: Number of unique combinations to sample.
        num_sets: Number of sets of combinations to generate.
        device: Target device for returned tensor.
    """
    total_tuples = math.comb(int(n), k)
    assert sample_size <= total_tuples, (
        f"Not enough unique {k}-tuples: need {sample_size}, have {total_tuples}"
        )
    ranks = sample_unique_ranks(total_tuples, sample_size, num_sets, device)
    comb_indices = unrank_combinations_batched(n, k, 
                                                ranks.view(-1)).view(num_sets, 
                                                                    sample_size, k)
    return comb_indices


def take_tuples(
    x: torch.Tensor,
    tuple_size: int,
    start: int = 0,
    stride_within: int = 1,
    step_between: int = 1,
):
    """
    Take k-tuples from the last dimension of x with full control over:
      - start index
      - stride within each tuple
      - step between tuple starts

    x:            (..., N)
    tuple_size:   k (number of elements per tuple)
    start:        first index used for the first tuple
    stride_within:distance between elements inside a tuple
    step_between: distance between starts of consecutive tuples
                  (if None, defaults to tuple_size * stride_within)

    Returns:
        y with shape (..., tuple_size, num_groups)
        where last two dims are (k, num_groups).
    """
    N = x.size(-1)

    if step_between is None:
        step_between = tuple_size * stride_within

    # Maximum starting index that still fits a full tuple
    max_start = N - (tuple_size - 1) * stride_within
    if max_start <= start:
        raise ValueError("Not enough elements to take a single tuple "
                         f"with start={start}, tuple_size={tuple_size}, stride_within={stride_within}.")

    # All possible starting positions
    starts = torch.arange(start, max_start, step_between, device=x.device)  # (num_groups,)
    num_groups = starts.numel()

    # Offsets inside each tuple
    offsets = torch.arange(tuple_size, device=x.device) * stride_within     # (tuple_size,)

    # Build index matrix: shape (tuple_size, num_groups)
    idx = starts.unsqueeze(0) + offsets.unsqueeze(1)

    # Gather: result (..., tuple_size, num_groups)
    y = x[..., idx]   # advanced indexing broadcasts over leading dims

    return y


def id_to_truth_table(id: torch.Tensor, rank: int, device):
    lut_entries = 1 << rank
    bits = torch.arange(lut_entries, device=device)
    # Reshape for broadcasting: id becomes [..., 1], bits becomes [lut_entries]
    truth01 = (id.unsqueeze(-1) >> (lut_entries - 1 - bits)) & 1
    return 1 - 2 * truth01   # {0,1} → {+1,-1}


def id_to_walsh_coefficients(id: torch.Tensor, rank: int, device):
    truth = id_to_truth_table(id, rank, device=device)
    coeffs = walsh_hadamard_transform(truth, n=rank)
    coeffs = coeffs / (1 << rank)
    return coeffs
