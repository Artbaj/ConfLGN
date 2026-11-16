import torch

from .base import Connections
from .fixed_connections import get_random_unique_connections
from ..functional import softmax


class LearnableConnectionFunction(torch.autograd.Function):
    """Autograd function for learnable connections.
    Implements the forward and backward pass for learnable connections
    using Gumbel-Softmax for differentiable sampling.
    """
    @staticmethod
    def forward(ctx, x, weights, tau, gumbel, indices):
        if gumbel:
            u = torch.rand_like(weights)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        else:
            g = torch.zeros_like(weights)
        connections = (weights + g).argmax(dim=0)
        l = torch.arange(weights.shape[1], device=x.device).unsqueeze(1)
        o = torch.arange(weights.shape[2], device=x.device).unsqueeze(0)
        output = x[:, indices[connections, l, o]]
        ctx.save_for_backward(x, weights, tau, g, indices)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, tau, g, indices = ctx.saved_tensors
        # compute gradient w.r.t. to learnable weights
        weights_grad = torch.einsum("bclo,blo->clo", 2*x[:,indices]-1, output_grad)
        # compute gradient w.r.t. to input with sparse scatter_add method
        input_grad = torch.zeros_like(x)
        temp = softmax((weights + g)/tau, dim=0).unsqueeze(0) * output_grad.unsqueeze(1)
        input_grad.scatter_add_(dim=1,
                                index=indices.expand(x.shape[0], -1, -1, -1).reshape(x.shape[0], -1),
                                src=temp.reshape(x.shape[0], -1))
        return input_grad, weights_grad, None, None, None


class LearnableDenseConnections(Connections):
    """Learnable dense connections implementation.
    Each neuron connects to ``lut_rank`` input features chosen from a set of candidates.
    The connections are learnable parameters optimized during training.

    Args:
        in_dim: Input feature dimension.
        out_dim: Number of neurons (output dimension).
        lut_rank: Number of input features each neuron connects to.
        temperature: Temperature parameter for Gumbel-Softmax.
        num_candidates: Number of candidate input features per neuron.
            If -1, all input features are candidates.
        gumbel: Whether to use Gumbel noise for sampling.
        device: Device to store the connection indices tensor.
        init_method: Method to initialize connections. Options are:
            - "random": Randomly select input features (with replacement).
            - "random-unique": Randomly select unique input features (without replacement).
    """
    def __init__(
            self,
            in_dim,
            out_dim,
            lut_rank=2,
            temperature=0.001,
            num_candidates=-1,
            gumbel=False,
            device=None,
            init_method="random",
            **kwargs
        ):
        super().__init__(
            lut_rank=lut_rank,
            device=device,
            init_method=init_method,
            **kwargs
        )

        self.temperature = temperature
        self.num_candidates = num_candidates
        self.lut_rank = lut_rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.gumbel = gumbel
        if num_candidates == -1:
            num_candidates = in_dim
            self.num_candidates = num_candidates
            self.indices = torch.arange(in_dim, device=self.device).view(
                in_dim, 1, 1).expand(in_dim, lut_rank, out_dim)
        else:
            assert num_candidates > 0, "num_candidates must be bigger than 0"
            self.num_candidates = num_candidates
            self.indices = self._init_connections()
        self.weights = torch.nn.Parameter(torch.rand(
            num_candidates, lut_rank, out_dim, dtype=torch.float32), requires_grad=True)

    def update_temperature(self, temperature: float):
        self.temperature = temperature

    def forward(self, x):
        return LearnableConnectionFunction.apply(x, self.weights, torch.tensor(self.temperature),
                                                 self.gumbel, self.indices)

    def _init_connections(self):
        """Constructs possible input–neuron connection indices.

        Each neuron takes ``lut_rank`` input features chosen out of ``lut_rank * num_candidates``
        possibilities. This function returns a tensor encoding which input indices are connected
        to which neuron.

        Returns:
            A tensor of shape ``(num_candidates, lut_rank, out_dim)`` with integer indices into
            the last dimension of the input.
        """
        assert self.in_dim >= self.num_candidates * self.lut_rank, (
            f"Cannot have num_candidates * lut_rank > in_dim "
            f"({self.num_candidates * self.lut_rank} > {self.in_dim})"
        )
        assert self.out_dim * self.lut_rank >= self.in_dim, (
                f"Need out_dim * lut_rank >= in_dim to cover all inputs "
                f"({self.out_dim} * {self.lut_rank} < {self.in_dim})."
                )

        if self.init_method == "random":
            # With this method both inputs can stem from the same input feature
            c = torch.randperm(self.lut_rank * self.out_dim * self.num_candidates,
                               device=self.device) % self.in_dim
            c = c.reshape(self.num_candidates, self.lut_rank, self.out_dim)
        elif self.init_method == "random-unique":
            c = get_random_unique_connections(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                n=self.lut_rank*self.num_candidates
            )
            c = c.reshape(self.num_candidates, self.lut_rank, self.out_dim)
        else:
            raise ValueError(self.connections)
        c = c.contiguous().to(torch.int64).to(self.device)
        return c
