from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.hook_points import HookPoint

class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.n_latents = n_latents
        self.n_inputs = n_inputs
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        self.latent_outs = HookPoint()
        if tied:
            self.decoder = TiedTranspose(self.encoder, self.latent_outs)  # type: Union[nn.Linear, TiedTranspose]
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)

        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [..., n_inputs])
        :return: autoencoder latents before activation (shape: [..., n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias
        return latents_pre_act

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [..., n_inputs])
        :return: autoencoder latents (shape: [..., n_latents])
        """
        return self.activation(self.encode_pre_act(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [..., n_latents])
        :return: reconstructed data (shape: [..., n_inputs])
        """
        W_decode = self.decoder.weight
        latent_outs = self.latent_outs(torch.einsum("...l,dl->...ld", latents, W_decode))
        latent_out = latent_outs.sum(dim=-2)
        return latent_out + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [..., n_inputs])
        :return:  autoencoder latents pre activation (shape: [..., n_latents])
                  autoencoder latents (shape: [..., n_latents])
                  reconstructed data (shape: [..., n_inputs])
        """
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).flatten(end_dim=-2).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "Autoencoder":
        n_latents, d_model = state_dict["encoder.weight"].shape
        autoencoder = cls(n_latents, d_model)
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear, latent_outs: HookPoint):
        super().__init__()
        self.linear = linear
        self.latent_outs = latent_outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None

        W_decode = self.weight
        latent_outs = self.latent_outs(torch.einsum("...l,dl->...ld", x, W_decode))
        latent_out = latent_outs.sum(dim=-2)
        return latent_out

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias
