import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch


def absmax_quantize(x: Tensor, bits: int = 8):
    """
    Absmax Quantization

    Args:
        x (torch.Tensor): Input tensor
        bits (int, optional): Number of bits. Defaults to 8.

    """
    Qb = 2 ** (bits - 1) - 1
    scale = Qb / torch.max(torch.abs(x))
    quant = (scale * x).round()
    dequant = quant / scale
    return quant.to(torch.int8), dequant


class BitLinear158(nn.Module):
    """
    A custom implementation of a linear layer that applies ternary weight quantization, quantizing weights to -1, 0, or +1.
    This layer is designed to offer a balance between computational efficiency and model expressiveness, potentially reducing
    the model's memory footprint and computational cost during inference.

    The quantization process is based on an absolute mean (absmean) approach, where the weights are scaled by the mean
    absolute value of the weights tensor and then rounded to the nearest integer within the set {-1, 0, +1}.

    Parameters:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to `True`, the layer will learn an additive bias. Default: `True`.

    Attributes:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (torch.nn.Parameter or None): The bias parameter. If `None`, no bias will be applied.
        weight (torch.nn.Parameter): The weight parameter before quantization.
        eps (float): A small epsilon value added for numerical stability during quantization.

    Example usage:
        >>> bit_linear_layer = BitLinear158(in_features=128, out_features=64, bias=True)
        >>> input_tensor = torch.randn(10, 128)
        >>> output_tensor = bit_linear_layer(input_tensor)
        The output tensor will have shape (10, 64) and will be computed using ternary quantized weights.
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the BitLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(BitLinear158, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.eps = 1e-6  # Small epsilon for numerical stability

    def forward(self, x):
        """
        Forward pass through the BitLinear layer with ternary weight quantization.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features), computed using ternary quantized weights.
        """
        # x = torch.sign(x)
        quantized_weight = self.quantize_weights(self.weight)
        return F.linear(x, quantized_weight)

    def quantize_weights(self, W):
        """
        Quantizes the weights to -1, 0, or +1 using an absolute mean (absmean) approach.

        Args:
            W (Tensor): The weight tensor to be quantized.

        Returns:
            Tensor: The quantized weight tensor.
        """
        gamma = torch.mean(torch.abs(W)) + self.eps
        W_scaled = W / gamma
        W_quantized = torch.sign(W_scaled) * torch.clamp(
            torch.abs(W_scaled).round(), max=1.0
        )
        return W_quantized

    def extra_repr(self):
        """
        Returns a string with the additional representation of the module, including the in_features,
        out_features, and quantization strategy, for debugging and logging purposes.
        """
        return "in_features={}, out_features={}, quantization=ternary".format(
            self.in_features, self.out_features
        )
