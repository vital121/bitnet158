import torch
import torch.nn as nn


class BitLinear(nn.Linear):
    """
    A custom linear layer that implements weight binarization and optional activation quantization to enable more
    efficient computation, particularly suited for deployment on hardware that benefits from quantized computations.

    This layer extends the standard `nn.Linear` layer by binarizing weights to 1 bit and optionally quantizing
    activations to a specified bit-width.

    Parameters:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
        dtype (torch.dtype, optional): Data type of the weights. Default is torch.bfloat16 for efficient computation.
        num_groups (int, optional): The number of groups for dividing the weights and activations during quantization,
                                    allowing for group-wise quantization. Default: 1.

    Attributes:
        quantized_weights (torch.Tensor): The buffer that stores quantized weights in 1-bit format.
        dtype (torch.dtype): The data type to which weights are cast during dequantization for computation.
        num_groups (int): The number of groups for group-wise weight binarization and activation quantization.
        eps (float): A small epsilon value to prevent division by zero during activation quantization.

    Methods:
        weight (property): Overrides the `weight` property to return dequantized weights for computation.
        dequantize_weights (self): Converts the quantized weights back to the specified dtype and computes the scaling
                                   factor (alpha) for the weights.
        ste_binarize (self, x): Applies the Straight Through Estimator (STE) method to binarize inputs while preserving
                                gradients during backpropagation.
        binarize_weights_groupwise (self): Performs group-wise binarization of weights using STE.
        quantize_activations_groupwise (self, x, b=8): Quantizes activations group-wise to a specified bit-width.

    Example:
        >>> layer = BitLinear(512, 512, bias=True, dtype=torch.bfloat16, num_groups=4)
        >>> input = torch.randn(10, 512)
        >>> output = layer(input)
        The output tensor will have gone through binarized weight multiplication and optionally quantized activations.
    """
    def __init__(self, in_features, out_features, bias=True, dtype=None, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.dtype = dtype if dtype is not None else torch.bfloat16
        # print(f"Using {self.dtype} for BitLinear {dtype}")
        self.num_groups = num_groups
        self.eps = 1e-5

        # Initialize 1-bit quantized weights and store them as int8
        self.register_buffer(
            "quantized_weights", torch.sign(self.weight.data).to(torch.int8)
        )
        # Clear the original weights to save memory
        del self.weight

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.dequantize_weights()

    @weight.setter
    def weight(self, value):
        # Update the quantized_weights when the weight property is set
        self.quantized_weights.data = torch.sign(value).to(torch.int8)

    def dequantize_weights(self):
        # Convert quantized_weights back to bfloat16 and compute alpha for the weights
        bfloat16_weights = self.quantized_weights.to(self.dtype)
        alpha = bfloat16_weights.mean()
        return bfloat16_weights * alpha

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Dequantize the weights before binarization
        weights = self.dequantize_weights()

        # Divide weights into groups
        group_size = weights.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(weights)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = weights[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output
