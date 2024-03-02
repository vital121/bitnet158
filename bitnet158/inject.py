import torch
import torch.nn as nn
from bitnet158 import BitLinear


def inject(model: nn.Module, copy_weights: bool = True, module_class=BitLinear):
    """
    Recursively traverses a PyTorch model and replaces all instances of `nn.Linear` layers with a custom layer specified
    by `module_class`. Optionally, the original weights and biases can be copied to the new custom layers.

    Parameters:
        model (nn.Module): The model to be modified.
        copy_weights (bool, optional): If `True`, the weights and biases from the original `nn.Linear` layers are copied to the new custom layers. Default is `True`.
        module_class (nn.Module, optional): The custom layer class to replace `nn.Linear` layers with. Default is `BitLinear`.

    This function modifies the model in-place and does not return a value.

    Example:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> inject(model, copy_weights=True, module_class=BitLinear)
        This example replaces all `nn.Linear` layers in a ResNet-18 model with `BitLinear` layers, copying over the original weights and biases.
    """
    for name, child in model.named_modules():
        if isinstance(child, nn.Linear) and "." not in name and (not isinstance(child, module_class)):
            # Replace the nn.Linear with BitLinear matching in features and and out_features, and add it to the model
            bitlinear = module_class(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                dtype=child.weight.dtype,
            )
            # copy the weights and bias
            if copy_weights:
                bitlinear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    bitlinear.bias.data.copy_(child.bias.data)

            setattr(
                model,
                name,
                bitlinear,
            )
        elif isinstance(child, nn.Module) and name != "":
            # print(name)
            inject(child, copy_weights=copy_weights, module_class=module_class)
