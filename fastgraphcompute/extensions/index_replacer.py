import torch
import os.path as osp

# Load the shared libraries
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cpu.so'))
if torch.cuda.is_available():
    torch.ops.load_library(osp.join(osp.dirname(osp.realpath(__file__)), 'index_replacer_cuda.so'))

@torch.jit.script
def index_replacer_torch(to_be_replaced: torch.Tensor, replacements: torch.Tensor) -> torch.Tensor:
    """
    Replace values in `to_be_replaced` using the `replacements` tensor.
    
    Args:
        to_be_replaced (torch.Tensor): Tensor of indices to be replaced.
        replacements (torch.Tensor): Tensor of replacement values.
    
    Returns:
        torch.Tensor: Tensor with replaced values.
    """
    # Ensure the input tensors are on the same device and dtype
    if to_be_replaced.device != replacements.device:
        raise AssertionError("Tensors must be on the same device.")
    if to_be_replaced.dtype != torch.int32 or replacements.dtype != torch.int32:
        raise AssertionError("Tensors must be int32 type.")

    # Initialize output tensor
    replaced = to_be_replaced.clone()

    # Identify valid indices (those within range of replacements)
    valid_mask = (to_be_replaced >= 0) & (to_be_replaced < replacements.size(0))

    # Replace valid indices
    replaced[valid_mask] = replacements[to_be_replaced[valid_mask]]

    return replaced

@torch.jit.script
def index_replacer(to_be_replaced: torch.Tensor, replacements: torch.Tensor) -> torch.Tensor:
    """
    Replace values in `to_be_replaced` using the `replacements` tensor.
    
    Args:
        to_be_replaced (torch.Tensor): Tensor of indices to be replaced.
        replacements (torch.Tensor): Tensor of replacement values.
    
    Returns:
        torch.Tensor: Tensor with replaced values.
    """
    # Ensure the input tensors are on the same device and dtype
    if to_be_replaced.device != replacements.device:
        raise AssertionError("Tensors must be on the same device.")
    if to_be_replaced.dtype != torch.int32 or replacements.dtype != torch.int32:
        raise AssertionError("Tensors must be int32 type.")

    if to_be_replaced.device.type == 'cuda':
        return torch.ops.index_replacer_cuda.index_replacer_cuda(to_be_replaced, replacements)
    else:
        return torch.ops.index_replacer_cpu.index_replacer_cpu(to_be_replaced, replacements)