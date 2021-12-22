import torch
import torch.nn.functional as F
from pytracking.libs.tensorlist import tensor_operation, TensorList


@tensor_operation
def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride=1, padding=0, dilation=1, groups=1, mode=None):
    """Standard conv2d. Returns the input if weight=None."""

    if padding == 0:
        padding2 = padding
    else:
        padding2 = (padding[0], padding[0], padding[1], padding[1])
    ind = None
    if mode is not None:
        if padding != 0:
            raise ValueError('Cannot input both padding and mode.')
        if mode == 'same':
            padding = (weight.shape[2]//2, weight.shape[3]//2)
            padding2 = (weight.shape[2] // 2, weight.shape[2] // 2, weight.shape[3] // 2, weight.shape[3] // 2)
            if weight.shape[2] % 2 == 0 or weight.shape[3] % 2 == 0:
                ind = (slice(-1) if weight.shape[2] % 2 == 0 else slice(None),
                       slice(-1) if weight.shape[3] % 2 == 0 else slice(None))
        elif mode == 'valid':
            padding = (0, 0)
            padding2 = 0
        elif mode == 'full':
            padding = (weight.shape[2]-1, weight.shape[3]-1)
            padding2 = (weight.shape[2] - 1, weight.shape[2] - 1, weight.shape[3] - 1, weight.shape[3] - 1)
        else:
            raise ValueError('Unknown mode for padding.')
    padding_operation = torch.nn.ZeroPad2d(padding2)
    input2 = padding_operation(input)
    if weight is None:
        return input
    out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if ind is None:
        if mode is not None:
            return tuple([out, torch.mean(input2, 0, True)])
        else:
            return out
    inputshape = input.shape
    input2shape = input2.shape
    # inputmean = torch.mean(input2, 0, True)
    if mode is not None:
        # return tuple([out[:,:,ind[0],ind[1]], torch.mean(input2, 0, True)])
        return tuple([out[:, :, ind[0], ind[1]], input2])
    else:
        return out[:,:,ind[0],ind[1]]


@tensor_operation
def conv1x1(input: torch.Tensor, weight: torch.Tensor):
    """Do a convolution with a 1x1 kernel weights. Implemented with matmul, which can be faster than using conv."""

    if weight is None:
        return input

    return torch.matmul(weight.view(weight.shape[0], weight.shape[1]),
                        input.view(input.shape[0], input.shape[1], -1)).view(input.shape[0], weight.shape[0], input.shape[2], input.shape[3])
