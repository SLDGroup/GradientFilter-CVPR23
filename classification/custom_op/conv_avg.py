import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, avg_pool2d
import torch.nn as nn
from math import ceil


class Conv2dAvgOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, weight, bias, stride, dilation, padding, order, groups = args
        x_h, x_w = x.shape[-2:]
        k_h, k_w = weight.shape[-2:]
        y = conv2d(x, weight, bias, stride, padding,
                   dilation=dilation, groups=groups)
        h, w = y.shape[-2:]
        p_h, p_w = ceil(h / order), ceil(w / order)
        weight_sum = weight.sum(dim=(-1, -2))
        x_order_h, x_order_w = order * stride[0], order * stride[1]
        x_pad_h, x_pad_w = ceil(
            (p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)
        x_sum = avg_pool2d(x, kernel_size=(x_order_h, x_order_w),
                           stride=(x_order_h, x_order_w),
                           padding=(x_pad_h, x_pad_w), divisor_override=1)
        cfgs = th.tensor([bias is not None, groups != 1,
                          stride[0], stride[1],
                          x_pad_h, x_pad_w,
                          k_h, k_w,
                          x_h, x_w, order])
        ctx.save_for_backward(x_sum, weight_sum, cfgs)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x_sum, weight_sum, cfgs = ctx.saved_tensors
        has_bias, grouping,\
            s_h, s_w,\
            x_pad_h, x_pad_w,\
            k_h, k_w,\
            x_h, x_w, order = [int(c) for c in cfgs]
        n, c_in, p_h, p_w = x_sum.shape
        grad_y, = grad_outputs
        _, c_out, gy_h, gy_w = grad_y.shape
        grad_y_pad_h, grad_y_pad_w = ceil(
            (p_h * order - gy_h) / 2), ceil((p_w * order - gy_w) / 2)
        grad_y_avg = avg_pool2d(grad_y, kernel_size=order, stride=order,
                                padding=(grad_y_pad_h, grad_y_pad_w),
                                count_include_pad=False)
        if grouping:
            grad_x_sum = grad_y_avg * weight_sum.view(1, c_out, 1, 1)
            grad_w_sum = (x_sum * grad_y_avg).sum(dim=(0, 2, 3))
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, 1, 1, 1), (c_out, 1, k_h, k_w)).clone()
        else:
            grad_x_sum = (
                weight_sum.t() @ grad_y_avg.flatten(start_dim=2)).view(n, c_in, p_h, p_w)
            gy = grad_y_avg.permute(1, 0, 2, 3).flatten(start_dim=1)
            gx = x_sum.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
            grad_w_sum = gy @ gx
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, c_in, 1, 1), (c_out, c_in, k_h, k_w)).clone()
        grad_x = th.broadcast_to(grad_x_sum.view(n, c_in, p_h, p_w, 1, 1),
                                 (n, c_in, p_h, p_w, order * s_h, order * s_w))
        grad_x = grad_x.permute(0, 1, 2, 4, 3, 5).reshape(
            n, c_in, p_h * order * s_h, p_w * order * s_w)
        grad_x = grad_x[..., x_pad_h:x_pad_h + x_h, x_pad_w:x_pad_w + x_w]
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 2, 3))
        else:
            grad_b = None

        return grad_x, grad_w, grad_b, None, None, None, None, None


class Conv2dDilatedOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, weight, bias, stride, dilation, padding, order, groups = args
        x_h, x_w = x.shape[-2:]
        k_h, k_w = weight.shape[-2:]
        y = conv2d(x, weight, bias, stride, padding,
                   dilation=dilation, groups=groups)
        h, w = y.shape[-2:]
        p_h, p_w = ceil(h / order), ceil(w / order)
        x_order_h, x_order_w = order * stride[0], order * stride[1]
        x_pad_h, x_pad_w = ceil(
            (p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)
        x_sum = avg_pool2d(x, kernel_size=(x_order_h, x_order_w),
                           stride=(x_order_h, x_order_w),
                           padding=(x_pad_h, x_pad_w), divisor_override=1)
        cfgs = th.tensor([bias is not None, groups != 1,
                          stride[0], stride[1],
                          x_pad_h, x_pad_w,
                          k_h, k_w,
                          x_h, x_w, order, dilation[0]])
        ctx.save_for_backward(x_sum, weight, cfgs)
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        x_sum, weight, cfgs = ctx.saved_tensors
        has_bias, grouping,\
            s_h, s_w,\
            x_pad_h, x_pad_w,\
            k_h, k_w,\
            x_h, x_w, order, dil = [int(c) for c in cfgs]
        n, c_in, p_h, p_w = x_sum.shape
        grad_y, = grad_outputs
        _, c_out, gy_h, gy_w = grad_y.shape
        grad_y_pad_h, grad_y_pad_w = ceil(
            (p_h * order - gy_h) / 2), ceil((p_w * order - gy_w) / 2)
        grad_y_avg = avg_pool2d(grad_y, kernel_size=order, stride=order,
                                padding=(grad_y_pad_h, grad_y_pad_w),
                                count_include_pad=False)
        equ_dil = dil // order
        if grouping:
            rot_weight = th.flip(weight, (2, 3))
            grad_x_sum = conv2d(grad_y_avg, rot_weight, padding=equ_dil,
                                dilation=equ_dil, groups=weight.shape[0])
            grad_w_sum = (x_sum * grad_y_avg).sum(dim=(0, 2, 3))
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, 1, 1, 1), (c_out, 1, k_h, k_w)).clone()
        else:
            rot_weight = th.flip(weight.permute(1, 0, 2, 3), (2, 3))
            grad_x_sum = conv2d(grad_y_avg, rot_weight,
                                padding=equ_dil, dilation=equ_dil)
            gy = grad_y_avg.permute(1, 0, 2, 3).flatten(start_dim=1)
            gx = x_sum.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
            grad_w_sum = gy @ gx
            grad_w = th.broadcast_to(grad_w_sum.view(
                c_out, c_in, 1, 1), (c_out, c_in, k_h, k_w)).clone()
        grad_x = th.broadcast_to(grad_x_sum.view(n, c_in, p_h, p_w, 1, 1),
                                 (n, c_in, p_h, p_w, order * s_h, order * s_w))
        grad_x = grad_x.permute(0, 1, 2, 4, 3, 5).reshape(
            n, c_in, p_h * order * s_h, p_w * order * s_w)
        grad_x = grad_x[..., x_pad_h:x_pad_h + x_h, x_pad_w:x_pad_w + x_w]
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 2, 3))
        else:
            grad_b = None

        return grad_x, grad_w, grad_b, None, None, None, None, None


class Conv2dAvg(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            order=4,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2dAvg, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.order = order

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            if self.dilation[0] == 1 or self.dilation[0] < self.order:
                y = Conv2dAvgOp.apply(x, self.weight, self.bias, self.stride, self.dilation,
                                      self.padding, self.order, self.groups)
            else:
                y = Conv2dDilatedOp.apply(x, self.weight, self.bias, self.stride, self.dilation,
                                          self.padding, self.order, self.groups)
        else:
            y = super().forward(x)
        return y


class Conv2dAvgQAT(Conv2dAvg):
    _FLOAT_MODULE = Conv2dAvg

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 order=4,
                 stride=1,
                 groups=1,
                 bias=True,
                 padding=0,
                 device=None,
                 dtype=None,
                 activate=False,
                 qconfig=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size,
                         order, stride, groups, bias, padding,
                         device, dtype, activate)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)
        Conv2dAvgQAT._FLOAT_MODULE = Conv2dAvg

    def forward(self, x: th.Tensor):
        weight = self.weight_fake_quant(self.weight)
        if self.activate:
            y = Conv2dAvgOp.apply(x, weight, self.bias, self.stride,
                                  self.padding, self.order, self.groups)
        else:
            y = conv2d(x, weight, self.bias, self.stride, self.padding)
        return y

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(
            mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       mod.order, mod.stride, mod.groups, mod.bias is not None,
                       mod.padding, activate=mod.activate, qconfig=qconfig)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        conv = Conv2dAvg(
            self.in_channels,
            self.out_channels,
            self.kernel_size,  # type: ignore[arg-type]
            self.order,
            self.stride,  # type: ignore[arg-type]
            self.groups,
            self.bias is not None,
            self.padding,  # type: ignore[arg-type]
            self.device,
            self.dtype,
            self.activate
        )
        conv.weight = nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = nn.Parameter(self.bias.detach())
        return conv


def wrap_conv_layer(conv, radius, active):
    new_conv = Conv2dAvg(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         order=radius,
                         activate=active
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv
