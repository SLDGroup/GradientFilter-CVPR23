"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

from ast import Invert
import torchvision
import torch.nn as nn

from ._base import EncoderMixin

from torch.quantization import fuse_modules
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops.misc import ConvNormActivation


class MobileNetV1Encoder(nn.Module, EncoderMixin):
    def __init__(self, out_channels, depth=5, relu6=True, log_grad=False):
        super(MobileNetV1Encoder, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.ReLU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        def conv_dw(inp, oup, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(relu6),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(relu6),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2, relu6), 
            conv_dw( 32,  64, 1, relu6),
            conv_dw( 64, 128, 2, relu6),
            conv_dw(128, 128, 1, relu6),
            conv_dw(128, 256, 2, relu6),
            conv_dw(256, 256, 1, relu6),
            conv_dw(256, 512, 2, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 512, 1, relu6),
            conv_dw(512, 1024, 2, relu6),
            conv_dw(1024, 1024, 1, relu6),
            # nn.AvgPool2d(7),
        )
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        self.log_grad = log_grad

    def get_stages(self):
        return [
            nn.Identity(),
            self.model[:2],
            self.model[2:4],
            self.model[4:6],
            self.model[6:11],
            self.model[11:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'fc' in k:
                continue
            new_state_dict[k[7:]] = v
        super().load_state_dict(new_state_dict, **kwargs)

    def fuse_modules(self):
        for blk in self.model:
            fuse_modules(blk, [["0", "1"]], inplace=True)
            if len(blk) > 3:
                fuse_modules(blk, [["3", "4"]], inplace=True)


class MobileNetV2Encoder(torchvision.models.MobileNetV2, EncoderMixin):
    def __init__(self, out_channels, depth=5, log_grad=False, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        self.log_grad = log_grad
        del self.classifier
        for m in self.modules():
            if type(m) == InvertedResidual:
                def new_forward(self, x):
                    if self.use_res_connect:
                        return self.quat_op.add(x, self.conv(x))
                    else:
                        return self.conv(x)
                m.quat_op = nn.quantized.FloatFunctional()
                m.forward = new_forward.__get__(m, InvertedResidual)

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias", None)
        state_dict.pop("classifier.1.weight", None)
        super().load_state_dict(state_dict, **kwargs)

    def fuse_modules(self):
        for module in self.features:
            if type(module) == InvertedResidual:
                count = len(module.conv)
                fuse_modules(module, [[f"conv.{count - 2}", f"conv.{count - 1}"]], inplace=True)
                for i in range(count - 2):
                    fuse_modules(module, [[f'conv.{i}.0', f'conv.{i}.1']], inplace=True)
            elif type(module) == ConvNormActivation:
                fuse_modules(module, [['0', '1']], inplace=True)


mobilenet_encoders = {
    "mobilenet_v1": {
        "encoder": MobileNetV1Encoder,
        "params": {
            "out_channels": (64, 128, 256, 512, 1024)
        }
    },
    "mobilenet_v2": {
        "encoder": MobileNetV2Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 24, 32, 96, 1280),
        },
    },
}
