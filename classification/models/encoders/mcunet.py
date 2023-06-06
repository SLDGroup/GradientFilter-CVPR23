import torch.nn as nn
from mcunet.model_zoo import build_model
from torch.quantization import fuse_modules
from ._base import EncoderMixin
from mcunet.tinynas.nn.networks.proxyless_nets import MobileInvertedResidualBlock
from mcunet.tinynas.nn.modules.layers import ZeroLayer


class MCUNetEncoder(nn.Module, EncoderMixin):
    def __init__(self, out_channels, depth=5, pretrained=True, log_grad=False):
        super(MCUNetEncoder, self).__init__()
        model = build_model("mcunet-5fps", pretrained=pretrained)[0]
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        self.model = nn.Sequential(*([model.first_conv] + list(model.blocks)))
        for m in self.modules():
            if type(m) == MobileInvertedResidualBlock:
                def new_forward(self, x):
                    if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
                        res = x
                    elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
                        res = self.mobile_inverted_conv(x)
                    else:
                        res = self.quat_op.add(self.mobile_inverted_conv(x), self.shortcut(x))
                    return res
                m.quat_op = nn.quantized.FloatFunctional()
                m.forward = new_forward.__get__(m, MobileInvertedResidualBlock)

    def get_stages(self):
        return [
            nn.Identity(),
            self.model[:2],
            self.model[2:4],
            self.model[4:6],
            self.model[6:11],
            self.model[11:]
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def fuse_modules(self):
        fuse_modules(self.model[0], [['conv', 'bn']], inplace=True)
        for i in range(1, len(self.model)):
            if self.model[i].mobile_inverted_conv.inverted_bottleneck is not None:
                fuse_modules(self.model[i].mobile_inverted_conv.inverted_bottleneck, [['conv', 'bn']], inplace=True)
            if self.model[i].mobile_inverted_conv.depth_conv is not None: 
                fuse_modules(self.model[i].mobile_inverted_conv.depth_conv, [['conv', 'bn']], inplace=True)
            if self.model[i].mobile_inverted_conv.point_linear is not None:
                fuse_modules(self.model[i].mobile_inverted_conv.point_linear, [['conv', 'bn']], inplace=True)


mcunet_encoders = {
    "mcunet": {
        "encoder": MCUNetEncoder,
        "params": {
            "out_channels": (8, 16, 24, 48, 160)
        }
    },
}
