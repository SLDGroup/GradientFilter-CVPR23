def freeze_layers(module, freeze_cfgs):
    if not isinstance(freeze_cfgs, list):
        print("No Freeze Required")
        return
    for cfg in freeze_cfgs:
        path = cfg['path'].split(' ')
        layer = module
        for p in path:
            if p.startswith('[') and p.endswith(']'):
                if p[1:-1].isdigit():
                    layer = layer[int(p[1:-1])]
                else:
                    layer = layer[p[1:-1]]
            else:
                layer = getattr(layer, p)
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False


def grad_logger(dst, name):
    def hook(grad):
        dst[name] = grad
    return hook
