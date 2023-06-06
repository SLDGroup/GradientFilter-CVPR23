import logging
from functools import reduce
import torch.nn as nn
from .conv_avg import wrap_conv_layer


DEFAULT_CFG = {
    "path": "",
    "radius": 8,
    "type": "",
}


def add_grad_filter(module: nn.Module, cfg):
    if cfg['type'] == 'cbr':
        module.conv = wrap_conv_layer(module.conv, cfg['radius'], True)
    elif cfg['type'] == 'resnet_basic_block':
        module.conv1 = wrap_conv_layer(module.conv1, cfg['radius'], True)
        module.conv2 = wrap_conv_layer(module.conv2, cfg['radius'], True)
    elif cfg['type'] == 'conv':
        module = wrap_conv_layer(module, cfg['radius'], True)
    else:
        raise NotImplementedError
    return module


def register_filter(module, cfgs):
    filter_install_cfgs = cfgs['filter_install']
    logging.info("Registering Filter")
    if not isinstance(filter_install_cfgs, list):
        logging.info("No Filter Required")
        return
    # Install filter
    for cfg in filter_install_cfgs:
        assert "path" in cfg.keys()
        for k in cfg.keys():
            assert k in DEFAULT_CFG.keys(), f"Filter registration: {k} not found"
        for k in DEFAULT_CFG.keys():
            if k not in cfg.keys():
                cfg[k] = DEFAULT_CFG[k]
        path_seq = cfg['path'].split('.')
        target = reduce(getattr, path_seq, module)
        upd_layer = add_grad_filter(target, cfg)
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)
