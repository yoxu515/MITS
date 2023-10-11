from networks.models.aot import AOT
from networks.models.mits import MITS



def build_vos_model(name, cfg, **kwargs):

    if name == 'aot':
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'mits':
        return MITS(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
