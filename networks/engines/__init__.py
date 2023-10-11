from networks.engines.aot_engine import AOTEngine, AOTInferEngine
from networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine
from networks.engines.mits_engine import MITSEngine, MITSInferEngine

def build_engine(name, phase='train', **kwargs):
    if name == 'aotengine':
        if phase == 'train':
            return AOTEngine(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    if name == 'deaotengine':
        if phase == 'train':
            return DeAOTEngine(**kwargs)
        elif phase == 'eval':
            return DeAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'mitsengine':
        if phase == 'train':
            return MITSEngine(**kwargs)
        elif phase == 'eval':
            return MITSInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
