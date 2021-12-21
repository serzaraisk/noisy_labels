class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def _getattr(self, key):
        target = self
        for dot in key.split('.'):
            target = target[dot]
        return target

    def _setattr(self, key, value):
        target = self
        for dot in key.split('.')[:-1]:
            target = target[dot]
        target[key.split('.')[-1]] = value