from .transformers_patch import register_transformers

__all__ = ["DotsMOCRParser", "register_transformers"]


def __getattr__(name):
    if name == "DotsMOCRParser":
        from .cli import DotsMOCRParser

        return DotsMOCRParser
    raise AttributeError(name)
