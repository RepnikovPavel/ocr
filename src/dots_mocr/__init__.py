__all__ = ["DotsMOCRParser", "register_transformers"]


def __getattr__(name):
    if name == "register_transformers":
        from .transformers_patch import register_transformers

        return register_transformers
    if name == "DotsMOCRParser":
        from .cli import DotsMOCRParser

        return DotsMOCRParser
    raise AttributeError(name)
