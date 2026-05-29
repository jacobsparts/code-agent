__version__ = "0.1.0"

__all__ = [
    "CodeAgent",
    "main",
]


def __getattr__(name):
    if name in {"CodeAgent", "main"}:
        from .agent import CodeAgent, main
        return {"CodeAgent": CodeAgent, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")