try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Para Python < 3.8
    from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__)
except PackageNotFoundError:
    __version__ = "0+unknown"