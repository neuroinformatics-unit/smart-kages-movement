from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smart-kages-movement")
except PackageNotFoundError:
    # package is not installed
    pass
