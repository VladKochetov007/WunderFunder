"""E427 overlay namespace; resolve unmodified package modules from the repo."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
