"""Isolated E426 overlay package that falls through to the frozen WNN sources."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
