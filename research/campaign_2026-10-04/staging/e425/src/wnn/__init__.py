"""Isolated E425 WNN overlay with fallback to the frozen root package."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
