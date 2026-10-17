from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def environment_bound_class(
    target: type[Any], environment_factory: Callable[[], Any], module_name: str
) -> type[Any]:
    """Adapt a legacy class constructor without changing its class-valued API."""

    @wraps(target.__init__)
    def initialize(instance: Any, *args: Any, **kwargs: Any) -> None:
        """Inject the current legacy environment unless the caller provides one."""
        kwargs.setdefault("environment", environment_factory())
        target.__init__(instance, *args, **kwargs)

    return type(
        target.__name__,
        (target,),
        {"__doc__": target.__doc__, "__init__": initialize, "__module__": module_name},
    )
