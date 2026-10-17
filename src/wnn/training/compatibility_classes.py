from __future__ import annotations

import copyreg
import importlib
from collections.abc import Callable
from functools import wraps

# Legacy class adapters preserve heterogeneous constructor and serialized state shapes.
from typing import Any

from wnn.training.environment import TrainingEnvironment


# Constructor arguments and instance state vary across the legacy class boundary.
# The target remains the canonical, importable class used by spawned workers.
def _new_canonical_instance(
    target: type[Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Recreate a canonical target instance without running its initializer."""
    return target.__new__(target, *args, **kwargs)


def _is_importable_class(target: type[Any]) -> bool:
    """Check whether its module and qualified name resolve to the same class."""
    module_name = target.__module__
    qualified_name = target.__qualname__
    if not module_name or "<locals>" in qualified_name:
        return False
    try:
        resolved: Any = importlib.import_module(module_name)
        for component in qualified_name.split("."):
            resolved = getattr(resolved, component)
    except (AttributeError, ImportError, ValueError):
        return False
    return resolved is target


def environment_bound_class(
    target: type[Any],
    environment_factory: Callable[[], TrainingEnvironment],
    module_name: str,
) -> type[Any]:
    """Adapt a legacy class constructor while preserving pickle identity.

    Importable facade and subclass instances keep their class identity. An
    unimportable facade falls back to its canonical target with captured state,
    which supports custom-name imports in spawn workers without module globals.
    """

    facade_type: type[Any] | None = None

    @wraps(target.__init__)
    def initialize(instance: Any, *args: Any, **kwargs: Any) -> None:
        """Inject the current legacy environment unless the caller provides one."""
        if "environment" not in kwargs:
            kwargs["environment"] = environment_factory()
        target.__init__(instance, *args, **kwargs)

    def reduce_ex(instance: Any, protocol: int) -> tuple[Any, ...]:
        """Preserve importable class identity; fall back for transient facades."""
        reduction = target.__reduce_ex__(instance, protocol)
        if type(instance) is not facade_type:
            return reduction
        if facade_type is None or _is_importable_class(facade_type):
            return reduction
        if len(reduction) < 2:
            return reduction
        rebuild, args, *state = reduction
        if rebuild is copyreg.__newobj__ and args and args[0] is facade_type:
            return _new_canonical_instance, (target, args[1:], {}), *state
        newobj_ex = getattr(copyreg, "__newobj_ex__", None)
        if rebuild is newobj_ex and args and args[0] is facade_type:
            return _new_canonical_instance, (target, args[1], args[2]), *state
        return reduction

    facade_type = type(
        target.__name__,
        (target,),
        {
            "__doc__": target.__doc__,
            "__init__": initialize,
            "__module__": module_name,
            "__reduce_ex__": reduce_ex,
        },
    )
    return facade_type
