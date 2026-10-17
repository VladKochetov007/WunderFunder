from __future__ import annotations

import copyreg
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


def environment_bound_class(
    target: type[Any],
    environment_factory: Callable[[], TrainingEnvironment],
    module_name: str,
) -> type[Any]:
    """Adapt a legacy class constructor while preserving canonical pickle state.

    Instances of the transient facade pickle as the canonical target class with
    their captured state. This keeps custom-name imports usable by spawn workers
    without registering the facade as a module global.
    """

    @wraps(target.__init__)
    def initialize(instance: Any, *args: Any, **kwargs: Any) -> None:
        """Inject the current legacy environment unless the caller provides one."""
        if "environment" not in kwargs:
            kwargs["environment"] = environment_factory()
        target.__init__(instance, *args, **kwargs)

    def reduce_ex(instance: Any, protocol: int) -> tuple[Any, ...]:
        """Serialize the facade as its importable canonical base class."""
        reduction = target.__reduce_ex__(instance, protocol)
        if len(reduction) < 2:
            return reduction
        rebuild, args, *state = reduction
        facade = type(instance)
        if rebuild is copyreg.__newobj__ and args and args[0] is facade:
            return _new_canonical_instance, (target, args[1:], {}), *state
        newobj_ex = getattr(copyreg, "__newobj_ex__", None)
        if rebuild is newobj_ex and args and args[0] is facade:
            return _new_canonical_instance, (target, args[1], args[2]), *state
        return reduction

    return type(
        target.__name__,
        (target,),
        {
            "__doc__": target.__doc__,
            "__init__": initialize,
            "__module__": module_name,
            "__reduce_ex__": reduce_ex,
        },
    )
