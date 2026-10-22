"""Run the preregistered paired from-scratch geometry experiment."""

from __future__ import annotations

if __package__:
    from scripts.geometry_r4.runner import main
else:
    from geometry_r4.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
