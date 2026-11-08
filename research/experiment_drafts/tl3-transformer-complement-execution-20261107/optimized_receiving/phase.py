from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run one declared sequence of canonical argv under the accepted owned relay."""
    commands = json.loads(Path(sys.argv[1]).read_text())
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
