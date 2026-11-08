"""Allow ``python -m swarm_agent`` as an alias for the ``swarm-agent`` CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
