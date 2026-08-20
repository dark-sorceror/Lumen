"""`uv run python -m workbench.cli [ws://host:port/ws]`"""
import sys

from workbench.cli.session import main

if __name__ == "__main__":
    sys.exit(main())
