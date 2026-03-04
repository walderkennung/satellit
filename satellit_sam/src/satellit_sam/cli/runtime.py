"""Runtime helpers for CLI command handlers."""

from __future__ import annotations

import shlex
import sys


def current_cli_command() -> str:
    """Return a shell-safe command string for the current CLI invocation."""
    return shlex.join(["satellit", *sys.argv[1:]])
