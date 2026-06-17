"""Append-only debug log helpers for runtime and command diagnostics."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_LOG_PATH = PROJECT_ROOT / "debug.txt"
_LOCK = threading.Lock()
_HOOKS_INSTALLED = False


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _command_text(cmd: str | Sequence[object]) -> str:
    if isinstance(cmd, str):
        return cmd
    return subprocess.list2cmdline([str(part) for part in cmd])


def append_debug(message: object, source: str = "app", level: str = "INFO") -> None:
    """Append timestamped text to debug.txt without interrupting the app."""
    try:
        text = str(message)
        lines = text.splitlines() or [""]
        with _LOCK:
            DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
                for line in lines:
                    f.write(f"[{_timestamp()}] [{level}] [{source}] {line}\n")
    except Exception:
        pass


def log_exception(context: str, exc: BaseException | None = None, source: str = "app") -> None:
    """Log an exception with traceback details."""
    if exc is None:
        append_debug(f"{context}\n{traceback.format_exc()}", source=source, level="ERROR")
    else:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        append_debug(f"{context}: {exc}\n{details}", source=source, level="ERROR")


def install_exception_hooks(source: str = "app") -> None:
    """Record unhandled main-thread and worker-thread exceptions."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True

    previous_excepthook = sys.excepthook

    def excepthook(exc_type, exc, tb):
        append_debug(
            "Unhandled exception\n" + "".join(traceback.format_exception(exc_type, exc, tb)),
            source=source,
            level="ERROR",
        )
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = excepthook

    if hasattr(threading, "excepthook"):
        previous_threading_hook = threading.excepthook

        def threading_hook(args):
            append_debug(
                f"Unhandled thread exception in {args.thread.name}\n"
                + "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
                source=source,
                level="ERROR",
            )
            previous_threading_hook(args)

        threading.excepthook = threading_hook


def check_imports(import_names: Iterable[str], source: str = "dependency-check") -> list[str]:
    """Log whether runtime imports are available and return missing names."""
    missing = []
    for name in import_names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)

    if missing:
        append_debug(f"Missing Python modules: {', '.join(missing)}", source=source, level="ERROR")
    else:
        append_debug("All checked Python modules are importable.", source=source)
    return missing


def log_subprocess_start(cmd: str | Sequence[object], source: str = "subprocess") -> None:
    append_debug(f"CMD start: {_command_text(cmd)}", source=source)


def log_subprocess_result(
    cmd: str | Sequence[object],
    returncode: int | None,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    source: str = "subprocess",
    max_chars: int = 4000,
) -> None:
    """Log command completion and bounded stdout/stderr tails."""
    append_debug(f"CMD exit {returncode}: {_command_text(cmd)}", source=source)
    for label, value in (("stdout", stdout), ("stderr", stderr)):
        if not value:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = value.strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[-max_chars:]
            text = f"...<truncated to last {max_chars} chars>\n{text}"
        append_debug(f"CMD {label}:\n{text}", source=source)
