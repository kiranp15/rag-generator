"""Dependency-free test runner: discovers test_*.py modules under tests/,
runs every function named test_*, and reports pass/fail counts.

Run with:  python -m tests.run_tests
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback

TESTS_DIR = os.path.dirname(__file__)


class _MonkeyPatch:
    """Minimal pytest-style monkeypatch substitute (setattr + auto-undo)."""

    def __init__(self):
        self._restores = []

    def setattr(self, obj, name, value):
        self._restores.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._restores):
            setattr(obj, name, old)


def discover_test_modules():
    for name in sorted(os.listdir(TESTS_DIR)):
        if name.startswith("test_") and name.endswith(".py"):
            yield f"tests.{name[:-3]}"


def main() -> int:
    sys.path.insert(0, os.path.dirname(TESTS_DIR))
    passed = 0
    failed = 0
    failures = []

    for module_name in discover_test_modules():
        module = importlib.import_module(module_name)
        for attr_name in sorted(dir(module)):
            if not attr_name.startswith("test_"):
                continue
            func = getattr(module, attr_name)
            if not callable(func):
                continue
            full_name = f"{module_name}.{attr_name}"
            mp = None
            try:
                params = inspect.signature(func).parameters
                if "monkeypatch" in params:
                    mp = _MonkeyPatch()
                    func(mp)
                else:
                    func()
                print(f"PASS  {full_name}")
                passed += 1
            except Exception:
                print(f"FAIL  {full_name}")
                failures.append((full_name, traceback.format_exc()))
                failed += 1
            finally:
                if mp is not None:
                    mp.undo()

    print(f"\n{passed} passed, {failed} failed")
    if failures:
        print("\n--- Failure details ---")
        for name, tb in failures:
            print(f"\n{name}:\n{tb}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
