"""Generated smoke tests from Testing Workbench.

These tests are intentionally conservative. They prefer static checks and safe
imports before trying anything that might open a real application window.
"""

import ast
import configparser
import importlib.util
import json
import os
import re
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORE_DIRS = {
    ".git", ".venv", ".build_venv", "venv", "env", "__pycache__", "build",
    "release", "dist", "app_libs", "vendor", "vendors", "site-packages",
    "test_logs", ".test_storage_tmp", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "wix_staging", "_internal",
}
IGNORE_DIR_PREFIXES = ("build_libs", "tmp")
IGNORE_FILE_PATTERNS = (
    re.compile(r".*backup.*\.py$", re.IGNORECASE),
    re.compile(r".*\.bak$", re.IGNORECASE),
)
CONFIG_SUFFIXES = {".ini", ".cfg", ".conf"}
TEXT_DATA_SUFFIXES = {".txt", ".md", ".csv", ".xml", ".yaml", ".yml"}
COMMON_TK_METHODS = {"destroy", "quit", "update", "withdraw", "deiconify", "focus", "focus_set"}


def should_ignore_path(path):
    if any(is_ignored_dir_name(part) for part in path.parts):
        return True
    return any(pattern.match(path.name) for pattern in IGNORE_FILE_PATTERNS)


def is_ignored_dir_name(name):
    lowered = name.lower()
    return lowered in IGNORE_DIRS or any(lowered.startswith(prefix) for prefix in IGNORE_DIR_PREFIXES)


def project_files(suffix):
    for root, dirnames, filenames in os.walk(PROJECT_ROOT, onerror=lambda _error: None):
        dirnames[:] = sorted(name for name in dirnames if not is_ignored_dir_name(name))
        root_path = Path(root)
        for filename in sorted(filenames):
            if not filename.lower().endswith(suffix):
                continue
            path = root_path / filename
            if should_ignore_path(path):
                continue
            yield path


def python_files():
    yield from project_files(".py")


def is_test_file(path):
    return path.name.startswith("test_") or "tests" in path.parts


class GeneratedSmokeTests(unittest.TestCase):
    def test_python_files_compile(self):
        for path in python_files():
            with self.subTest(path=str(path.relative_to(PROJECT_ROOT))):
                source = path.read_text(encoding="utf-8", errors="replace")
                compile(source, str(path), "exec")

    def test_json_files_parse(self):
        for path in project_files(".json"):
            with self.subTest(path=str(path.relative_to(PROJECT_ROOT))):
                json.loads(path.read_text(encoding="utf-8"))

    def test_config_files_parse(self):
        for suffix in CONFIG_SUFFIXES:
            for path in project_files(suffix):
                with self.subTest(path=str(path.relative_to(PROJECT_ROOT))):
                    parser = configparser.ConfigParser()
                    parser.read(path, encoding="utf-8")

    def test_text_data_files_are_readable(self):
        for suffix in TEXT_DATA_SUFFIXES:
            for path in project_files(suffix):
                with self.subTest(path=str(path.relative_to(PROJECT_ROOT))):
                    if path.stat().st_size:
                        path.read_text(encoding="utf-8")

    def test_tkinter_button_callbacks_are_not_missing_methods(self):
        for path in python_files():
            source = path.read_text(encoding="utf-8", errors="ignore")
            if "Button" not in source and "button" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            class_methods = {
                node.name: {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            }
            for class_node in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
                for call in [node for node in ast.walk(class_node) if isinstance(node, ast.Call)]:
                    func_name = getattr(call.func, "attr", getattr(call.func, "id", ""))
                    if func_name not in {"Button", "Checkbutton", "Radiobutton", "Menubutton"}:
                        continue
                    command = next((kw.value for kw in call.keywords if kw.arg == "command"), None)
                    if isinstance(command, ast.Attribute) and isinstance(command.value, ast.Name) and command.value.id == "self":
                        with self.subTest(path=str(path.relative_to(PROJECT_ROOT)), line=call.lineno, command=command.attr):
                            if command.attr in COMMON_TK_METHODS:
                                continue
                            self.assertIn(command.attr, class_methods.get(class_node.name, set()))

    def test_likely_entry_imports_quickly(self):
        candidates = [
            path for path in python_files()
            if not is_test_file(path) and path.name.lower() in {"main.py", "app.py", "testing_workbench.py"}
        ]
        for path in candidates[:3]:
            with self.subTest(path=str(path.relative_to(PROJECT_ROOT))):
                start = time.perf_counter()
                spec = importlib.util.spec_from_file_location("generated_smoke_target", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertLess(time.perf_counter() - start, 12)


if __name__ == "__main__":
    unittest.main()
