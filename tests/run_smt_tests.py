#!/usr/bin/env python3
import importlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    failures = []
    count = 0
    for module_name in ('tests.test_smt_symmetry', 'tests.test_smt_metrics'):
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith('test_'):
                continue
            count += 1
            try:
                function()
                print(f'PASS {module_name}.{name}')
            except Exception as exc:
                failures.append((module_name, name, repr(exc)))
                print(f'FAIL {module_name}.{name}: {exc!r}')
    print(f'tests={count} failures={len(failures)}')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
