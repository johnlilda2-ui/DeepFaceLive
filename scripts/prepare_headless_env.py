#!/usr/bin/env python3
"""Prepare or repair the experiment's pip-free venv without deleting packages."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


def ensure_environment(directory):
    directory = Path(directory).resolve()
    interpreter = directory / 'bin/python'
    if directory.exists() and not (directory / 'pyvenv.cfg').is_file():
        raise ValueError(f'Refusing to overwrite a directory that is not a venv: {directory}')
    if (directory / 'pyvenv.cfg').is_file():
        try:
            result = subprocess.run([str(interpreter), '--version'], check=True,
                                    capture_output=True, text=True, timeout=20)
            if result.stdout.startswith(f'Python {sys.version_info.major}.{sys.version_info.minor}.'):
                return interpreter
        except (OSError, subprocess.SubprocessError):
            pass
        # Kaggle persistence can restore interpreter symlinks as empty files.
        # Recreate only interpreter aliases; retain installed packages and models.
        for name in ('python', 'python3', f'python{sys.version_info.major}.{sys.version_info.minor}'):
            (directory / 'bin' / name).unlink(missing_ok=True)
    venv.EnvBuilder(with_pip=False, symlinks=False).create(directory)
    subprocess.run([str(interpreter), '--version'], check=True, timeout=20)
    return interpreter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(ensure_environment(parser.parse_args().directory))
