"""Rebuild the exploration and forecasting steps in order."""
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    scripts = [
        'run_exploration.py',
        'prepare_forecasting_data.py',
        'run_forecasting.py',
    ]
    for script in scripts:
        print(f'Running {script}', flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / script)],
            cwd=ROOT,
            check=True,
        )

    path = ROOT / 'notebooks/02_forecasting.ipynb'
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=-1,
        kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}},
    ).execute()
    nbformat.write(notebook, path)
    print('Exploration and forecasting are complete.', flush=True)


if __name__ == '__main__':
    main()
