"""Rebuild exploration artifacts and execute the exploration notebook."""
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    for script in ['inspect_data.py', 'prepare_data.py', 'analyze_patterns.py', 'evaluate_baselines.py']:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script)], cwd=ROOT, check=True)
    path = ROOT / 'notebooks/01_data_exploration.ipynb'
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, path)
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'matplotlib', 'pyarrow', 'nbformat', 'nbclient', 'ipykernel']}
    versions['python'] = sys.version
    (ROOT / 'reports/environment.json').write_text(json.dumps(versions, indent=2))
    print('Notebook executed and saved.', flush=True)


if __name__ == '__main__':
    main()
