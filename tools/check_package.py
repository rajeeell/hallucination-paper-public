"""Check syntax, notebook structure, outputs and credential patterns offline.

Does not import or execute experiment code. Python syntax checks cannot prove
that optional model/service dependencies or GPU inference work.
"""
from pathlib import Path
import json
import re
import sys
import warnings

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parent.parent
SKIP = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'release_verification'}
PATTERNS = [re.compile(rb'hf_[A-Za-z0-9]{25,}'), re.compile(rb'gsk_[A-Za-z0-9]{25,}'),
            re.compile(rb'sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{25,}'),
            re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
            re.compile(rb'AKIA[A-Z0-9]{16}'),
            re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]

def main():
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    transformer = TransformerManager()
    errors = []
    counts = {'python': 0, 'notebooks': 0, 'json': 0}
    for path in ROOT.rglob('*'):
        if not path.is_file() or SKIP.intersection(path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        if any(b'xxxxx' not in match.group().lower() for pattern in PATTERNS for match in pattern.finditer(data)):
            errors.append({'path': rel, 'error': 'credential pattern; value withheld'})
        if path.suffix in {'.tex', '.bib', '.bst', '.sty', '.cls', '.docx', '.zip'}:
            errors.append({'path': rel, 'error': 'manuscript/template/document file in candidate'})
        try:
            if path.suffix == '.py':
                compile(data.decode('utf-8-sig'), rel, 'exec')
                counts['python'] += 1
            elif path.suffix == '.json':
                json.loads(data.decode('utf-8'))
                counts['json'] += 1
            elif path.suffix == '.ipynb':
                notebook = nbformat.reads(data.decode('utf-8'), as_version=4)
                nbformat.validate(notebook)
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == 'code':
                        if cell.outputs or cell.execution_count is not None:
                            raise ValueError(f'Cell {index} has saved outputs or execution count')
                        compile(transformer.transform_cell(cell.source), f'{rel}:cell-{index}', 'exec')
                counts['notebooks'] += 1
        except Exception as error:
            errors.append({'path': rel, 'error': str(error)})
    print(json.dumps({'validated': counts, 'errors': errors}, indent=2))
    return int(bool(errors))

if __name__ == '__main__':
    sys.exit(main())
