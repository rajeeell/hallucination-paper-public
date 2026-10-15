"""Offline reanalysis of the final ICNLSP headline metrics; no model inference."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import pickle
import socket
import sys

# A guard for this verification process only. The research inference runners
# remain separate and are never imported here.
def _deny_network(*args, **kwargs):
    raise RuntimeError('Network access is disabled during offline verification')

socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
socket.create_connection = _deny_network
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

class ArrayUnpickler(pickle.Unpickler):
    """Read the project's numeric pickles without allowing arbitrary globals."""
    def find_class(self, module, name):
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name in {'_reconstruct', 'scalar'}:
            return getattr(np.core.multiarray, name)
        if module == 'numpy' and name in {'ndarray', 'dtype'}:
            return getattr(np, name)
        if module in {'numpy.core.numeric', 'numpy._core.numeric'} and name == '_frombuffer':
            return np.core.numeric._frombuffer
        if module == 'builtins' and name in {'set', 'frozenset', 'slice', 'complex'}:
            import builtins
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')

def load_arrays(path):
    with Path(path).open('rb') as stream:
        return ArrayUnpickler(stream).load()

def ece(y, p, bins=10):
    result = 0.0
    for i in range(bins):
        mask = (p >= i / bins) & ((p <= 1) if i == bins - 1 else (p < (i + 1) / bins))
        if mask.any():
            result += mask.mean() * abs(float(p[mask].mean() - y[mask].mean()))
    return float(result)

def paper_predictions(X, y):
    # Original calibration/RAGTruth analysis protocol (C=1, raw perplexity).
    # This differs from tests/09's controlled AUROC notebook protocol.
    # Fitting the scaler globally is a documented limitation.
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    return cross_val_predict(lr, Xs, y, cv=5, method='predict_proba')[:, 1]

def main():
    config = json.loads((ROOT / 'configs/experiments.json').read_text(encoding='utf-8'))
    checks, controlled, calibration, external = [], [], [], []
    def check(name, actual, expected, hard=0.00051, soft=None):
        delta = abs(actual - expected)
        status = 'PASS' if np.isfinite(actual) and delta <= hard else 'WARN' if soft is not None and np.isfinite(actual) and delta <= soft else 'FAIL'
        row = {'name': name, 'actual': float(actual), 'published': float(expected), 'delta': float(delta), 'status': status}
        checks.append(row)
        print(f'{status:4} {name}: {actual:.6f} (published {expected:.6f})')
    for cell in config['controlled_cells']:
        d = load_arrays(ROOT / cell['features'])
        y = np.asarray(d['labels'], dtype=int)
        arrays = [np.asarray(d[key], dtype=float) for key in ['perplexity', 'mean_prob', 'min_prob', 'SDS']]
        if set(np.unique(y)) != {0, 1} or any(x.shape != y.shape or not np.isfinite(x).all() for x in arrays):
            raise ValueError(f'Invalid features or labels in {cell["features"]}')
        X = np.column_stack(arrays)
        name = cell['cell']
        check(name + ' sample count', len(y), cell['n'], hard=0)
        ppl_auc = float(roc_auc_score(y, arrays[0]))
        # Reproduce the actual tests/09 notebook: C=0.1, PPL clipped to [1,100],
        # feature order SDS, PPL, mean probability, minimum probability.
        clean_ppl = np.nan_to_num(np.clip(arrays[0], 1, 100), nan=0.0, posinf=1000.0, neginf=-1000.0)
        notebook_features = np.column_stack([arrays[3], clean_ppl, arrays[1], arrays[2]])
        original_lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        combined_auc = float(cross_val_score(original_lr, StandardScaler().fit_transform(notebook_features), y,
                                            cv=5, scoring='roc_auc').mean())
        two_feature_auc = float(cross_val_score(original_lr, StandardScaler().fit_transform(notebook_features[:, :2]), y,
                                               cv=5, scoring='roc_auc').mean())
        saved_results = {row['Method']: float(row['AUROC']) for row in d['results']}
        # Compare unrounded metrics with the configured table values and
        # compare the four-feature refit with its saved notebook score.
        check(name + ' PPL AUROC', ppl_auc, cell['published_ppl_auc'], hard=0.0005, soft=0.001)
        check(name + ' four-feature combined vs published', combined_auc, cell['published_combined_auc'], hard=0.0005)
        check(name + ' notebook refit vs saved four-feature result', combined_auc,
              saved_results['SDS + All Uncertainty'], hard=1e-10)
        controlled.append({'cell': name, 'n': len(y), 'ppl_auc': ppl_auc,
                           'combined_four_feature_notebook_auc': combined_auc,
                           'combined_two_feature_notebook_auc': two_feature_auc,
                           'saved_four_feature_auc': saved_results['SDS + All Uncertainty'],
                           'published_combined_auc': cell['published_combined_auc'],
                           'published_matches_two_feature_at_3dp': round(two_feature_auc, 3) == cell['published_combined_auc'],
                           'published_matches_four_feature_at_3dp': round(combined_auc, 3) == cell['published_combined_auc']})
        pp = paper_predictions(X[:, :1], y)
        pc = paper_predictions(X, y)
        calibration.append({'cell': name, 'ppl_ece': ece(y, pp), 'combined_ece': ece(y, pc),
                            'ppl_brier': float(brier_score_loss(y, pp)), 'combined_brier': float(brier_score_loss(y, pc))})
    for metric, published in {'ppl_ece': 0.072, 'combined_ece': 0.035, 'ppl_brier': 0.187, 'combined_brier': 0.150}.items():
        check('Controlled mean ' + metric, np.mean([x[metric] for x in calibration]), published, hard=0.005, soft=0.020)
    for cell in config['natural_reversals']:
        records = json.loads((ROOT / cell['path']).read_text(encoding='utf-8'))
        y = np.array([x['hallucinated'] for x in records])
        score = np.array([x['perplexity'] for x in records])
        check(cell['cell'] + ' natural k=1 PPL AUROC', roc_auc_score(y, score), cell['published_auc'])
        check(cell['cell'] + ' unsupported count', y.sum(), cell['unsupported'], hard=0)
    for model in config['models']:
        path = ROOT / 'analysis/ragtruth_results/ragtruth' / model['slug'] / 'ragtruth_qa.json'
        records = json.loads(path.read_text(encoding='utf-8'))['records']
        y = np.array([x['hallucinated'] for x in records])
        X = np.array([[r[k] for k in ['perplexity', 'mean_prob', 'min_prob', 'sds']] for r in records])
        check(model['name'] + ' RAGTruth sample count', len(y), 875, hard=0)
        external.append({'model': model['name'], 'n': len(y), 'ppl_auc': float(roc_auc_score(y, X[:, 0])),
                         'combined_oof_auc': float(roc_auc_score(y, paper_predictions(X, y)))})
    check('RAGTruth mean PPL AUROC', np.mean([x['ppl_auc'] for x in external]), 0.584)
    check('RAGTruth mean combined OOF AUROC', np.mean([x['combined_oof_auc'] for x in external]), 0.643, hard=0.005, soft=0.020)
    result = {'scope': 'Saved-artifact offline reanalysis; no new inference or judge labels',
              'checks': checks, 'controlled': controlled, 'calibration': calibration, 'ragtruth': external,
              'summary': {s: sum(c['status'] == s for c in checks) for s in ['PASS', 'WARN', 'FAIL']}}
    out = ROOT / 'release_verification/icnlsp_metrics.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('SUMMARY', result['summary'])
    return int(result['summary']['FAIL'] > 0)

if __name__ == '__main__':
    sys.exit(main())
