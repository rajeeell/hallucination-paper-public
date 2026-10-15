"""Small tests for numerical helpers and judge parsing, without inference."""
import ast
import importlib.util
from pathlib import Path
import socket

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('A test attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)

def calibration_module():
    spec = importlib.util.spec_from_file_location('release_calibration', ROOT / 'analysis/calibration_analysis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_ece_includes_probability_one():
    m = calibration_module()
    y = np.array([0, 1])
    assert m.expected_calibration_error(y, np.array([0., 1.])) == 0
    assert m.expected_calibration_error(y, np.array([1., 0.])) == 1

def test_precision_threshold_uses_highest_recall_candidate():
    m = calibration_module()
    y = np.array([0, 1, 1, 0])
    threshold, precision, recall = m.threshold_at_precision(y, np.array([.1, .9, .8, .2]), .95)
    assert threshold == .8
    assert precision == 1
    assert recall == 1

def judge_functions():
    # Extract only pure parser/vote functions, avoiding the hosted API module
    # import and all client initialization. No module top-level code executes.
    tree = ast.parse((ROOT / 'tools/label_vary_k.py').read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'parse_label', 'conservative_vote'}]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + functions, type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {}
    exec(compile(module, '<pure-judge-functions>', 'exec'), scope)
    return scope

@pytest.mark.parametrize('text, expected', [('FAITHFUL', 0), ('`HALLUCINATED`', 1), ('unsure', None)])
def test_judge_parser(text, expected):
    assert judge_functions()['parse_label'](text) == expected

@pytest.mark.parametrize('a,b,expected', [(0,0,0), (0,1,1), (1,0,1), (None,0,1), (0,None,1)])
def test_judge_conservative_vote(a, b, expected):
    assert judge_functions()['conservative_vote'](a, b) == expected
