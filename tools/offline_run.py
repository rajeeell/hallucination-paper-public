"""Run a selected analysis script with socket connections disabled."""
from pathlib import Path
import os
import runpy
import socket
import sys

ALLOWED = {
    'analysis/calibration_analysis.py',
    'analysis/make_perplexity_distribution_fig.py',
    'analysis/protocol_sensitivity.py',
    'analysis/rigor_audit_r10_r6.py',
    'analysis/bootstrap_aurocs_vary_k.py',
    'tools/verify_paper_metrics.py',
    'tools/verify_icnlsp.py',
}

def deny(*args, **kwargs):
    raise RuntimeError('Network access is disabled during offline analysis')

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ALLOWED:
        raise SystemExit('Specify one supported offline script: ' + ', '.join(sorted(ALLOWED)))
    root = Path(__file__).resolve().parent.parent
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']:
        os.environ[key] = '1'
    for key in ['HF_TOKEN', 'GROQ_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY']:
        os.environ.pop(key, None)
    os.environ['MPLBACKEND'] = 'Agg'
    script = root / sys.argv[1]
    sys.argv = [str(script)]
    sys.path.insert(0, str(root))
    runpy.run_path(str(script), run_name='__main__')

if __name__ == '__main__':
    main()
