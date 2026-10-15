"""Label retrieval-depth samples with a Claude judge.

Each sample receives two judge votes. Final label 0 requires both votes to
be faithful; a hallucinated, uncertain or unparseable vote gives label 1.
The judgment concerns support from the supplied context.

Input:
  analysis/vary_k_results/vary_k/<model>/<dataset>/vary_k_n*_k*.json
Outputs:
  analysis/vary_k_labeled_<judge>/vary_k/ (labeled records and summaries)

The --multiseed option uses the corresponding multiseed_k1 directories.
Outputs are separated by judge model to support cross-judge comparisons.
See SYSTEM_PROMPT for the complete labeling instructions.

Usage:
  python3 tools/label_vary_k.py --dry-run
  python3 tools/label_vary_k.py
  python3 tools/label_vary_k.py --cell gemma-3-4b

Requires ANTHROPIC_API_KEY in the environment. Labeling calls the hosted API;
--dry-run lists the planned work without making API requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from typing import Optional

try:
    import anthropic
except ImportError:
    sys.exit("missing dependency: pip install --user anthropic")


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Default input/output for the vary_k pipeline. The script also supports the
# multi-seed §5.7 protocol-sensitivity sweep when run with --multiseed; in
# that mode RAW_DIR and OUT_DIR are swapped to the multiseed_k1 hierarchy.
RAW_DIR = REPO_ROOT / "analysis" / "vary_k_results" / "vary_k"

MODEL = "claude-opus-4-6"
MAX_CONCURRENCY = 20           # parallel Anthropic API calls in flight
MAX_TOKENS_PER_JUDGE = 32      # 1 label + maybe a short rationale token
TIMEOUT_SECONDS = 60

# Output is namespaced by judge model so multiple judges can be compared
# (e.g. Cohen's kappa across claude-sonnet-4-6 vs claude-opus-4-6).
OUT_DIR = REPO_ROOT / "analysis" / f"vary_k_labeled_{MODEL}" / "vary_k"

# Multi-seed paths used when --multiseed is passed.
MULTISEED_RAW_DIR = REPO_ROOT / "analysis" / "multiseed_k1_results" / "multiseed_k1"
MULTISEED_OUT_DIR = REPO_ROOT / "analysis" / f"multiseed_k1_labeled_{MODEL}" / "multiseed_k1"
MULTISEED_GLOB = "*/*/k1_s*.json"
VARY_K_GLOB = "*/*/vary_k_n*_k*.json"

# Conservative double-vote system prompt. Designed to be cache-stable:
# every byte after the last user-injected variable is identical across calls.
SYSTEM_PROMPT = """You are a strict hallucination judge for retrieval-augmented question answering.

You will be given:
  - a QUESTION
  - the CONTEXT that was retrieved for that question (one or more passages)
  - a GOLD answer (the human-verified reference)
  - a GENERATED answer produced by a language model conditioned on the CONTEXT

Decide whether the GENERATED answer is HALLUCINATED relative to the CONTEXT.

Definitions:
  - faithful (label = 0): every factual claim in the GENERATED answer is directly
    supported by the CONTEXT. It is fine if the GENERATED answer omits details;
    it is fine if it paraphrases.
  - hallucinated (label = 1): the GENERATED answer contains at least one claim
    that is not supported by the CONTEXT, contradicts the CONTEXT, or appears
    to be fabricated from prior knowledge.

Special rules (apply in order, top to bottom):
  1. Ignore external world knowledge. Judge only against the CONTEXT.
  2. If the CONTEXT is too sparse to support the GENERATED answer's claims even
     though the claims happen to be true in the real world: label hallucinated.
  3. If the GENERATED answer is mostly correct but adds an unsupported detail
     (date, name, number, attribution): label hallucinated.
  4. If the GOLD answer is itself not supported by the CONTEXT, that does not
     change the rules above. Judge the GENERATED answer against the CONTEXT.
  5. If you are unsure: label hallucinated (conservative default).

Output format. Reply with one of these literal tokens and nothing else:
  FAITHFUL
  HALLUCINATED

Do not include explanations, punctuation, or extra whitespace."""


# ----- parsing + voting -----

def parse_label(text: str) -> Optional[int]:
    """Parse a single judge response into 0 (faithful), 1 (hallucinated), or None (unparseable)."""
    s = text.strip().upper()
    # tolerate small wrapper noise
    s = s.strip("`*. \n\t\"'")
    if s.startswith("FAITHFUL"):
        return 0
    if s.startswith("HALLUCINATED"):
        return 1
    return None


def conservative_vote(label_a: Optional[int], label_b: Optional[int]) -> int:
    """Conservative voting rule from the existing tests/12/super notebook.

    - both faithful (0/0)              -> 0
    - any hallucinated or unparseable  -> 1
    """
    if label_a == 0 and label_b == 0:
        return 0
    return 1


# ----- prompt construction -----

def build_user_prompt(question: str, context, gold_answer: str, generated_answer: str) -> str:
    context_str = "\n\n".join(context) if isinstance(context, list) else str(context)
    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "CONTEXT:\n"
        f"{context_str}\n\n"
        "GOLD ANSWER:\n"
        f"{gold_answer}\n\n"
        "GENERATED ANSWER:\n"
        f"{generated_answer}\n\n"
        "Reply with one token: FAITHFUL or HALLUCINATED."
    )


# ----- single-sample async pipeline -----

async def judge_one_sample(
    client: anthropic.AsyncAnthropic,
    sample: dict,
    sem: asyncio.Semaphore,
) -> dict:
    """Run two independent judge calls for one sample, apply conservative vote, return updated sample."""
    user_prompt = build_user_prompt(
        question=sample["question"],
        context=sample.get("context", ""),
        gold_answer=sample.get("gold_answer", ""),
        generated_answer=sample.get("generated_answer", ""),
    )

    async def _call_judge() -> tuple[Optional[int], str]:
        # Two-message system prompt with cache_control on the second block: the
        # first call pays a 1.25x write premium, every subsequent call (across
        # all 1800) pays ~0.1x for the system prefix.
        async with sem:
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS_PER_JUDGE,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                    timeout=TIMEOUT_SECONDS,
                )
            except anthropic.APIError as e:
                return None, f"<api error: {type(e).__name__}: {e}>"
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text
        return parse_label(text), text.strip()

    # two independent calls (each gets its own non-deterministic sampling)
    judge_a, raw_a = await _call_judge()
    judge_b, raw_b = await _call_judge()
    label = conservative_vote(judge_a, judge_b)

    out = dict(sample)
    out["hallucinated"] = label
    out["judge"] = {
        "judge_model": MODEL,
        "judge_a_label": judge_a,
        "judge_b_label": judge_b,
        "judge_a_raw": raw_a,
        "judge_b_raw": raw_b,
        "voting": "conservative_double",
    }
    return out


# ----- file pipeline -----

async def label_file(
    client: anthropic.AsyncAnthropic,
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    sem: asyncio.Semaphore,
) -> dict:
    samples = json.loads(in_path.read_text())
    if not isinstance(samples, list):
        raise ValueError(f"expected a JSON list in {in_path}, got {type(samples).__name__}")

    # idempotent resume: if the output already has labels for some samples, reuse them
    existing: dict[int, dict] = {}
    if out_path.exists():
        try:
            for prev in json.loads(out_path.read_text()):
                if prev.get("hallucinated") is not None:
                    existing[prev["id"]] = prev
        except Exception:
            pass

    todo = [s for s in samples if s["id"] not in existing]
    if not todo:
        labeled = [existing[s["id"]] for s in samples]
    else:
        print(f"  judging {len(todo)} of {len(samples)} samples (resuming {len(existing)}) ...")
        tasks = [judge_one_sample(client, s, sem) for s in todo]
        new_labels = await asyncio.gather(*tasks)
        by_id = {s["id"]: s for s in new_labels}
        labeled = [existing.get(s["id"]) or by_id[s["id"]] for s in samples]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labeled, indent=2))

    n_hall = sum(1 for s in labeled if s["hallucinated"] == 1)
    return {
        "in_path": str(in_path.relative_to(REPO_ROOT)),
        "out_path": str(out_path.relative_to(REPO_ROOT)),
        "n": len(labeled),
        "n_hallucinated": n_hall,
        "hall_rate": round(n_hall / len(labeled), 3) if labeled else 0.0,
    }


# ----- orchestrator -----

async def main_async(args) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("missing ANTHROPIC_API_KEY (source .env first)")

    if args.multiseed:
        raw_dir = MULTISEED_RAW_DIR
        out_dir = MULTISEED_OUT_DIR
        glob = MULTISEED_GLOB
    else:
        raw_dir = RAW_DIR
        out_dir = OUT_DIR
        glob = VARY_K_GLOB

    cells = sorted(raw_dir.glob(glob))
    if args.cell:
        cells = [c for c in cells if args.cell in str(c)]
    cells = [c for c in cells if "_n1_" not in c.name]  # skip 1-sample smoke tests

    if not cells:
        sys.exit(f"no input files matched under {raw_dir}")

    print(f"found {len(cells)} cell file(s) under {raw_dir}")
    for c in cells:
        n_samples = len(json.loads(c.read_text()))
        print(f"  - {c.relative_to(REPO_ROOT)}  ({n_samples} samples)")

    if args.dry_run:
        print("\n[dry run] not calling the API.")
        return 0

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    client = anthropic.AsyncAnthropic()

    t0 = time.time()
    results = []
    for c in cells:
        rel = c.relative_to(raw_dir)
        out_path = out_dir / rel
        print(f"\n[cell] {rel}")
        summary = await label_file(client, c, out_path, sem)
        print(f"  -> hall_rate = {summary['hall_rate']}  ({summary['n_hallucinated']}/{summary['n']})")
        results.append(summary)

    print("\nsummary:")
    print(f"  total samples: {sum(r['n'] for r in results)}")
    print(f"  total hallucinated: {sum(r['n_hallucinated'] for r in results)}")
    print(f"  wall time: {time.time() - t0:.1f}s")

    summary_path = out_dir.parent / "labeling_summary.json"
    summary_path.write_text(json.dumps({
        "judge_model": MODEL,
        "voting": "conservative_double",
        "max_concurrency": MAX_CONCURRENCY,
        "cells": results,
    }, indent=2))
    print(f"\nwrote {summary_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default=None, help="substring filter (e.g. 'gemma-3-4b')")
    p.add_argument("--dry-run", action="store_true", help="list inputs and exit; no API calls")
    p.add_argument("--multiseed", action="store_true",
                   help="label the multi-seed §5.7 sweep (analysis/multiseed_k1_results/) instead of vary_k")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
