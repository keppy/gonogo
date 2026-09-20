"""Rate a fine-tuned encoder classifier on the Banking77 canary pilot.

The counterpart to `banking77_routing.py` (TF-IDF baseline) and
`banking77_claude_agent.py` (frontier LLM): this agent is a ModernBERT-small
encoder fine-tuned by the thomas pipeline (`thomas/examples/banking77_encoder.py`,
trained on Modal). Same seed, same 250-case slice, so all three reports score
the identical case set and can be read side by side.

The confidence is a real posterior: `max(softmax(logits / T))` over the full
77-class distribution, with the temperature T fit on a held-out calibration
split by the training pipeline. That is what makes gonogo's ECE and operating
point meaningful here -- a calibrated classifier lets the report answer "how
much can we automate, at what precision, with the rest going to a human."

    python examples/banking77_encoder_eval.py --model-dir <artifact-dir>

The artifact dir is whatever the thomas driver exported: an HF
`save_pretrained` output plus `label2id.json`, `temperature.json` and
`metrics.json` (see the cross-repo contract in the thomas repo's plan).
"""

from __future__ import annotations

import argparse
import json
import os
import random

from gonogo import Case, Prediction, evaluate
from gonogo.scoring import exact

from banking77_routing import PILOT_SIZE, SEED, fetch

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - the message is the point
    raise SystemExit("this example needs transformers + torch: pip install transformers torch")

BATCH = 64
MAX_LENGTH = 128


def build_agent(model_dir: str):
    """Load the exported classifier; return Case -> Prediction.

    One confidence definition, shared with the training pipeline: softmax over
    the full 77-class distribution, temperature applied to the logits BEFORE
    the softmax, confidence = max class probability.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    with open(os.path.join(model_dir, "label2id.json"), encoding="utf-8") as fh:
        label2id = json.load(fh)
    id2label = {i: l for l, i in label2id.items()}

    with open(os.path.join(model_dir, "temperature.json"), encoding="utf-8") as fh:
        temperature = float(json.load(fh)["temperature"])
    if not temperature > 0:
        raise SystemExit(f"bad temperature in {model_dir}: {temperature}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def agent(case: Case) -> Prediction:
        enc = tokenizer(case.input, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits[0]
        probs = torch.softmax(logits / temperature, dim=-1)
        conf, idx = probs.max(dim=-1)
        return Prediction(output=id2label[int(idx)], confidence=round(float(conf), 3))

    return agent


def batch_agent(agent, cases: list[Case]) -> dict[str, Prediction]:
    """The same predictions, one case at a time via the shared agent.

    Kept simple (per-case forward) -- 250 short texts is seconds even on CPU.
    """
    return {c.id: agent(c) for c in cases}


def confidence_aware():
    """Score = confidence when correct, 0 when wrong.

    A plain exact() scorer would leave ECE as NaN and the operating point
    empty; passing the classifier's real confidence through is what lets the
    risk-coverage curve answer the automation question.
    """

    def score(output, expected):
        label, confidence = output
        ok = label == expected
        return ok, confidence if ok else 0.0, "" if ok else f"expected {expected!r}, got {label!r}"

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="artifact dir exported by thomas/examples/banking77_encoder.py")
    parser.add_argument("--limit", type=int, default=PILOT_SIZE,
                        help=f"number of cases (default: {PILOT_SIZE})")
    parser.add_argument("--target", type=float, default=0.95,
                        help="precision target the agent must clear (default: 0.95)")
    parser.add_argument("--baseline", default="banking77_baseline.json",
                        help="baseline report JSON to pair against (skip if absent)")
    args = parser.parse_args()

    _, test = fetch("train"), fetch("test")

    # Same seed and same slice as banking77_routing.py, so the runs score the
    # identical case set and the reports can be read side by side.
    pilot = random.Random(SEED).sample(test, args.limit)
    cases = [Case(input=t, expected=l, id=f"msg-{i:04d}") for i, (t, l) in enumerate(pilot)]

    with open(os.path.join(args.model_dir, "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    print(f"[model metrics from training: {json.dumps(metrics, indent=None)}]")

    report = evaluate(
        build_agent(args.model_dir),
        cases,
        scorer=confidence_aware(),
        task=f"Route customer message to the right queue (encoder: {os.path.basename(args.model_dir.rstrip('/'))})",
        target=args.target,
    )
    print(report.markdown())

    here = os.path.dirname(__file__)
    out = os.path.join(here, "banking77_encoder_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report.html())

    # Per-case outcomes, so this run can be paired against the baseline:
    #   from gonogo import compare
    #   compare(json.load(open("banking77_baseline.json")),
    #           json.load(open("banking77_encoder.json")))
    data = os.path.join(here, "banking77_encoder.json")
    with open(data, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"\n[html report written to {out}]")
    print(f"[per-case results written to {data}]")

    baseline = os.path.join(here, args.baseline)
    if os.path.exists(baseline):
        try:
            from gonogo import compare
            print("\n[vs TF-IDF baseline]")
            compare(json.load(open(baseline)), report.to_dict())
        except Exception as exc:  # pairing is a bonus, never a blocker
            print(f"[baseline comparison skipped: {exc!r}]")


if __name__ == "__main__":
    main()
