"""A real measurement on a real public dataset.

Banking77 is 13,083 genuine customer-service messages from a retail bank,
labelled into 77 intents. Routing a message to the right queue is exactly the
kind of workflow a pilot gets scoped around, so it makes an honest stand-in for
"can we automate our support triage?"

The agent here is a deliberately ordinary baseline: TF-IDF nearest-centroid,
pure stdlib, no model API. That is the point. A plausible-looking baseline is
what most demos are built on, and the question is whether the harness tells you
the truth about it.

    python examples/banking77_routing.py

Data is fetched once from the PolyAI repository and cached next to this file.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from gonogo import Case, Prediction, evaluate
from gonogo.scoring import exact

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
CACHE = Path(__file__).parent / "data"
TOKEN = re.compile(r"[a-z0-9']+")

# A pilot ships with a case set a human actually labelled, not a full benchmark
# split. 250 is a realistic size and it keeps the small-n lesson visible.
PILOT_SIZE = 250
SEED = 11


def fetch(name: str) -> list[tuple[str, str]]:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"banking77_{name}.csv"
    if not path.exists():
        print(f"downloading banking77 {name} split...")
        urllib.request.urlretrieve(f"{BASE}/{name}.csv", path)
    with path.open(encoding="utf-8", newline="") as fh:
        return [(row["text"], row["category"]) for row in csv.DictReader(fh)]


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class NearestCentroid:
    """TF-IDF nearest-centroid classifier with a softmax confidence.

    Confidence is the softmax over centroid similarities. It is a real signal --
    ambiguous messages do score lower -- but it was never calibrated against
    anything, which is exactly the situation most agents are in.
    """

    def __init__(self, temperature: float = 12.0):
        self.temperature = temperature
        self.idf: dict[str, float] = {}
        self.centroids: dict[str, dict[str, float]] = {}

    def fit(self, rows: list[tuple[str, str]]) -> "NearestCentroid":
        docs = [tokenize(t) for t, _ in rows]
        df = Counter()
        for tokens in docs:
            df.update(set(tokens))
        n = len(docs)
        self.idf = {term: math.log(n / (1 + count)) + 1.0 for term, count in df.items()}

        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Counter = Counter()
        for tokens, (_, label) in zip(docs, rows):
            for term, weight in self._vector(tokens).items():
                sums[label][term] += weight
            counts[label] += 1

        for label, vec in sums.items():
            centroid = {t: w / counts[label] for t, w in vec.items()}
            self.centroids[label] = self._normalize(centroid)
        return self

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm else vec

    def predict(self, text: str) -> tuple[str, float]:
        vec = self._vector(tokenize(text))
        scores = {
            label: sum(w * centroid.get(t, 0.0) for t, w in vec.items())
            for label, centroid in self.centroids.items()
        }
        if not scores:
            return "", 0.0
        best = max(scores, key=scores.get)
        top = max(scores.values())
        total = sum(math.exp(self.temperature * (s - top)) for s in scores.values())
        return best, 1.0 / total


def main() -> None:
    train = fetch("train")
    test = fetch("test")
    model = NearestCentroid().fit(train)

    pilot = random.Random(SEED).sample(test, PILOT_SIZE)
    cases = [
        Case(input=text, expected=label, id=f"msg-{i:04d}")
        for i, (text, label) in enumerate(pilot)
    ]

    def agent(case: Case) -> Prediction:
        label, confidence = model.predict(case.input)
        return Prediction(output=label, confidence=round(confidence, 3))

    report = evaluate(
        agent,
        cases,
        scorer=exact(),
        task="Route customer message to the right queue",
        target=0.95,
    )
    print(report.markdown())

    out = Path(__file__).parent / "banking77_report.html"
    out.write_text(report.html(), encoding="utf-8")

    # Per-case outcomes, so this baseline can be paired against a model run.
    data = Path(__file__).parent / "banking77_baseline.json"
    data.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\n[html report written to {out}]")
    print(f"[per-case results written to {data}]")

    # For context: the same agent measured on the full held-out split.
    full = [Case(input=t, expected=l, id=str(i)) for i, (t, l) in enumerate(test)]
    full_report = evaluate(agent, full, scorer=exact(), task="full test split", target=0.95)
    print(f"[full split, n={full_report.n}: {full_report.decision.pass_rate}]")


if __name__ == "__main__":
    main()
