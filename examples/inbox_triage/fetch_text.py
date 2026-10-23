"""Rebuild the email text for the shipped cases from the public Enron corpus.

    python examples/inbox_triage/fetch_text.py --tar enron_mail_20150507.tar.gz

The repo ships case ids, labels, the gates' cached answers and the reports, but no
email text. Offline replay (`run_gate.py --offline`, `cost_rule.py`, `holdout_check.py`)
doesn't need it. Live gate runs, `--dry-run` and `agent.py` do: this script reads
exactly the messages the cases name (each case carries its maildir `path`) from the CMU
tarball and writes `data/messages.jsonl`.

The corpus: https://www.cs.cmu.edu/~enron/ (May 7 2015 version,
`enron_mail_20150507.tar.gz`, about 443 MB). It is the FERC public release; CMU asks
users to respect the privacy of the people in it. Phone and SSN-like numbers are
replaced the same way they were when the cases were labeled, and bodies are cut at
2,000 characters, so the text matches what every gate saw. Each rebuilt id is checked
against `sha1(Message-ID)`.
"""

from __future__ import annotations

import argparse
import email
import email.policy
import hashlib
import json
import re
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MAX_BODY = 2000

PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,5})?(?!\w)")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")


def scrub(text: str) -> str:
    text = CARD.sub("[NUMBER]", text)
    text = SSN.sub("[NUMBER]", text)
    return PHONE.sub("[PHONE]", text)


def body_of(msg) -> str:
    if msg.is_multipart():
        parts = [p for p in msg.walk() if p.get_content_type() == "text/plain"]
        payload = parts[0].get_content() if parts else ""
    else:
        payload = msg.get_content()
    return payload if isinstance(payload, str) else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", type=Path, required=True, help="enron_mail_20150507.tar.gz")
    ap.add_argument("--out", type=Path, default=DATA / "messages.jsonl")
    args = ap.parse_args()

    wanted: dict[str, tuple[str, str]] = {}  # tar path -> (case id, split)
    for split in ("eval", "holdout"):
        for line in (DATA / "cases" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                wanted[c["path"]] = (c["id"], split)

    out, bad = [], []
    with tarfile.open(args.tar, "r:gz") as tf:
        for m in tf:
            if m.name not in wanted:
                continue
            case_id, split = wanted[m.name]
            fh = tf.extractfile(m)
            if fh is None:
                continue
            msg = email.message_from_bytes(fh.read(), policy=email.policy.default)
            message_id = str(msg.get("Message-ID", "")).strip()
            if "enron-" + hashlib.sha1(message_id.encode()).hexdigest()[:12] != case_id:
                bad.append(case_id)
                continue
            body = body_of(msg).strip()
            out.append({
                "id": case_id, "split": split, "spot": False, "user": m.name.split("/")[1],
                "date": str(msg.get("Date", "")),
                "from": str(msg.get("X-From") or msg.get("From", "")),
                "to": str(msg.get("X-To") or msg.get("To", "")),
                "cc": str(msg.get("X-cc") or msg.get("Cc", "")),
                "subject": str(msg.get("Subject", "")),
                "body": scrub(body[:MAX_BODY]),
                "truncated": len(body) > MAX_BODY,
            })

    missing = len(wanted) - len(out) - len(bad)
    if bad or missing:
        raise SystemExit(f"{len(bad)} ids did not match their Message-ID and {missing} paths were not in "
                         f"the tarball; is it the May 7 2015 version?")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in sorted(out, key=lambda r: r["id"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} messages to {args.out}")


if __name__ == "__main__":
    main()
