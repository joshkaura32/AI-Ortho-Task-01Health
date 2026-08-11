"""Self-eval CLI — score a plans directory against this pack's gold.

From the pack root:
    python starter/score.py --plans submissions/mine --split train

Train cases carry their gold (gold_transforms.json), so split=train is your dev loop.
Eval gold is withheld — if you send plans (suggested), emit one per (case, instruction)
in eval/; we score them with this same metrics.py against the hidden gold.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import format_report, score_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(Path(__file__).resolve().parents[1]),
                    help="pack root (default: this kit's parent directory)")
    ap.add_argument("--plans", required=True, help="directory of <case>__<iid>.json plans")
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    args = ap.parse_args()
    report = score_pack(args.pack, args.plans, split=args.split)
    print(format_report(report, Path(args.plans).name))


if __name__ == "__main__":
    main()
