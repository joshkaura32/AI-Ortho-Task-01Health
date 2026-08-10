"""A complete, VALID (do-nothing) submission — the shape of what you send back.

Replace plan_case() with your model. Everything else — case loading, the instruction
battery loop, file naming, the plan format — is already exactly what our harness expects.

    python starter/example_predict.py --out submissions/mine            # eval plans
    python starter/example_predict.py --out submissions/dev --split train
    python starter/score.py --plans submissions/dev --split train       # self-eval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from contract import Case, Plan, case_instructions, load_case, plan_filename, save_plan  # noqa: E402


def plan_case(case: Case, instruction: str) -> dict:
    """YOUR MODEL HERE.

    Return {fdi: (t_mm (3,), quat_xyzw (4,))} for EVERY tooth in the case.
    This placeholder moves nothing — a valid plan, and your self-eval floor.
    """
    return {f: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])) for f in case.fdis}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", required=True, help="directory for the emitted plan files")
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    args = ap.parse_args()

    n = 0
    for cdir in sorted((Path(args.pack) / args.split).iterdir()):
        if not cdir.is_dir():
            continue
        case = load_case(cdir)
        for instr in case_instructions(cdir):
            plan = Plan(case_id=case.case_id, instruction_id=instr["id"],
                        instruction=instr["text"],
                        transforms=plan_case(case, instr["text"]),
                        meta={"model": "example-identity"})
            save_plan(plan, Path(args.out) / plan_filename(case.case_id, instr["id"]))
            n += 1
    print(f"wrote {n} plans -> {args.out}")


if __name__ == "__main__":
    main()
