# 01Health - AI Research Task: Instruction-Conditioned Treatment Planning

> **Timeline:** 7 days from receiving this pack - sized so ideas can marinate, training
> (if any) can run overnight, and the task fits around your job. Stopping early with a
> sharp writeup beats pushing on.

This is the only document you need - the task, the data, the formats, and what to send,
in one read.

**Start in five minutes.** Python ≥ 3.10 and numpy are the only requirements. From the
pack root:

```bash
pip install numpy
python starter/example_predict.py --out submissions/dev --split train   # valid do-nothing plans
python starter/score.py --plans submissions/dev --split train           # scored: your floor
```

Replace `plan_case()` in `starter/example_predict.py` with your approach and iterate.
(Optional: `starter/parser.py` has an LLM parsing path behind `pip install anthropic` + an
API key - never required.)

## 1. The problem

01Health makes clear-aligner treatment accessible through general dentists, supported by
orthodontic specialists. For every case a dentist submits a 3D scan of the patient's teeth
plus a free-text prescription, and a specialist designs the plan: where every tooth should
end up. That design step is expert, slow, and what this role will help automate - safely,
with a clinician always approving the output.

Your task is a scoped slice of it. Build a model or pipeline:

```
predict(arch_geometry, instruction_text) → per-tooth rigid movements
```

Two properties are load-bearing:

1. **Teeth are rigid.** You output per-tooth rigid transforms - never deformed geometry.
2. **The text must matter.** The same mouth under a different instruction must produce a
   materially, *correctly* different plan.

**Building or training a neural model is optional.** Any approach that produces plans is in
bounds: learned, geometric, LLM-in-the-loop, hybrid. Public datasets and pretrained weights
are allowed if documented; ortho-specific commercial tools are not. Babysitting training
runs buys nothing here - we are hiring for how you think about **3D + text multimodal
modeling**, and the deepest part of your submission is the *proposed architecture* in your
writeup, which may go well beyond whatever you had time to prototype.

## 2. The data

Anonymised, downsampled derivatives of real cases (do not redistribute; delete after the
process).

```
train/  (50 cases)   geometry + the real dentist instruction + the designer's real plan
eval/   (25 cases)   geometry + instructions ONLY - you predict the rest
starter/             loader, instruction parser, the public scorer, and example_predict.py
                     (a complete valid do-nothing submission - replace one function)
```

Per case:

| file | contents |
|---|---|
| `points.npz` | labeled 3D point clouds - arrays keyed `t<FDI>`, 1024 points per tooth, float32, mm, scan frame (subsample freely) |
| `teeth.json` | per tooth: `centroid` [x,y,z] and `frame_q` [x,y,z,w] - the local coordinate frame, provided so you don't spend budget on geometry preprocessing |
| `instruction.txt` | (train) the real dentist's free-text request |
| `gold_transforms.json` | (train) the designer's real plan: `{"transforms": {"<fdi>": {"t_mm": [3], "q": [4]}}}` |
| `instructions.json` | (eval) `[{"id": "i0", "text": "..."}, ...]` - the eval battery; one plan per entry if you send plans |
| `meta.json` | case id, arch type, complexity band |

And the starter kit - five flat Python files, numpy-only:

| file | what it is |
|---|---|
| `starter/contract.py` | case/plan loading and saving, quaternion helpers |
| `starter/parser.py` | a keyword instruction parser - a starting point, improve or replace it freely |
| `starter/metrics.py` | the scorer; the judge runs this same code against the hidden eval gold |
| `starter/score.py` | self-eval CLI: `python starter/score.py --plans <dir> --split train` |
| `starter/example_predict.py` | a complete, valid (do-nothing) submission - replace `plan_case()` |

**Conventions** (everything you need; the starter kit implements all of it):

- Units millimetres; quaternions `[x, y, z, w]`, unit norm.
- A transform places the tooth's start geometry `x` at **`x' = R(q) @ (x − c) + c + t`**,
  where `c` is the tooth's start centroid. Identity transform = tooth does not move.
- FDI numbering, 28 teeth per case (no third molars): upper `17…27`, lower `47…37`.
  Quadrants: 1 upper-right, 2 upper-left, 3 lower-left, 4 lower-right (patient's view).
  Unit digit: 1–2 incisors, 3 canine, 4–5 premolars, 6–7 molars - "1.6" = FDI 16.
  Anterior = unit digits 1–3, posterior = 4–7.
- `frame_q` rotates a scan-frame vector INTO the tooth frame (`v_tooth = R(frame_q) @
  v_scan`). Tooth axes: x mesiodistal (toward distal), y buccolingual (outward),
  z occluso-apical (up).

Load a case with the starter kit: `from contract import load_case`.

**Provenance, honestly stated.** The instruction is the dentist's prescription, written
when the case was submitted - after seeing the scan, before any plan existed. The gold is
the orthodontic specialist's *planned* final tooth positions (dentist-approved in most
cases), not the clinical outcome. Plans sometimes went through revision rounds whose
feedback isn't in the instruction text - so treat the text as a strong but incomplete
explanation of the gold. This is real clinical data, noise included; reasoning well about
that noise is part of the task.

## 3. The instructions

Free text, varying along three deliberate axes. You don't need to parse them into this
taxonomy - but your *outputs* must respond to these axes, because the gold plans do:

1. **Protection** - teeth that must not move ("do not move 1.6, 2.6", "avoid moving the
   molars"). A protected tooth's correct transform is the identity.
2. **Scope** - which arch/region the work applies to ("lower arch only" ⇒ every upper
   tooth stays put; "front teeth" ⇒ incisors + canines).
3. **Objective trade-off** - speed ↔ safety ↔ aesthetics.

Example, same style as the pack: *"Align and level the lower arch only; leave the upper
arch completely untouched. Resolve crowding primarily by proclination rather than IPR."*

Each eval case carries **multiple instructions**, including pairs that contradict each
other on these axes. We score whether your plans *differ between instructions the way the
gold plans differ* - in magnitude and direction - plus shuffled-text controls. Run your own
**text-shuffle control** - feed your system the instructions shuffled across cases (works
for any approach, learned or not) - and report both numbers in your writeup.

## 4. The plan format

If you send plans (suggested - see Deliverables): one file per (case, instruction),
`plans/<case_id>__<instruction_id>.json`

```json
{
  "format": "taskplan-1",
  "case_id": "prod_0021",
  "instruction_id": "i1",
  "instruction": "the text you were given",
  "transforms": {
    "11": {"t_mm": [0.42, -1.10, 0.00], "q": [0.0, 0.0, 0.087, 0.996]},
    "...": {}
  },
  "meta": {"model": "yours", "anything": "you want us to see"}
}
```

Every tooth in the case must appear (identity transform for teeth you don't move).
`starter/example_predict.py` already emits exactly this for the whole eval set - replace
its `plan_case()` with your model and the format is guaranteed right. Self-evaluate any
time: `python starter/score.py --plans <dir> --split train` (train cases carry their gold;
the scorer is the same metric code we run).

If the core is done and you have appetite, two open extensions: **staging** (decompose
movements into clinically-sized per-stage steps, collision-free at every stage - the
stage budgets and refinements some instructions mention matter only here) and
**multi-plan** (2–3 distinct valid plans for the same case and instruction). Unfinished
extensions cost nothing.

## 5. Deliverables

One thing is required - **your writeup**. Plan files are suggested, and code is optional.

1. **The writeup (required)** - no required format or length: most good ones end up
   somewhere around 2–4 pages, but write what the thinking needs. These are the questions
   we'll be reading for:
   - Describe your model architecture and why you chose it.
   - How does your prototype turn (geometry, text) into plans - and how does the text
     actually steer the output?
   - What do your shuffle-control numbers show - and how much of your sensitivity comes
     from explicit parsing/rules versus learned behaviour, if both are present?
   - What further data could be introduced to make a model perform better and why?
   - If given unlimited time and budget, what would you build differently and why?

2. **`plans/` (suggested)** - plan JSON for the 25 eval cases × every listed instruction
   (`starter/example_predict.py` emits the exact format). Not required, but well worth
   sending: plans let us score your ideas objectively alongside the writeup, and they give
   the follow-up conversation real material.

3. **Code or weights (optional)** - only if you're happy to share; reproducibility is a
   plus, never a requirement. What you built stays yours.

## 6. Practical

- **Compute:** a laptop CPU is enough for everything here (a small learned model on this
  pack trains in minutes, and non-learned pipelines need none).
- **Questions:** email louisa.rumble@01health.ai
- **Submission:** your writeup (plus `plans/` and anything else you'd like us to see - zip
  or link) to louisa.rumble@01health.ai, within 7 days of receiving this pack. 

Good luck - we're looking forward to seeing how you think.
