from pathlib import Path
import json

from ollama import chat, ChatResponse


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "llama3.2"  # larger model

DATA_PATH = Path("2_data/processed/synthetic_patient_triage.json")

# Standardised triage texts per NTS label (for patient-facing output)
TRIAGE_TEXTS = {
    "U0": "Life-threatening emergency, start CPR and call 112 immediately.",
    "U1": "Very urgent, call 112 or go to the emergency department immediately.",
    "U2": "Serious problem, you should be seen by a doctor as soon as possible (within 1 hour).",
    "U3": "Needs medical attention today, contact your GP within a few hours.",
    "U4": "Mild problem, GP contact within 24 hours is sufficient.",
    "U5": "No urgent risk, suitable for self-care or routine GP contact.",
}

SYSTEM_TRIAGE = (
    "You are a GP triage assistant in a Dutch primary care context.\n"
    "Your task is to classify the urgency of the patient's situation using the Dutch NTS.\n"
    "Always respond in English.\n\n"
    "You MUST respond in the following plain-text format, on four separate lines:\n"
    "LINE 1: LABEL: one of U0, U1, U2, U3, U4, U5\n"
    "LINE 2: TRIAGE: one short English sentence for the urgency level (for the patient)\n"
    "LINE 3: RATIONALE: one short English sentence why this NTS level applies\n"
    "LINE 4: ADVICE: one short English sentence with clear action advice\n\n"
    "Example format (do NOT copy the content, only the structure):\n"
    "LABEL: U2\n"
    "TRIAGE: Serious problem, you should be seen by a doctor as soon as possible (within 1 hour).\n"
    "RATIONALE: Sudden chest pain and shortness of breath may indicate a heart problem.\n"
    "ADVICE: Call your GP immediately or 112 if symptoms worsen.\n\n"
    "Do not add any extra text before or after these four lines.\n\n"
    "NTS urgency levels (clinical meaning):\n"
    "U0: Failure of Airway, Breathing, Circulation, or consciousness (ABCD). "
    "Resuscitation context (e.g. cardiac arrest, no normal breathing, no response). "
    "Requires CPR and 112 immediately.\n"
    "U1: Unstable ABCD with direct life threat. Critically ill patient who may collapse "
    "without immediate intervention. Requires immediate ambulance / emergency department.\n"
    "U2: Serious threat to ABCD or risk of organ damage if not seen quickly, but no immediate "
    "cardiac arrest. Requires urgent medical assessment as soon as possible.\n"
    "U3: Real chance of harm or important humane reasons, but no acute threat to ABCD. "
    "Should be seen within a few hours (same day).\n"
    "U4: Negligible risk of harm. Evaluation within 24 hours is sufficient.\n"
    "U5: No realistic risk of harm in the short term. Suitable for self-care or routine GP contact "
    "(e.g. next working day).\n"
)


# ---------------------------------------------------------------------
# Helper: call Ollama
# ---------------------------------------------------------------------


def _ollama_chat(messages: list[dict[str, str]]) -> str:
    """
    Small helper to call Ollama's chat API.
    """
    response: ChatResponse = chat(
        model=MODEL_NAME,
        messages=messages,
        stream=False,
    )
    return response.message.content


def simple_ollama_test() -> None:
    """
    Send a tiny test message to the model and print the reply.
    Use this to check that everything is wired correctly.
    """
    messages = [
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]
    reply = _ollama_chat(messages)
    print("Model replied:")
    print(reply)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_cases(path: Path = DATA_PATH) -> list[dict]:
    """
    Load synthetic triage cases from JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list):
        raise ValueError("Expected a list of cases in synthetic_patient_triage.json")

    return cases


def split_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split cases into train and eval according to the 'split' field.
    Returns (train_cases, eval_cases).
    """
    train_cases: list[dict] = []
    eval_cases: list[dict] = []

    for c in cases:
        split = c.get("split")
        if split == "train":
            train_cases.append(c)
        elif split == "eval":
            eval_cases.append(c)

    return train_cases, eval_cases


# ---------------------------------------------------------------------
# Parsing structured (non-JSON) model output
# ---------------------------------------------------------------------


def _parse_structured_response(raw: str) -> dict:
    """
    Parse model output of the form:

        LABEL: U2
        TRIAGE: ...
        RATIONALE: ...
        ADVICE: ...

    Returns a dict with keys: label, triage, rationale, advice.

    If parsing fails, raises ValueError.
    """
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"Not enough lines in model response:\n{raw}")

    # Helper to extract "KEY: value"
    def extract(prefix: str, line: str) -> str:
        if not line.upper().startswith(prefix):
            raise ValueError(f"Expected line starting with '{prefix}' but got: {line!r}")
        # split only on the first ':'
        parts = line.split(":", 1)
        if len(parts) < 2:
            raise ValueError(f"No ':' found in line: {line!r}")
        return parts[1].strip()

    label = extract("LABEL", lines[0])
    triage = extract("TRIAGE", lines[1])
    rationale = extract("RATIONALE", lines[2])
    advice = extract("ADVICE", lines[3])

    # Standardise label and triage text
    label = label.upper()
    triage_text = TRIAGE_TEXTS.get(label, triage)

    return {
        "label": label,
        "triage": triage_text,
        "rationale": rationale,
        "advice": advice,
    }


def print_user_friendly(result: dict) -> None:
    """
    Print a user-friendly view of a triage result.
    """
    label = result.get("label", "")
    triage = result.get("triage", "")
    rationale = result.get("rationale", "")
    advice = result.get("advice", "")

    print("\nUser-friendly:\n")
    print(f"Urgency: {label}")
    print(triage)
    print("\nAdvice:")
    print(advice)
    print("\nWhy this urgency level:")
    print(rationale)


# ---------------------------------------------------------------------
# Baseline triage (zero-shot)
# ---------------------------------------------------------------------


def triage_baseline(text: str) -> dict:
    """
    Baseline triage: zero-shot classification into NTS label (U0–U5),
    returning a structured dict with:
      - label
      - triage (standardised text)
      - rationale
      - advice
    """
    messages = [
        {"role": "system", "content": SYSTEM_TRIAGE},
        {
            "role": "user",
            "content": (
                "Patient description:\n"
                f"{text}\n\n"
                "Respond ONLY with four lines in the exact format described above "
                "(LABEL, TRIAGE, RATIONALE, ADVICE), in English."
            ),
        },
    ]

    raw = _ollama_chat(messages)
    return _parse_structured_response(raw)


# ---------------------------------------------------------------------
# Few-shot support
# ---------------------------------------------------------------------


def format_example_case(case: dict) -> str:
    """
    Format one synthetic case as a few-shot example:
    - question_en
    - ideal structured answer in the same 4-line format.
    """
    question = case["question_en"]
    nts_code = case["nts_code"]
    out = case["output_en"]

    # We only care about label, we will still standardise triage text later
    example = (
        f"LABEL: {nts_code}\n"
        f"TRIAGE: {out['triage']}\n"
        f"RATIONALE: {out['rationale']}\n"
        f"ADVICE: {out['advice']}\n"
    )

    return (
        "Patient question:\n"
        f"{question}\n\n"
        "Ideal answer:\n"
        f"{example}"
    )


def build_few_shot_block(
    train_cases: list[dict],
    max_examples: int | None = None,
) -> str:
    """
    Build a few-shot block from the train cases.

    If max_examples is not None, we use at most that many cases.
    """
    if not train_cases:
        return ""

    if max_examples is not None:
        examples = train_cases[:max_examples]
    else:
        examples = train_cases

    formatted = [format_example_case(c) for c in examples]

    return (
        "Here are example triage cases with their ideal answers.\n\n"
        + "\n---\n\n".join(formatted)
        + "\nNow handle the next patient in the SAME FORMAT.\n"
    )


def triage_fewshot(
    text: str,
    train_cases: list[dict],
    max_examples: int | None = None,
) -> dict:
    """
    Triage with few-shot in-context examples:

    - Uses the same SYSTEM_TRIAGE as triage_baseline.
    - Adds up to max_examples synthetic training examples (from the train split)
      to the user prompt, before the new patient description.
    """
    few_shot_block = build_few_shot_block(train_cases, max_examples=max_examples)

    messages = [
        {"role": "system", "content": SYSTEM_TRIAGE},
        {
            "role": "user",
            "content": (
                few_shot_block
                + "\n\nNow a new patient:\n"
                f"{text}\n\n"
                "Respond ONLY with four lines in the exact format described above "
                "(LABEL, TRIAGE, RATIONALE, ADVICE), in English."
            ),
        },
    ]

    raw = _ollama_chat(messages)
    return _parse_structured_response(raw)


# ---------------------------------------------------------------------
# Manual test runner
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Simple connectivity test
    print("=== Simple connectivity test ===")
    simple_ollama_test()

    # Load synthetic cases
    print("\n=== Loading synthetic cases ===")
    try:
        cases = load_cases()
        train_cases, eval_cases = split_cases(cases)
        print(
            f"Loaded {len(cases)} cases "
            f"({len(train_cases)} train / {len(eval_cases)} eval)"
        )
    except Exception as e:
        print(f"Could not load cases: {e}")
        cases = []
        train_cases, eval_cases = [], []

    example_text = "Man, 58, sudden chest pain and shortness of breath for 15 minutes."

    # Baseline example
    print("\n=== Baseline triage example (zero-shot) ===")
    try:
        baseline_result = triage_baseline(example_text)
        print(baseline_result)
        print_user_friendly(baseline_result)
    except Exception as e:
        print(f"Baseline failed: {e}")

    # Few-shot example (if we have training cases)
    if train_cases:
        print("\n=== Few-shot triage example ===")
        try:
            fewshot_result = triage_fewshot(
                example_text,
                train_cases=train_cases,
                max_examples=12,
            )
            print(fewshot_result)
            print_user_friendly(fewshot_result)
        except Exception as e:
            print(f"Few-shot failed: {e}")
