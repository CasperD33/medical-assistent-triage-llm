# Medical Assistent Triage LLM

## Project Objective

Experiment for my AI assignment: This project implements a medical triage assistent using LLM in the context of Dutch primary care, inspired by the NTS/NHG triage systems.

Goal of the implemented LLM: decide whether a user should contact a doctor or not. A layperson describes their symptoms in natural language → the LLM triage assistant:

1) Assigns an urgency category (U0–U5, based on NTS logic),
2) Explains the reasoning (rationale),
3) Provides advice (self-care vs GP vs emergency).

Important: 
- This system is a research / educational prototype, not a medical device whatsoever.
- It is designed to showcase: 
    - How an LLM can approximate NTS-style triage. 
    - How chain-of-thought prompting and structured outputs can be used for clinical decision support.
    - How synthetic data can be used to avoid privacy issues.

## NTS Labels (Dutch Triage Standard)

- U0 - Immediate resuscitation context
    - No effective breathing and/or circulation and/or consciousness.
    - CPR and direct life-saving action required.
- U1 - Immediate threat to airway, breathing, circulation, or consciousness
    - Patient is alive, but ABCD is unstable.
    - Direct life-saving action needed without delay.
- U2 - Serious risk of deterioration or organ damage:
    - Potentially serious pathology (e.g. suspected appendicitis, DVT, acute eye emergencies).
    - Needs evaluation as soon as possible (very urgent GP / emergency care).
- U3 - Relevant medical risk or strong humane reasons, for example
    - Significant pain, high distress, worrisome symptoms without immediate life threat.
    - Needs evaluation within hours (same day GP)
- U4 - Very small risk of harm
    - Complaints that should be assessed, but not urgently.
    - Evaluation within 24 hours is sufficient (next day GP / phone consult).
- U5 - No realistic medical risk
    - Mild, self-limiting complaints without red flags.
    - Suitable for self-care or non-urgent GP contact (information, reassurance, routine appointment).

The job of the LLM is to map the question of the patient in one of these codes and support that decision with explanation and advice.

## Synthetic Dataset

To avoid patient data and privacy issues, we use a fully synthetic dataset of 100 cases. Each case is meant to be clinically plausible but not based on real individuals.

Example Entry: 

{
  "id": "case_090",
  "question_en": "For one day I have had mild pain in my ear after swimming, but no fever or discharge. Can I wait and see?",
  "complaint_norm_en": "Adult with mild external ear discomfort after swimming, no systemic symptoms.",
  "nts_code": "U5",
  "split": "train",
  "output_en": {
    "triage": "No realistic risk of harm; suitable for self-care or non-urgent GP contact.",
    "rationale": "Mild ear discomfort after swimming may represent irritation or early swimmer's ear and is not immediately dangerous.",
    "advice": "Keep the ear dry and avoid inserting objects; see the GP if pain increases, discharge appears or fever develops."
  }
}

### Description: 

- id: Unique identifier for the case, e.g. "case_000" … "case_099".
- question_en: The patient-style description of the complaint in English: how a layperson might actually ask the question. At inference time, the model sees only this field (question_en) as input.
- complaint_norm_en: A normalized, clinician-style summary of the complaint in English. This is not used as input to the model during inference. Instead, it serves as:
    - Metadata for the authors to check clinical correctness of each case.
    - A compact description for debugging and documentation.
    - A potential future target for a normalization step (e.g. model that converts patient text → clinical summary).
This field stays in the dataset, but the triage assistant itself does not consume it at runtime.
- nts_code: The ground truth urgency label, one of "U0", "U1", "U2", "U3", "U4", "U5". This is what the model is evaluated against.
- split: Experimental split, either:
    - "train" – cases that may be used as in-context examples (few-shot prompts) or for prompt tuning.
    - "eval" – held-out cases, never used in the model’s examples, only for evaluation.

    (Hier misschien nog wat toevoegen)
- output_en: Expected behaviour of the assistant (in English). Contains:
    - triage: Standardized textual description of the NTS category for this case (e.g. “No realistic risk of harm; suitable for self-care or non-urgent GP contact.”). This is tied directly to the NTS code, not free-form.
    - rationale: Short clinical reasoning that explains why this case belongs to that NTS category (symptoms, red flags, risk assessment).
    - advice: Concrete guidance to the patient:
        - Self-care vs GP vs emergency
        - Escalation criteria (e.g. “if X happens, call 112 / GP immediately”).
These fields are used as reference behaviour: how we want the assistant to respond (content-wise) when it picks that NTS code.

## Model Behaviour & output format

At inference time, the workflow is:
1) User provides a free-text complaint (question_en style).
2) The LLM is instructed via a system prompt to:
    - Assign a NTS label (U0–U5).
    - Provide a structured explanation and advice.

The assistant is expected to respond in structured JSON, for example:

{
  "label": "U5",
  "triage": "No realistic risk of harm; suitable for self-care or non-urgent GP contact.",
  "rationale": "Based on the mild ear discomfort without fever or discharge, there are no red flags for serious disease.",
  "advice": "You can monitor at home, keep the ear dry and see your GP if pain increases, discharge appears or fever develops."
}

## Scope & limitations
- This project is a prototype, built with:
    - A small synthetic dataset (100 cases).
    - No real patient data.
- NTS/NHG guidelines are complex and depend on:
    - Context, comorbidities, social situation, and dynamic clinical judgment.
- The assistant is therefore:
    - Not validated for clinical use.
    - Not to be used as a standalone medical decision tool.

In case of doubt or serious symptoms, users must always be instructed to: 
*Call emergency services (112) or contact their GP / out-of-hours service immediately.*

## Future Work
Language of the input for now is only English, with English evaluation, but with Dutch logic (applying the dutch NTS guidelines). Later on, additionally, I would like to implement the Dutch language as well (future work)






