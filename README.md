# Medical Assistent Triage LLM (Dutch NTS)

## Project Objective

Experiment for my AI assignment: This project implements a medical triage assistent using LLM in the context of Dutch primary care, inspired by the NTS (Dutch) triage systems.

Goal of the implemented LLM: decide whether a user should contact a doctor or not. A layperson describes their symptoms in natural language → the LLM triage assistant:

1) Assigns an urgency category (U0–U5, based on NTS logic),
2) Explains the reasoning (rationale),
3) Provides advice (self-care vs GP vs emergency).

Disclaimer: This system is a research / educational prototype, not a medical device whatsoever!!!!

It is designed to showcase how an LLM can approximate NTS-style triage. 


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
  "nts_code": "U5",
  "split": "train",
  "output_en": {
    "triage": "No realistic risk of harm; suitable for self-care or non-urgent GP contact.",
    "rationale": "Mild ear discomfort after swimming may represent irritation or early swimmer's ear and is not immediately dangerous.",
    "advice": "Keep the ear dry and avoid inserting objects; see the GP if pain increases, discharge appears or fever develops."
  }
}

## Repository Structure

It is organized as follows:
- 1_src/
    - demo_triage.py # demo of the bot works
    - eval_triage.py # evaluation scripts, such as the metrics, confusion matrices, JSON logs
- 2_data/ 
    - processed/ 
        - synthetic_patient_triage.json # synthetic dataset partially based on HealthCareMagic dataset (kaggle)
- 3_results:
    - confusion_baseline.png  # confusion matrix baseline model
    - confusion_fewshot.png   # confusion matrix fewshot model
    - baseline_classification_report.txt #precision    recall  f1-score   support for baseline
    - fewshot_classification_report.txt #precision    recall  f1-score   support for fewshot 
    - baseline_per_case.json # evaluation per case with true label and output label
    - fewshot_per_case.json # evaluation per case with true label and output label


## Future Work
Language of the input for now is only English, with English evaluation, but with Dutch logic (applying the dutch NTS guidelines). Later on, additionally, I would like to implement the Dutch language as well (future work)






