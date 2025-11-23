from triage_model import load_cases, split_cases, triage_baseline, triage_fewshot, print_user_friendly

def main():
    all_cases = load_cases()
    train_cases, _ = split_cases(all_cases)

    print("Triage demo (type 'quit' to stop)\n")

    while True:
        text = input("Describe your symptoms: ")
        if text.strip().lower() in {"q", "quit", "exit"}:
            break

        print("\n--- BASELINE (zero-shot) ---")
        try:
            res_base = triage_baseline(text)
            print_user_friendly(res_base)
        except Exception as e:
            print(f"Baseline failed: {e}")

        print("\n--- FEW-SHOT ---")
        try:
            res_fs = triage_fewshot(text, train_cases, max_examples=12)
            print_user_friendly(res_fs)
        except Exception as e:
            print(f"Few-shot failed: {e}")

        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
