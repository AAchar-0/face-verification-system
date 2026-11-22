"""Step 3: turn the counts from a hand-run test (you and consenting friends) into rates with honest ranges.

Example: 14 of 15 genuine attempts were accepted, and 0 of 20 impostor attempts were accepted:
    python src/03_summarise_manual_test.py --genuine 14 15 --impostor 0 20

Genuine attempt  = the enrolled person tries to verify as themselves.
Impostor attempt = a different person tries to verify as someone else.
"""
import argparse

from evaluation import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise a manual verification test.")
    parser.add_argument("--genuine", nargs=2, type=int, metavar=("ACCEPTED", "TOTAL"), required=True)
    parser.add_argument("--impostor", nargs=2, type=int, metavar=("ACCEPTED", "TOTAL"), required=True)
    args = parser.parse_args()

    accepted_genuine, total_genuine = args.genuine
    accepted_impostor, total_impostor = args.impostor
    if not (0 <= accepted_genuine <= total_genuine and 0 <= accepted_impostor <= total_impostor and total_genuine > 0 and total_impostor > 0):
        raise SystemExit("Counts must satisfy 0 <= accepted <= total, with totals above 0.")

    rejected_genuine = total_genuine - accepted_genuine
    frr_low, frr_high = wilson_interval(rejected_genuine, total_genuine)
    far_low, far_high = wilson_interval(accepted_impostor, total_impostor)
    print(f"False reject rate: {rejected_genuine / total_genuine:.1%}  ({rejected_genuine} of {total_genuine}; 95% range {frr_low:.1%} to {frr_high:.1%})")
    print(f"False accept rate: {accepted_impostor / total_impostor:.1%}  ({accepted_impostor} of {total_impostor}; 95% range {far_low:.1%} to {far_high:.1%})")
    print("\nWith so few attempts the ranges are wide. Report them, and never report a single rate on its own.")


if __name__ == "__main__":
    main()
