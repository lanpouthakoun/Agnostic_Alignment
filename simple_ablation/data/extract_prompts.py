import argparse
import re
from typing import List, Optional

from datasets import load_dataset


HUMAN_BLOCK_REGEX = re.compile(
    r"Human:(.*?)(?=\n\nAssistant:|\Z)", flags=re.S
)


def extract_first_human(text: str) -> Optional[str]:
    """
    Extract the first 'Human:' block from a conversation string.
    Returns None if not found.
    """
    m = HUMAN_BLOCK_REGEX.search(text)
    if not m:
        return None
    return m.group(1).strip()


def extract_all_humans(text: str) -> List[str]:
    """
    Extract all 'Human:' blocks from a conversation string.
    Returns a list of stripped strings (may be empty).
    """
    blocks = HUMAN_BLOCK_REGEX.findall(text)
    return [b.strip() for b in blocks if b.strip()]


def normalize_single_line(s: str) -> str:
    """
    Collapse newlines and extra whitespace so each prompt is a single line.
    """
    return " ".join(s.split()).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Human: prompts from Anthropic/hh-rlhf dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="harmless-base",
        help="Which subset to load (harmless-base or red-team-attempts).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split, e.g. 'train'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="unsafe_spans.txt",
        help="Output text file path",
    )
    parser.add_argument(
        "--all_human_turns",
        action="store_true",
        help="Extract ALL Human: turns (default: only first turn)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate prompts (preserve order)",
    )

    args = parser.parse_args()

    print(f"Loading dataset: Anthropic/hh-rlhf, data_dir={args.data_dir}, split={args.split}")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=args.data_dir, split=args.split)

    prompts: List[str] = []

    print("Extracting Human: prompts...")
    for row in ds:
        conv = row["chosen"]  # conversation text

        if args.all_human_turns:
            humans = extract_all_humans(conv)
            prompts.extend(humans)
        else:
            first = extract_first_human(conv)
            if first:
                prompts.append(first)

    # normalize to single-line strings
    prompts = [normalize_single_line(p) for p in prompts if p.strip()]

    if args.dedupe:
        seen = set()
        deduped = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        prompts = deduped

    print(f"Extracted {len(prompts)} prompts. Writing to {args.output}")

    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
