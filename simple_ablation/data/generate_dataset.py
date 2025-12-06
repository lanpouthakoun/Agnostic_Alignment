import argparse
import json
import random
from typing import List, Tuple, Optional


def load_lines(path: str) -> List[str]:
    """Load non-empty, stripped lines from a text file."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def build_prompt_with_position(
    unsafe_span: str,
    filler_pool: List[str],
    target_bucket: str,
    min_fillers: int = 1,
    max_fillers: int = 6,
) -> Tuple[str, int, int]:
    """
    Build a prompt where `unsafe_span` is placed roughly at the front, middle,
    or back of the text, controlled by `target_bucket`.
    Returns (text, span_start, span_end).
    """

    if min_fillers < 0:
        raise ValueError("min_fillers must be >= 0")
    if max_fillers < min_fillers:
        raise ValueError("max_fillers must be >= min_fillers")

    # Choose how many filler chunks to use in total
    total_fillers = random.randint(min_fillers, max_fillers)

    # If we have no fillers, the whole prompt is just the unsafe span
    if total_fillers == 0 or not filler_pool:
        text = unsafe_span.strip()
        return text, 0, len(text)

    # Sample fillers with replacement so we don't run out
    fillers = [random.choice(filler_pool) for _ in range(total_fillers)]

    # Decide how many fillers go on the left based on bucket
    if target_bucket == "front":
        # unsafe_span near the beginning → more right fillers
        left = 0
    elif target_bucket == "back":
        # unsafe_span near the end → more left fillers
        left = total_fillers
    elif target_bucket == "middle":
        # unsafe_span somewhere in the middle
        # small random jitter so it's not always exactly centered
        base = total_fillers // 2
        jitter = random.randint(-1, 1)
        left = min(max(base + jitter, 0), total_fillers)
    else:
        raise ValueError(f"Unknown bucket: {target_bucket}")

    left_chunks = fillers[:left]
    right_chunks = fillers[left:]

    pieces = []
    if left_chunks:
        pieces.append(" ".join(left_chunks))
    pieces.append(unsafe_span.strip())
    if right_chunks:
        pieces.append(" ".join(right_chunks))

    text = " ".join(pieces).strip()

    # Locate the span
    span_start = text.index(unsafe_span.strip())
    span_end = span_start + len(unsafe_span.strip())

    return text, span_start, span_end


def make_dataset(
    unsafe_spans: List[str],
    filler_pool: List[str],
    out_path: str,
    k_per_span_per_bucket: int = 3,
    seed: Optional[int] = None,
) -> None:
    """
    Create a dataset with ONLY UNSAFE examples, where an unsafe span is placed
    at front/middle/back. Every example has an unsafe span and valid span indices.
    """

    if seed is not None:
        random.seed(seed)

    buckets = ["front", "middle", "back"]
    examples = []
    ex_id = 0

    # Generate UNSAFE examples (every example contains an unsafe span)
    for span_idx, span in enumerate(unsafe_spans):
        for bucket in buckets:
            for _ in range(k_per_span_per_bucket):
                text, s, e = build_prompt_with_position(
                    unsafe_span=span,
                    filler_pool=filler_pool,
                    target_bucket=bucket,
                )
                ex = {
                    "id": ex_id,
                    "text": text,
                    "label": 1,  # unsafe
                    "span_start": s,
                    "span_end": e,
                    "span_type": "unknown",  # you can customize by category
                    "position_bucket": bucket,
                    "source_span_index": span_idx,
                }
                examples.append(ex)
                ex_id += 1

    # Shuffle before writing
    random.shuffle(examples)

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")
    print(f"  Unsafe: {len(examples)}")
    print("  Safe:   0 (disabled)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset where every example contains an unsafe span."
    )
    parser.add_argument(
        "--unsafe_file",
        type=str,
        required=True,
        help="Path to a text file with one unsafe span per line.",
    )
    parser.add_argument(
        "--filler_file",
        type=str,
        required=True,
        help="Path to a text file with benign filler sentences, one per line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--k_per_span_per_bucket",
        type=int,
        default=3,
        help="How many unsafe prompts to generate for each (span, bucket).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    unsafe_spans = load_lines(args.unsafe_file)
    filler_pool = load_lines(args.filler_file)

    if not unsafe_spans:
        raise ValueError("No unsafe spans loaded. Check --unsafe_file.")
    if not filler_pool:
        raise ValueError("No filler lines loaded. Check --filler_file.")

    make_dataset(
        unsafe_spans=unsafe_spans,
        filler_pool=filler_pool,
        out_path=args.output,
        k_per_span_per_bucket=args.k_per_span_per_bucket,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()