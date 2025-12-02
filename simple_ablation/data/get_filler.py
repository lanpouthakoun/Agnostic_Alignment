#!/usr/bin/env python3
from datasets import load_dataset
import nltk
import argparse

# Make sure you have punkt
nltk.download("punkt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext", help="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-103-v1")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="fillers.txt")
    args = parser.parse_args()

    print(f"Loading {args.dataset}/{args.subset} ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)

    fillers = []

    print("Extracting sentences ...")
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        # Sentence tokenize
        sentences = nltk.sent_tokenize(text)
        for s in sentences:
            s = s.strip()
            if len(s.split()) >= 5:     # skip very short fragments
                fillers.append(s)

    print(f"Total filler sentences extracted: {len(fillers)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for s in fillers:
            f.write(s + "\n")

    print(f"Wrote fillers to {args.output}")


if __name__ == "__main__":
    main()
