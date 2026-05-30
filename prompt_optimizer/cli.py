"""
cli.py
Command-line interface for prompt-optimizer.

Usage examples:
    prompt-optimizer --list-models
    prompt-optimizer --input prompt.txt --target llama
    prompt-optimizer --input prompt.txt --target claude --show-diff
    echo "You are a helpful assistant. Think step by step." | prompt-optimizer --target gemini
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import PromptAnalyzer
from .rules_engine import RulesEngine
from .transformer import PromptTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-optimizer",
        description=(
            "Adapt and lint prompts for specific LLM models using rule-based "
            "transformations. No LLMs used — fully deterministic."
        ),
    )
    parser.add_argument(
        "--target", "-t",
        metavar="MODEL",
        help="Target model ID (e.g. llama, claude, gpt, mistral, gemini, qwen)",
    )
    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Path to prompt file. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write adapted prompt to this file. Defaults to stdout.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model profiles and exit.",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Print a summary of transformations applied and lint warnings.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print detected features of the input prompt and exit (no transformation).",
    )
    parser.add_argument(
        "--model-variant",
        metavar="VARIANT",
        help="Optional model variant (e.g. 'o1', '3', '7b-instruct') for conditional rules.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    engine = RulesEngine()

    # --list-models
    if args.list_models:
        print("Available model profiles:")
        for model_id in engine.list_models():
            profile = engine.get_profile(model_id)
            sub_families = profile.sub_families
            if sub_families:
                sf_list = ", ".join(sub_families.keys())
                print(f"  {model_id:<12} — {profile.display_name} (sub-families: {sf_list})")
            else:
                print(f"  {model_id:<12} — {profile.display_name}")
        sys.exit(0)

    # Read prompt
    if args.input:
        prompt_text = Path(args.input).read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            print("Reading prompt from stdin (Ctrl+D to finish):", file=sys.stderr)
        prompt_text = sys.stdin.read()

    if not prompt_text.strip():
        print("Error: empty prompt.", file=sys.stderr)
        sys.exit(1)

    analyzer = PromptAnalyzer()
    features = analyzer.analyze(prompt_text)

    # --analyze only
    if args.analyze:
        print(features.summary())
        sys.exit(0)

    if not args.target:
        parser.error("--target is required unless using --list-models or --analyze")

    profile = engine.get_profile(args.target)
    transformer = PromptTransformer()
    result = transformer.transform(prompt_text, profile, features, args.model_variant)

    # Output adapted prompt
    if args.output:
        Path(args.output).write_text(result.adapted, encoding="utf-8")
    else:
        print(result.adapted)

    # Diff summary
    if args.show_diff:
        print("\n" + "─" * 60, file=sys.stderr)
        print(result.diff_summary(), file=sys.stderr)


if __name__ == "__main__":
    main()
