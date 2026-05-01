"""Command-line entry point.

Subcommands:
    decepticons fit      — fit a byte-latent predictive coder on a corpus and optionally sample
    decepticons score    — score a corpus with a fresh-fit model and print bits/byte
    decepticons info     — show installed version and where the package is installed

Run ``decepticons --help`` or ``decepticons <subcommand> --help`` for details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .codecs import ByteCodec
from .model import ByteLatentPredictiveCoder


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decepticons",
        description="A backend-neutral kernel of predictive primitives — fit and sample from a byte-latent predictive coder.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"decepticons {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{fit,score,info}")

    fit = subparsers.add_parser(
        "fit",
        help="fit on a text file and optionally sample",
        description="Fit a ByteLatentPredictiveCoder on a UTF-8 text corpus and optionally sample bytes from it.",
    )
    fit.add_argument("--input", required=True, type=Path, help="Path to a UTF-8 text file.")
    fit.add_argument("--prompt", default="", help="Prompt used for sampling after fit.")
    fit.add_argument("--generate", type=int, default=0, metavar="N", help="Number of bytes to generate after the prompt (0 = no sampling).")
    fit.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (ignored under --greedy).")
    fit.add_argument("--greedy", action="store_true", help="Use greedy (argmax) decoding instead of sampling.")
    fit.add_argument("--seed", type=int, default=None, metavar="N", help="Seed for reproducible sampling (default: unseeded).")

    score = subparsers.add_parser(
        "score",
        help="fit on a text file and print bits/byte",
        description="Fit a ByteLatentPredictiveCoder on a UTF-8 text corpus and print bits/byte. Same as 'fit' without sampling.",
    )
    score.add_argument("--input", required=True, type=Path, help="Path to a UTF-8 text file.")

    subparsers.add_parser(
        "info",
        help="show installed version and package location",
        description="Print the installed decepticons version and the path to the installed package.",
    )

    return parser


def _cmd_fit(args: argparse.Namespace) -> int:
    text = args.input.read_text(encoding="utf-8")
    model = ByteLatentPredictiveCoder()
    report = model.fit(text)
    print(f"train bits/byte: {report.train_bits_per_byte:.4f}")
    print(f"patches: {report.patches}")
    print(f"mean patch size: {report.mean_patch_size:.2f}")
    if args.generate > 0:
        prompt = ByteCodec.encode_text(args.prompt or text[:16])
        sample = model.generate(
            prompt,
            steps=args.generate,
            temperature=args.temperature,
            greedy=args.greedy,
            seed=args.seed,
        )
        print(ByteCodec.decode_text(sample))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    text = args.input.read_text(encoding="utf-8")
    model = ByteLatentPredictiveCoder()
    report = model.fit(text)
    print(f"train bits/byte: {report.train_bits_per_byte:.4f}")
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    import decepticons

    print(f"decepticons {__version__}")
    print(f"installed at: {Path(decepticons.__file__).parent}")
    print(f"public surface: {len(decepticons.__all__)} symbols")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        return _cmd_fit(args)
    if args.command == "score":
        return _cmd_score(args)
    if args.command == "info":
        return _cmd_info(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
