"""Single entry point for the project.

    python main.py                    show what is trained and what to run next
    python main.py status             same as above
    python main.py demo [--no-explain] [--rules-dir PATH]
                                      run the trained models live on Grid2Op:
                                      ENN KPIs, t+12 failure forecasts, and
                                      per-line rule explanations
    python main.py train [options]    train every model (stages are reused when valid;
                                      a valid ENN/classifier/rule set is replaced only
                                      with an explicit --force-stage)
    python main.py train --profile [PATH]
                                      train with LLM explanations, all options from
                                      one JSON profile (default: src/llm_config.json)
    python main.py smoke [--llm-config PATH]
                                      tiny end-to-end training run into artifacts_smoke/

Settings come from .env (see project_config.py).  Run ``python main.py
<command> --help`` for the options of each command.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Workflow modules (and project_config) are imported only inside the chosen
# command: ``smoke`` must set its overrides before the configuration is read.


def _chosen_command(argv: list[str]) -> str | None:
    return next((arg for arg in argv if not arg.startswith("-")), None)


def build_parser(command: str | None = None) -> argparse.ArgumentParser:
    """Build the CLI; ``train`` options are attached only when ``train`` runs."""
    parser = argparse.ArgumentParser(
        description="Failure forecasting for RL grid-operation agents.",
        epilog="Without a command, 'status' is shown.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    commands.add_parser("status", help="show trained artifacts and the next step")
    demo_parser = commands.add_parser(
        "demo",
        help="run the trained models live on Grid2Op",
        description="ENN uncertainty -> t+12 failure forecast -> per-line rule "
                    "explanations (when failure_forecast/explainability/rules/ "
                    "exists). Needs no training rows, OOF file, or LLM config.",
    )
    demo_parser.add_argument(
        "--no-explain",
        action="store_true",
        help="skip rule explanations (HGB forecasts are identical either way)",
    )
    demo_parser.add_argument(
        "--rules-dir",
        type=Path,
        default=None,
        help="distilled-rule directory (default: "
             "<artifacts>/failure_forecast/explainability/rules)",
    )

    train_parser = commands.add_parser(
        "train",
        help="train every model; valid existing stages are reused",
        description="Run the configured ENN and failure-forecast pipeline.",
    )
    if command == "train":
        from src.workflows import llm_profile, train

        train.add_arguments(train_parser)
        train_parser.add_argument(
            "--profile",
            type=Path,
            nargs="?",
            const=llm_profile.DEFAULT_CONFIG_PATH,
            default=None,
            help="run with LLM explanations, taking every option from this JSON "
                 "profile (default: src/llm_config.json); "
                 "cannot be combined with other train options",
        )

    smoke_parser = commands.add_parser(
        "smoke", help="tiny end-to-end training run into artifacts_smoke/"
    )
    smoke_parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="include SHAP ranking, LLM rule generation, and explained inference",
    )
    return parser


def _run_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from src.workflows import llm_profile, train

    if args.profile is None:
        train.run_pipeline(args)
        return
    defaults = vars(train.build_parser().parse_args([]))
    changed = sorted(name for name, value in defaults.items()
                     if getattr(args, name) != value)
    if changed:
        parser.error("--profile takes every option from the profile; remove: "
                     + ", ".join("--" + name.replace("_", "-") for name in changed))
    try:
        profile = llm_profile.load_profile(args.profile)
    except llm_profile.ProfileError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None
    train.run_pipeline(llm_profile.profile_to_pipeline_args(profile))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser(_chosen_command(argv))
    args = parser.parse_args(argv)
    command = args.command or "status"

    if command == "status":
        from src.workflows import status

        status.run()
    elif command == "demo":
        from src.workflows import demo

        demo.run(explain=not args.no_explain, rules_dir=args.rules_dir)
    elif command == "train":
        _run_train(args, parser)
    elif command == "smoke":
        from src.workflows import smoke

        smoke.run(args.llm_config)


if __name__ == "__main__":
    main()
