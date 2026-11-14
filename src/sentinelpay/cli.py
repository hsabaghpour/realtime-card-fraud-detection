"""Single entry point: ``sentinelpay <step>``.

The Makefile, the Airflow DAGs and the Databricks jobs all call these same
commands, so there is exactly one implementation of every pipeline step.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from sentinelpay.data.generator import DRIFT_SCENARIOS
from sentinelpay.utils.config import Settings, load_settings
from sentinelpay.utils.io import write_json
from sentinelpay.utils.logging import get_logger, timed

log = get_logger(__name__)


def cmd_generate(settings: Settings, args: argparse.Namespace) -> None:
    from sentinelpay.data import generator

    scenarios = list(DRIFT_SCENARIOS) if args.drift_scenario == "all" else [args.drift_scenario]
    generator.run(settings, scenarios)


def _with_spark(fn: Callable) -> Callable[[Settings, argparse.Namespace], None]:
    """Wrap a ``run(spark, settings)`` step so Spark only starts when needed."""

    def _cmd(settings: Settings, args: argparse.Namespace) -> None:
        from sentinelpay.utils.spark import get_spark

        fn(get_spark(settings), settings)

    return _cmd


def cmd_bronze(settings: Settings, args: argparse.Namespace) -> None:
    from sentinelpay.pipeline import bronze

    _with_spark(bronze.run)(settings, args)


def cmd_silver(settings: Settings, args: argparse.Namespace) -> None:
    from sentinelpay.pipeline import silver

    _with_spark(silver.run)(settings, args)


def cmd_demo(settings: Settings, args: argparse.Namespace) -> None:
    """Run every implemented step in order and record how long each took."""
    from sentinelpay.data import generator
    from sentinelpay.pipeline import bronze, silver
    from sentinelpay.utils.spark import get_spark

    timings: dict[str, float] = {}
    with timed("generate", timings):
        generator.run(settings, list(DRIFT_SCENARIOS))
    with timed("spark_start", timings):
        spark = get_spark(settings)
    with timed("bronze", timings):
        bronze.run(spark, settings)
    with timed("silver", timings):
        silver.run(spark, settings)

    timings["total"] = round(sum(timings.values()), 2)
    write_json(settings.paths.report("timings.json"), timings)
    log.info("demo pipeline finished in %.1fs (see reports/)", timings["total"])


COMMANDS: dict[str, tuple[Callable[[Settings, argparse.Namespace], None], str]] = {
    "generate": (cmd_generate, "generate synthetic history + production feeds"),
    "bronze": (cmd_bronze, "ingest the history landing zone into the bronze Delta table"),
    "silver": (cmd_silver, "validate, quarantine and de-duplicate bronze into silver"),
    "demo": (cmd_demo, "run the full pipeline end to end"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinelpay", description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="config environment (default: $SENTINELPAY_ENV or local)")
    parser.add_argument("--sample-size", type=int, help="override data.sample_size")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        if name == "generate":
            p.add_argument(
                "--drift-scenario",
                choices=[*DRIFT_SCENARIOS, "all"],
                default="all",
                help="production-period drift scenario to generate (default: all)",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {"data": {"sample_size": args.sample_size}} if args.sample_size else None
    settings = load_settings(env=args.env, overrides=overrides)
    handler, _ = COMMANDS[args.command]
    handler(settings, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
