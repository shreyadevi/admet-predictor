"""Command-line interface.

    admet benchmark            compare all models, random vs scaffold split
    admet train                select, evaluate and save the production models
    admet predict SMILES ...   predict properties for molecules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import ENDPOINTS
from .predict import DEFAULT_MODEL_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]


def cmd_benchmark(args) -> int:
    from .pipeline import benchmark, save_json, summarize
    from .report import benchmark_markdown, plot_benchmark

    seeds = list(range(args.seeds))
    results = benchmark(args.endpoints, args.splits, seeds, fast=args.fast)
    summary = summarize(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "benchmark_raw.csv", index=False)
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    (out / "benchmark.md").write_text(benchmark_markdown(summary, len(seeds)))
    plot_benchmark(summary, out)
    save_json({"seeds": seeds, "fast": args.fast, "summary": summary.to_dict(orient="records")},
              out / "benchmark.json")
    print(f"\nWrote {out / 'benchmark.md'}")
    print((out / "benchmark.md").read_text())
    return 0


def cmd_train(args) -> int:
    from .pipeline import save_json, train_final
    from .report import plot_parity

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in args.endpoints:
        summary = train_final(name, Path(args.model_dir), split=args.split, fast=args.fast)
        if name == "solubility":
            pred = summary["test_predictions"]["logS"]
            plot_parity(pred["y_true"], pred["y_pred"], f"Solubility: {args.split}-split test set",
                        ENDPOINTS[name].units, out / "parity_solubility.png")
        summary.pop("test_predictions")
        summaries[name] = summary
    save_json(summaries, out / "final_models.json")
    print(json.dumps({k: {"model": v["model"], "split": v["split"]} for k, v in summaries.items()}, indent=2))
    return 0


def cmd_predict(args) -> int:
    from .predict import Predictor

    predictor = Predictor.load(args.model_dir)
    print(json.dumps(predictor.predict(args.smiles), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="admet", description="ADMET property prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="Compare models under random and scaffold splits")
    b.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS), choices=list(ENDPOINTS))
    b.add_argument("--splits", nargs="+", default=["random", "scaffold"], choices=["random", "scaffold"])
    b.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    b.add_argument("--fast", action="store_true", help="Smaller models, for CI and quick checks")
    b.add_argument("--out", default=str(REPO_ROOT / "reports"))
    b.set_defaults(func=cmd_benchmark)

    t = sub.add_parser("train", help="Select, evaluate and save the production models")
    t.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS), choices=list(ENDPOINTS))
    t.add_argument("--split", default="scaffold", choices=["random", "scaffold"])
    t.add_argument("--fast", action="store_true")
    t.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    t.add_argument("--out", default=str(REPO_ROOT / "reports"))
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Predict ADMET properties for SMILES")
    p.add_argument("smiles", nargs="+")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
