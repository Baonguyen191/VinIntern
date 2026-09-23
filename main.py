import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Electricity load anomaly detection pipeline"
    )
    parser.add_argument("--user", default="U317", help="User ID (default: U317)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument(
        "--pipeline",
        choices=["towt", "degree-day", "bdg2", "statistical"],
        default="degree-day",
        help="Pipeline type (default: degree-day)",
    )
    parser.add_argument(
        "--building",
        default="Rat_office_Colby",
        help="BDG2 building ID (default: Rat_office_Colby)",
    )
    parser.add_argument(
        "--eta", type=float, default=3.0, help="Iterative cleaning multiplier (default: 3.0)"
    )

    # TOWT-specific
    parser.add_argument(
        "--breakpoints", type=int, default=6, help="Temperature breakpoints for TOWT (default: 6)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization for TOWT (default: 1.0)"
    )

    # Degree-day-specific
    parser.add_argument(
        "--k", type=float, default=3.0, help="Anomaly threshold multiplier (default: 3.0)"
    )
    parser.add_argument(
        "--balance-step", type=float, default=0.5,
        help="Balance point search step in °F (default: 0.5)",
    )
    parser.add_argument(
        "--window", type=int, default=28,
        help="Rolling window size in days for statistical baseline (default: 28)",
    )

    args = parser.parse_args()

    if args.pipeline == "statistical":
        from src.pipelines.statistical import run_statistical_pipeline

        run_statistical_pipeline(
            building_id=args.building,
            output_dir=args.output or "results/bdg2/statistical",
            k=args.k,
            window=args.window,
        )
    elif args.pipeline == "bdg2":
        from src.pipelines.bdg2 import run_bdg2_pipeline

        run_bdg2_pipeline(
            building_id=args.building,
            output_dir=args.output or "results/bdg2/degree_day",
            k=args.k,
            eta=args.eta,
            balance_step=args.balance_step,
        )
    elif args.pipeline == "towt":
        from src.pipelines.towt import run_pipeline

        run_pipeline(
            user_id=args.user,
            output_dir=args.output or "results/eweld/towt",
            eta=args.eta,
            n_breakpoints=args.breakpoints,
            alpha=args.alpha,
        )
    else:
        from src.pipelines.degree_day import run_pipeline

        run_pipeline(
            user_id=args.user,
            output_dir=args.output or "results/eweld/degree_day",
            k=args.k,
            eta=args.eta,
            balance_step=args.balance_step,
        )


if __name__ == "__main__":
    main()
