import argparse
import sys
from typing import Optional, Sequence


def parse_arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set the parameters to run the model.")

    # -- Model parameters
    parser_group_model = parser.add_argument_group("Model and BCD parameters")
    parser_group_model.add_argument(
        "--model",
        type=str,
        help="The type of model.",
        choices=["linear", "logistic"],
        default="linear",
    )
    parser_group_model.add_argument(
        "--nr-parties",
        type=int,
        help="The number of parties. This is also the number of slices that is created from the dataset.",
        default=2,
    )
    parser_group_model.add_argument(
        "--central-iter",
        type=int,
        help="Maximum iterations for sklearn and (single-party) BCD runs on centralized data.",
        default=1000,
    )
    parser_group_model.add_argument(
        "--BCD-iter",
        type=int,
        help="Maximum number of iterations over all participating parties for BCD.",
        default=1000,
    )
    parser_group_model.add_argument(
        "--local-iter",
        type=int,
        help="Maximum number of (local) IRLS iterations per participating party.",
        default=100,
    )
    parser_group_model.add_argument(
        "--tol",
        type=float,
        help="Tolerance level for the Euclidean distance between the betas of two consecutive iterations.",
        default=0.01,
    )

    # -- Data parameters
    parser_group_data = parser.add_argument_group("Data parameters")
    parser_group_data.add_argument(
        "--dataset",
        type=str,
        help="The dataset.",
        choices=[
            "fish",
            "iris",
            "wine",
            "diabetes",
            "random_linear",
            "random_logistic",
            "forestfires",
            "diabetes_big",
        ],
        default="fish",
    )
    parser_group_data.add_argument(
        "--nr-samples",
        type=int,
        help="Number of samples if we are constructing a dataset.",
        default=100,
    )
    parser_group_data.add_argument(
        "--nr-features",
        type=int,
        help="Number of features if we are constructing a dataset.",
        default=10,
    )
    parser_group_data.add_argument(
        "-c", "--centering", help="center the data matrix", action="store_true"
    )
    parser_group_data.add_argument(
        "-i", "--remove-intercept", help="Do not fit an intercept", action="store_true"
    )

    # -- Result parameters
    parser_group_output = parser.add_argument_group("Output parameters")
    parser_group_output.add_argument(
        "-g",
        "--generate-plots",
        help="Generate plots of the final models",
        action="store_true",
    )
    parser_group_output.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    if isinstance(argv, list):
        argv.extend(sys.argv[1:])
    args = parser.parse_args(argv)
    return args
