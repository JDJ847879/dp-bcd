import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bcd
import libraries
import numpy.typing as npt
from experiment_parser import parse_arguments

import data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bcd_glm")


def get_data_sets(
    args: Any,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], List[npt.NDArray[Any]]]:
    X, y = data.dataset(
        dataset_name=args.dataset,
        centering=args.centering,
        add_intercept=not args.remove_intercept,
        n=args.nr_samples,
        m=args.nr_features,
    )
    X_splitted = data.split_data(X, args.nr_parties)
    return X, y, X_splitted


def calculate_betas(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    X_splitted: List[npt.NDArray[Any]],
    args: Any,
    betas_init: Optional[List[npt.NDArray[Any]]] = None,
) -> Dict[str, npt.NDArray[Any]]:
    betas, _, _ = calculate_betas_and_performance(
        X=X, y=y, X_splitted=X_splitted, args=args, betas_init=betas_init
    )
    return betas


def calculate_betas_and_performance(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    X_splitted: List[npt.NDArray[Any]],
    args: Any,
    betas_init: Optional[List[npt.NDArray[Any]]] = None,
) -> Tuple[Dict[str, npt.NDArray[Any]], Dict[str, float], Dict[str, float]]:
    betas: Dict[str, npt.NDArray[Any]] = {}
    ex_time: Dict[str, float] = {}
    iterations: Dict[str, float] = {}
    fit_intercept = not (args.remove_intercept)
    fit_intercept = False
    t = time.perf_counter()
    betas["sklearn"] = libraries.sklearn_beta(
        X=X,
        y=y,
        model=args.model,
        fit_intercept=fit_intercept,
        nr_iterations=args.central_iter,
        tol=args.tol,
    )
    logger.info(
        "Calculated betas for sklearn models %s - %s",
        str(betas["sklearn"]),
        time.ctime(),
    )
    logger.info(f"Training took {time.perf_counter() - t} s")
    t = time.perf_counter()

    players = bcd.initialize_players(
        model=args.model,
        X_splitted=[X],
        y=y,
        beta_init_splitted=betas_init,
    )
    betas["IRLS_central"], it = bcd.fit_glm(
        players=players,
        outer_iterations=1,
        inner_iterations=args.central_iter,
        tol=args.tol,
    )
    ex_time["IRLS_central"] = round((time.perf_counter() - t), 3)
    iterations["IRLS_central"] = it
    logger.info(
        "Calculated betas for IRLS central models %s - %s",
        str(betas["IRLS_central"]),
        time.ctime(),
    )
    logger.info(f"Training took {time.perf_counter() - t} s")
    t = time.perf_counter()
    players = bcd.initialize_players(
        model=args.model,
        X_splitted=X_splitted,
        y=y,
        beta_init_splitted=betas_init,
    )
    betas["IRLS_bcd"], it = bcd.fit_glm(
        players=players,
        outer_iterations=args.BCD_iter,
        inner_iterations=args.local_iter,
        tol=args.tol,
    )
    ex_time["IRLS_bcd"] = round((time.perf_counter() - t), 3)
    iterations["IRLS_bcd"] = it
    logger.info(
        "Calculated betas for IRLS bcd models %s - %s",
        str(betas["IRLS_bcd"]),
        time.ctime(),
    )
    logger.info(f"Training took {time.perf_counter() - t} s")
    return betas, ex_time, iterations


def main(
    argv: Optional[Sequence[str]] = None, log_result: bool = False
) -> Dict[str, npt.NDArray[Any]]:
    args = parse_arguments(argv)
    print(args)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    X, y, X_splitted = get_data_sets(args)
    betas = calculate_betas(X, y, X_splitted, args)

    if log_result:
        for k, v in betas.items():
            logger.info(f"{k}: {v}")

    return betas


if __name__ == "__main__":
    main()
