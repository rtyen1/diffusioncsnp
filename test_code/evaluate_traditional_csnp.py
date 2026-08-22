#!/usr/bin/env python3
"""Evaluate classical causal-discovery baselines on a CSNP NPY manifest.

The manifest must contain ``fp_data`` and ``fp_graph`` columns.  Data matrices
have shape [observations, variables], and graph matrices use graph[i, j] = 1
for i -> j.

PC and GES produce CPDAGs.  The other supported methods produce DAGs.  Every
valid DAG is also converted to its CPDAG, so CPDAG metrics are the common
comparison across all methods.  Directed DAG precision/recall/F1 and
TabCausal-style SHD are only reported for DAG outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Literal

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_csnp_4var_gp_benchmark_summary import (  # noqa: E402
    compute_cpdag_metrics,
    compute_dag_metrics,
    is_acyclic,
    maybe_standardize,
    run_ges_cpdag,
    run_pc_cpdag,
)
from test_code.evaluate_variable_joint_cpdag_vs_traditional import (  # noqa: E402
    dag_to_cpdag_fast,
)


METHODS = {
    "pc_fisherz",
    "pc_kci",
    "ges",
    "direct_lingam",
    "ica_lingam",
    "score",
    "das",
    "nogam",
    "notears_linear",
    "notears_nonlinear",
    "dagma_linear",
    "dagma_nonlinear",
}

CPDAG_METRICS = {
    "CPDAG_SHD": "cpdag_shd_pairwise",
    "CPDAG_exact_match": "cpdag_exact_match",
    "Skeleton_Precision": "skeleton_precision",
    "Skeleton_Recall": "skeleton_recall",
    "Skeleton_F1": "skeleton_f1",
    "CPDAG_Directed_Precision": "directed_precision",
    "CPDAG_Directed_Recall": "directed_recall",
    "CPDAG_Directed_F1": "directed_f1",
}
DAG_METRICS = {
    "DAG_SHD": "dag_shd_pairwise",
    "DAG_exact_match": "dag_exact_match",
    "DAG_Precision": "directed_edge_precision",
    "DAG_Recall": "directed_edge_recall",
    "DAG_F1": "directed_edge_f1",
    "Pred_is_DAG": "pred_is_dag",
}
SUMMARY_METRICS = [*CPDAG_METRICS, *DAG_METRICS, "Time"]


@dataclass
class Prediction:
    kind: Literal["cpdag", "dag"]
    directed: np.ndarray
    undirected: np.ndarray | None = None


@dataclass
class MethodSpec:
    name: str
    output_type: str
    variant: str
    backend: str
    runner: Callable[[np.ndarray, argparse.Namespace], Prediction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default="pc_fisherz,ges,direct_lingam",
        help=f"Comma-separated list from: {', '.join(sorted(METHODS))}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--print-errors", action="store_true")

    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--pc-uc-rule", type=int, default=0)
    parser.add_argument("--pc-uc-priority", type=int, default=2)
    parser.add_argument("--ges-score-func", default="local_score_BIC")

    parser.add_argument(
        "--lingam-backend",
        choices=("auto", "official", "causal-learn"),
        default="official",
        help="Use the official lingam package by default; auto may fall back to causal-learn",
    )
    parser.add_argument("--direct-lingam-measure", default="pwling")
    parser.add_argument("--ica-lingam-max-iter", type=int, default=1000)
    parser.add_argument("--lingam-weight-threshold", type=float, default=0.0)

    parser.add_argument("--toporder-alpha", type=float, default=0.05)
    parser.add_argument("--toporder-n-splines", type=int, default=10)
    parser.add_argument("--nogam-crossval", type=int, default=5)

    parser.add_argument("--notears-lambda1", type=float, default=0.1)
    parser.add_argument("--notears-nonlinear-lambda1", type=float, default=0.01)
    parser.add_argument("--notears-nonlinear-lambda2", type=float, default=0.01)
    parser.add_argument("--notears-max-iter", type=int, default=100)
    parser.add_argument("--notears-hidden", type=int, default=10)
    parser.add_argument("--notears-weight-threshold", type=float, default=0.3)

    parser.add_argument("--dagma-linear-lambda1", type=float, default=0.03)
    parser.add_argument("--dagma-nonlinear-lambda1", type=float, default=0.02)
    parser.add_argument("--dagma-nonlinear-lambda2", type=float, default=0.005)
    parser.add_argument("--dagma-hidden", type=int, default=10)
    parser.add_argument("--dagma-weight-threshold", type=float, default=0.3)
    parser.add_argument("--dagma-t", type=int, default=None, help="Use package default if omitted")
    parser.add_argument("--dagma-warm-iter", type=int, default=None)
    parser.add_argument("--dagma-max-iter", type=int, default=None)
    return parser.parse_args()


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if not methods:
        raise ValueError("--methods must not be empty")
    if len(methods) != len(set(methods)):
        raise ValueError("--methods contains duplicates")
    return methods


def package_version(package: str, fallback: str = "unknown") -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def resolve_lingam_backend(requested: str):
    if requested in ("auto", "official"):
        try:
            from lingam import DirectLiNGAM, ICALiNGAM

            return DirectLiNGAM, ICALiNGAM, f"lingam {package_version('lingam')}"
        except ImportError:
            if requested == "official":
                raise ImportError(
                    "Official LiNGAM backend requested but `lingam` is not installed. "
                    "Install the official package with `pip install lingam`."
                ) from None
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM, ICALiNGAM
    except ImportError as exc:
        raise ImportError(
            "LiNGAM methods require either `lingam` or causal-learn's bundled LiNGAM."
        ) from exc
    return (
        DirectLiNGAM,
        ICALiNGAM,
        f"causal-learn bundled LiNGAM {getattr(sys.modules[DirectLiNGAM.__module__.rsplit('.', 1)[0]], '__version__', 'unknown')}",
    )


def binary_from_weights(weights: np.ndarray, threshold: float, transpose: bool = False) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if transpose:
        weights = weights.T
    result = (np.abs(weights) > threshold).astype(np.int8)
    np.fill_diagonal(result, 0)
    return result


def run_lingam(
    data: np.ndarray,
    args: argparse.Namespace,
    algorithm: str,
    classes: tuple,
) -> Prediction:
    DirectLiNGAM, ICALiNGAM = classes
    x = maybe_standardize(data, standardize=not args.no_standardize)
    if algorithm == "direct":
        model = DirectLiNGAM(random_state=args.seed, measure=args.direct_lingam_measure)
    elif algorithm == "ica":
        model = ICALiNGAM(random_state=args.seed, max_iter=args.ica_lingam_max_iter)
    else:
        raise ValueError(f"Unknown LiNGAM algorithm: {algorithm}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x)

    # LiNGAM B[target, predictor] is predictor -> target, opposite to this project.
    weights = np.asarray(model.adjacency_matrix_, dtype=np.float64)
    return Prediction(
        "dag",
        binary_from_weights(weights, args.lingam_weight_threshold, transpose=True),
    )


def run_toporder(data: np.ndarray, args: argparse.Namespace, algorithm: str) -> Prediction:
    import pandas as pd
    from dodiscover.context_builder import make_context
    from dodiscover.toporder import DAS, NoGAM, SCORE

    classes = {"score": SCORE, "das": DAS, "nogam": NoGAM}
    kwargs = {
        "alpha": args.toporder_alpha,
        "n_splines": args.toporder_n_splines,
    }
    if algorithm == "nogam":
        kwargs["n_crossval"] = args.nogam_crossval
    model = classes[algorithm](**kwargs)
    x = maybe_standardize(data, standardize=not args.no_standardize)
    columns = list(range(x.shape[1]))
    data_frame = pd.DataFrame(x, columns=columns)
    # The fixed DoDiscover revision does not pass ``data_frame`` to its
    # implicit ContextBuilder, so an empty context cannot infer the variables.
    # Build the equivalent observational context explicitly from the columns.
    context = make_context().variables(data=data_frame).build()
    model.learn_graph(data_frame, context=context)
    directed = nx.to_numpy_array(model.graph_, nodelist=columns, dtype=np.int8)
    np.fill_diagonal(directed, 0)
    return Prediction("dag", directed.astype(np.int8))


def run_notears(data: np.ndarray, args: argparse.Namespace, nonlinear: bool) -> Prediction:
    x = maybe_standardize(data, standardize=not args.no_standardize)
    np.random.seed(args.seed)
    if not nonlinear:
        from notears.linear import notears_linear

        weights = notears_linear(
            x.copy(),
            lambda1=args.notears_lambda1,
            loss_type="l2",
            max_iter=args.notears_max_iter,
            w_threshold=args.notears_weight_threshold,
        )
    else:
        import torch
        from notears.nonlinear import NotearsMLP, notears_nonlinear

        torch.manual_seed(args.seed)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.double)
            model = NotearsMLP(dims=[x.shape[1], args.notears_hidden, 1], bias=True)
            weights = notears_nonlinear(
                model,
                x.copy(),
                lambda1=args.notears_nonlinear_lambda1,
                lambda2=args.notears_nonlinear_lambda2,
                max_iter=args.notears_max_iter,
                w_threshold=args.notears_weight_threshold,
            )
        finally:
            torch.set_default_dtype(previous_dtype)
    return Prediction("dag", binary_from_weights(weights, 0.0))


def run_dagma(data: np.ndarray, args: argparse.Namespace, nonlinear: bool) -> Prediction:
    x = maybe_standardize(data, standardize=not args.no_standardize)
    np.random.seed(args.seed)
    fit_kwargs: dict[str, object] = {"w_threshold": args.dagma_weight_threshold}
    if args.dagma_t is not None:
        fit_kwargs["T"] = args.dagma_t
    if args.dagma_warm_iter is not None:
        fit_kwargs["warm_iter"] = args.dagma_warm_iter
    if args.dagma_max_iter is not None:
        fit_kwargs["max_iter"] = args.dagma_max_iter

    if not nonlinear:
        from dagma.linear import DagmaLinear

        model = DagmaLinear(loss_type="l2", verbose=False)
        weights = model.fit(x.copy(), lambda1=args.dagma_linear_lambda1, **fit_kwargs)
    else:
        import torch
        from dagma.nonlinear import DagmaMLP, DagmaNonlinear

        torch.manual_seed(args.seed)
        previous_dtype = torch.get_default_dtype()
        try:
            equation_model = DagmaMLP(dims=[x.shape[1], args.dagma_hidden, 1], bias=True)
            model = DagmaNonlinear(equation_model, verbose=False)
            weights = model.fit(
                x.copy(),
                lambda1=args.dagma_nonlinear_lambda1,
                lambda2=args.dagma_nonlinear_lambda2,
                **fit_kwargs,
            )
        finally:
            torch.set_default_dtype(previous_dtype)
    return Prediction("dag", binary_from_weights(weights, 0.0))


def check_optional_dependency(method: str) -> None:
    try:
        if method in {"score", "das", "nogam"}:
            from dodiscover.toporder import DAS, NoGAM, SCORE  # noqa: F401
        elif method.startswith("notears_"):
            from notears.linear import notears_linear  # noqa: F401
            if method == "notears_nonlinear":
                from notears.nonlinear import NotearsMLP  # noqa: F401
        elif method.startswith("dagma_"):
            from dagma.linear import DagmaLinear  # noqa: F401
            if method == "dagma_nonlinear":
                from dagma.nonlinear import DagmaMLP  # noqa: F401
    except ImportError as exc:
        package = "dodiscover" if method in {"score", "das", "nogam"} else method.split("_", 1)[0]
        raise ImportError(
            f"Method `{method}` cannot start because its optional dependency `{package}` "
            f"is unavailable or incompatible: {exc}"
        ) from exc


def method_specs(methods: list[str], args: argparse.Namespace) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    traditional_args = SimpleNamespace(
        standardize_traditional=not args.no_standardize,
        pc_alpha=args.pc_alpha,
        pc_test="fisherz",
        pc_stable=True,
        pc_uc_rule=args.pc_uc_rule,
        pc_uc_priority=args.pc_uc_priority,
        ges_score_func=args.ges_score_func,
    )

    if "pc_fisherz" in methods:
        def pc_fisherz(data: np.ndarray, _args: argparse.Namespace) -> Prediction:
            traditional_args.pc_test = "fisherz"
            directed, undirected = run_pc_cpdag(data, traditional_args)
            return Prediction("cpdag", directed, undirected)

        specs.append(MethodSpec("PC-fisherz", "CPDAG", f"alpha={args.pc_alpha}", f"causal-learn {package_version('causal-learn')}", pc_fisherz))
    if "pc_kci" in methods:
        def pc_kci(data: np.ndarray, _args: argparse.Namespace) -> Prediction:
            traditional_args.pc_test = "kci"
            directed, undirected = run_pc_cpdag(data, traditional_args)
            return Prediction("cpdag", directed, undirected)

        specs.append(MethodSpec("PC-kci", "CPDAG", f"alpha={args.pc_alpha}", f"causal-learn {package_version('causal-learn')}", pc_kci))
    if "ges" in methods:
        def ges_runner(data: np.ndarray, _args: argparse.Namespace) -> Prediction:
            directed, undirected = run_ges_cpdag(data, traditional_args)
            return Prediction("cpdag", directed, undirected)

        specs.append(MethodSpec("GES-BIC", "CPDAG", args.ges_score_func, f"causal-learn {package_version('causal-learn')}", ges_runner))

    lingam_methods = set(methods) & {"direct_lingam", "ica_lingam"}
    if lingam_methods:
        DirectLiNGAM, ICALiNGAM, backend = resolve_lingam_backend(args.lingam_backend)
        classes = (DirectLiNGAM, ICALiNGAM)
        if "direct_lingam" in methods:
            specs.append(MethodSpec("Direct-LiNGAM", "DAG", f"measure={args.direct_lingam_measure}; threshold={args.lingam_weight_threshold}", backend, lambda data, a: run_lingam(data, a, "direct", classes)))
        if "ica_lingam" in methods:
            specs.append(MethodSpec("ICA-LiNGAM", "DAG", f"max_iter={args.ica_lingam_max_iter}; threshold={args.lingam_weight_threshold}", backend, lambda data, a: run_lingam(data, a, "ica", classes)))
    for method, display_name in (("score", "SCORE"), ("das", "DAS"), ("nogam", "NoGAM")):
        if method in methods:
            check_optional_dependency(method)
            extra = f"; cv={args.nogam_crossval}" if method == "nogam" else ""
            specs.append(MethodSpec(display_name, "DAG", f"alpha={args.toporder_alpha}; n_splines={args.toporder_n_splines}{extra}", f"dodiscover {package_version('dodiscover')}", lambda data, a, name=method: run_toporder(data, a, name)))

    if "notears_linear" in methods:
        check_optional_dependency("notears_linear")
        specs.append(MethodSpec("NOTEARS-linear", "DAG", f"lambda1={args.notears_lambda1}; threshold={args.notears_weight_threshold}", f"notears {package_version('notears')}", lambda data, a: run_notears(data, a, False)))
    if "notears_nonlinear" in methods:
        check_optional_dependency("notears_nonlinear")
        specs.append(MethodSpec("NOTEARS-nonlinear", "DAG", f"lambda1={args.notears_nonlinear_lambda1}; lambda2={args.notears_nonlinear_lambda2}; hidden={args.notears_hidden}; threshold={args.notears_weight_threshold}", f"notears {package_version('notears')}", lambda data, a: run_notears(data, a, True)))
    if "dagma_linear" in methods:
        check_optional_dependency("dagma_linear")
        specs.append(MethodSpec("DAGMA-linear", "DAG", f"lambda1={args.dagma_linear_lambda1}; threshold={args.dagma_weight_threshold}", f"dagma {package_version('dagma')}", lambda data, a: run_dagma(data, a, False)))
    if "dagma_nonlinear" in methods:
        check_optional_dependency("dagma_nonlinear")
        specs.append(MethodSpec("DAGMA-nonlinear", "DAG", f"lambda1={args.dagma_nonlinear_lambda1}; lambda2={args.dagma_nonlinear_lambda2}; hidden={args.dagma_hidden}; threshold={args.dagma_weight_threshold}", f"dagma {package_version('dagma')}", lambda data, a: run_dagma(data, a, True)))

    # Preserve the user-specified order instead of the construction order above.
    canonical_to_display = {
        "pc_fisherz": "PC-fisherz", "pc_kci": "PC-kci", "ges": "GES-BIC",
        "direct_lingam": "Direct-LiNGAM", "ica_lingam": "ICA-LiNGAM",
        "score": "SCORE", "das": "DAS", "nogam": "NoGAM",
        "notears_linear": "NOTEARS-linear", "notears_nonlinear": "NOTEARS-nonlinear",
        "dagma_linear": "DAGMA-linear", "dagma_nonlinear": "DAGMA-nonlinear",
    }
    by_name = {spec.name: spec for spec in specs}
    return [by_name[canonical_to_display[method]] for method in methods]


def validate_example(data_raw: np.ndarray, graph_raw: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    if data_raw.ndim != 2 or graph_raw.shape != (data_raw.shape[1], data_raw.shape[1]):
        raise ValueError(f"Invalid shapes at row {index}: data={data_raw.shape}, graph={graph_raw.shape}")
    if not np.issubdtype(data_raw.dtype, np.number) or not np.isfinite(data_raw).all():
        raise ValueError(f"Invalid data values at row {index}")
    if not np.isin(graph_raw, (0, 1)).all() or np.any(np.diag(graph_raw)):
        raise ValueError(f"Invalid adjacency at row {index}")
    true = graph_raw.astype(np.int8, copy=False)
    if not nx.is_directed_acyclic_graph(nx.from_numpy_array(true, create_using=nx.DiGraph)):
        raise ValueError(f"Cyclic ground truth at row {index}")
    return data_raw.astype(np.float64, copy=False), true


def validate_prediction(prediction: Prediction, nodes: int) -> Prediction:
    directed = np.asarray(prediction.directed, dtype=np.int8)
    if directed.shape != (nodes, nodes) or not np.isin(directed, (0, 1)).all():
        raise ValueError(f"Invalid predicted directed matrix: shape={directed.shape}")
    np.fill_diagonal(directed, 0)
    undirected = None if prediction.undirected is None else np.asarray(prediction.undirected, dtype=np.int8)
    if undirected is not None:
        if undirected.shape != (nodes, nodes) or not np.isin(undirected, (0, 1)).all():
            raise ValueError(f"Invalid predicted undirected matrix: shape={undirected.shape}")
        np.fill_diagonal(undirected, 0)
    return Prediction(prediction.kind, directed, undirected)


def evaluate_prediction(
    true_dag: np.ndarray,
    true_dir: np.ndarray,
    true_undir: np.ndarray,
    prediction: Prediction,
) -> dict[str, float]:
    output = {metric: float("nan") for metric in SUMMARY_METRICS}
    prediction = validate_prediction(prediction, true_dag.shape[0])
    pred_dag: np.ndarray | None = None
    if prediction.kind == "cpdag":
        pred_dir = prediction.directed
        pred_undir = prediction.undirected
        if pred_undir is None:
            raise ValueError("CPDAG prediction is missing its undirected matrix")
    else:
        pred_dag = prediction.directed.copy()
        dag_metrics = compute_dag_metrics(true_dag, pred_dag)
        for output_name, source_name in DAG_METRICS.items():
            output[output_name] = float(dag_metrics[source_name])

        if is_acyclic(pred_dag):
            pred_dir, pred_undir = dag_to_cpdag_fast(pred_dag)
        else:
            pred_dir = pred_undir = None

    if pred_dir is not None and pred_undir is not None:
        cpdag_metrics = compute_cpdag_metrics(true_dir, true_undir, pred_dir, pred_undir)
        for output_name, source_name in CPDAG_METRICS.items():
            output[output_name] = float(cpdag_metrics[source_name])
    return output


def mean_std(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), 0
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0, len(array)


def summarize(
    values: dict[tuple[int, str], dict[str, list[float]]],
    successes: dict[tuple[int, str], int],
    errors: dict[tuple[int, str], int],
    specs: list[MethodSpec],
    standardize: bool,
) -> list[dict[str, object]]:
    specs_by_name = {spec.name: spec for spec in specs}
    rows = []
    for (nodes, method), method_values in sorted(values.items()):
        spec = specs_by_name[method]
        row: dict[str, object] = {
            "f": nodes,
            "method": method,
            "output_type": spec.output_type,
            "variant": spec.variant,
            "backend": spec.backend,
            "standardize": standardize,
            "n": successes[(nodes, method)],
            "n_errors": errors[(nodes, method)],
        }
        for output_name in SUMMARY_METRICS:
            mean, std, count = mean_std(method_values[output_name])
            row[f"{output_name}_mean"] = mean
            row[f"{output_name}_std"] = std
            if output_name == "CPDAG_SHD":
                row["n_cpdag_scored"] = count
            elif output_name == "DAG_SHD":
                row["n_dag_scored"] = count
        rows.append(row)
    return rows


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not 0 < args.pc_alpha < 1 or not 0 < args.toporder_alpha < 1:
        raise ValueError("alpha values must be in (0, 1)")
    if args.max_datasets is not None and args.max_datasets <= 0:
        raise ValueError("--max-datasets must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    for name in ("lingam_weight_threshold", "notears_weight_threshold", "dagma_weight_threshold"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("dagma_t", "dagma_warm_iter", "dagma_max_iter"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    methods = parse_methods(args.methods)
    specs = method_specs(methods, args)  # Dependency failures happen before the long run.

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if args.max_datasets is not None:
        manifest_rows = manifest_rows[: args.max_datasets]
    if not manifest_rows:
        raise ValueError("Manifest is empty")
    missing_columns = {"fp_data", "fp_graph"} - set(manifest_rows[0])
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    print("Methods:", ", ".join(f"{spec.name} [{spec.backend}]" for spec in specs), flush=True)
    values: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in SUMMARY_METRICS}
    )
    successes: dict[tuple[int, str], int] = defaultdict(int)
    errors: dict[tuple[int, str], int] = defaultdict(int)
    started = time.perf_counter()

    for index, manifest_row in enumerate(manifest_rows, start=1):
        data_path = Path(manifest_row["fp_data"])
        graph_path = Path(manifest_row["fp_graph"])
        if not data_path.is_file() or not graph_path.is_file():
            raise FileNotFoundError(f"Missing data or graph at manifest row {index}")
        data, true_dag = validate_example(
            np.load(data_path, allow_pickle=False),
            np.load(graph_path, allow_pickle=False),
            index,
        )
        nodes = data.shape[1]
        true_dir, true_undir = dag_to_cpdag_fast(true_dag)

        for spec in specs:
            group = (nodes, spec.name)
            _ = values[group]
            try:
                method_started = time.perf_counter()
                prediction = spec.runner(data.copy(), args)
                method_elapsed = time.perf_counter() - method_started
                metrics = evaluate_prediction(true_dag, true_dir, true_undir, prediction)
                for output_name in SUMMARY_METRICS:
                    values[group][output_name].append(
                        method_elapsed if output_name == "Time" else metrics[output_name]
                    )
                successes[group] += 1
            except Exception as exc:
                errors[group] += 1
                if args.print_errors:
                    print(f"[ERROR] row={index} method={spec.name}: {exc!r}", flush=True)

        if index == 1 or index % args.progress_every == 0 or index == len(manifest_rows):
            print(f"[{index}/{len(manifest_rows)}] elapsed={time.perf_counter() - started:.1f}s", flush=True)

    output_rows = summarize(values, successes, errors, specs, not args.no_standardize)
    if not output_rows:
        raise RuntimeError("No output rows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
