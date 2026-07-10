import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bsr import BSR
from typing import List, Tuple, Optional, Dict, Any, Union

# Benchmark formulas from the BSR paper: page 5
def f1(x):
    return 2.5 * x[:, 0] ** 4 - 1.3 * x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - 1.7 * x[:, 1]
def f2(x):
    return 8.0 * x[:, 0] ** 2 + 8.0 * x[:, 1] ** 3 - 15.0
def f3(x):
    return 0.2 * x[:, 0] ** 3 + 0.5 * x[:, 1] ** 3 - 1.2 * x[:, 1] - 0.5 * x[:, 0]
def f4(x):
    return 1.5 * np.exp(x[:, 0]) + 5.0 * np.cos(x[:, 1])
def f5(x):
    return 6.0 * np.sin(x[:, 0]) * np.cos(x[:, 1])
def f6(x):
    return 1.35 * x[:, 0] * x[:, 1] + 5.5 * np.sin((x[:, 0] - 1.0) * (x[:, 1] - 1.0))
def f_eml_test1(x):
    return x[:, 0] ** 2
def f_eml_test2(x):
    return x[:, 0] ** 3
def f_eml_test3(x):
    return x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 0] * x[:, 1]
def f_eml_test4(x):
    return np.cos(x[:, 0]) + np.sin(x[:, 1])
def f_xy(x):
    return x[:, 0] * x[:, 1]
def f_cos(x):
    return np.cos(x[:, 0])





FORMULAS = {
    "f1": f1,
    "f2": f2,
    "f3": f3,
    "f4": f4,
    "f5": f5,
    "f6": f6,
    "f_eml_test1": f_eml_test1,
    "f_eml_test2": f_eml_test2,
    "f_eml_test3": f_eml_test3,
    "f_eml_test4": f_eml_test4,
    "f_xy": f_xy,
    "f_cos": f_cos,
}

FORMULA_BOUNDS = {
    "f1": (0.1, 3),
    "f2": (0.1, 3),
    "f3": (0.1, 3),
    "f4": (0.1, 3),
    "f5": (0.1, 3),
    "f6": (-3.0, 3.0),
    "f_eml_test1": (0.1, 3),
    "f_eml_test2": (0.1, 3),
    "f_eml_test3": (0.1, 3),
    "f_eml_test4": (0.1, 3),
    "f_xy": (0.1, 3),
    "f_cos": (0.1, 3),
}


# MCM-SymReg > codes > funcs.py > grow()
# prob = 1 / np.power((1 + depth), -beta)

OPERATION_SETS = {
    # "paper": {"beta": -1.0, "proposals": 100, "max_complexity": None, "max_depth": None},
    # "exp_log": {"beta": -0.5, "proposals": 2000, "max_complexity": 180, "max_depth": 7},
    "default": {"beta": -1.0, "proposals": 100, "max_complexity": None, "max_depth": None, "tree_dtype": "float"},
    "paper": {"beta": -1.0, "proposals": 500, "max_complexity": None, "max_depth": None, "tree_dtype": "float"},
    "exp_log": {"beta": -0.25, "proposals": 500, "max_complexity": 500, "max_depth": 40, "tree_dtype": "complex"},
}

FORMULA_EML_CONFIGS = {
    "f1": {"beta": -0.5, "max_complexity": 90, "max_depth": 18, "proposals": 500, "T_start": 200.0, "cool_fraction": 0.8, "trees": 4},
    "f2": {"beta": -0.5, "max_complexity": 50, "max_depth": 15, "proposals": 500, "T_start": 300.0, "cool_fraction": 0.8, "trees": 2},
    "f3": {"beta": -0.75, "max_complexity": 75, "max_depth": 15, "proposals": 500, "T_start": 100.0, "cool_fraction": 0.8, "trees": 4},
    "f4": {"beta": -0.3,  "max_complexity": 170, "max_depth": 25, "proposals": 500, "T_start": 100.0, "cool_fraction": 0.8, "trees": 4},
    "f5": {"beta": -0.25, "max_complexity": 500, "max_depth": 35, "proposals": 500, "T_start": 250.0, "cool_fraction": 0.85, "trees": 4},
    "f6": {"beta": -0.25, "max_complexity": 190, "max_depth": 26, "proposals": 500, "T_start": 250.0, "cool_fraction": 0.85, "trees": 4},
    "f_xy": {"beta": -0.5, "max_complexity": 20, "max_depth": 12, "proposals": 500, "T_start": 50.0, "cool_fraction": 0.8, "trees": 1},
    "f_cos": {"beta": -0.5, "max_complexity": 160, "max_depth": 25, "proposals": 500, "T_start": 100.0, "cool_fraction": 0.8, "trees": 2},
    "f_eml_test1": {"beta": -1.0, "max_complexity": 20, "max_depth": None, "proposals": 500, "T_start": 50.0, "cool_fraction": 0.8, "trees": 1},
    "f_eml_test2": {"beta": -1.0, "max_complexity": 30, "max_depth": None, "proposals": 500, "T_start": 50.0, "cool_fraction": 0.8, "trees": 1},
    "f_eml_test3": {"beta": -1.0, "max_complexity": 70, "max_depth": None, "proposals": 500, "T_start": 50.0, "cool_fraction": 0.8, "trees": 3},
    "f_eml_test4": {"beta": -0.5, "max_complexity": 150, "max_depth": None, "proposals": 500, "T_start": 100.0, "cool_fraction": 0.8, "trees": 4},
}


def sample_data(func, n_train, n_test, low, high):
    x_train = np.random.uniform(low, high, size=(n_train, 2))
    x_test = np.random.uniform(low, high, size=(n_test, 2))
    y_train = func(x_train)
    y_test = func(x_test)

    return (
        pd.DataFrame(x_train),
        pd.Series(y_train),
        pd.DataFrame(x_test),
        pd.Series(y_test),
    )


def rmse(y_true: List[float]|Tuple[float], y_pred: List[float]|Tuple[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred).reshape(-1) - np.asarray(y_true)) ** 2)))


def format_final_expression(coefficients: list[float]|tuple[float], expressions: list[str]|tuple[str]) -> str:
    """:param coefficients: List of coefficients for each tree in the model. The first coefficient is the intercept term, and the rest correspond to the trees in order.
        :param expressions: List of string expressions for each tree in the model, in order.
        :return: A string representation of the final expression in the form of "c0 + c1*(tree1) + c2*(tree2) + ..."
    """
    terms = [f"{coefficients[0]:.6g}"]
    for coefficient, expression in zip(coefficients[1:], expressions):
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f"{sign} {abs(coefficient):.6g}*({expression})")
    return " ".join(terms)


def normalize_tree_expression(expression: str) -> str:
    expression = re.sub(r"x\[(\d+)\]", r"x\1", expression)
    expression = expression.replace("^", "**")
    expression = expression.replace("[", "(").replace("]", ")")
    return expression


def simplify_final_expression(coefficients: list[float]|tuple[float], expressions: list[str]|tuple[str], tolerance: Optional[float]=None) -> str:
    """
    Simplifies the final expression using sympy, if available.
    If tolerance is provided, coefficients with absolute value below the tolerance will be dropped.
    """
    try:
        import sympy as sp
    except ImportError:
        return "Install sympy to generate this field"

    symbols = sp.symbols("x0:20")
    local_dict = {f"x{i}": symbol for i, symbol in enumerate(symbols)}
    local_dict.update({"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log, "abs": sp.Abs, "Re": sp.re})

    try:
        combined = sp.Float(coefficients[0])
        for coefficient, expression in zip(coefficients[1:], expressions):
            if tolerance is not None and abs(coefficient) < tolerance:
                continue
            tree = sp.sympify(normalize_tree_expression(expression), locals=local_dict)
            combined += sp.Float(coefficient) * tree
        
        # Check expression complexity to prevent hanging in SymPy
        expr_str = str(combined)
        if len(expr_str) > 250 or expr_str.count("exp") + expr_str.count("log") > 5:
            return expr_str
            
        return str(sp.simplify(sp.expand(combined)))
    except Exception as exc:
        return f"Could not simplify: {exc}"


def value_for_operation_set(values, operation_set):
    if values is None:
        return OPERATION_SETS[operation_set]

    defaults = OPERATION_SETS[operation_set].copy()
    for item in values:
        name, raw_value = item.split("=", maxsplit=1)
        if name != operation_set:
            continue
        for assignment in raw_value.split(","):
            key, value = assignment.split(":", maxsplit=1)
            if key == "beta":
                defaults[key] = float(value)
            elif key == "proposals":
                defaults[key] = int(value)
            elif key == "max_complexity":
                defaults[key] = None if value.lower() == "none" else int(value)
            elif key == "max_depth":
                defaults[key] = None if value.lower() == "none" else int(value)
            elif key == "tree_dtype":
                defaults[key] = value
            else:
                raise ValueError(f"Unknown per-operation setting {key!r}")
    return defaults


def run_one(name, func, operation_set, args):
    operation_config = value_for_operation_set(args.operation_config, operation_set)
    
    # Merge formula-specific EML configs
    if operation_set == "exp_log" and name in FORMULA_EML_CONFIGS:
        for k, v in FORMULA_EML_CONFIGS[name].items():
            operation_config[k] = v
    
    low = args.low if args.low is not None else FORMULA_BOUNDS[name][0]
    high = args.high if args.high is not None else FORMULA_BOUNDS[name][1]

    x_train, y_train, x_test, y_test = sample_data(
        func=func,
        n_train=args.n_train,
        n_test=args.n_test,
        low=low,
        high=high,
    )

    trees = min(operation_config.get("trees", args.trees), args.trees)
    model = BSR(
        treeNum=trees,
        itrNum=args.iterations,
        beta=args.beta if args.beta is not None else operation_config["beta"],
        disp=args.verbose,
        val=args.proposals if args.proposals is not None else operation_config["proposals"],
        operation_set=operation_set,
        max_complexity=operation_config["max_complexity"],
        max_depth=operation_config["max_depth"],
        tree_dtype=np.complex128 if operation_config.get("tree_dtype") == "complex" else np.float64,
        n_jobs=args.n_jobs,
        T_start=operation_config.get("T_start"),
        cool_fraction=operation_config.get("cool_fraction"),
    )
    print(f"DEBUG: trees={trees}, beta={model.beta}, itrNum={model.itrNum}, proposals={model.val}, T_start={model.T_start}")
    model.fit(x_train, y_train)

    train_rmse = rmse(y_train, model.predict(x_train))
    test_rmse = rmse(y_test, model.predict(x_test))
    expressions = model.model()
    coefficients = model.betas_[-1].reshape(-1).tolist()
    final_expression = format_final_expression(coefficients, expressions)
    stable_coefficients, stable_expressions = zip(*[(c, e) for c, e in zip(coefficients[1:], expressions) if abs(c) > 1e-6 or len(coefficients) <2])
    stable_coefficients = [coefficients[0]] + list(stable_coefficients)
    simplified_expression = format_final_expression(stable_coefficients, stable_expressions) if operation_set == "exp_log" else simplify_final_expression(coefficients, expressions)

    return {
        "formula": name,
        "operation_set": operation_set,
        "x_low": args.low,
        "x_high": args.high,
        "x_low": low,
        "x_high": high,
        "beta": args.beta if args.beta is not None else operation_config["beta"],
        "proposals": args.proposals if args.proposals is not None else operation_config["proposals"],
        "max_complexity": operation_config["max_complexity"],
        "max_depth": operation_config["max_depth"],
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "complexity": model.complexity(),
        "expressions": expressions,
        "coefficients": coefficients,
        "final_expression": final_expression,
        "simplified_expression": simplified_expression,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run BSR on the six benchmark formulas from the paper.")
    parser.add_argument("--formulas", nargs="+", choices=FORMULAS.keys(), default=list(FORMULAS.keys()))
    parser.add_argument("--operation-sets", nargs="+", choices=OPERATION_SETS.keys(), default=OPERATION_SETS.keys())
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--low", type=float, default=0.1)
    parser.add_argument("--high", type=float, default=3)
    parser.add_argument("--trees", type=int, default=3, help="Indicates maximum number of trees used to approximate the expression")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--proposals", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--operation-config",
        nargs="+",
        default=None,
        help="Per-set overrides like paper=beta:-1,proposals:100 exp_log=beta:-0.5,proposals:2000,max_complexity:180,max_depth:7",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments") / "results")
    return parser.parse_args()


import multiprocessing
from datetime import datetime
cpu_count = multiprocessing.cpu_count

FIELDNAMES = [
    "formula",
    "operation_set",
    "x_low",
    "x_high",
    "beta",
    "proposals",
    "max_complexity",
    "max_depth",
    "train_rmse",
    "test_rmse",
    "complexity",
    "coefficients",
    "final_expression",
    "simplified_expression",
]

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"paper_formulas_summary_{timestamp}.csv"
    expressions_path = args.out_dir / f"paper_formulas_expressions_{timestamp}.txt"

    print(f"Running experiments with formulas: {args.formulas} and operation sets: {args.operation_sets}.")
    print(f"Number of threads: { cpu_count() if args.n_jobs==-1 else args.n_jobs }, number of iterations: {args.iterations}, number of trees: {args.trees if args.trees is not None else 'formula-specific'}")
    print(f"Incremental results will be saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  TXT: {expressions_path}")
    print("-" * 60)

    for name in args.formulas:
        for operation_set in args.operation_sets:
            if name.startswith("f_eml_test") and operation_set == "default":
                continue
            print(f"Running {name} with {operation_set} operations...")
            result = run_one(name, FORMULAS[name], operation_set, args)
            print(f"  Train RMSE: {result['train_rmse']:.6g}, Test RMSE: {result['test_rmse']:.6g}, Complexity: {result['complexity']}")
            
            # 1. Write/append to CSV
            row = {
                "formula": result["formula"],
                "operation_set": result["operation_set"],
                "x_low": result["x_low"],
                "x_high": result["x_high"],
                "beta": result["beta"],
                "proposals": result["proposals"],
                "max_complexity": result["max_complexity"],
                "max_depth": result["max_depth"],
                "train_rmse": result["train_rmse"],
                "test_rmse": result["test_rmse"],
                "complexity": result["complexity"],
                "coefficients": result["coefficients"],
                "final_expression": result["final_expression"],
                "simplified_expression": result["simplified_expression"],
            }
            csv_exists = csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
                if not csv_exists:
                    writer.writeheader()
                writer.writerow(row)

            # 2. Write/append to Expressions Text File
            step_lines = [
                f"{name} / {operation_set}",
                f"  x_domain: [{result['x_low']}, {result['x_high']}]",
                f"  beta: {result['beta']}",
                f"  proposals: {result['proposals']}",
                f"  max_complexity: {result['max_complexity']}",
                f"  max_depth: {result['max_depth']}",
                f"  train_rmse: {result['train_rmse']:.6g}",
                f"  test_rmse: {result['test_rmse']:.6g}",
                f"  complexity: {result['complexity']}",
                f"  coefficients: {result['coefficients']}",
                f"  final_expression: {result['final_expression']}",
                f"  simplified_expression: {result['simplified_expression']}",
            ]
            for i, expr in enumerate(result["expressions"], start=1):
                step_lines.append(f"  tree_{i}: {expr}")
            step_lines.append("")
            
            with expressions_path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(step_lines) + "\n")

    print("-" * 60)
    print(f"All runs completed successfully!")
    print(f"Final summary saved to: {csv_path}")
    print(f"Final expressions saved to: {expressions_path}")


if __name__ == "__main__":
    main()
