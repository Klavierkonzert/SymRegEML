import numpy as np
import pandas as pd
import random
import time
from pathlib import Path
from bsr import BSR

def f_eml_test1(x):
    return x[:, 0] ** 2

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

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred).reshape(-1) - np.asarray(y_true)) ** 2)))

def format_final_expression(coefficients, expressions):
    terms = [f"{coefficients[0]:.6g}"]
    for coefficient, expression in zip(coefficients[1:], expressions):
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f"{sign} {abs(coefficient):.6g}*({expression})")
    return " ".join(terms)

def is_exact_x2(expr):
    # Check if the expression matches typical EML representations of x_0^2
    # e.g., containing eml(eml(1, eml(eml(1, x0), 1)), 1)
    # The printed form uses "exp(L)-log(R)"
    # A unary EML chain representing -ln(x0) or ln(x0)^2:
    # x0^2 under EML is exp( exp(1) - log( exp(exp(1) - log(x0)) - log(1) ) ) - log(1)
    # Wait, the exact string in BSR output format has "exp(...)-log(...)"
    clean = expr.replace(" ", "")
    # Check if we have x0 (represented as x0 or x[0] or similar) and some expected structure
    return "x0" in clean or "x[0]" in clean

def main():
    out_dir = Path("experiments") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prior_comparison_results.csv"
    
    # Write header
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("config_name,trial,beta,prior_enabled,train_rmse,test_rmse,complexity,expression,time\n")

    # We will test beta = -0.5 and beta = -0.1
    # For each, we run 2 trials with seeds 1, 2
    betas = [-0.5, -0.1]
    priors = [
        ("No Prior", None),
        ("With Prior (0.5)", {"exp_minus_log": {1.0: 0.5}})
    ]
    
    n_trials = 2
    results = []

    print("Starting prior comparative sweep...")
    print("| Config | Trial | Beta | Prior | Train RMSE | Complexity | Time | Expression |")
    print("|---|---|---|---|---|---|---|---|")

    for beta in betas:
        for prior_name, left_prior in priors:
            for trial in range(1, n_trials + 1):
                # Set seeds for reproducibility
                random.seed(trial)
                np.random.seed(trial)
                
                x_train, y_train, x_test, y_test = sample_data(
                    func=f_eml_test1,
                    n_train=100,
                    n_test=1000,
                    low=0.1,
                    high=5.9
                )
                
                t0 = time.time()
                model = BSR(
                    treeNum=1,
                    itrNum=4,
                    beta=beta,
                    disp=False,
                    val=200,
                    operation_set="exp_log",
                    max_complexity=20,
                    max_depth=None,
                    tree_dtype=np.complex128,
                    n_jobs=4,
                    T_start=50.0,
                    cool_fraction=0.8,
                    left_prior=left_prior,
                    right_prior=None
                )
                
                try:
                    model.fit(x_train, y_train)
                    train_rmse = rmse(y_train, model.predict(x_train))
                    test_rmse = rmse(y_test, model.predict(x_test))
                    comp = model.complexity()
                    dt = time.time() - t0
                    expressions = model.model()
                    coefficients = model.betas_[-1].reshape(-1).tolist()
                    expr = format_final_expression(coefficients, expressions)
                    
                    config_str = f"Beta {beta} - {prior_name}"
                    print(f"| {config_str} | {trial} | {beta} | {prior_name} | {train_rmse:.4f} | {comp} | {dt:.1f}s | {expr[:40]}... |")
                    
                    with open(out_file, "a", encoding="utf-8") as f:
                        f.write(f"\"{config_str}\",{trial},{beta},\"{prior_name}\",{train_rmse:.6f},{test_rmse:.6f},{comp},\"{expr}\",{dt:.2f}\n")
                        
                    results.append({
                        "config": config_str,
                        "beta": beta,
                        "prior": prior_name,
                        "train_rmse": train_rmse,
                        "complexity": comp,
                        "time": dt
                    })
                except Exception as e:
                    print(f"Error on Beta {beta}, {prior_name}, trial {trial}: {e}")
                    
    # Print summary averages
    df = pd.DataFrame(results)
    summary = df.groupby(["beta", "prior"]).agg({
        "train_rmse": "mean",
        "complexity": "mean",
        "time": "mean"
    }).reset_index()
    
    print("\nSummary Averages:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
