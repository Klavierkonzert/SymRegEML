import numpy as np
import pandas as pd
import random
from pathlib import Path
import time
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

def main():
    random.seed(1)
    np.random.seed(1)

    # Grid Search for f_eml_test1 (x_0^2) with max_depth=None and max_complexity=20
    grid = [
        # (beta, max_complexity, max_depth, T_start)
        (-1.0, 20, None, None),
        (-1.0, 20, None, 50.0),
        (-1.0, 20, None, 200.0),
        
        (-0.5, 20, None, None),
        (-0.5, 20, None, 50.0),
        (-0.5, 20, None, 200.0),
        
        (-0.1, 20, None, None),
        (-0.1, 20, None, 50.0),
        (-0.1, 20, None, 200.0),
    ]

    print(f"Total grid search runs: {len(grid)}")
    
    out_dir = Path("experiments") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tune_x2_results.csv"
    
    # Write header
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("run_id,beta,max_complexity,max_depth,T_start,train_rmse,test_rmse,complexity,coefficients,expression\n")

    x_train, y_train, x_test, y_test = sample_data(
        func=f_eml_test1,
        n_train=100,
        n_test=1000,
        low=0.1,
        high=5.9
    )

    for run_id, (beta, max_complexity, max_depth, T_start) in enumerate(grid, 1):
        print(f"\n--- Run {run_id}/{len(grid)}: beta={beta}, max_complexity={max_complexity}, max_depth={max_depth}, T_start={T_start} ---")
        
        t0 = time.time()
        model = BSR(
            treeNum=1,
            itrNum=8,   # 8 chains
            beta=beta,
            disp=False,
            val=1000,   # 1000 proposals per chain
            operation_set="exp_log",
            max_complexity=max_complexity,
            max_depth=max_depth,
            tree_dtype=np.complex128,
            n_jobs=8,   # 8 jobs parallel
            T_start=T_start,
            cool_fraction=0.8
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
            
            print(f"  Completed in {dt:.1f}s. Train RMSE: {train_rmse:.6g}, Complexity: {comp}")
            print(f"  Expr: {expr}")
            
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(f"{run_id},{beta},{max_complexity},{max_depth},{T_start},{train_rmse:.6f},{test_rmse:.6f},{comp},\"{coefficients}\",\"{expr}\"\n")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
