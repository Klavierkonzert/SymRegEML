import numpy as np
import pandas as pd
import random
from pathlib import Path
from bsr import BSR
import itertools
import time

def f1(x):
    return 2.5 * x[:, 0] ** 4 - 1.3 * x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - 1.7 * x[:, 1]

def f2(x):
    return 8.0 * x[:, 0] ** 2 + 8.0 * x[:, 1] ** 3 - 15.0

FORMULAS = {"f1": f1, "f2": f2}
FORMULA_BOUNDS = {"f1": (0.1, 5.9), "f2": (0.1, 5.9)}

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

def main():
    random.seed(1)
    np.random.seed(1)

    # Grid search parameters
    T_starts = [200.0, 1000.0]
    cool_fractions = [0.8]
    betas = [-0.3, -0.5, -0.7]
    proposals_list = [2000]
    
    # We will test f1 and f2
    formulas_to_test = ["f1", "f2"]
    
    results = []
    
    # Generate all combinations
    combinations = list(itertools.product(formulas_to_test, T_starts, cool_fractions, betas, proposals_list))
    print(f"Total combinations to run: {len(combinations)}")
    
    out_dir = Path("experiments") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tuning_results.csv"
    
    # Write header
    with open(out_file, "w") as f:
        f.write("formula,T_start,cool_fraction,beta,proposals,train_rmse,test_rmse,complexity\n")

    for formula_name, T_start, cool_fraction, beta, proposals in combinations:
        print(f"\n--- Running {formula_name} with T_start={T_start}, cool_fraction={cool_fraction}, beta={beta}, proposals={proposals} ---")
        
        low, high = FORMULA_BOUNDS[formula_name]
        x_train, y_train, x_test, y_test = sample_data(
            func=FORMULAS[formula_name],
            n_train=100,
            n_test=1000,
            low=low,
            high=high
        )
        
        # Max complexity / depth configurations based on typical runs
        max_complexity = 250 if formula_name == "f1" else 150
        max_depth = 35 if formula_name == "f1" else 30
        
        t0 = time.time()
        model = BSR(
            treeNum=3,
            itrNum=8,
            beta=beta,
            disp=False,
            val=proposals,
            operation_set="exp_log",
            max_complexity=max_complexity,
            max_depth=max_depth,
            tree_dtype=np.complex128,
            n_jobs=8,
            T_start=T_start,
            cool_fraction=cool_fraction
        )
        
        try:
            model.fit(x_train, y_train)
            train_rmse = rmse(y_train, model.predict(x_train))
            test_rmse = rmse(y_test, model.predict(x_test))
            comp = model.complexity()
            dt = time.time() - t0
            print(f"  Completed in {dt:.1f}s. Train RMSE: {train_rmse:.4f}, Complexity: {comp}")
            
            with open(out_file, "a") as f:
                f.write(f"{formula_name},{T_start},{cool_fraction},{beta},{proposals},{train_rmse:.6f},{test_rmse:.6f},{comp}\n")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
