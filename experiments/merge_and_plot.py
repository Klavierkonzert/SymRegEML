import os
import glob
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt

def merge_and_plot():
    master_csv = r"experiments/results/final/final_run.csv"
    backup_csv = r"experiments/results/final/final_run_backup.csv"
    
    # 1. Find the latest generated summary CSV
    summary_files = glob.glob(r"experiments/results/paper_formulas_summary_*.csv")
    if not summary_files:
        print("Error: No paper_formulas_summary_*.csv files found in experiments/results/")
        return
        
    latest_summary = max(summary_files, key=os.path.getmtime)
    print(f"Found latest summary file: {latest_summary}")
    
    # Backup master CSV if it exists
    if os.path.exists(master_csv):
        shutil.copy2(master_csv, backup_csv)
        print(f"Backed up {master_csv} to {backup_csv}")
    
    # Read old master rows
    master_rows = []
    if os.path.exists(backup_csv):
        with open(backup_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            master_rows = list(reader)
            
    # Read new summary rows
    new_rows = []
    with open(latest_summary, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        new_rows = list(reader)
        
    # Merge rows
    # We want to replace any "exp_log" rows for the formulas present in new_rows,
    # and add f_xy if it is not present.
    new_formulas = {row["formula"] for row in new_rows}
    
    merged_rows = []
    # Add all non-exp_log rows, or exp_log rows of formulas that were NOT in the new run
    for row in master_rows:
        if row["operation_set"] == "exp_log" and row["formula"] in new_formulas:
            continue
        merged_rows.append(row)
        
    # Append the new exp_log rows
    for row in new_rows:
        if row["operation_set"] == "exp_log":
            merged_rows.append(row)
            
    # Write back to master CSV
    fieldnames = [
        "formula", "operation_set", "x_low", "x_high", "beta", "proposals",
        "max_complexity", "max_depth", "train_rmse", "test_rmse", "complexity",
        "coefficients", "final_expression", "simplified_expression"
    ]
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            # Clean up keys to match fieldnames
            cleaned_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(cleaned_row)
            
    print(f"Successfully merged new EML results into {master_csv}")
    
    # 2. Plot results
    # Read data from master CSV for plotting
    data = {}
    with open(master_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            formula = row[0]
            op_set = row[1]
            train_rmse = float(row[8]) if row[8] else 1e-10
            test_rmse = float(row[9]) if row[9] else 1e-10
            complexity = int(row[10]) if row[10] else 0
            
            if formula not in data:
                data[formula] = {}
            data[formula][op_set] = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "complexity": complexity
            }
            
    formulas = ["f1", "f2", "f3", "f4", "f5", "f6", "f_eml_test1", "f_eml_test2", "f_eml_test3", "f_eml_test4", "f_xy"]
    op_sets = ["default", "paper", "exp_log"]
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['text.color'] = '#1F2937'
    plt.rcParams['axes.labelcolor'] = '#1F2937'
    plt.rcParams['xtick.color'] = '#4B5563'
    plt.rcParams['ytick.color'] = '#4B5563'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = {
        "default": "#4F46E5",
        "paper": "#F43F5E",
        "exp_log": "#10B981"
    }
    
    labels = {
        "default": "Default Set (with sin, cos, x^2, x^3)",
        "paper": "Paper Set (standard operators)",
        "exp_log": "EML Set (exp(x) - ln(y) only)"
    }
    
    x = np.arange(len(formulas))
    width = 0.25
    
    # 1. Test RMSE Plot (Log Scale)
    for idx, op_set in enumerate(op_sets):
        rmse_vals = []
        for f in formulas:
            val = data.get(f, {}).get(op_set, {}).get("test_rmse", None)
            if val is None:
                rmse_vals.append(np.nan)
            elif val < 1e-8:
                rmse_vals.append(1e-8)
            else:
                rmse_vals.append(val)
                
        bar_pos = x + (idx - 1) * width
        ax1.bar(bar_pos, rmse_vals, width, label=labels[op_set], color=colors[op_set], alpha=0.9, edgecolor='none')
        
    ax1.set_yscale('log')
    ax1.set_ylabel('Test RMSE (Log Scale)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_title('Model Accuracy (Test RMSE) across Operation Sets', fontsize=14, fontweight='bold', pad=15, color='#111827')
    ax1.set_xticks(x)
    ax1.set_xticklabels(formulas, rotation=15, ha='right', fontsize=10)
    ax1.grid(True, which="both", ls="--", color="#E5E7EB", alpha=0.7)
    ax1.set_axisbelow(True)
    
    # 2. Complexity Plot (Linear Scale)
    for idx, op_set in enumerate(op_sets):
        comp_vals = []
        for f in formulas:
            val = data.get(f, {}).get(op_set, {}).get("complexity", None)
            if val is None:
                comp_vals.append(np.nan)
            else:
                comp_vals.append(val)
                
        bar_pos = x + (idx - 1) * width
        ax2.bar(bar_pos, comp_vals, width, label=labels[op_set], color=colors[op_set], alpha=0.9, edgecolor='none')
        
    ax2.set_ylabel('Model Complexity (Number of Nodes)', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title('Model Complexity (Node Count) across Operation Sets', fontsize=14, fontweight='bold', pad=15, color='#111827')
    ax2.set_xticks(x)
    ax2.set_xticklabels(formulas, rotation=15, ha='right', fontsize=10)
    ax2.grid(True, ls="--", color="#E5E7EB", alpha=0.7)
    ax2.set_axisbelow(True)
    
    handles, labels_list = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_list, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB', fontsize=10)
    
    plt.subplots_adjust(top=0.90, hspace=0.35)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_path = r"experiments/results/final/final_run_visualization.png"
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to: {output_path}")
    
    # Also save to conversation artifacts directory if it exists
    artifacts_dir = r"C:\Users\alex3\.gemini\antigravity-ide\brain\fa80e6d4-1edd-4ef8-8d4a-cf9235e25524"
    if os.path.exists(artifacts_dir):
        artifacts_plot_path = os.path.join(artifacts_dir, "final_run_visualization.png")
        shutil.copy2(output_path, artifacts_plot_path)
        print(f"Also copied visualization to artifacts folder: {artifacts_plot_path}")

if __name__ == "__main__":
    merge_and_plot()
