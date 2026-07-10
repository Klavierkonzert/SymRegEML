import os
import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = r"experiments/results/final/final_run.csv"

# Read data
data = {}
with open(csv_path, "r", encoding="utf-8") as f:
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

# List of formulas in order
formulas = ["f1", "f2", "f3", "f4", "f5", "f6", "f_eml_test1", "f_eml_test2", "f_eml_test3", "f_eml_test4"]
op_sets = ["default", "paper", "exp_log"]

# Set up matplotlib style for premium look
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['text.color'] = '#1F2937'
plt.rcParams['axes.labelcolor'] = '#1F2937'
plt.rcParams['xtick.color'] = '#4B5563'
plt.rcParams['ytick.color'] = '#4B5563'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

# Colors
colors = {
    "default": "#4F46E5",  # Indigo
    "paper": "#F43F5E",    # Rose/Coral
    "exp_log": "#10B981"   # Emerald
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
        # Use 1e-10 as a floor for visualization of zero/near-zero values
        if val is None:
            rmse_vals.append(np.nan)
        elif val < 1e-8:
            rmse_vals.append(1e-8)
        else:
            rmse_vals.append(val)
            
    # Shift bars
    bar_pos = x + (idx - 1) * width
    # Filter out NaNs for bars
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

# Legend and layout
handles, labels_list = ax1.get_legend_handles_labels()
fig.legend(handles, labels_list, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB', fontsize=10)

# Add some padding at the top of subplots for the legend
plt.subplots_adjust(top=0.90, hspace=0.35)
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Save plot
output_dir = r"experiments/results/final"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "final_run_visualization.png")
plt.savefig(output_path, dpi=300)
print(f"Visualization saved to: {output_path}")
