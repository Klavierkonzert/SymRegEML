import sys
import os
import io
import numpy as np
import pandas as pd

# Add codes to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MCMC-SymReg/codes')))
from bsr_class import BSR

def test_annealing():
    # Dummy data
    x = np.random.uniform(0.1, 5.9, size=(50, 2))
    y = 8.0 * x[:, 0]**2 + 8.0 * x[:, 1]**3 - 15.0
    
    # 1. Test Default Set (should have temp = 1.0)
    print("Testing 'default' operation set (should remain un-annealed at 1.0):")
    model_default = BSR(treeNum=1, itrNum=1, val=10, operation_set="default", disp=True)
    
    # Capture stdout
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    try:
        model_default.fit(x, y)
    finally:
        sys.stdout = sys.__stdout__
        
    output_default = stdout_capture.getvalue()
    lines_default = [line for line in output_default.split('\n') if "newProp" in line]
    print(f"Captured {len(lines_default)} newProp calls.")
    for line in lines_default:
        print(f"  {line.strip()}")
        assert "temp=1.000" in line, f"Expected temperature 1.000, got: {line}"
    print("Default set test PASSED!\n")

    # 2. Test EML Set (should have temp starting at 50.0 and cooling)
    print("Testing 'exp_log' operation set (should start at 50.0 and cool down):")
    model_eml = BSR(treeNum=1, itrNum=1, val=20, operation_set="exp_log", disp=True)
    
    stdout_capture_eml = io.StringIO()
    sys.stdout = stdout_capture_eml
    try:
        model_eml.fit(x, y)
    finally:
        sys.stdout = sys.__stdout__
        
    output_eml = stdout_capture_eml.getvalue()
    lines_eml = [line for line in output_eml.split('\n') if "newProp" in line]
    print(f"Captured {len(lines_eml)} newProp calls.")
    for line in lines_eml:
        print(f"  {line.strip()}")
        
    # Check that early temperatures are > 1.0, and later is 1.0
    temps = []
    for line in lines_eml:
        parts = line.split("temp=")
        if len(parts) > 1:
            t_val = float(parts[1].split(")...")[0])
            temps.append(t_val)
            
    assert temps[0] == 50.0, f"Expected starting temp 50.0, got {temps[0]}"
    assert temps[-1] == 1.0, f"Expected final temp 1.0, got {temps[-1]}"
    assert all(temps[i] >= temps[i+1] for i in range(len(temps)-1)), "Temperature schedule should be non-increasing!"
    print("EML set annealing test PASSED!\n")
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_annealing()
