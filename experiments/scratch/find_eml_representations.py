import cmath
import sympy as sp

# Evaluation points (using complex numbers to avoid logs of negative numbers and real-only singularities)
PTS = [2.1 + 0.3j, 3.5 - 0.7j, 5.2 + 1.1j]

def eml_eval(l, r):
    # Avoid extreme exponents
    if l.real > 50:
        raise OverflowError
    # log(0) or log of very small numbers
    if abs(r) < 1e-15:
        raise ZeroDivisionError
    val = cmath.exp(l) - cmath.log(r)
    if cmath.isnan(val) or cmath.isinf(val):
        raise ValueError
    return val

def get_key(vals):
    # Round to 6 decimal places to group equivalent values
    return tuple(complex(round(v.real, 6), round(v.imag, 6)) for v in vals)

def search_eml():
    x = sp.Symbol('x')
    
    # dp[c] stores a dict of {key: (sympy_expr, float_vals_list, repr_str)} for complexity c
    dp = {}
    
    # Terminals (Complexity 1)
    dp[1] = {}
    
    # 1.0 terminal
    t1_vals = [1.0, 1.0, 1.0]
    dp[1][get_key(t1_vals)] = (sp.Integer(1), t1_vals, "1")
    
    # x terminal
    tx_vals = list(PTS)
    dp[1][get_key(tx_vals)] = (x, tx_vals, "x")
    
    # Targets we want to find
    targets = {
        "x^2": (lambda val: val**2, x**2),
        "x^3": (lambda val: val**3, x**3),
        "0": (lambda val: 0.0, sp.Integer(0)),
        "sin(x)": (lambda val: cmath.sin(val), sp.sin(x)),
        "cos(x)": (lambda val: cmath.cos(val), sp.cos(x)),
    }
    
    # Precompute target keys
    target_keys = {}
    for target_name, (target_func, target_sp) in list(targets.items()):
        target_vals = [target_func(v) for v in PTS]
        target_keys[get_key(target_vals)] = (target_name, target_sp)
        
    found = {}
    
    MAX_COMPLEXITY = 35
    print("Starting optimized EML search using observational equivalence...")
    for c in range(3, MAX_COMPLEXITY + 1, 2):
        dp[c] = {}
        # Try all partitions c1 + c2 = c - 1
        for c1 in range(1, c, 2):
            c2 = c - 1 - c1
            if c1 in dp and c2 in dp:
                for key1, (expr1, vals1, repr1) in dp[c1].items():
                    for key2, (expr2, vals2, repr2) in dp[c2].items():
                        try:
                            # Evaluate numerically on the points
                            new_vals = [eml_eval(vals1[i], vals2[i]) for i in range(3)]
                            new_key = get_key(new_vals)
                            
                            # Check if we already have this function at smaller complexity
                            already_exists = False
                            for prev_c in range(1, c, 2):
                                if new_key in dp[prev_c]:
                                    already_exists = True
                                    break
                            
                            if not already_exists and new_key not in dp[c]:
                                # Construct SymPy expression
                                sym_expr = sp.exp(expr1) - sp.log(expr2)
                                repr_str = f"eml({repr1}, {repr2})"
                                dp[c][new_key] = (sym_expr, new_vals, repr_str)
                                
                                # Check if it matches any target
                                if new_key in target_keys:
                                    target_name, target_sp = target_keys[new_key]
                                    # Double check with SymPy
                                    simplified = sp.simplify(sym_expr)
                                    if simplified == target_sp:
                                        found[target_name] = (c, repr_str)
                                        print(f"\nFOUND {target_name} at complexity {c}!")
                                        print(f"  Expression: {repr_str}")
                                        del target_keys[new_key]
                        except (OverflowError, ZeroDivisionError, ValueError, IndexError):
                            pass
        
        print(f"Complexity {c}: found {len(dp[c])} unique functions.")
        if not target_keys:
            break
            
    print("\nSearch Summary:")
    for name, (c, repr_str) in found.items():
        print(f" - {name}: complexity {c}, expression: {repr_str}")
    if target_keys:
        print(f"Remaining targets not found: {list(target_keys.values())}")

if __name__ == "__main__":
    search_eml()
