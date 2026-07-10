import numpy as np

def p_split(depth, beta=-0.5):
    return 1 / np.power((1 + depth), -beta)

def get_scale(is_left, sibling_type):
    if is_left:
        # Default symmetric scale for the left child
        # To make L and R symmetric, we could scale L by 1.0 (so it has base probability)
        # Or we can scale L such that it has exactly 50% chance to split. Let's just use 1.0.
        return 1.0 
    else:
        # Right child scales based on left child's state
        if sibling_type == 0: 
            # Left child terminated. Right child MUST split.
            return 20.0
        else: 
            # Left child split. Right child MUST terminate.
            return 0.05

print(f"{'Depth':<10}{'Base p':<10}{'P(Term, Term)':<15}{'P(Term, Split)':<15}{'P(Split, Term)':<15}{'P(Split, Split)':<15}")
for d in range(0, 5):
    p_base = p_split(d)
    
    # Left child
    p_L_split = min(0.9999, p_base * get_scale(True, None))
    p_L_term = 1 - p_L_split
    
    # Right child | L is term
    p_R_split_given_L_term = min(0.9999, p_base * get_scale(False, 0))
    p_R_term_given_L_term = 1 - p_R_split_given_L_term
    
    # Right child | L is split
    p_R_split_given_L_split = min(0.9999, p_base * get_scale(False, 1))
    p_R_term_given_L_split = 1 - p_R_split_given_L_split
    
    # Joint probabilities
    p_term_term = p_L_term * p_R_term_given_L_term
    p_term_split = p_L_term * p_R_split_given_L_term
    p_split_term = p_L_split * p_R_term_given_L_split
    p_split_split = p_L_split * p_R_split_given_L_split
    
    print(f"{d:<10}{p_base:<10.3f}{p_term_term:<15.4f}{p_term_split:<15.4f}{p_split_term:<15.4f}{p_split_split:<15.4f}")
