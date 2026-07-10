import sympy as sp

x = sp.Symbol('x')

def eml(l, r):
    return sp.exp(l) - sp.log(r)

# Define components
# 1. A1 = eml(1, x) -> e - log(x)
A1 = eml(sp.Integer(1), x)

# 2. A2 = eml(1, A1) -> e - log(e - log(x))
A2 = eml(sp.Integer(1), A1)

# 3. A3 = eml(A2, 1) -> exp(A2) -> e^e / (e - log(x))? No, eml(A2, 1) = exp(A2) - log(1) = exp(A2)
# Wait, let's evaluate A2 = eml(1, eml(1, x)) -> e - log(e - log(x))
# So eml(A2, 1) = exp(e - log(e - log(x))) = e^e / (e - log(x))
A3 = eml(A2, sp.Integer(1))

# 4. A4 = eml(1, A3) -> e - log(e^e / (e - log(x))) = e - (e - log(e - log(x))) = log(e - log(x))
A4 = eml(sp.Integer(1), A3)

# 5. Y = eml(A4, x) -> exp(A4) - log(x) = (e - log(x)) - log(x) = e - 2*log(x) = e - log(x^2)
Y = eml(A4, x)

# 6. L3 = log(Y) -> eml(1, eml(eml(1, Y), 1))
L3 = eml(sp.Integer(1), eml(eml(sp.Integer(1), Y), sp.Integer(1)))

# 7. Z = eml(L3, x) -> exp(L3) - log(x) = Y - log(x) = e - 2*log(x) - log(x) = e - 3*log(x) = e - log(x^3)
Z = eml(L3, x)

# 8. Z2 = eml(Z, 1) -> exp(Z) = e^e / x^3
Z2 = eml(Z, sp.Integer(1))

# 9. Z3 = eml(1, Z2) -> e - log(e^e / x^3) = log(x^3)
Z3 = eml(sp.Integer(1), Z2)

# 10. Cube = eml(Z3, 1) -> exp(Z3) = x^3
Cube = eml(Z3, sp.Integer(1))

print("Simplified Cube:")
print(sp.simplify(Cube))
