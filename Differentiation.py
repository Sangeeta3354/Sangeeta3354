import sympy as sp

# Define a symbolic variable
x = sp.symbols('x')

# Define a function
f = x**2 + 3*x + 2

# Find the derivative of the function with respect to x
f_prime = sp.diff(f, x)

# Print the derivative
print("The derivative of f(x) is:", f_prime)
