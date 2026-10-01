"""
FreeLunch algebraic attack on Griffin-pi (SageMath implementation).
-> Using rounds bypassing as explained in appendix B of reference paper

Reference: "The Algebraic FreeLunch: Efficient Gröbner Basis Attacks Against
Arithmetization-Oriented Primitives"

Attack overview:
  1. sysGen  : build the FreeLunch polynomial system P_G
  2. matGen  : compute the multiplication matrix T0 from the FreeLunch basis
  3. polyDet : compute det(x0^alpha0 * I_DH + sum_{i=0}^{alpha0-1} x0^i * M_i)
  4. uniSol  : find roots of the resulting univariate polynomial
"""

import sys
from sage.all import *
import time
import numpy as np

# Griffin parameters
p        = 28407454060060787          # 55-bit prime
n_rounds = 6                          # number of rounds to attack by default
t        = 12                         # number of branches
alpha    = 3                          # smallest integer with gcd(alpha, p-1)=1
inv_a  = pow(alpha, -1, p - 1)
print("inv_a : ", inv_a)

Fp     = FiniteField(p)

# Quadratic-function parameters for the non-linear layer
deltas = [0, 0] + [Fp(4 * (i + 1)) for i in range(t-2)]
mus    = [0, 0] + [Fp(7 * (i + 1)**2) for i in range(t-2)]

# MDS matrix M
M_rows = [
    [10,14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3],
    [ 8,12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1],
    [ 2, 6,10,14, 1, 3, 5, 7, 1, 3, 5, 7],
    [ 2, 2, 8,12, 1, 1, 4, 6, 1, 1, 4, 6],
    [ 5, 7, 1, 3,10,14, 2, 6, 5, 7, 1, 3],
    [ 4, 6, 1, 1, 8,12, 2, 2, 4, 6, 1, 1],
    [ 1, 3, 5, 7, 2, 6,10,14, 1, 3, 5, 7],
    [ 1, 1, 4, 6, 2, 2, 8,12, 1, 1, 4, 6],
    [ 5, 7, 1, 3, 5, 7, 1, 3,10,14, 2, 6],
    [ 4, 6, 1, 1, 4, 6, 1, 1, 8,12, 2, 2],
    [ 1, 3, 5, 7, 1, 3, 5, 7, 2, 6,10,14],
    [ 1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8,12],
]
M_mat = matrix(Fp, M_rows)
M_mat_inv = M_mat.inverse()

# Round constants
set_random_seed(int.from_bytes(b"Griffin", "little"))
RC = []
for _ in range(n_rounds - 1):
    RC.append(vector(Fp, [Fp.random_element() for _ in range(t)]))
RC.append(vector(Fp, [Fp(0)] * t))   # last round constant is 0

d = 2 * alpha + 1          # degree per round for Griffin
weights = [1] + [d**(i + 1) + 1 for i in range(n_rounds - 3)] # -3 thanks to the bypassing technique
to = TermOrder('wdeglex', tuple(weights))
names = ["x0"] + [f"u{i}" for i in range(1, n_rounds - 3 + 1)]
PR_fl = PolynomialRing(Fp, names, order=to)
gens = PR_fl.gens()
x0 = gens[0]
U  = list(gens[1:])

A_ring = PolynomialRing(Fp, 'a2,a4,a6,a8,a11', order="lex")
a2, a4, a6, a8, a11 = A_ring.gens()
input_ais = list()
B_ring = PolynomialRing(Fp, 'b2,b4,b6,b8,b11,u0', order="lex")
b2, b4, b6, b8, b11, u0 = B_ring.gens()
input_bis = list()
X_ring = PolynomialRing(B_ring, 'x')
x = X_ring.gen()
M_mat_x = matrix(X_ring, M_mat)

# Griffin non-linear layer
def L_i(z0, z1, z2, i):
    return Fp(i - 1) * z0 + z1 + (z2 if i > 2 else 0)

def griffin_nl(state, zero=False, leading_term=False):
    """
    Apply Griffin non-linear layer to a state

    Parameters
    * state : list of polynomials or numbers

    Returns
    * nl : output of NL layer
    """
    nl = [0, 0]
    if not zero:
        #print("state[0] : ", state[0])
        nl = [Fp(state[0])**inv_a, state[1]**alpha]
        #print("state[0]**inv_a : ", Fp(state[0])**inv_a)
    for i in range(2, t):
        li_val = L_i(nl[0], nl[1], state[i-1], i)
        #print(f"L_{i} : ", li_val)
        if not leading_term:
            nl.append(state[i] * (li_val**2 + deltas[i] * li_val + mus[i]))
            #print(f"nl[{i}] = ", nl[i])
        else:
            nl.append(state[i] * li_val**2)

    return nl

def griffin_nl_with_new_var(state, u_new, system, reduction=False):
    """
    Apply Griffin non-linear layer and take care of introduction of new variable and equation in system

    Parameters
    * state  : list of t polynomials (current state before this NL layer)
    * u_new  : fresh polynomial variable for x0^{1/alpha} in this round

    Returns
    * new_state : list of t polynomials (output of NL layer)
    * p_eq      : polynomial equation  u_new^alpha - state[0]
    """
    # Equation for the 1/alpha branch: u_new^alpha = state[0]
    p_eq = 0
    if not reduction:
        p_eq = state[0][0] - u_new**alpha
    else:
        p_eq = state[0].reduce(system) - u_new**alpha

    y0 = u_new
    y1 = 0
    if not reduction:
        y1 = state[1][0]**alpha
    else:
        y1 = state[1]**alpha   # y1 = state[1]^alpha  (alpha branch)

    new_state = [y0, y1]
    for i in range(2, t):
        li = L_i(y0, y1, state[i - 1], i)
        y_i = 0
        if not reduction:
            y_i = (state[i] * (li**2 + deltas[i] * li + mus[i]))
        else:
            y_i = (state[i].reduce(system) * (li**2 + deltas[i] * li + mus[i]).reduce(system)).reduce(system)
        new_state.append(y_i)

    return new_state, p_eq

def griffin_affine(state, rnd_idx, ring):
    """
    Parameters
    * state    : list of t polynomials
    * rnd_idx  : round index

    Returns
    * new_state : list of t polynomials
    """
    sv = vector(ring, state)
    if rnd_idx is None:
        rc = 0
    else:
        rc = vector(ring, [ring(c) for c in RC[rnd_idx]])
    result = M_mat * sv + rc
    return list(result)

def griffin_round_with_new_var(state, rnd_idx, u_new, system):
    """
    One full Griffin round: NL layer then affine layer.

    Parameters
    * state   : list of t polynomials (input state)
    * rnd_idx : round index
    * u_new   : auxiliary variable for this round

    Returns
    * new_state : list of t polynomials
    * p_eq      : equation u_new^alpha - (input to 1/alpha branch)
    """
    state_nl, p_eq = griffin_nl_with_new_var(state, u_new, system, True)
    #print("NL layer state : ", state_nl)
    new_state = griffin_affine(state_nl, rnd_idx, PR_fl)
    #print("After round state : ", new_state)
    return new_state, p_eq


# Step 1: sysGen: build the FreeLunch polynomial system
# CICO problem: find raw_input with raw_input[0] = 0 and output[0] = 0.
def build_initial_state(system):
    init_a = [0, 0, a2, 0, a4, 0, a6, 0, a8, 0, 0, a11]
    input_a = M_mat_inv * vector(init_a)
    #print("input a : ", input_a)
    one_nl_a = griffin_nl(init_a, True)
    one_round_a = griffin_affine(one_nl_a, None, A_ring)
    #print("One round a : ", one_round_a)
    two_nl_a = griffin_nl(one_round_a, True, True)
    #print("Two NL a : ", two_nl_a)
    two_rounds_a = griffin_affine(two_nl_a, None, A_ring)
    #print("Two rounds a : ", two_rounds_a)
    a_system = [input_a[0], one_round_a[0], one_round_a[1], two_rounds_a[0], a11 - 2]
    #print()
    #print("System in a : ")
    #for eq in a_system:
    #    print("Equation : ", eq)
    #print()
    ideal = A_ring.ideal(a_system)
    ideal_variety = ideal.variety()
    A2, A4, A6, A8, A11 = ideal_variety[-1]['a2'], ideal_variety[-1]['a4'], ideal_variety[-1]['a6'], ideal_variety[-1]['a8'], ideal_variety[-1]['a11']
    ai_s = vector(Fp, [0, 0, A2, 0, A4, 0, A6, 0, A8, 0, 0, A11])
    print("ai_s : ", ai_s)
    input_ais.extend(list(M_mat_inv * ai_s))
    init_b = [0, 0, b2, 0, b4, 0, b6, 0, b8, 0, 0, b11]
    init_x = [ai_s[i]*x + init_b[i] for i in range(t)]
    input_x = M_mat_inv * vector(init_x)
    one_nl_x = griffin_nl(init_x)
    #print("one nl x : ", one_nl_x)
    one_round_x = griffin_affine(one_nl_x, 0, X_ring)
    #print("one round x : ", one_round_x)
    two_nl_x, eq_u0 = griffin_nl_with_new_var(one_round_x, u0, system)
    #print("two_nl_x : ", two_nl_x)
    two_rounds_x = griffin_affine(two_nl_x, 1, X_ring)
    #print("two rounds x : ", two_rounds_x)
    b_system = [input_x[0][0], two_rounds_x[0][1], two_rounds_x[0][2], b2-1, b4-1, eq_u0]
    #print()
    #print("System in b : ")
    #for eq in b_system:
    #    print("Equation : ", eq)
    #print()
    ideal = B_ring.ideal(b_system)
    ideal_variety = ideal.variety()
    #print("ideal variety : ", ideal_variety)
    B2, B4, B6, B8, B11, U0 = ideal_variety[-1]["b2"],ideal_variety[-1]["b4"],ideal_variety[-1]["b6"],ideal_variety[-1]["b8"],ideal_variety[-1]["b11"],ideal_variety[-1]["u0"]
    bi_s = vector(Fp, [0, 0, B2, 0, B4, 0, B6, 0, B8, 0, 0, B11])
    print("bi_s : ", bi_s)
    input_bis.extend(list(M_mat_inv * bi_s))
    input_state = [input_ais[i]*x0 + input_bis[i] for i in range(t)]
    print("input_state : ", input_state)
    M_input_state = [ai_s[i]*x0 + bi_s[i] for i in range(t)]
    print("M * input_state : ", M_input_state)
    one_nl_state = griffin_nl(M_input_state, False)
    #print("one_nl_state : ", one_nl_state)
    one_round_state = griffin_affine(one_nl_state, 0, PR_fl)
    #print("One round state : ", one_round_state)
    two_nl_state = griffin_nl(one_round_state, False)
    #print("two_nl_state : ", two_nl_state)
    two_rounds_state = griffin_affine(two_nl_state, 1, PR_fl)
    #print("two_rounds_state : ", two_rounds_state)
    three_nl_state = griffin_nl(two_rounds_state, False)
    #print("three_nl_state : ", three_nl_state)
    three_rounds_state = griffin_affine(three_nl_state, 2, PR_fl)
    return input_state, three_rounds_state

def sys_gen(n_rounds, PR_fl):
    """
    Generate the FreeLunch polynomial system for Griffin with n_rounds rounds, using the bypassing technique introduced in Appendix B of the reference paper

    Returns
    * system : list of system polynomials
    """
    system = []
    state, three_rnds_state  = build_initial_state(system)
    state = three_rnds_state
    n_extra_rounds = n_rounds-3
    for rnd in range(n_extra_rounds):
        u_new = U[rnd]
        state, p_eq = griffin_round_with_new_var(state, rnd+3, u_new, system)
        system.append(p_eq)
        print("New equation nbr ", len(system))
    # g = first component of the final state (CICO requires this = 0)
    #print("state[0] = ", state[0])
    system.append(state[0])
    return system


# Step 2 : matGen: compute the multiplication matrix T0
# We compute the M_i blocks by reducing x0^{alpha0} * phi_j modulo the ideal and reading off the coefficients.
def mat_gen(system_fl, PR_fl, n_rounds):
    """
    Compute the block matrices M0..M_{alpha0-1} defining T0.

    Parameters
    * system_fl : list of polynomials in PR_fl (the FreeLunch system)
    * PR_fl     : PolynomialRing with FreeLunch order
    * n_rounds  : number of Griffin rounds

    Returns
    * alpha0  : int     (degree of x0 in the leading monomial of g)
    * DH      : int     (dimension of the H subspace = product of alpha_i)
    * Mi_list : list of alpha0 matrices over Fp  (M0 .. M_{alpha0-1})
    * B_H     : list of monomials (basis of H, i.e. monomials not involving x0)
    """
    x0_fl = PR_fl.gen(0)

    # Normal basis
    # We need the standard monomials (basis of R/I) w.r.t. the FreeLunch order.
    I_fl  = PR_fl.ideal(system_fl)
    B_all = I_fl.normal_basis()   # monomials not in LT(I) under FreeLunch order

    # H-basis: monomials with x0-exponent == 0
    B_H = [m for m in B_all if m.degree(x0_fl) == 0]
    print("B_H : ", B_H)

    DH     = len(B_H)
    DI     = len(B_all)
    alpha0 = DI // DH      # degree of x0 in LM(g)

    print(f"DI = {DI}, DH = {DH}, alpha0 = {alpha0}")

    B_H_idx = {m: j for j, m in enumerate(B_H)}
    print("B_H_idx : ", B_H_idx)

    GB = system_fl   # FreeLunch system is already a GB in its order

    # Build M_i blocks
    # From the reduction: x0^{alpha0} * phi_j \equiv rem (mod I)
    # where rem = sum_{k,l} c_{k,l} * x0^k * B_H[l],
    # and by Lemma 1:  rem = -sum_k sum_l M_k[l,j] * x0^k * B_H[l]
    # => M_k[l,j] = -c_{k,l}
    Mi_list = [matrix(Fp, DH, DH) for _ in range(alpha0)]

    for j, phi in enumerate(B_H):
        poly_to_reduce = x0_fl**alpha0 * phi
        # .reduce(GB) performs division with remainder w.r.t. the ring's term order, returning the normal form.
        rem = poly_to_reduce.reduce(GB)
        for mono, coef in zip(rem.monomials(), rem.coefficients()):
            k_exp = mono.degree(x0_fl)
            # Divide out x0^k to isolate the B_H component
            base = PR_fl(mono // (x0_fl**k_exp)) if k_exp > 0 else PR_fl(mono)
            if base in B_H_idx:
                l = B_H_idx[base]
                Mi_list[k_exp][l, j] = -Fp(coef)

    return alpha0, DH, Mi_list, B_H


# Step 3: polyDet: compute det via Lemma 2
def poly_det(alpha0, DH, Mi_list):
    """
    Compute the determinant using the Bareiss algorithm (fraction-free Gaussian elimination)
    directly on the matrix with polynomial entries.
    """
    Fx = PolynomialRing(Fp, 'x')
    x = Fx.gen()
    # Construct the Matrix with Polynomial Entries
    # A(x) = x^alpha0 * I + sum(x^i * Mi)
    print("  Constructing polynomial matrix...")
    A = matrix(Fx, DH, DH)

    # Set diagonal: x^alpha0
    x_alpha0 = x**alpha0
    for i in range(DH):
        A[i, i] = x_alpha0

    # Add Mi blocks
    # Idea for optimization : use previous coeff to compute power + use blocks of coefficients
    coeff = 1
    for i in range(alpha0):
        Mi = Mi_list[i]
        A += coeff * Mi
        #for r in range(DH):
        #    for c in range(DH):
        #        val = Mi[r, c]
        #        if val != 0:
        #            A[r, c] += coeff * val
        coeff *= x

    start_time = time.time()
    f = A.det()

    end_time = time.time()
    print(f"  Determinant computed in {end_time - start_time:.2f} seconds.")
    print(f"  Resulting polynomial degree: {f.degree()}")
    return f


# Step 4: uniSol: find roots of f(x0)
def uni_sol(f):
    """
    Return all roots of the univariate polynomial f over Fp.
    """
    return [r for r, _ in f.roots()]


# Verification
def griffin_eval(input_vec, n_rounds):
    """
    Evaluate Griffin-pi numerically on input_vec (list of Fp elements).
    Returns the output vector.
    """
    state = list(M_mat * vector(Fp, input_vec))

    for rnd in range(n_rounds):
        # Non-linear layer S
        nl = griffin_nl(state)
        # Affine layer
        state = griffin_affine(nl, rnd, Fp)

    return state


def verify_cico(x0_val, n_rounds):
    """
    Check that raw_input is a CICO solution.
    """
    inp = [Fp(0)] + [input_ais[i]*x0_val + input_bis[i] for i in range(1, t)]
    print("input in x0_val : ", inp)
    out = griffin_eval(inp, n_rounds)
    return out[0] == Fp(0), out


# Main attack
def freelunch_attack(n_rounds):
    start_time = time.time()
    print("=" * 60)
    print(f"FreeLunch attack on Griffin-pi")
    print(f"  p={p}, t={t}, alpha={alpha}, R={n_rounds}")
    print("=" * 60)

    # Step 1: sysGen
    print("\n[1/4] sysGen: building FreeLunch polynomial system...")
    system = sys_gen(n_rounds, PR_fl)
    print(f"  Generated {len(system)} polynomials in {PR_fl.ngens()} variables.")
    print(f"  Equations:")
    for i, eq in enumerate(system):
        lm = eq.lm()
        print(f"    p[{i}]: LM = {lm},  #terms = {len(eq.monomials())}")

    system_fl = system
    print("  FreeLunch ring order:", PR_fl.term_order())
    print("  Leading monomials in FreeLunch order:")
    for i, eq in enumerate(system_fl):
        print(f"    p[{i}]: LM = {eq.lm()}")

    # Confirm it is a FreeLunch system: all leading monomials should be pure powers of a single variable.
    lms = [eq.lm() for eq in system_fl]
    print(f"\n  FreeLunch check: LMs = {lms}")
    is_fl = all(sum(1 for e in lm.exponents()[0] if e > 0) == 1 for lm in lms)
    print(f"  Is FreeLunch system: {is_fl}")
    if not is_fl:
        print("  WARNING: Not a FreeLunch system => check encoding!")

    # Step 2: matGen
    print("\n[2/4] matGen: computing multiplication matrix T0...")
    alpha0, DH, Mi_list, B_H = mat_gen(system_fl, PR_fl, n_rounds)
    print(f"  alpha0 = {alpha0},  DH = {DH}")
    print(f"  M_i blocks (sizes {DH}x{DH}):  {len(Mi_list)} blocks")

    # Step 3: polyDet
    print("\n[3/4] polyDet: computing det(x0^alpha0 * I + sum x0^i * M_i)...")
    f = poly_det(alpha0, DH, Mi_list)
    print(f"  Degree of f(x0) = {f.degree()}")
    #print(f"  f(x0) = {f}")

    # Step 4: uniSol
    print("\n[4/4] uniSol: finding roots of f(x0)...")
    roots = uni_sol(f)
    end_time = time.time()
    print(f"  Found {len(roots)} root(s): {roots}")
    print("Duration of the attack : ", end_time - start_time, "seconds")

    # Verification
    print("\nVerification:")

    for rv in roots:
        ok, out = verify_cico(rv, n_rounds)
        inp = [Fp(0)] + [input_ais[i]*rv + input_bis[i] for i in range(1, t)]
        print(f"  x0 = {rv}")
        print(f"  input[0]  = {inp[0]},  output[0] = {out[0]},  CICO OK: {ok}")
        print(f"Input state = {inp}")
        print(f"Output state = {out}")

    return roots


roots = freelunch_attack(n_rounds)
