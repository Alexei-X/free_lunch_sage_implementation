"""
FreeLunch algebraic attack on Griffin-pi (SageMath implementation).

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

# Griffin parameters
p        = 28407454060060787          # 55-bit prime
n_rounds = 4                          # number of rounds to attack by default
t        = 12                         # number of branches
alpha    = 3                          # smallest integer with gcd(alpha, p-1)=1

Fp     = GF(p)

# Quadratic-function parameters for the non-linear layer
deltas = {i: Fp(4 * (i + 1)) for i in range(2, t)}
mus    = {i: Fp(7 * (i + 1)**2) for i in range(2, t)}

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

# Round constants
set_random_seed(int.from_bytes(b"Griffin", "little"))
RC = []
for _ in range(n_rounds - 1):
    RC.append(vector(Fp, [Fp.random_element() for _ in range(t)]))
RC.append(vector(Fp, [Fp(0)] * t))   # last round constant is 0

d = 2 * alpha + 1          # degree per round for Griffin
weights = [1]              # for x0
for i in range(1, n_rounds + 1):
    weights.append(d**(i - 1))   # = alpha * wt(u_i), integer
to = TermOrder('wdeglex', tuple(weights))
names = ["x0"] + [f"u{i}" for i in range(1, n_rounds + 1)]
PR_fl = PolynomialRing(Fp, names, order=to)
gens = PR_fl.gens()
x0 = gens[0]
U  = list(gens[1:])

# Griffin non-linear layer
def L_i(y0, y1, y_prev, i):
    return Fp(i - 1) * y0 + y1 + y_prev


def griffin_nl(state, u_new, system):
    """
    Apply Griffin non-linear layer.

    Parameters
    * state  : list of t polynomials (current state before this NL layer)
    * u_new  : fresh polynomial variable for x0^{1/alpha} in this round

    Returns
    * new_state : list of t polynomials (output of NL layer)
    * p_eq      : polynomial equation  u_new^alpha - state[0]
    """
    # Equation for the 1/alpha branch: u_new^alpha = state[0]
    p_eq = u_new**alpha - state[0].reduce(system)

    y0 = u_new
    y1 = state[1]**alpha   # y1 = state[1]^alpha  (alpha branch)

    new_state = [y0, y1]
    for i in range(2, t):
        li = L_i(y0, y1, state[i - 1], i)
        y_i = (state[i] * (li**2 + deltas[i] * li + mus[i])).reduce(system)
        new_state.append(y_i)

    return new_state, p_eq


def griffin_affine(state, rnd_idx):
    """
    Parameters
    * state    : list of t polynomials
    * rnd_idx  : round index

    Returns
    * new_state : list of t polynomials
    """
    sv = vector(PR_fl, state)
    rc = vector(PR_fl, [PR_fl(c) for c in RC[rnd_idx]])
    result = M_mat * sv + rc
    return list(result)


def griffin_round(state, rnd_idx, u_new, system):
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
    state_nl, p_eq = griffin_nl(state, u_new, system)
    new_state = griffin_affine(state_nl, rnd_idx)
    return new_state, p_eq


# Step 1: sysGen: build the FreeLunch polynomial system
# CICO problem: find raw_input with raw_input[0] = 0 and output[0] = 0.

def build_initial_state():
    """
    Returns a list of t affine polynomials in x0 (the state after the initial linear layer, before round 1).
    """
    raw = [PR_fl(0), 31*x0+3, 3*x0, x0+1, 7*x0+43, 2*x0+1, 4*x0+2, 6*x0+12, 9*x0+36, 3*x0+3, 4*x0+7, 2*x0+10]
    print(f"Input as a x0 state = {raw}")
    raw_vec = vector(PR_fl, raw)
    z0 = list(M_mat * raw_vec)
    #print(f"z0 = {z0}")
    return z0


def sys_gen(n_rounds, PR_fl):
    """
    Generate the FreeLunch polynomial system for Griffin with n_rounds rounds.

    Returns
    * system : list of n_rounds+1 polynomials
             system[0..n_rounds-1] are the p_i  (u_i^alpha - arg_to_inversion_i)
             system[n_rounds]      is  g = output[0]  (must be 0 for CICO)
    """
    state  = build_initial_state()
    system = []
    for rnd in range(n_rounds):
        u_new = U[rnd]
        state, p_eq = griffin_round(state, rnd, u_new, system)
        system.append(p_eq)
    # g = first component of the final state (CICO requires this = 0)
    system.append(state[0].reduce(system))
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
    for i in range(alpha0):
        coeff = x**i
        Mi = Mi_list[i]
        for r in range(DH):
            for c in range(DH):
                val = Mi[r, c]
                if val != 0:
                    A[r, c] += coeff * val

    start_time = time.time()
    # Compute Determinant using Bareiss Algorithm
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
        y0_val = state[0]**((p - 1 + alpha) // alpha % (p - 1))  # x^{1/alpha} in Fp
        inv_a  = pow(int(alpha), -1, p - 1)
        y0_val = pow(int(state[0]), inv_a, p)
        y1_val = pow(int(state[1]), alpha, p)

        nl = [Fp(y0_val), Fp(y1_val)]
        for i in range(2, t):
            li_val = Fp((i - 1)) * Fp(y0_val) + Fp(y1_val) + state[i - 1]
            nl.append(state[i] * (li_val**2 + deltas[i] * li_val + mus[i]))

        # Affine layer
        state = list(M_mat * vector(Fp, nl) + RC[rnd])

    return state


def verify_cico(x0_val, n_rounds):
    """
    Check that raw_input is a CICO solution.
    """
    #inp = [Fp(0), Fp(x0_val)] + [Fp(0)] * (t - 2)
    inp = [Fp(0), Fp(31*x0_val+3), Fp(3*x0_val), Fp(x0_val+1), Fp(7*x0_val+43), Fp(2*x0_val+1), Fp(4*x0_val+2),
        Fp(6*x0_val+12), Fp(9*x0_val+36), Fp(3*x0_val+3), Fp(4*x0_val+7), Fp(2*x0_val+10)]
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
    print(f"  f(x0) = {f}")

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
        #inp = [Fp(0)] + [Fp(rv)] + [Fp(0)] * (t - 2)
        inp = [Fp(0), Fp(31*rv+3), Fp(3*rv), Fp(rv+1), Fp(7*rv+43), Fp(2*rv+1), Fp(4*rv+2),
            Fp(6*rv+12), Fp(9*rv+36), Fp(3*rv+3), Fp(4*rv+7), Fp(2*rv+10)]
        print(f"  x0 = {rv}")
        print(f"  input[0]  = {inp[0]},  output[0] = {out[0]},  CICO OK: {ok}")
        print(f"Input state = {inp}")
        print(f"Output state = {out}")

    return roots


if __name__ == "__main__":
    roots = freelunch_attack(n_rounds)
