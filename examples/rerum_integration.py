"""
Integration example: Using rerum with symlik

This demonstrates how rerum's DSL and rewriting engine can be used
to express and simplify likelihood expressions symbolically.

rerum provides:
- Human-readable DSL for rules
- Pattern matching and term rewriting
- Tracing and debugging
- CLI/REPL for interactive exploration

symlik provides:
- Automatic differentiation for likelihoods
- MLE computation
- Standard errors and Fisher information
- DataFrame support

Together they enable mathematicians to:
1. Express models in readable DSL format
2. Symbolically manipulate and simplify expressions
3. Fit models to data
"""

try:
    from rerum import RuleEngine, E, ARITHMETIC_PRELUDE, MATH_PRELUDE
    RERUM_AVAILABLE = True
except ImportError:
    RERUM_AVAILABLE = False
    print("Note: rerum not installed. Install with: pip install rerum")

from symlik import LikelihoodModel, diff, simplify


# =============================================================================
# Example 1: Using rerum to simplify likelihood expressions
# =============================================================================

def example_simplify_with_rerum():
    """Use rerum's DSL to define simplification rules for likelihoods."""
    if not RERUM_AVAILABLE:
        return

    # Define rules for likelihood manipulation
    engine = RuleEngine.from_dsl('''
        # Algebraic simplification
        @add-zero: (+ ?x 0) => :x
        @mul-one: (* ?x 1) => :x
        @mul-zero: (* ?x 0) => 0
        @neg-neg: (* -1 (* -1 ?x)) => :x

        # Log rules
        @log-exp: (log (exp ?x)) => :x
        @exp-log: (exp (log ?x)) => :x
        @log-product: (log (* ?a ?b)) => (+ (log :a) (log :b))
        @log-quotient: (log (/ ?a ?b)) => (- (log :a) (log :b))
        @log-power: (log (^ ?a ?n)) => (* :n (log :a))

        # Exponential rules
        @exp-sum: (exp (+ ?a ?b)) => (* (exp :a) (exp :b))
        @exp-neg: (exp (* -1 ?x)) => (/ 1 (exp :x))

        # Power rules
        @pow-zero: (^ ?x 0) => 1
        @pow-one: (^ ?x 1) => :x
    ''')

    # Example: Simplify a Weibull hazard expression
    # h(t) = (k/θ) * (t/θ)^(k-1)
    weibull_hazard = E("(* (/ k theta) (^ (/ t theta) (- k 1)))")
    print("Weibull hazard:", weibull_hazard)

    # If we have k=1 (exponential special case), simplify
    # (^ x 0) => 1, so h(t) = k/θ = 1/θ = λ
    exp_special = E("(* (/ 1 theta) (^ (/ t theta) 0))")
    result = engine(exp_special)
    print(f"Exponential special case (k=1): {exp_special} => {result}")

    return engine


# =============================================================================
# Example 2: Define distribution contributions via DSL
# =============================================================================

def example_distribution_dsl():
    """Express distribution contributions in readable DSL format."""
    if not RERUM_AVAILABLE:
        return

    # Rules that expand distribution names into log-likelihood formulas
    distribution_rules = '''
        # Exponential: f(t|λ) = λ exp(-λt)
        @exponential "Log-likelihood for exponential":
            (loglik exponential ?rate ?t) =>
                (+ (log :rate) (* -1 (* :rate :t)))

        # Right-censored exponential: S(t|λ) = exp(-λt)
        @exponential-censored "Log-survival for exponential":
            (logsurv exponential ?rate ?t) =>
                (* -1 (* :rate :t))

        # Normal: f(x|μ,σ²) = (2πσ²)^(-1/2) exp(-(x-μ)²/(2σ²))
        @normal "Log-likelihood for normal":
            (loglik normal ?mu ?sigma2 ?x) =>
                (+ (* -0.5 (log (* 2 (* pi :sigma2))))
                   (* -1 (/ (^ (- :x :mu) 2) (* 2 :sigma2))))

        # Poisson: f(k|λ) = λ^k exp(-λ) / k!
        @poisson "Log-likelihood for Poisson":
            (loglik poisson ?lambda ?k) =>
                (+ (* :k (log :lambda)) (- 0 :lambda) (* -1 (logfact :k)))
    '''

    engine = RuleEngine.from_dsl(distribution_rules)

    # Expand exponential log-likelihood
    expr = E("(loglik exponential lambda t)")
    result = engine(expr)
    print(f"Exponential: {expr} => {result}")

    # Expand normal log-likelihood
    expr = E("(loglik normal mu sigma2 x)")
    result = engine(expr)
    print(f"Normal: {expr}")
    print(f"    => {result}")

    return engine


# =============================================================================
# Example 3: Differentiation rules via rerum
# =============================================================================

def example_derivative_rules():
    """Use rerum for symbolic differentiation."""
    if not RERUM_AVAILABLE:
        return

    # Differentiation rules (dd = d/d)
    deriv_rules = RuleEngine.from_dsl('''
        # Basic rules
        @dd-const[100]: (dd ?c:const ?v:var) => 0
        @dd-var-same[100]: (dd ?x:var ?x) => 1
        @dd-var-diff[90]: (dd ?y:var ?x:var) => 0

        # Linearity
        @dd-sum: (dd (+ ?f ?g) ?v) => (+ (dd :f :v) (dd :g :v))
        @dd-const-mult: (dd (* ?c:const ?f) ?v) => (* :c (dd :f :v))
        @dd-neg: (dd (* -1 ?f) ?v) => (* -1 (dd :f :v))

        # Product rule
        @dd-product: (dd (* ?f ?g) ?v) => (+ (* (dd :f :v) :g) (* :f (dd :g :v)))

        # Chain rule
        @dd-power: (dd (^ ?f ?n:const) ?v) => (* :n (* (^ :f (- :n 1)) (dd :f :v)))
        @dd-log: (dd (log ?f) ?v) => (/ (dd :f :v) :f)
        @dd-exp: (dd (exp ?f) ?v) => (* (exp :f) (dd :f :v))
    ''')

    # Simplification rules
    simplify_rules = RuleEngine.from_dsl('''
        @add-zero: (+ ?x 0) => :x
        @add-zero-r: (+ 0 ?x) => :x
        @mul-one: (* ?x 1) => :x
        @mul-zero: (* ?x 0) => 0
        @mul-zero-r: (* 0 ?x) => 0
    ''')

    # Chain the engines: differentiate, then simplify
    normalize = deriv_rules >> simplify_rules

    # Differentiate exponential log-likelihood: log(λ) - λt
    # d/dλ [log(λ) - λt] = 1/λ - t
    expr = E("(dd (+ (log lambda) (* -1 (* lambda t))) lambda)")
    result = normalize(expr)
    print(f"d/dλ [log(λ) - λt] = {result}")

    # Differentiate Weibull cumulative hazard: (t/θ)^k
    # d/dθ [(t/θ)^k] = k * (t/θ)^(k-1) * (-t/θ²)
    expr = E("(dd (^ (/ t theta) k) theta)")
    result, trace = deriv_rules(expr, trace=True)
    print(f"\nd/dθ [(t/θ)^k]:")
    print(f"  Raw: {result}")
    print(f"  Rules applied: {trace.rules_applied()}")

    return normalize


# =============================================================================
# Example 4: Load rules from file
# =============================================================================

def example_load_rules_file():
    """Load likelihood rules from a .rules file."""
    if not RERUM_AVAILABLE:
        return

    import os
    rules_path = os.path.join(os.path.dirname(__file__), "likelihood.rules")

    if os.path.exists(rules_path):
        engine = RuleEngine.from_file(rules_path)
        print(f"Loaded {len(engine)} rules from {rules_path}")
        print(f"Groups: {engine.groups()}")

        # List some rules
        print("\nSample rules:")
        for rule, meta in list(engine)[:5]:
            print(f"  {meta.name}: {meta.description or '(no description)'}")

        return engine
    else:
        print(f"Rules file not found: {rules_path}")


# =============================================================================
# Example 5: Interactive exploration with trace
# =============================================================================

def example_trace():
    """Show step-by-step rewriting with tracing."""
    if not RERUM_AVAILABLE:
        return

    engine = RuleEngine.from_dsl('''
        @add-zero: (+ ?x 0) => :x
        @mul-one: (* ?x 1) => :x
        @mul-zero: (* ?x 0) => 0
    ''')

    # Trace a multi-step simplification
    expr = E("(+ (* y 1) (* x 0))")
    result, trace = engine(expr, trace=True)

    print(f"Simplify: {expr}")
    print(f"Result: {result}")
    print(f"\nTrace:\n{trace}")
    print(f"\nSummary: {trace.summary()}")


# =============================================================================
# Example 6: Compare rerum vs symlik differentiation
# =============================================================================

def example_compare_differentiation():
    """Compare differentiation results between rerum and symlik."""
    print("\n" + "="*60)
    print("Comparing differentiation: rerum vs symlik")
    print("="*60)

    # Expression: x^2 + 2x + 1
    expr = ["+", ["+", ["^", "x", 2], ["*", 2, "x"]], 1]

    # Using symlik's diff
    symlik_result = diff(expr, "x")
    print(f"\nsymlik diff of (x² + 2x + 1):")
    print(f"  Raw: {symlik_result}")
    print(f"  Simplified: {simplify(symlik_result)}")

    if RERUM_AVAILABLE:
        # Using rerum
        deriv_engine = RuleEngine.from_dsl('''
            @dd-const: (dd ?c:const ?v:var) => 0
            @dd-var-same: (dd ?x:var ?x) => 1
            @dd-var-diff: (dd ?y:var ?x:var) => 0
            @dd-sum: (dd (+ ?f ?g) ?v) => (+ (dd :f :v) (dd :g :v))
            @dd-const-mult: (dd (* ?c:const ?f) ?v) => (* :c (dd :f :v))
            @dd-power: (dd (^ ?f ?n:const) ?v) => (* :n (* (^ :f (- :n 1)) (dd :f :v)))
        ''')

        simplify_engine = RuleEngine.from_dsl('''
            @add-zero: (+ ?x 0) => :x
            @mul-one: (* ?x 1) => :x
            @mul-zero: (* ?x 0) => 0
            @pow-zero: (^ ?x 0) => 1
        ''', fold_funcs=ARITHMETIC_PRELUDE)

        # Wrap expr in dd operator
        dd_expr = ["dd", expr, "x"]
        rerum_result = (deriv_engine >> simplify_engine)(dd_expr)
        print(f"\nrerum diff of (x² + 2x + 1):")
        print(f"  Result: {rerum_result}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RERUM + SYMLIK Integration Examples")
    print("="*60)

    if not RERUM_AVAILABLE:
        print("\nTo run these examples, install rerum:")
        print("  pip install rerum")
        print("\nShowing symlik-only example:")
        example_compare_differentiation()
    else:
        print("\n--- Example 1: Simplification ---")
        example_simplify_with_rerum()

        print("\n--- Example 2: Distribution DSL ---")
        example_distribution_dsl()

        print("\n--- Example 3: Derivative Rules ---")
        example_derivative_rules()

        print("\n--- Example 4: Load Rules File ---")
        example_load_rules_file()

        print("\n--- Example 5: Tracing ---")
        example_trace()

        print("\n--- Example 6: Compare Differentiation ---")
        example_compare_differentiation()

    print("\n" + "="*60)
    print("Done!")
