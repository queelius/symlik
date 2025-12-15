"""Tests for symlik.evaluate module."""

import math
import pytest
from symlik.evaluate import evaluate, STANDARD_OPS, expr_to_function


class TestConstants:
    """Test evaluation of constants."""

    def test_integer(self):
        assert evaluate(42, {}) == 42.0

    def test_float(self):
        assert evaluate(3.14, {}) == 3.14

    def test_negative(self):
        assert evaluate(-5, {}) == -5.0

    def test_zero(self):
        assert evaluate(0, {}) == 0.0


class TestVariables:
    """Test evaluation of variables."""

    def test_simple_variable(self):
        assert evaluate("x", {"x": 5}) == 5.0

    def test_multiple_variables(self):
        assert evaluate("y", {"x": 1, "y": 2, "z": 3}) == 2.0

    def test_builtin_e(self):
        assert evaluate("e", {}) == pytest.approx(math.e)

    def test_builtin_pi(self):
        assert evaluate("pi", {}) == pytest.approx(math.pi)

    def test_unbound_variable_raises(self):
        with pytest.raises(ValueError, match="Unbound variable"):
            evaluate("x", {})


class TestArithmetic:
    """Test arithmetic operators."""

    def test_addition(self):
        assert evaluate(["+", 2, 3], {}) == 5.0

    def test_addition_multiple(self):
        assert evaluate(["+", 1, 2, 3, 4], {}) == 10.0

    def test_subtraction(self):
        assert evaluate(["-", 5, 3], {}) == 2.0

    def test_negation(self):
        assert evaluate(["-", 5], {}) == -5.0

    def test_multiplication(self):
        assert evaluate(["*", 4, 5], {}) == 20.0

    def test_multiplication_multiple(self):
        assert evaluate(["*", 2, 3, 4], {}) == 24.0

    def test_division(self):
        assert evaluate(["/", 10, 2], {}) == 5.0

    def test_division_by_zero(self):
        assert evaluate(["/", 1, 0], {}) == float("inf")

    def test_power(self):
        assert evaluate(["^", 2, 3], {}) == 8.0

    def test_power_fractional(self):
        assert evaluate(["^", 4, 0.5], {}) == 2.0


class TestTrigonometric:
    """Test trigonometric functions."""

    def test_sin_zero(self):
        assert evaluate(["sin", 0], {}) == pytest.approx(0.0)

    def test_sin_pi_half(self):
        assert evaluate(["sin", ["/", "pi", 2]], {}) == pytest.approx(1.0)

    def test_cos_zero(self):
        assert evaluate(["cos", 0], {}) == pytest.approx(1.0)

    def test_cos_pi(self):
        assert evaluate(["cos", "pi"], {}) == pytest.approx(-1.0)

    def test_tan_zero(self):
        assert evaluate(["tan", 0], {}) == pytest.approx(0.0)

    def test_arcsin(self):
        assert evaluate(["arcsin", 1], {}) == pytest.approx(math.pi / 2)

    def test_arccos(self):
        assert evaluate(["arccos", 1], {}) == pytest.approx(0.0)

    def test_arctan(self):
        assert evaluate(["arctan", 1], {}) == pytest.approx(math.pi / 4)


class TestHyperbolic:
    """Test hyperbolic functions."""

    def test_sinh_zero(self):
        assert evaluate(["sinh", 0], {}) == pytest.approx(0.0)

    def test_cosh_zero(self):
        assert evaluate(["cosh", 0], {}) == pytest.approx(1.0)

    def test_tanh_zero(self):
        assert evaluate(["tanh", 0], {}) == pytest.approx(0.0)


class TestExpLog:
    """Test exponential and logarithmic functions."""

    def test_exp_zero(self):
        assert evaluate(["exp", 0], {}) == pytest.approx(1.0)

    def test_exp_one(self):
        assert evaluate(["exp", 1], {}) == pytest.approx(math.e)

    def test_log_one(self):
        assert evaluate(["log", 1], {}) == pytest.approx(0.0)

    def test_log_e(self):
        assert evaluate(["log", "e"], {}) == pytest.approx(1.0)

    def test_ln_alias(self):
        assert evaluate(["ln", "e"], {}) == pytest.approx(1.0)

    def test_log_negative(self):
        assert evaluate(["log", -1], {}) == float("-inf")

    def test_sqrt(self):
        assert evaluate(["sqrt", 4], {}) == pytest.approx(2.0)

    def test_sqrt_negative(self):
        assert math.isnan(evaluate(["sqrt", -1], {}))


class TestOther:
    """Test other functions."""

    def test_abs_positive(self):
        assert evaluate(["abs", 5], {}) == 5.0

    def test_abs_negative(self):
        assert evaluate(["abs", -5], {}) == 5.0

    def test_lgamma(self):
        # lgamma(5) = log(4!) = log(24)
        assert evaluate(["lgamma", 5], {}) == pytest.approx(math.log(24))

    def test_gamma(self):
        # gamma(5) = 4! = 24
        assert evaluate(["gamma", 5], {}) == pytest.approx(24.0)

    def test_erf_zero(self):
        assert evaluate(["erf", 0], {}) == pytest.approx(0.0)


class TestSummation:
    """Test sum special form."""

    def test_sum_simple(self):
        # sum_{i=1}^{3} i = 1 + 2 + 3 = 6
        assert evaluate(["sum", "i", 3, "i"], {}) == 6.0

    def test_sum_with_data(self):
        # sum_{i=1}^{n} x[i]
        expr = ["sum", "i", ["len", "x"], ["@", "x", "i"]]
        assert evaluate(expr, {"x": [10, 20, 30]}) == 60.0

    def test_sum_squares(self):
        # sum_{i=1}^{3} i^2 = 1 + 4 + 9 = 14
        assert evaluate(["sum", "i", 3, ["^", "i", 2]], {}) == 14.0

    def test_sum_empty(self):
        assert evaluate(["sum", "i", 0, "i"], {}) == 0.0

    def test_sum_with_external_var(self):
        # sum_{i=1}^{3} (i * x) = x * (1+2+3) = 6x
        assert evaluate(["sum", "i", 3, ["*", "i", "x"]], {"x": 2}) == 12.0


class TestProduct:
    """Test prod special form."""

    def test_prod_simple(self):
        # prod_{i=1}^{4} i = 1 * 2 * 3 * 4 = 24
        assert evaluate(["prod", "i", 4, "i"], {}) == 24.0

    def test_prod_empty(self):
        assert evaluate(["prod", "i", 0, "i"], {}) == 1.0

    def test_prod_with_data(self):
        expr = ["prod", "i", ["len", "x"], ["@", "x", "i"]]
        assert evaluate(expr, {"x": [2, 3, 4]}) == 24.0


class TestIndexing:
    """Test @ and index special forms."""

    def test_at_indexing(self):
        assert evaluate(["@", "x", 1], {"x": [10, 20, 30]}) == 10.0

    def test_at_indexing_middle(self):
        assert evaluate(["@", "x", 2], {"x": [10, 20, 30]}) == 20.0

    def test_at_indexing_last(self):
        assert evaluate(["@", "x", 3], {"x": [10, 20, 30]}) == 30.0

    def test_index_function(self):
        assert evaluate(["index", "x", 2], {"x": [10, 20, 30]}) == 20.0


class TestLength:
    """Test len special form."""

    def test_len_simple(self):
        assert evaluate(["len", "x"], {"x": [1, 2, 3, 4, 5]}) == 5.0

    def test_len_empty(self):
        assert evaluate(["len", "x"], {"x": []}) == 0.0


class TestTotal:
    """Test total special form."""

    def test_total_simple(self):
        assert evaluate(["total", "x"], {"x": [1, 2, 3, 4, 5]}) == 15.0

    def test_total_empty(self):
        assert evaluate(["total", "x"], {"x": []}) == 0.0


class TestDerivative:
    """Test dd special form (numerical differentiation)."""

    def test_dd_linear(self):
        # d/dx(2x) = 2
        expr = ["dd", ["*", 2, "x"], "x"]
        assert evaluate(expr, {"x": 5.0}) == pytest.approx(2.0, rel=1e-5)

    def test_dd_quadratic(self):
        # d/dx(x^2) = 2x, at x=3 => 6
        expr = ["dd", ["^", "x", 2], "x"]
        assert evaluate(expr, {"x": 3.0}) == pytest.approx(6.0, rel=1e-5)

    def test_dd_exp(self):
        # d/dx(e^x) = e^x, at x=1 => e
        expr = ["dd", ["exp", "x"], "x"]
        assert evaluate(expr, {"x": 1.0}) == pytest.approx(math.e, rel=1e-5)


class TestIntegral:
    """Test int special form (numerical integration)."""

    def test_int_definite(self):
        # integral of x from 0 to 2 = x^2/2 |_0^2 = 2
        expr = ["int", "x", "x", [0, 2]]
        assert evaluate(expr, {}) == pytest.approx(2.0, rel=1e-3)

    def test_int_quadratic(self):
        # integral of x^2 from 0 to 3 = x^3/3 |_0^3 = 9
        expr = ["int", ["^", "x", 2], "x", [0, 3]]
        assert evaluate(expr, {}) == pytest.approx(9.0, rel=1e-3)


class TestConditional:
    """Test if special form."""

    def test_if_true(self):
        assert evaluate(["if", 1, 10, 20], {}) == 10.0

    def test_if_false(self):
        assert evaluate(["if", 0, 10, 20], {}) == 20.0

    def test_if_with_comparison(self):
        # Note: this requires a comparison operator
        # For now, just test with truthy/falsy values
        assert evaluate(["if", 5, "x", "y"], {"x": 100, "y": 200}) == 100.0


class TestNestedExpressions:
    """Test nested and complex expressions."""

    def test_nested_arithmetic(self):
        # (2 + 3) * (4 - 1) = 5 * 3 = 15
        expr = ["*", ["+", 2, 3], ["-", 4, 1]]
        assert evaluate(expr, {}) == 15.0

    def test_complex_expression(self):
        # sin(x)^2 + cos(x)^2 = 1
        expr = ["+", ["^", ["sin", "x"], 2], ["^", ["cos", "x"], 2]]
        assert evaluate(expr, {"x": 1.5}) == pytest.approx(1.0)

    def test_log_likelihood_style(self):
        # sum of log(lambda) - lambda * x[i]
        expr = ["sum", "i", ["len", "x"],
                ["+", ["log", "lambda"],
                 ["*", -1, ["*", "lambda", ["@", "x", "i"]]]]]
        result = evaluate(expr, {"x": [1, 2, 3], "lambda": 0.5})
        expected = 3 * math.log(0.5) - 0.5 * (1 + 2 + 3)
        assert result == pytest.approx(expected)


class TestExprToFunction:
    """Test expr_to_function helper."""

    def test_simple_function(self):
        f = expr_to_function(["^", "x", 2], "x")
        assert f(3) == 9.0
        assert f(4) == 16.0

    def test_function_with_ops(self):
        f = expr_to_function(["sin", "x"], "x")
        assert f(0) == pytest.approx(0.0)
        assert f(math.pi / 2) == pytest.approx(1.0)


class TestCustomOps:
    """Test custom operator prelude."""

    def test_custom_op(self):
        custom_ops = {**STANDARD_OPS, "double": lambda args: 2 * args[0]}
        assert evaluate(["double", 5], {}, ops=custom_ops) == 10.0

    def test_override_op(self):
        custom_ops = {**STANDARD_OPS, "+": lambda args: sum(args) + 100}
        assert evaluate(["+", 1, 2], {}, ops=custom_ops) == 103.0


class TestIndefiniteIntegral:
    """Test indefinite integral paths."""

    def test_int_indefinite_with_var_in_env(self):
        # ['int', inner_expr, var] without bounds - uses env[var] as upper bound
        expr = ["int", "x", "x"]
        result = evaluate(expr, {"x": 2.0})
        # Integral of x from 0 to 2 = x^2/2 |_0^2 = 2
        assert result == pytest.approx(2.0, rel=1e-2)

    def test_int_indefinite_without_var_raises(self):
        expr = ["int", "x", "x"]
        with pytest.raises(ValueError, match="Cannot evaluate indefinite integral"):
            evaluate(expr, {})


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_integration_equal_bounds(self):
        # a == b returns 0
        expr = ["int", "x", "x", [5, 5]]
        assert evaluate(expr, {}) == 0.0

    def test_operator_exception_returns_nan(self):
        # Catch exceptions in operator evaluation and return nan
        custom_ops = {**STANDARD_OPS, "bad": lambda args: 1/0}
        result = evaluate(["bad", 1], {}, ops=custom_ops)
        assert math.isnan(result)


class TestExprToMultivarFunction:
    """Test expr_to_multivar_function helper."""

    def test_multivar_function_basic(self):
        from symlik.evaluate import expr_to_multivar_function
        f = expr_to_multivar_function(["+", ["^", "x", 2], ["^", "y", 2]], ["x", "y"])
        assert f([3, 4]) == 25.0

    def test_multivar_function_product(self):
        from symlik.evaluate import expr_to_multivar_function
        f = expr_to_multivar_function(["*", "x", "y"], ["x", "y"])
        assert f([2, 3]) == 6.0

    def test_multivar_function_three_vars(self):
        from symlik.evaluate import expr_to_multivar_function
        f = expr_to_multivar_function(["+", ["+", "x", "y"], "z"], ["x", "y", "z"])
        assert f([1, 2, 3]) == 6.0


class TestAdditionalFunctions:
    """Test additional mathematical functions."""

    def test_erfc(self):
        assert evaluate(["erfc", 0], {}) == pytest.approx(1.0)
        assert evaluate(["erfc", 100], {}) == pytest.approx(0.0, abs=1e-10)


class TestErrorHandling:
    """Test error handling."""

    def test_unknown_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            evaluate(["unknown_op", 1, 2], {})

    def test_invalid_expression(self):
        with pytest.raises(ValueError, match="Invalid expression"):
            evaluate([], {})
