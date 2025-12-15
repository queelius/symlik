"""Tests for symlik.calculus module."""

import math
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from symlik.calculus import (
    simplify,
    diff,
    integrate,
    gradient,
    hessian,
    jacobian,
    laplacian,
    diff_at,
    gradient_at,
    hessian_at,
)


class TestSimplify:
    """Test algebraic simplification."""

    def test_add_zero_right(self):
        assert simplify(["+", "x", 0]) == "x"

    def test_add_zero_left(self):
        assert simplify(["+", 0, "x"]) == "x"

    def test_mul_one_right(self):
        assert simplify(["*", "x", 1]) == "x"

    def test_mul_one_left(self):
        assert simplify(["*", 1, "x"]) == "x"

    def test_mul_zero(self):
        assert simplify(["*", "x", 0]) == 0

    def test_pow_zero(self):
        assert simplify(["^", "x", 0]) == 1

    def test_pow_one(self):
        assert simplify(["^", "x", 1]) == "x"

    def test_sub_self(self):
        assert simplify(["-", "x", "x"]) == 0

    def test_div_self(self):
        assert simplify(["/", "x", "x"]) == 1

    def test_double_negation(self):
        assert simplify(["-", ["-", "x"]]) == "x"

    def test_nested_simplification(self):
        # (x * 1) + 0 = x
        result = simplify(["+", ["*", "x", 1], 0])
        assert result == "x"


class TestDiff:
    """Test symbolic differentiation."""

    def test_diff_constant(self):
        assert diff(5, "x") == 0

    def test_diff_variable(self):
        assert diff("x", "x") == 1

    def test_diff_other_variable(self):
        assert diff("y", "x") == 0

    def test_diff_power(self):
        # d/dx(x^2) = 2x - verify numerically
        result = diff(["^", "x", 2], "x")
        from symlik import evaluate
        for x_val in [0, 1, 2, -1, 0.5]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(2 * x_val)

    def test_diff_power_cube(self):
        # d/dx(x^3) = 3x^2 - verify numerically
        result = diff(["^", "x", 3], "x")
        from symlik import evaluate
        for x_val in [0, 1, 2, -1, 0.5]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(3 * x_val**2)

    def test_diff_sum(self):
        # d/dx(x + y) = 1 + 0 = 1
        result = diff(["+", "x", "y"], "x")
        from symlik import evaluate
        assert evaluate(result, {"x": 5, "y": 10}) == pytest.approx(1.0)

    def test_diff_product(self):
        # d/dx(x * y) = y (since dy/dx = 0)
        result = diff(["*", "x", "y"], "x")
        from symlik import evaluate
        for y_val in [1, 2, 5, -3]:
            assert evaluate(result, {"x": 10, "y": y_val}) == pytest.approx(y_val)

    def test_diff_exp(self):
        # d/dx(e^x) = e^x - verify numerically
        result = diff(["exp", "x"], "x")
        from symlik import evaluate
        for x_val in [0, 1, -1, 0.5]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(math.exp(x_val))

    def test_diff_log(self):
        # d/dx(ln(x)) = 1/x - verify numerically
        result = diff(["log", "x"], "x")
        from symlik import evaluate
        for x_val in [0.5, 1, 2, 5]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(1 / x_val)

    def test_diff_sin(self):
        # d/dx(sin(x)) = cos(x) - verify numerically
        result = diff(["sin", "x"], "x")
        from symlik import evaluate
        for x_val in [0, math.pi/4, math.pi/2, math.pi]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(math.cos(x_val))

    def test_diff_cos(self):
        # d/dx(cos(x)) = -sin(x)
        result = diff(["cos", "x"], "x")
        # Result could be ["*", -1, ["sin", "x"]] or ["-", ["sin", "x"]]
        # Just verify it simplifies correctly when evaluated
        from symlik import evaluate
        assert evaluate(result, {"x": 0}) == pytest.approx(0.0)
        assert evaluate(result, {"x": math.pi/2}) == pytest.approx(-1.0)

    def test_diff_chain_rule(self):
        # d/dx(sin(x^2)) = cos(x^2) * 2x
        result = diff(["sin", ["^", "x", 2]], "x")
        from symlik import evaluate
        # At x=1: cos(1) * 2 ≈ 1.08
        assert evaluate(result, {"x": 1.0}) == pytest.approx(2 * math.cos(1), rel=1e-5)


class TestIntegrate:
    """Test symbolic integration."""

    def test_integrate_constant(self):
        # ∫5 dx = 5x - verify numerically
        result = integrate(5, "x")
        from symlik import evaluate
        for x_val in [0, 1, 2, -1]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(5 * x_val)

    def test_integrate_x(self):
        # ∫x dx = x^2/2 - verify numerically
        result = integrate("x", "x")
        from symlik import evaluate
        for x_val in [0, 1, 2, 3]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(x_val**2 / 2)

    def test_integrate_x_squared(self):
        # ∫x^2 dx = x^3/3 - verify numerically
        result = integrate(["^", "x", 2], "x")
        from symlik import evaluate
        for x_val in [0, 1, 2, 3]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(x_val**3 / 3)

    def test_integrate_exp(self):
        # ∫e^x dx = e^x - verify numerically
        result = integrate(["exp", "x"], "x")
        from symlik import evaluate
        for x_val in [0, 1, -1, 0.5]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(math.exp(x_val))

    def test_integrate_sin(self):
        # ∫sin(x) dx = -cos(x) - verify numerically
        result = integrate(["sin", "x"], "x")
        from symlik import evaluate
        for x_val in [0, math.pi/2, math.pi]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(-math.cos(x_val))

    def test_integrate_cos(self):
        # ∫cos(x) dx = sin(x) - verify numerically
        result = integrate(["cos", "x"], "x")
        from symlik import evaluate
        for x_val in [0, math.pi/4, math.pi/2]:
            assert evaluate(result, {"x": x_val}) == pytest.approx(math.sin(x_val))


class TestGradient:
    """Test gradient computation."""

    def test_gradient_single_var(self):
        # ∇(x^2) = [2x] - verify numerically
        result = gradient(["^", "x", 2], ["x"])
        from symlik import evaluate
        for x_val in [0, 1, 2, -1]:
            assert evaluate(result[0], {"x": x_val}) == pytest.approx(2 * x_val)

    def test_gradient_two_vars(self):
        # ∇(x^2 + y^2) = [2x, 2y] - verify numerically
        expr = ["+", ["^", "x", 2], ["^", "y", 2]]
        result = gradient(expr, ["x", "y"])
        from symlik import evaluate
        for x_val, y_val in [(1, 2), (3, 4), (0, 0)]:
            assert evaluate(result[0], {"x": x_val, "y": y_val}) == pytest.approx(2 * x_val)
            assert evaluate(result[1], {"x": x_val, "y": y_val}) == pytest.approx(2 * y_val)

    def test_gradient_mixed_terms(self):
        # ∇(x*y) = [y, x] - verify numerically
        result = gradient(["*", "x", "y"], ["x", "y"])
        from symlik import evaluate
        for x_val, y_val in [(2, 3), (5, 7), (1, 1)]:
            assert evaluate(result[0], {"x": x_val, "y": y_val}) == pytest.approx(y_val)
            assert evaluate(result[1], {"x": x_val, "y": y_val}) == pytest.approx(x_val)

    def test_gradient_complex(self):
        # ∇(x^2 + xy + y^2) = [2x + y, x + 2y] - verify numerically
        expr = ["+", ["+", ["^", "x", 2], ["*", "x", "y"]], ["^", "y", 2]]
        result = gradient(expr, ["x", "y"])
        from symlik import evaluate
        for x_val, y_val in [(1, 2), (3, 4), (0, 5)]:
            assert evaluate(result[0], {"x": x_val, "y": y_val}) == pytest.approx(2*x_val + y_val)
            assert evaluate(result[1], {"x": x_val, "y": y_val}) == pytest.approx(x_val + 2*y_val)


class TestHessian:
    """Test Hessian computation."""

    def test_hessian_quadratic(self):
        # H(x^2 + y^2) = [[2, 0], [0, 2]]
        expr = ["+", ["^", "x", 2], ["^", "y", 2]]
        result = hessian(expr, ["x", "y"])
        assert result == [[2, 0], [0, 2]]

    def test_hessian_mixed(self):
        # H(xy) = [[0, 1], [1, 0]]
        result = hessian(["*", "x", "y"], ["x", "y"])
        assert result == [[0, 1], [1, 0]]

    def test_hessian_complex(self):
        # H(x^2 + 2xy + 3y^2) = [[2, 2], [2, 6]]
        expr = ["+", ["+", ["^", "x", 2], ["*", 2, ["*", "x", "y"]]], ["*", 3, ["^", "y", 2]]]
        result = hessian(expr, ["x", "y"])
        from symlik import evaluate
        env = {"x": 0, "y": 0}  # Hessian is constant for quadratic
        H = [[evaluate(result[i][j], env) for j in range(2)] for i in range(2)]
        assert H == [[2, 2], [2, 6]]


class TestJacobian:
    """Test Jacobian computation."""

    def test_jacobian_identity(self):
        # J([x, y]) = [[1, 0], [0, 1]]
        result = jacobian(["x", "y"], ["x", "y"])
        assert result == [[1, 0], [0, 1]]

    def test_jacobian_linear(self):
        # J([x + y, x - y]) = [[1, 1], [1, -1]]
        exprs = [["+", "x", "y"], ["-", "x", "y"]]
        result = jacobian(exprs, ["x", "y"])
        from symlik import evaluate
        env = {"x": 0, "y": 0}
        J = [[evaluate(result[i][j], env) for j in range(2)] for i in range(2)]
        assert J[0] == [1, 1]
        # Second row: d(x-y)/dx = 1, d(x-y)/dy = -1
        assert J[1][0] == 1
        assert J[1][1] == pytest.approx(-1)

    def test_jacobian_nonlinear(self):
        # J([x*y, x^2]) = [[y, x], [2x, 0]]
        exprs = [["*", "x", "y"], ["^", "x", 2]]
        result = jacobian(exprs, ["x", "y"])
        from symlik import evaluate
        env = {"x": 3, "y": 4}
        J = [[evaluate(result[i][j], env) for j in range(2)] for i in range(2)]
        assert J == [[4, 3], [6, 0]]


class TestLaplacian:
    """Test Laplacian computation."""

    def test_laplacian_quadratic(self):
        # ∇²(x^2 + y^2) = 2 + 2 = 4
        expr = ["+", ["^", "x", 2], ["^", "y", 2]]
        result = laplacian(expr, ["x", "y"])
        assert result == 4

    def test_laplacian_single_var(self):
        # ∇²(x^2) = 2
        result = laplacian(["^", "x", 2], ["x"])
        assert result == 2

    def test_laplacian_three_vars(self):
        # ∇²(x^2 + y^2 + z^2) = 2 + 2 + 2 = 6
        expr = ["+", ["+", ["^", "x", 2], ["^", "y", 2]], ["^", "z", 2]]
        result = laplacian(expr, ["x", "y", "z"])
        from symlik import evaluate
        assert evaluate(result, {"x": 0, "y": 0, "z": 0}) == 6

    def test_laplacian_empty_vars(self):
        # Edge case: empty vars list should return 0
        result = laplacian(["^", "x", 2], [])
        assert result == 0


class TestNumericalEvaluation:
    """Test numerical evaluation helpers."""

    def test_diff_at(self):
        # d/dx(x^2) at x=3 = 6
        result = diff_at(["^", "x", 2], "x", {"x": 3.0})
        assert result == pytest.approx(6.0)

    def test_gradient_at(self):
        # ∇(x^2 + y^2) at (3, 4) = [6, 8]
        expr = ["+", ["^", "x", 2], ["^", "y", 2]]
        result = gradient_at(expr, ["x", "y"], {"x": 3.0, "y": 4.0})
        np.testing.assert_array_almost_equal(result, [6.0, 8.0])

    def test_hessian_at(self):
        # H(x^2 + y^2) at any point = [[2, 0], [0, 2]]
        expr = ["+", ["^", "x", 2], ["^", "y", 2]]
        result = hessian_at(expr, ["x", "y"], {"x": 1.0, "y": 1.0})
        np.testing.assert_array_almost_equal(result, [[2, 0], [0, 2]])


class TestEdgeCases:
    """Test edge cases and special situations."""

    def test_diff_constant_expression(self):
        # d/dx(5 + 3) = 0
        result = diff(["+", 5, 3], "x")
        assert result == 0

    def test_gradient_empty_vars(self):
        result = gradient(["^", "x", 2], [])
        assert result == []

    def test_hessian_single_var(self):
        # H(x^3) = [[6x]]
        result = hessian(["^", "x", 3], ["x"])
        from symlik import evaluate
        assert evaluate(result[0][0], {"x": 2}) == pytest.approx(12)

    def test_nested_derivatives(self):
        # d/dx(d/dx(x^3)) = d/dx(3x^2) = 6x
        first = diff(["^", "x", 3], "x")  # 3x^2
        second = diff(first, "x")  # 6x
        from symlik import evaluate
        assert evaluate(second, {"x": 5}) == pytest.approx(30)


class TestPropertyBasedDifferentiation:
    """Property-based tests for mathematical correctness."""

    @given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_diff_constant_is_zero(self, c):
        """d/dx(c) = 0 for any constant."""
        result = diff(c, "x")
        from symlik import evaluate
        assert evaluate(result, {"x": 1}) == pytest.approx(0)

    @given(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_diff_x_is_one(self, x_val):
        """d/dx(x) = 1 everywhere."""
        result = diff("x", "x")
        from symlik import evaluate
        assert evaluate(result, {"x": x_val}) == pytest.approx(1)

    @given(st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_exp_is_own_derivative(self, x_val):
        """d/dx(e^x) = e^x."""
        expr = ["exp", "x"]
        deriv = diff(expr, "x")
        from symlik import evaluate
        expected = math.exp(x_val)
        assert evaluate(deriv, {"x": x_val}) == pytest.approx(expected, rel=1e-5)

    @given(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_log_derivative(self, x_val):
        """d/dx(ln(x)) = 1/x."""
        deriv = diff(["log", "x"], "x")
        from symlik import evaluate
        assert evaluate(deriv, {"x": x_val}) == pytest.approx(1/x_val, rel=1e-5)

    @given(st.integers(min_value=1, max_value=5),
           st.floats(min_value=0.5, max_value=3, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_power_rule(self, n, x_val):
        """d/dx(x^n) = n*x^(n-1)."""
        deriv = diff(["^", "x", n], "x")
        from symlik import evaluate
        expected = n * (x_val ** (n - 1))
        assert evaluate(deriv, {"x": x_val}) == pytest.approx(expected, rel=1e-5)

    @given(st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_sin_cos_derivative_relationship(self, x_val):
        """d/dx(sin(x)) = cos(x) and d/dx(cos(x)) = -sin(x)."""
        from symlik import evaluate
        sin_deriv = diff(["sin", "x"], "x")
        cos_deriv = diff(["cos", "x"], "x")
        assert evaluate(sin_deriv, {"x": x_val}) == pytest.approx(math.cos(x_val), rel=1e-5)
        assert evaluate(cos_deriv, {"x": x_val}) == pytest.approx(-math.sin(x_val), rel=1e-5)

    @given(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
           st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=1000)
    def test_hessian_symmetry(self, x_val, y_val):
        """Hessian matrix should be symmetric: H[i,j] = H[j,i]."""
        # f(x,y) = x^2*y + x*y^2 (mixed terms)
        expr = ["+", ["*", ["^", "x", 2], "y"], ["*", "x", ["^", "y", 2]]]
        hess = hessian(expr, ["x", "y"])
        from symlik import evaluate
        env = {"x": x_val, "y": y_val}
        h01 = evaluate(hess[0][1], env)
        h10 = evaluate(hess[1][0], env)
        assert h01 == pytest.approx(h10, rel=1e-5)
