# Distributions Reference

Pre-built likelihood models for common statistical distributions.

All functions return a `LikelihoodModel` instance ready for fitting.

## exponential

```python
from symlik.distributions import exponential

model = exponential(data_var='x', param='lambda')
```

**Distribution:** \(f(x; \lambda) = \lambda e^{-\lambda x}\)

**Parameters:**

- `data_var` (str): Name of data variable. Default: `'x'`
- `param` (str): Name of rate parameter. Default: `'lambda'`

**MLE:** \(\hat\lambda = 1/\bar{x}\)

---

## normal

```python
from symlik.distributions import normal

model = normal(data_var='x', mean='mu', var='sigma2')
```

**Distribution:** \(f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/(2\sigma^2)}\)

**Parameters:**

- `data_var` (str): Name of data variable. Default: `'x'`
- `mean` (str): Name of mean parameter. Default: `'mu'`
- `var` (str): Name of variance parameter. Default: `'sigma2'`

**MLE:** \(\hat\mu = \bar{x}\), \(\hat\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2\)

**Note:** Use bounds to keep variance positive: `bounds={'sigma2': (0.01, None)}`

---

## normal_mean

```python
from symlik.distributions import normal_mean

model = normal_mean(data_var='x', mean='mu', known_var=1.0)
```

Normal distribution with **known** variance. Estimates mean only.

**Parameters:**

- `data_var` (str): Name of data variable. Default: `'x'`
- `mean` (str): Name of mean parameter. Default: `'mu'`
- `known_var` (float): Known variance value. Default: `1.0`

**MLE:** \(\hat\mu = \bar{x}\)

---

## poisson

```python
from symlik.distributions import poisson

model = poisson(data_var='x', param='lambda')
```

**Distribution:** \(P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}\)

**Parameters:**

- `data_var` (str): Name of data variable. Default: `'x'`
- `param` (str): Name of rate parameter. Default: `'lambda'`

**MLE:** \(\hat\lambda = \bar{x}\)

**Note:** The log-likelihood ignores the \(\log(k!)\) term (constant w.r.t. \(\lambda\)).

---

## bernoulli

```python
from symlik.distributions import bernoulli

model = bernoulli(data_var='x', param='p')
```

**Distribution:** \(P(X = 1) = p\), \(P(X = 0) = 1 - p\)

**Parameters:**

- `data_var` (str): Name of data variable (0/1 values). Default: `'x'`
- `param` (str): Name of probability parameter. Default: `'p'`

**MLE:** \(\hat{p} = \bar{x}\) (proportion of 1s)

**Note:** Use bounds to avoid log(0): `bounds={'p': (0.01, 0.99)}`

---

## binomial

```python
from symlik.distributions import binomial

model = binomial(successes='k', trials='n', param='p')
```

Binomial distribution for a **single observation** of k successes in n trials.

**Distribution:** \(P(K = k) = \binom{n}{k} p^k (1-p)^{n-k}\)

**Parameters:**

- `successes` (str): Name of success count variable. Default: `'k'`
- `trials` (str): Name of trials count variable. Default: `'n'`
- `param` (str): Name of probability parameter. Default: `'p'`

**MLE:** \(\hat{p} = k/n\)

---

## gamma

```python
from symlik.distributions import gamma

model = gamma(data_var='x', shape='alpha', rate='beta')
```

**Distribution:** \(f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}\)

Uses the **rate parameterization** (not scale).

**Parameters:**

- `data_var` (str): Name of data variable. Default: `'x'`
- `shape` (str): Name of shape parameter \(\alpha\). Default: `'alpha'`
- `rate` (str): Name of rate parameter \(\beta\). Default: `'beta'`

**Note:** No closed-form MLE. Use numerical optimization with good starting values.

---

## weibull

```python
from symlik.distributions import weibull

model = weibull(data_var='x', shape='k', scale='lambda')
```

**Distribution:** \(f(x; k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}\)

**Parameters:**

- `data_var` (str): Name of data variable. Default: `'x'`
- `shape` (str): Name of shape parameter \(k\). Default: `'k'`
- `scale` (str): Name of scale parameter \(\lambda\). Default: `'lambda'`

---

## beta

```python
from symlik.distributions import beta

model = beta(data_var='x', alpha='alpha', beta_param='beta')
```

**Distribution:** \(f(x; \alpha, \beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}\)

For data in the interval \((0, 1)\).

**Parameters:**

- `data_var` (str): Name of data variable. Default: `'x'`
- `alpha` (str): First shape parameter. Default: `'alpha'`
- `beta_param` (str): Second shape parameter. Default: `'beta'`

**Note:** Ensure data is strictly in \((0, 1)\), not including endpoints.
