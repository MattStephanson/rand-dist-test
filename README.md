# Statistics for libc++ [rand.dist] tests

libc++ contains a number of [unit tests](https://github.com/llvm/llvm-project/tree/main/libcxx/test/std/numerics/rand/rand.dist)
for the [random number distributions](https://eel.is/c++draft/rand.dist) found in the C++ Standard Library. They mostly
take two forms, either comparing the sample mean, variance, skew, and kurtosis to the analytic values, or applying the
Kolmogorov-Smirnov test. These test statistics are themselves random variables, and in most cases their distributions
are not known analytically. Therefore, it is not obvious what the critical values should be, even if a significance
level were agreed upon.

This program approximates the critical values by sampling each distribution several times with different seeds for the
underlying PRNG. Obviously this is circular, but these distributions have been part of the C++ Standard for over a
decade, so it seems unlikely that there are obvious errors in the major implementations. The libc++ tests specify the
maximum absolute difference from the expected value (mostly relative, sometimes absolute), so the critical value is the
half-width of the smallest symmetric interval around the median that contains $1-\alpha$ of the sample means (or
variances, skews, etc.). I've chosen $\alpha = 0.1$, but it can be changed in the code. A smaller value, however, will
necessitate more samples. For `piecewise_constant_distribution`, the maximum absolute and relative values among all the
intervals is given.
