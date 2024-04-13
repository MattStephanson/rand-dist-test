// Copyright 2024 Matt Stephanson.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The function `test_once` and `test_samp_once` are derived from the
// LLVM/libc++ test suite, specifically:
//
//      libcxx/test/std/numerics/rand/rand.dist/rand.dist.bern/rand.dist.bern.bernoulli/eval.pass.cpp
//      libcxx/test/std/numerics/rand/rand.dist/rand.dist.samp/rand.dist.samp.pconst/eval.pass.cpp
//
//      //===----------------------------------------------------------------------===//
//      //
//      // Part of the LLVM Project, under the Apache License v2.0 with LLVM
//      Exceptions.
//      // See https://llvm.org/LICENSE.txt for license information.
//      // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//      //
//      //===----------------------------------------------------------------------===//
//
//      Copyright (c) 2009-2014 by the contributors listed in CREDITS-LIBCXX.TXT
//
//      Permission is hereby granted, free of charge, to any person obtaining a
//      copy of this software and associated documentation files (the
//      "Software"), to deal in the Software without restriction, including
//      without limitation the rights to use, copy, modify, merge, publish,
//      distribute, sublicense, and/or sell copies of the Software, and to
//      permit persons to whom the Software is furnished to do so, subject to
//      the following conditions:
//
//      The above copyright notice and this permission notice shall be included
//      in all copies or substantial portions of the Software.
//
//      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//      OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//      MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//      IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//      CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//      TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//      SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

using std::cout;
using std::format;

template <class T> T sqr(T x) { return x * x; }

template <class Dist, class Gen, class... ParamsT>
void test_once(const unsigned seed, const int N, std::vector<double> &mmean,
               std::vector<double> &vvar, std::vector<double> &sskew,
               std::vector<double> &kkurtosis, ParamsT... params) {
  Gen g(seed == 0 ? Gen::default_seed : seed);
  Dist d(params...);
  std::vector<typename Dist::result_type> u(N);
  for (int i = 0; i < N; ++i) {
    u[i] = d(g);
  }
  double mean = std::accumulate(u.begin(), u.end(), double(0)) / u.size();
  double var = 0;
  double skew = 0;
  double kurtosis = 0;
  for (unsigned i = 0; i < u.size(); ++i) {
    double dbl = (u[i] - mean);
    double d2 = sqr(dbl);
    var += d2;
    skew += dbl * d2;
    kurtosis += d2 * d2;
  }
  var /= u.size();
  double dev = std::sqrt(var);
  if (var != 0.0) {
    skew /= u.size() * dev * var;
    kurtosis /= u.size() * var * var;
    kurtosis -= 3;
  } else {
    skew = 0.0;
    kurtosis = 0.0;
  }

#pragma omp critical
  {
    mmean.push_back(mean);
    vvar.push_back(var);
    sskew.push_back(skew);
    kkurtosis.push_back(kurtosis);
  }
}

template <class Dist, class Gen>
void test_samp_once(const unsigned seed, const int N,
                    std::vector<double> &interval_prob,
                    std::vector<double> &mmean, std::vector<double> &vvar,
                    std::vector<double> &sskew, std::vector<double> &kkurtosis,
                    const std::vector<double> &b,
                    const std::vector<double> &p) {
  Gen g(seed == 0 ? Gen::default_seed : seed);
  Dist d(b.begin(), b.end(), p.begin());
  std::vector<typename Dist::result_type> u(N);
  for (int i = 0; i < N; ++i) {
    u[i] = d(g);
  }

  const auto I = b.size() - 1; // number of intervals
  std::vector<double> x_prob = p;
  const double s = std::accumulate(x_prob.begin(), x_prob.end(), 0.0);
  std::for_each(x_prob.begin(), x_prob.end(), [s](double &p) { p /= s; });
  std::sort(u.begin(), u.end());

#pragma omp critical
  {
    for (size_t i = 0; i < I; ++i) {
      const auto lb = std::lower_bound(u.begin(), u.end(), b[i]);
      const auto ub = std::lower_bound(u.begin(), u.end(), b[i + 1]);
      const auto Ni = ub - lb; // number of samples in this interval

      double mean = std::accumulate(lb, ub, 0.0) / Ni;
      double var = 0;
      double skew = 0;
      double kurtosis = 0;
      for (auto i = lb; i != ub; ++i) {
        double dbl = (*i - mean);
        double d2 = sqr(dbl);
        var += d2;
        skew += dbl * d2;
        kurtosis += d2 * d2;
      }
      var /= Ni;
      double dev = std::sqrt(var);
      if (var != 0.0) {
        skew /= Ni * dev * var;
        kurtosis /= Ni * var * var;
        kurtosis -= 3;
      } else {
        skew = 0.0;
        kurtosis = 0.0;
      }

      interval_prob.push_back(static_cast<double>(Ni) / N);
      mmean.push_back(mean);
      vvar.push_back(var);
      sskew.push_back(skew);
      kkurtosis.push_back(kurtosis);
    }
  }
}

std::tuple<size_t, size_t, double>
find_median_interval(std::vector<double> &x) {
  const double p = 0.9;

  const size_t Nm1 = x.size() - 1;
  size_t lo;
  size_t hi;
  if (x.size() % 2 == 0) {
    lo = x.size() / 2 - 1;
    hi = lo + 1;
  } else {
    lo = (x.size() - 1) / 2;
    hi = lo;
  }

  const auto median = 0.5 * (x[lo] + x[hi]);

  // find the smallest symmetric interval around the median that contains p of
  // the samples.
  while (static_cast<double>(hi - lo) < p * x.size()) {
    if (lo == 0 && hi < Nm1) {
      ++hi;
    } else if (hi == Nm1 && lo > 0) {
      --lo;
    } else if (lo == 0 && hi == Nm1) {
      break;
    } else if (x[hi + 1] - median < median - x[lo - 1]) {
      ++hi;
    } else {
      --lo;
    }
  }

  return {lo, hi, median};
}

void print(const char *label, std::vector<double> &x) {
  const double p = 0.9;
  std::sort(x.begin(), x.end());

  size_t lo, hi;
  double median;
  std::tie(lo, hi, median) = find_median_interval(x);

  cout << format(
      "{:<8}: P({:+.3e} < X < {:+.3e}) = {:.2f} - {:.2f} = {:.2f},  {:+.3e} "
      "+/- {:.2f}%\n",
      label, x[lo], x[hi], static_cast<double>(hi) / x.size(),
      static_cast<double>(lo) / x.size(),
      static_cast<double>(hi - lo) / x.size(), median,
      100.0 * std::abs(std::max(x[hi] - median, median - x[lo]) / median));
}

void summary(const char *name, std::vector<double> &mean,
             std::vector<double> &var, std::vector<double> &skew,
             std::vector<double> &kurtosis) {
  cout << name << '\n';
  cout << "==============================================================="
          "=====================\n";
  print("mean", mean);
  print("variance", var);
  print("skew", skew);
  print("kurtosis", kurtosis);
  cout << '\n';
}

void summary_samp(const char *name, const double *max_abs_range,
                  const double *max_rel_range) {
  constexpr auto format_str = "{:<8}:   {:+.3e}   {:.2f}%\n";
  cout << name << '\n';
  cout << "======================================\n";
  cout << format(format_str, "p_i", max_abs_range[0], max_rel_range[0]);
  cout << format(format_str, "mean", max_abs_range[1], max_rel_range[1]);
  cout << format(format_str, "variance", max_abs_range[2], max_rel_range[2]);
  cout << format(format_str, "skew", max_abs_range[3], max_rel_range[3]);
  cout << format(format_str, "kurtosis", max_abs_range[4], max_rel_range[4]);
  cout << '\n';
}

int samples = 51;

template <class Dist, class Gen, class... ParamsT>
void test(const char *label, const int N, ParamsT... params) {
  std::vector<double> mean, var, skew, kurtosis;
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < samples; ++i) {
    test_once<Dist, Gen>(i, N, mean, var, skew, kurtosis, params...);
  }
  summary(label, mean, var, skew, kurtosis);
}

template <class Dist, class Gen>
void test_samp(const char *label, const int N,
               std::initializer_list<double> params) {
  const size_t I = (params.size() - 1) / 2; // number of intervals
  std::vector<double> b(params.begin(), params.begin() + I + 1);
  std::vector<double> p(params.begin() + I + 1, params.end());

  std::vector<double> interval_prob, mean, var, skew, kurtosis;
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < samples; ++i) {
    test_samp_once<Dist, Gen>(i, N, interval_prob, mean, var, skew, kurtosis, b,
                              p);
  }

  std::vector<double> interval_data[5];
  for (size_t i = 0; i < 5; ++i)
    interval_data[i].resize(samples);

  double max_abs_range[5] = {0.0};
  double max_rel_range[5] = {0.0};
  for (size_t i = 0; i < I; ++i) {
    for (int j = 0; j < samples; ++j) {
      interval_data[0][j] = interval_prob[i + j * I];
      interval_data[1][j] = mean[i + j * I];
      interval_data[2][j] = var[i + j * I];
      interval_data[3][j] = skew[i + j * I];
      interval_data[4][j] = kurtosis[i + j * I];
    }

    for (size_t i = 0; i < 5; ++i) {
      auto &x = interval_data[i];
      std::sort(x.begin(), x.end());
      size_t lo, hi;
      double median;
      std::tie(lo, hi, median) = find_median_interval(x);

      auto diff = std::max(x[hi] - median, median - x[lo]);
      if (diff > max_abs_range[i])
        max_abs_range[i] = diff;

      diff /= std::abs(median);
      if (diff > max_rel_range[i])
        max_rel_range[i] = diff;
    }
  }

  for (size_t i = 0; i < 5; ++i)
    max_rel_range[i] *= 100.0;

  summary_samp(label, max_abs_range, max_rel_range);
}

int main() {
  using namespace std;

  // rand.dist.bern.bernoulli
  test<bernoulli_distribution, minstd_rand>("rand.dist.bern.bernoulli (0.75)",
                                            100'000, 0.75);
  test<bernoulli_distribution, minstd_rand>("rand.dist.bern.bernoulli (0.25)",
                                            100'000, 0.25);

  // rand.dist.bern.bin
  test<binomial_distribution<int>, mt19937_64>("rand.dist.bern.bin (5, .75)",
                                               1'000'000, 5, .75);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (30, .03125)",
                                            100'000, 30, .03125);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (40, .25)",
                                            100'000, 40, .25);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (40, 0)",
                                            100'000, 40, 0);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (40, 1)",
                                            100'000, 40, 1);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (127, 0.5)",
                                            100'000, 127, 0.5);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (1, 0.5)",
                                            100'000, 1, 0.5);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (1, 0.005)",
                                            100'000, 1, 0.005);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (0, 0)",
                                            100'000, 0, 0);
  test<binomial_distribution<int>, mt19937>("rand.dist.bern.bin (0, 1)",
                                            100'000, 0, 1);
  test<binomial_distribution<int>, mt19937>(
      "rand.dist.bern.bin (128738942, 1.6941441471907126e-08)", 1'000'000,
      128738942, 1.6941441471907126e-08);

  // rand.dist.bern.geo
  test<geometric_distribution<int>, mt19937>("rand.dist.bern.geo (.03125)",
                                             1'000'000, .03125);
  test<geometric_distribution<int>, mt19937>("rand.dist.bern.geo (0.05)",
                                             1'000'000, 0.05);
  test<geometric_distribution<int>, minstd_rand>("rand.dist.bern.geo (.25)",
                                                 1'000'000, .25);
  test<geometric_distribution<int>, mt19937>("rand.dist.bern.geo (0.5)",
                                             1'000'000, 0.5);
  test<geometric_distribution<int>, mt19937>("rand.dist.bern.geo (0.75)",
                                             1'000'000, 0.75);
  test<geometric_distribution<int>, mt19937>("rand.dist.bern.geo (0.96875)",
                                             1'000'000, 0.96875);

  // rand.dist.bern.negbin
  test<negative_binomial_distribution<int>, minstd_rand>(
      "rand.dist.bern.negbin (5, .25)", 1'000'000, 5, .25);
  test<negative_binomial_distribution<int>, mt19937>(
      "rand.dist.bern.negbin (30, .03125)", 1'000'000, 30, .03125);
  test<negative_binomial_distribution<int>, mt19937>(
      "rand.dist.bern.negbin (40, .25)", 1'000'000, 40, .25);
  test<negative_binomial_distribution<int>, mt19937>(
      "rand.dist.bern.negbin (40, 1)", 1'000, 40, 1);
  test<negative_binomial_distribution<int>, mt19937>(
      "rand.dist.bern.negbin (127, 0.5)", 1'000'000, 127, 0.5);
  test<negative_binomial_distribution<int>, mt19937>(
      "rand.dist.bern.negbin (1, 0.05)", 1'000'000, 1, 0.05);
  test<negative_binomial_distribution<int>, minstd_rand>(
      "rand.dist.bern.negbin (5, .75)", 1'000'000, 5, .75);

  // rand.dist.norm.cauchy
  // N = 1'000'000, p = 0.1 --> K-S critical value D = 0.00122

  // rand.dist.norm.chisq
  test<chi_squared_distribution<>, minstd_rand>("rand.dist.norm.chisq (0.5)",
                                                1'000'000, 0.5);
  test<chi_squared_distribution<>, minstd_rand>("rand.dist.norm.chisq (1)",
                                                1'000'000, 1);
  test<chi_squared_distribution<>, mt19937>("rand.dist.norm.chisq (2)",
                                            1'000'000, 2);

  // rand.dist.norm.f
  // N = 100'000, p = 0.1 --> K-S critical value D = 0.00387

  // rand.dist.norm.lognormal
  test<lognormal_distribution<>, mt19937>(
      "rand.dist.norm.lognormal (-1. / 8192, 0.015625)", 1'000'000, -1. / 8192,
      0.015625);
  test<lognormal_distribution<>, mt19937>(
      "rand.dist.norm.lognormal (-1. / 32, 0.25)", 1'000'000, -1. / 32, 0.25);
  test<lognormal_distribution<>, mt19937>(
      "rand.dist.norm.lognormal (-1. / 8, 0.5)", 1'000'000, -1. / 8, 0.5);
  test<lognormal_distribution<>, mt19937>("rand.dist.norm.lognormal (default)",
                                          1'000'000);
  test<lognormal_distribution<>, mt19937>(
      "rand.dist.norm.lognormal (-0.78125, 1.25)", 1'000'000, -0.78125, 1.25);

  // rand.dist.norm.normal
  test<normal_distribution<>, minstd_rand>("rand.dist.norm.normal (5, 4)",
                                           1'000'000, 5, 4);

  // rand.dist.norm.t
  test<student_t_distribution<>, minstd_rand>("rand.dist.norm.t (5.5)",
                                              1'000'000, 5.5);
  test<student_t_distribution<>, minstd_rand>("rand.dist.norm.t (10)",
                                              1'000'000, 10);
  test<student_t_distribution<>, minstd_rand>("rand.dist.norm.t (100)",
                                              1'000'000, 100);

  // rand.dist.pois.exp
  test<exponential_distribution<>, mt19937>("rand.dist.pois.exp (0.75)",
                                            1'000'000, 0.75);
  test<exponential_distribution<>, mt19937>("rand.dist.pois.exp (1)", 1'000'000,
                                            1);
  test<exponential_distribution<>, mt19937>("rand.dist.pois.exp (10)",
                                            1'000'000, 10);
  test<exponential_distribution<>, mt19937>("rand.dist.pois.exp (2)", 1'000'000,
                                            2);

  // rand.dist.pois.extreme
  test<extreme_value_distribution<>, mt19937>("rand.dist.pois.extreme (0.5, 2)",
                                              1'000'000, 0.5, 2);
  test<extreme_value_distribution<>, mt19937>("rand.dist.pois.extreme (1, 2)",
                                              1'000'000, 1, 2);
  test<extreme_value_distribution<>, mt19937>("rand.dist.pois.extreme (1.5, 3)",
                                              1'000'000, 1.5, 3);
  test<extreme_value_distribution<>, mt19937>("rand.dist.pois.extreme (3, 4)",
                                              1'000'000, 3, 4);

  // rand.dist.pois.gamma
  test<gamma_distribution<>, mt19937>("rand.dist.pois.gamma (0.5, 2)",
                                      1'000'000, 0.5, 2);
  test<gamma_distribution<>, mt19937>("rand.dist.pois.gamma (1, 0.5)",
                                      1'000'000, 1, 0.5);
  test<gamma_distribution<>, mt19937>("rand.dist.pois.gamma (2, 3)", 1'000'000,
                                      2, 3);

  // rand.dist.pois.poisson
  test<poisson_distribution<>, minstd_rand>("rand.dist.pois.poisson (2)",
                                            100'000, 2);
  test<poisson_distribution<>, minstd_rand>("rand.dist.pois.poisson (0.75)",
                                            100'000, 0.75);
  test<poisson_distribution<>, mt19937>("rand.dist.pois.poisson (20)",
                                        1'000'000, 20);

  // rand.dist.pois.weibull
  test<weibull_distribution<>, mt19937>("rand.dist.pois.weibull (0.5, 2)",
                                        1'000'000, 0.5, 2);
  test<weibull_distribution<>, mt19937>("rand.dist.pois.weibull (1, .5)",
                                        1'000'000, 1, .5);
  test<weibull_distribution<>, mt19937>("rand.dist.pois.weibull (2, 3)",
                                        1'000'000, 2, 3);

  // rand.dist.uni.int
  test<uniform_int_distribution<>, minstd_rand>(
      "rand.dist.uni.int (0, INT_MAX)", 10'000, 0, INT_MAX);
  test<uniform_int_distribution<>, mt19937>("rand.dist.uni.int (0, INT_MAX)",
                                            10'000, 0, INT_MAX);

  // rand.dist.uni.real
  test<uniform_real_distribution<>, minstd_rand0>(
      "rand.dist.uni.real (default, minstd_rand0)", 100'000);
  test<uniform_real_distribution<>, minstd_rand>(
      "rand.dist.uni.real (default, minstd_rand)", 100'000);
  test<uniform_real_distribution<>, mt19937>(
      "rand.dist.uni.real (default, mt19937)", 100'000);
  test<uniform_real_distribution<>, mt19937_64>(
      "rand.dist.uni.real (default, mt19937_64)", 100'000);
  test<uniform_real_distribution<>, ranlux24_base>(
      "rand.dist.uni.real (default, ranlux24_base)", 100'000);
  test<uniform_real_distribution<>, ranlux48_base>(
      "rand.dist.uni.real (default, ranlux48_base)", 100'000);
  test<uniform_real_distribution<>, minstd_rand>("rand.dist.uni.real (-1, 1)",
                                                 100'000, -1, 1);
  test<uniform_real_distribution<>, minstd_rand>("rand.dist.uni.real (5.5, 25)",
                                                 100'000, 5.5, 25);

  // rand.dist.samp.discrete
  // N =  1'000'000, p = 0.1 --> K-S critical value D = 0.00122
  // N = 10'000'000, p = 0.1 --> K-S critical value D = 3.87e-4

  // rand.dist.samp.pconst
  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (25, 62.5, 12.5)", 1'000'000,
      {10, 14, 16, 17, 25, 62.5, 12.5});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (0, 62.5, 12.5)", 1'000'000,
      {10, 14, 16, 17, 25, 62.5, 12.5});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (25, 0, 12.5)", 1'000'000,
      {10, 14, 16, 17, 25, 0, 12.5});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (25, 62.5, 0)", 1'000'000,
      {10, 14, 16, 17, 25, 62.5, 0});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (25, 0, 0)", 100'000, {10, 14, 16, 17, 25, 0, 0});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (0, 25, 0)", 100'000, {10, 14, 16, 17, 0, 25, 0});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (0, 0, 1)", 100'000, {10, 14, 16, 17, 0, 0, 1});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (75, 25)", 100'000, {10, 14, 16, 75, 25});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (0, 25)", 100'000, {10, 14, 16, 0, 25});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (1, 0)", 100'000, {10, 14, 16, 1, 0});

  test_samp<piecewise_constant_distribution<>, mt19937_64>(
      "rand.dist.samp.pconst (1)", 100'000, {10, 14, 1});

  // rand.dist.samp.plinear
  // N = 1'000'000, p = 0.1 --> K-S critical value D = 0.00122

  return 0;
}
