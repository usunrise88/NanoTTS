#include "audio/resample.hpp"

#include <cmath>
#include <numeric>

namespace xvibe {
namespace {

double bessel_i0(double x) {
  double sum = 1.0, term = 1.0;
  for (int k = 1; k < 50; ++k) {
    term *= (x / 2.0) * (x / 2.0) / (static_cast<double>(k) * k);
    sum += term;
    if (term < 1e-12 * sum) break;
  }
  return sum;
}

int gcd_int(int a, int b) { return b == 0 ? a : gcd_int(b, a % b); }

}  // namespace

std::vector<float> resample(const std::vector<float>& in, int from_rate, int to_rate) {
  if (from_rate == to_rate || in.empty()) return in;

  const int g = gcd_int(from_rate, to_rate);
  const int up = to_rate / g;
  const int down = from_rate / g;

  // scipy: half_len = 10 * max(up, down), Kaiser beta 5.0.
  const int max_rate = up > down ? up : down;
  const int half = 10 * max_rate;
  const int taps = 2 * half + 1;
  const double beta = 5.0;
  const double cutoff = 1.0 / static_cast<double>(max_rate);

  std::vector<double> h(static_cast<size_t>(taps));
  const double denom = bessel_i0(beta);
  for (int i = 0; i < taps; ++i) {
    const double n = i - half;
    const double sinc = (n == 0.0) ? cutoff : std::sin(M_PI * cutoff * n) / (M_PI * n);
    const double r = static_cast<double>(2 * i) / (taps - 1) - 1.0;
    const double w = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / denom;
    h[static_cast<size_t>(i)] = sinc * w * up;
  }

  const size_t n_in = in.size();
  const size_t n_out = (n_in * static_cast<size_t>(up) + static_cast<size_t>(down) - 1) /
                       static_cast<size_t>(down);
  std::vector<float> out(n_out, 0.f);

  for (size_t m = 0; m < n_out; ++m) {
    const long long centre = static_cast<long long>(m) * down;  // position in upsampled grid
    double acc = 0.0;
    // Only samples where (centre - k) is a multiple of `up` contribute.
    const long long lo = centre - half;
    const long long hi = centre + half;
    long long first = lo + ((up - (lo % up)) % up);
    for (long long pos = first; pos <= hi; pos += up) {
      const long long src = pos / up;
      if (src < 0 || src >= static_cast<long long>(n_in)) continue;
      const long long tap = pos - centre + half;
      acc += static_cast<double>(in[static_cast<size_t>(src)]) * h[static_cast<size_t>(tap)];
    }
    // No division by `up` here: the filter already carries that gain, and only
    // every up-th tap contributes to a given output, so the sum is already at
    // unity. Dividing again attenuated by `up` — 38 dB for a 44.1 kHz source.
    out[m] = static_cast<float>(acc);
  }
  return out;
}

}  // namespace xvibe
