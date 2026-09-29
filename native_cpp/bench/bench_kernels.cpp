// Per-kernel timing at the production shapes (H = 128, 211 inputs, 2 GRU layers).
//   g++ -O3 -march=x86-64-v3 -fno-trapping-math -fno-math-errno -std=c++17 -Inative_cpp/include \
//       native_cpp/bench/bench_kernels.cpp -o /tmp/bench_kernels && taskset -c 75 /tmp/bench_kernels
#include <chrono>
#include <cstdio>
#include <random>

#include "wnn/activations.hpp"
#include "wnn/kernels.hpp"

using wnn::kernels::Vec;

template <class F>
double time_us(F&& f, int reps) {
  for (int i = 0; i < reps / 10; ++i) f();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) f();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

int main() {
  constexpr int H = 128, IN = 211, G = 3 * H;
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.f, 0.1f);
  auto fill = [&](Vec& v, std::size_t n) {
    v.resize(n);
    for (auto& x : v) x = nd(rng);
  };
  Vec w_in, b_in, w_g, b_g, w_h1, b_h1, x, h, gi, z, cat, head, gate(2 * H);
  fill(w_in, std::size_t(IN) * H);
  fill(b_in, H);
  fill(w_g, std::size_t(G) * H);
  fill(b_g, G);
  fill(w_h1, std::size_t(H) * 2 * H);
  fill(b_h1, H);
  fill(x, IN);
  for (int i = 112; i < IN; ++i)
    if (i % 5) x[i] = 0.f;  // event block mostly zero
  fill(h, H);
  gi.resize(G);
  z.resize(H);
  fill(cat, 2 * H);
  head.resize(H);
  volatile float sink = 0;
  const int reps = 200000;
  const double t_proj = time_us([&] { wnn::kernels::saxpy_sparse(w_in.data(), b_in.data(), x.data(), z.data(), H, IN); sink = z[0]; }, reps);
  const double t_gemv = time_us([&] { wnn::kernels::gemv(w_g.data(), b_g.data(), h.data(), gi.data(), G, H); sink = gi[0]; }, reps);
  const double t_head = time_us([&] { wnn::kernels::gemv(w_h1.data(), b_h1.data(), cat.data(), head.data(), H, 2 * H); sink = head[0]; }, reps);
  const double t_sig = time_us([&] { for (int k = 0; k < 2 * H; ++k) gate[k] = wnn::act::sigmoid_fast(gi[k] + gi[G - 1 - k]); sink = gate[0]; }, reps);
  const double t_gelu = time_us([&] { for (int k = 0; k < H; ++k) z[k] = wnn::act::gelu_fast(z[k] * 0.5f); sink = z[0]; }, reps);
  std::printf("input projection (sparse saxpy 211->128): %.2f us\n", t_proj);
  std::printf("GRU gemv 384x128 (x4 per row):          %.2f us  -> %.2f\n", t_gemv, 4 * t_gemv);
  std::printf("head gemv 128x256:                      %.2f us\n", t_head);
  std::printf("sigmoid x256 (x2 per row):              %.2f us\n", t_sig);
  std::printf("gelu x128 (x2 per row):                 %.2f us\n", t_gelu);
  std::printf("sum of kernels per row:                 %.2f us\n", t_proj + 4 * t_gemv + t_head + 2 * t_sig + 2 * t_gelu);
  return sink > 1e30f;
}
