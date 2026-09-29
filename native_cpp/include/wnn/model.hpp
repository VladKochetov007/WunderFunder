// Streaming GRU predictor: C++ twin of wnn.nn.RNNModel (enc="linear", cell="gru") evaluated one row
// at a time, with the event features computed in place.
//
//   x   = [raw 112 | event 99]                 (EventFeatures)
//   z   = gelu(W_inp ((x - mean) / std) + b)    (hidden H)
//   o   = GRU layers on z (PyTorch gate order r, z, n)
//   y   = W2 gelu(W1 [o | z] + b1) + b2         (2 outputs: t0, t1)
//
// Weights come as a name -> array map with the PyTorch state_dict names (Weights::from). At load time
// the input normalisation is folded into the projection (W' = W / std, b' = b - W' mean), which is
// stored transposed so zero inputs are skipped; the recurrent and head matrices stay row-major.
#pragma once

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "wnn/activations.hpp"
#include "wnn/event_features.hpp"
#include "wnn/kernels.hpp"

namespace wnn {

using kernels::Vec;

struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;
  int dim(int i) const { return shape.at(i); }
};

using WeightMap = std::map<std::string, Tensor>;

// All weights in one 64-byte aligned arena, in the order a step reads them (input projection, then
// per layer w_ih, b_ih, w_hh, b_hh, then the head), each segment starting on a 64-byte boundary, so
// the hardware prefetcher sees one forward stream per row.
struct Arena {
  Vec data;
  std::size_t push(const float* v, std::size_t n) {
    const std::size_t at = data.size();
    const std::size_t padded = (n + 15) / 16 * 16;
    data.resize(at + padded, 0.f);
    std::copy(v, v + n, data.begin() + static_cast<std::ptrdiff_t>(at));
    return at;
  }
  const float* at(std::size_t off) const { return data.data() + off; }
};

struct GruLayer {
  std::size_t w_ih, b_ih, w_hh, b_hh;  // arena offsets: row-major (3H x in), (3H), (3H x H), (3H)
  int in = 0;
};

struct Weights {
  int hidden = 0, head_hidden = 0;
  Arena arena;
  std::size_t w_inp_t = 0, b_inp = 0, w_h1 = 0, b_h1 = 0, w_h2 = 0, b_h2 = 0;
  std::vector<GruLayer> layers;
  // optional fp16 copies of the dense matrices (GRU w_ih / w_hh per layer, head W1)
  kernels::HalfVec half;
  std::vector<std::size_t> half_ih, half_hh;
  std::size_t half_h1 = 0;
  bool has_half = false;

#if WNN_HAVE_F16
  void make_half() {
    std::vector<std::pair<const float*, std::size_t>> parts;
    for (const GruLayer& g : layers) {
      parts.emplace_back(arena.at(g.w_ih), static_cast<std::size_t>(3 * hidden) * g.in);
      parts.emplace_back(arena.at(g.w_hh), static_cast<std::size_t>(3 * hidden) * hidden);
    }
    parts.emplace_back(arena.at(w_h1), static_cast<std::size_t>(head_hidden) * 2 * hidden);
    std::vector<std::size_t> offs;
    for (const auto& [ptr, n] : parts) {
      const kernels::HalfVec h = kernels::to_half(ptr, n);
      offs.push_back(half.size());
      half.insert(half.end(), h.begin(), h.end());
    }
    for (std::size_t l = 0; l < layers.size(); ++l) {
      half_ih.push_back(offs[2 * l]);
      half_hh.push_back(offs[2 * l + 1]);
    }
    half_h1 = offs.back();
    has_half = true;
  }
#endif

  static Weights from(const WeightMap& m) {
    auto get = [&](const std::string& k) -> const Tensor& {
      auto it = m.find(k);
      if (it == m.end()) throw std::invalid_argument("missing weight " + k);
      return it->second;
    };
    Weights w;
    const Tensor& inp = get("inp.weight");
    w.hidden = inp.dim(0);
    const int n_in = inp.dim(1);
    if (n_in != kInputs)
      throw std::invalid_argument("model input width " + std::to_string(n_in) + ", engine supports " +
                                  std::to_string(kInputs) + " (raw + event features)");
    const std::vector<float>& mean = get("mean").data;
    const std::vector<float>& sd = get("std").data;
    const std::vector<float>& bias = get("inp.bias").data;
    std::vector<float> folded(inp.data.size()), b_inp(w.hidden);
    for (int o = 0; o < w.hidden; ++o) {
      double b = bias[o];
      for (int i = 0; i < n_in; ++i) {
        const double wi = static_cast<double>(inp.data[static_cast<std::size_t>(o) * n_in + i]) / sd[i];
        folded[static_cast<std::size_t>(o) * n_in + i] = static_cast<float>(wi);
        b -= wi * mean[i];
      }
      b_inp[o] = static_cast<float>(b);
    }
    const Vec t = kernels::transpose(folded.data(), w.hidden, n_in);
    w.w_inp_t = w.arena.push(t.data(), t.size());
    w.b_inp = w.arena.push(b_inp.data(), b_inp.size());
    for (int l = 0;; ++l) {
      const std::string s = std::to_string(l);
      if (m.find("gru.weight_ih_l" + s) == m.end()) break;
      const Tensor& wih = get("gru.weight_ih_l" + s);
      if (wih.dim(0) != 3 * w.hidden) throw std::invalid_argument("GRU gate width mismatch");
      const Tensor& whh = get("gru.weight_hh_l" + s);
      const Tensor& bih = get("gru.bias_ih_l" + s);
      const Tensor& bhh = get("gru.bias_hh_l" + s);
      GruLayer g;
      g.in = wih.dim(1);
      g.w_ih = w.arena.push(wih.data.data(), wih.data.size());
      g.b_ih = w.arena.push(bih.data.data(), bih.data.size());
      g.w_hh = w.arena.push(whh.data.data(), whh.data.size());
      g.b_hh = w.arena.push(bhh.data.data(), bhh.data.size());
      w.layers.push_back(g);
    }
    if (w.layers.empty()) throw std::invalid_argument("no GRU layers");
    const Tensor& h1 = get("head.0.weight");
    const Tensor& h2 = get("head.2.weight");
    if (h1.dim(1) != 2 * w.hidden || h2.dim(1) != h1.dim(0)) throw std::invalid_argument("head shape mismatch");
    if (h2.dim(0) < 2) throw std::invalid_argument("head needs at least two outputs");
    w.head_hidden = h1.dim(0);
    const Tensor& b1 = get("head.0.bias");
    const Tensor& b2 = get("head.2.bias");
    w.w_h1 = w.arena.push(h1.data.data(), h1.data.size());
    w.b_h1 = w.arena.push(b1.data.data(), b1.data.size());
    // only the two target outputs (auxiliary heads are training-only)
    w.w_h2 = w.arena.push(h2.data.data(), 2 * static_cast<std::size_t>(h2.dim(1)));
    w.b_h2 = w.arena.push(b2.data.data(), 2);
    return w;
  }
};

class Predictor {
 public:
  // fast: vectorised activation approximations (errors < 5e-6, as in the production Rust engine);
  // false: libm exp/tanh/erf.
  // half: store the dense matrices as fp16 (x86-64-v3 builds only; ignored elsewhere).
  explicit Predictor(Weights w, bool fast = true, bool half = false) : w_(std::move(w)), fast_(fast) {
#if WNN_HAVE_F16
    if (half && (w_.hidden % 16) == 0) w_.make_half();
#else
    (void)half;
#endif
    const int H = w_.hidden, L = static_cast<int>(w_.layers.size());
    x_.resize(kInputs);
    h_.assign(static_cast<std::size_t>(L) * H, 0.f);
    gi_.resize(3 * H);
    gh_.resize(3 * H);
    gate_.resize(2 * H);
    cat_.resize(2 * H);  // [GRU output | z]
    head_.resize(w_.head_hidden);
  }

  void reset() {
    events_.reset();
    std::fill(h_.begin(), h_.end(), 0.f);
  }

  // One row in (112 floats), two predictions out.
  void step(const float* row, float out[2]) {
    const int H = w_.hidden;
    std::copy(row, row + kRaw, x_.begin());
    events_.update(row, x_.data() + kRaw);
    float* z = cat_.data() + H;  // the projection lives in the second half of the head input
    const Arena& A = w_.arena;
    kernels::saxpy_sparse(A.at(w_.w_inp_t), A.at(w_.b_inp), x_.data(), z, H, kInputs);
    gelu(z, H);

    const float* layer_in = z;
    for (std::size_t l = 0; l < w_.layers.size(); ++l) {
      const GruLayer& g = w_.layers[l];
      float* h = h_.data() + l * H;
      dense(g.w_ih, w_.has_half ? w_.half_ih[l] : 0, A.at(g.b_ih), layer_in, gi_.data(), 3 * H, g.in);
      dense(g.w_hh, w_.has_half ? w_.half_hh[l] : 0, A.at(g.b_hh), h, gh_.data(), 3 * H, H);
      gru_update(h, H);
      layer_in = h;
    }
    std::copy(layer_in, layer_in + H, cat_.begin());
    dense(w_.w_h1, w_.half_h1, A.at(w_.b_h1), cat_.data(), head_.data(), w_.head_hidden, 2 * H);
    gelu(head_.data(), w_.head_hidden);
    kernels::gemv(A.at(w_.w_h2), A.at(w_.b_h2), head_.data(), out, 2, w_.head_hidden);
  }

  int hidden() const { return w_.hidden; }
  bool fast() const { return fast_; }
  bool half() const { return w_.has_half; }

 private:
  // Dense layer from the fp32 arena, or from the fp16 copy when half storage is on.
  void dense(std::size_t off32, std::size_t off16, const float* b, const float* x, float* y, int out, int in) const {
#if WNN_HAVE_F16
    if (w_.has_half) {
      kernels::gemv_half(w_.half.data() + off16, b, x, y, out, in);
      return;
    }
#else
    (void)off16;
#endif
    kernels::gemv(w_.arena.at(off32), b, x, y, out, in);
  }

  void gelu(float* v, int n) const {
    if (fast_) {
      for (int i = 0; i < n; ++i) v[i] = act::gelu_fast(v[i]);
    } else {
      for (int i = 0; i < n; ++i) v[i] = act::gelu_exact(v[i]);
    }
  }

  // PyTorch GRU cell, gate order (r, z, n): h = (1 - z) n + z h.
  void gru_update(float* __restrict h, int H) {
    float* __restrict rz = gate_.data();
    const float* __restrict gi = gi_.data();
    const float* __restrict gh = gh_.data();
    if (fast_) {
      for (int k = 0; k < 2 * H; ++k) rz[k] = act::sigmoid_fast(gi[k] + gh[k]);
      for (int k = 0; k < H; ++k) {
        const float n = act::tanh_fast(gi[2 * H + k] + rz[k] * gh[2 * H + k]);
        h[k] = (1.f - rz[H + k]) * n + rz[H + k] * h[k];
      }
    } else {
      for (int k = 0; k < 2 * H; ++k) rz[k] = act::sigmoid_exact(gi[k] + gh[k]);
      for (int k = 0; k < H; ++k) {
        const float n = std::tanh(gi[2 * H + k] + rz[k] * gh[2 * H + k]);
        h[k] = (1.f - rz[H + k]) * n + rz[H + k] * h[k];
      }
    }
  }

  Weights w_;
  bool fast_;
  EventFeatures events_;
  Vec x_, h_, gi_, gh_, gate_, cat_, head_;
};

// Weighted sum of member predictions (coefficients already include any output scale).
class Ensemble {
 public:
  void add(Weights w, float coef, bool fast = true, bool half = false) {
    members_.emplace_back(std::move(w), fast, half);
    coefs_.push_back(coef);
  }
  void reset() {
    for (auto& m : members_) m.reset();
  }
  void step(const float* row, float out[2]) {
    out[0] = out[1] = 0.f;
    float tmp[2];
    for (std::size_t i = 0; i < members_.size(); ++i) {
      members_[i].step(row, tmp);
      out[0] += coefs_[i] * tmp[0];
      out[1] += coefs_[i] * tmp[1];
    }
  }
  std::size_t size() const { return members_.size(); }

 private:
  std::vector<Predictor> members_;
  std::vector<float> coefs_;
};

}  // namespace wnn
