// Candidate screening for wnn.symreg: |y|-weighted Pearson of transformed features with a
// residual, one pass over the rows per candidate family.
//
// Build: g++ -O3 -march=native -ffast-math -fopenmp -shared -fPIC symreg_screen.cpp -o libsymreg_screen.so
// Data layout: feature-major (each column contiguous, float32), rows n; accumulators in double.
// Operator codes match wnn.symreg.UNARY / BINARY.
#include <cmath>
#include <cstdint>

namespace {

inline float unary(int op, float v) {
    switch (op) {
        case 0: return v;
        case 1: return std::fabs(v);
        case 2: return v * v;
        case 3: return std::copysign(std::sqrt(std::fabs(v)), v);
        case 4: return std::copysign(std::log1p(std::fabs(v)), v);
        case 5: return std::tanh(v);
        case 6: return v > 0.f ? v : 0.f;
        case 7: return v < 0.f ? v : 0.f;
        case 8: return (v > 0.f) - (v < 0.f);
        default: return std::exp(-v * v);
    }
}

inline double corr(double sw, double swv, double swvv, double swvr, double swr, double swrr) {
    const double mv = swv / sw, mr = swr / sw;
    const double cov = swvr / sw - mv * mr;
    const double vv = swvv / sw - mv * mv, vr = swrr / sw - mr * mr;
    if (vv <= 1e-12 || vr <= 1e-12) return 0.0;
    return cov / std::sqrt(vv * vr);
}

}  // namespace

extern "C" {

// Z: k columns x n rows (column-major, contiguous per column); out: k x n_ops.
void screen_unary(const float* Z, int64_t n, int64_t k, const float* r, const float* w, int n_ops, double* out) {
    double sw = 0, swr = 0, swrr = 0;
    for (int64_t i = 0; i < n; ++i) {
        sw += w[i];
        swr += double(w[i]) * r[i];
        swrr += double(w[i]) * r[i] * r[i];
    }
#pragma omp parallel for schedule(dynamic)
    for (int64_t j = 0; j < k; ++j) {
        const float* col = Z + j * n;
        for (int op = 0; op < n_ops; ++op) {
            double s1 = 0, s2 = 0, s3 = 0;
#pragma omp simd reduction(+ : s1, s2, s3)
            for (int64_t i = 0; i < n; ++i) {
                const float v = unary(op, col[i]);
                const float wv = w[i] * v;
                s1 += wv;
                s2 += wv * v;
                s3 += wv * r[i];
            }
            out[j * n_ops + op] = corr(sw, s1, s2, s3, swr, swrr);
        }
    }
}

// V: m columns x n rows (column-major); out: m x m x 7 (a, b, op); commutative ops (add, mul,
// max, min) are evaluated only for a < b and left 0 otherwise, like the numba reference.
void screen_binary(const float* V, int64_t n, int64_t m, const float* r, const float* w, double* out) {
    double sw = 0, swr = 0, swrr = 0;
    for (int64_t i = 0; i < n; ++i) {
        sw += w[i];
        swr += double(w[i]) * r[i];
        swrr += double(w[i]) * r[i] * r[i];
    }
#pragma omp parallel for schedule(dynamic)
    for (int64_t a = 0; a < m; ++a) {
        const float* va = V + a * n;
        for (int64_t b = 0; b < m; ++b) {
            if (a == b) continue;
            const float* vb = V + b * n;
            double* o = out + (a * m + b) * 7;
            const bool comm = a < b;
            // pass 1: sub, div, gate (always)
            double p0 = 0, p1 = 0, p2 = 0, q0 = 0, q1 = 0, q2 = 0, g0 = 0, g1 = 0, g2 = 0;
#pragma omp simd reduction(+ : p0, p1, p2, q0, q1, q2, g0, g1, g2)
            for (int64_t i = 0; i < n; ++i) {
                const float x = va[i], y = vb[i], wi = w[i], ri = r[i];
                const float vs = x - y;
                const float vd = x / (1.f + std::fabs(y));
                const float vg = y > 0.f ? x : 0.f;
                p0 += wi * vs; p1 += wi * vs * vs; p2 += wi * vs * ri;
                q0 += wi * vd; q1 += wi * vd * vd; q2 += wi * vd * ri;
                g0 += wi * vg; g1 += wi * vg * vg; g2 += wi * vg * ri;
            }
            o[1] = corr(sw, p0, p1, p2, swr, swrr);
            o[3] = corr(sw, q0, q1, q2, swr, swrr);
            o[6] = corr(sw, g0, g1, g2, swr, swrr);
            if (!comm) continue;
            // pass 2: add, mul, max, min (once per unordered pair)
            double a0 = 0, a1 = 0, a2 = 0, m0 = 0, m1 = 0, m2 = 0, x0 = 0, x1 = 0, x2 = 0, n0 = 0, n1 = 0, n2 = 0;
#pragma omp simd reduction(+ : a0, a1, a2, m0, m1, m2, x0, x1, x2, n0, n1, n2)
            for (int64_t i = 0; i < n; ++i) {
                const float x = va[i], y = vb[i], wi = w[i], ri = r[i];
                const float va_ = x + y, vm = x * y, vx = x > y ? x : y, vn = x < y ? x : y;
                a0 += wi * va_; a1 += wi * va_ * va_; a2 += wi * va_ * ri;
                m0 += wi * vm; m1 += wi * vm * vm; m2 += wi * vm * ri;
                x0 += wi * vx; x1 += wi * vx * vx; x2 += wi * vx * ri;
                n0 += wi * vn; n1 += wi * vn * vn; n2 += wi * vn * ri;
            }
            o[0] = corr(sw, a0, a1, a2, swr, swrr);
            o[2] = corr(sw, m0, m1, m2, swr, swrr);
            o[4] = corr(sw, x0, x1, x2, swr, swrr);
            o[5] = corr(sw, n0, n1, n2, swr, swrr);
        }
    }
}

}  // extern "C"
