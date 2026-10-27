#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Host-only draft. sw contains exact float expansion of ACTUAL stored scales:
// fp32 copied verbatim; fp16 decoded from stored bits, never regenerated originals.
struct D2Reference { std::vector<double> value, denominator; };
inline D2Reference d2_reference(const int8_t* xq, const float* xs,
                               const uint8_t* wq, const float* sw,
                               int in, int out) {
    if (in <= 0 || in % 32 || out <= 0) throw std::runtime_error("invalid D2 shape");
    D2Reference r{std::vector<double>(out),std::vector<double>(out)};
    for (int o=0;o<out;o++) for (int k=0;k<in;k++) {
        size_t wi=(size_t)o*(in/2)+k/2;
        int nibble=(wq[wi] >> ((k&1)*4)) & 15;
        double weight=(double)(nibble-8)*(double)sw[(size_t)o*(in/32)+k/32];
        double activation=(double)xq[k]*(double)xs[k/32];
        double product=weight*activation;
        if (!std::isfinite(product)) throw std::runtime_error("nonfinite oracle input/product");
        r.value[o]+=product;
        r.denominator[o]+=std::abs(product);
    }
    return r;
}
// Use independently for G and U, never combine their denominators.
// Required pairs: candidate-old, candidate-CPUdouble, old-CPUdouble.
inline double d2_normalized_error(double a,double b,double denominator) {
    if (!std::isfinite(a) || !std::isfinite(b) ||
        !std::isfinite(denominator) || denominator<0)
        throw std::runtime_error("nonfinite result or invalid denominator");
    double numerator=std::abs(a-b);
    if (denominator==0) return numerator==0 ? 0 : std::numeric_limits<double>::infinity();
    return numerator/denominator;
}
