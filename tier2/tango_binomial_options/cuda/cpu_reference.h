#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

struct OptionData {
    float S;      // stock price
    float K;      // strike price
    float T;      // time to expiration
    float r;      // risk-free rate
    float sigma;  // volatility
};

namespace binomial_reference {
constexpr int num_steps = 1024;

// Explicit NaN handling preserves fmaxf's selection rule and allows the
// backward-induction loop to vectorize without enabling fast-math.
inline float maximum(float a, float b) {
    return std::isnan(a) ? b : std::isnan(b) ? a : a > b ? a : b;
}

inline float price(OptionData opt, int steps = num_steps) {
    if (steps < 1 || steps > num_steps)
        throw std::invalid_argument("binomial CPU reference steps must be in 1..1024");
    float dt   = opt.T / (float)steps;
    float u    = expf(opt.sigma * sqrtf(dt));
    float d    = 1.0f / u;
    float R    = expf(opt.r * dt);
    float Rinv = 1.0f / R;
    float p    = (R - d) / (u - d);
    float q    = 1.0f - p;

    std::array<float, num_steps + 1> vals;
    std::array<float, 2 * num_steps + 1> payoff;
    // Each exponent occurs many times, with identical operands. Cache its
    // original powf/payoff expression once without changing its arithmetic.
    for (int e = -steps; e <= steps; ++e) {
        float ST = opt.S * powf(u, (float)e);
        payoff[e + steps] = fmaxf(opt.K - ST, 0.0f);
    }
    for (int j = 0; j <= steps; ++j)
        vals[j] = payoff[2 * j];
    for (int step = steps; step > 0; --step) {
        for (int j = 0; j < step; ++j) {
            float cont = Rinv * (p * vals[j + 1] + q * vals[j]);
            float exercise = payoff[2 * j - (step - 1) + steps];
            vals[j] = maximum(cont, exercise);
        }
    }
    return vals[0];
}

inline std::vector<float> prices(const OptionData* options, int count,
        unsigned workers = std::thread::hardware_concurrency()) {
    if (count < 1)
        throw std::invalid_argument("binomial CPU reference needs positive option count");
    std::vector<float> result(count);
    workers = std::max(1u, std::min({workers, 16u, static_cast<unsigned>(count)}));
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    auto work = [&](unsigned worker) {
        const size_t first = static_cast<size_t>(count) * worker / workers;
        const size_t end = static_cast<size_t>(count) * (worker + 1) / workers;
        for (size_t i = first; i < end; ++i)
            result[i] = price(options[i]);
    };
    for (unsigned worker = 1; worker < workers; ++worker) {
        try { threads.emplace_back(work, worker); }
        catch (const std::exception&) { work(worker); }
    }
    work(0);
    for (auto& thread : threads) thread.join();
    return result;
}
}
