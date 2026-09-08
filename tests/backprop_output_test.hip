// Independent dense, double-precision reference for the actual production
// hybrid epoch. Non-saturating inputs exercise every dependency and momentum.
#define main benchmark_main
#include "../tier2/rodinia_backprop/rodinia_backprop.hip"
#undef main

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool close_value(double actual, double expected, const char* field, size_t index,
                        double absolute_tolerance = 2e-5) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > std::max(absolute_tolerance, 2e-5 * std::fabs(expected))) {
        fprintf(stderr, "%s[%zu]: got %.9g, expected %.12g\n", field, index, actual, expected);
        return false;
    }
    return true;
}

static float signed_random(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (float)((int)(state >> 16) - 32768) / 65536.0f;
}

static bool check_epoch(int in, bool upstream_inputs, bool momentum) {
    BackpropNetwork net(in);
    if (upstream_inputs) {
        initialize_network(net);
    } else {
        uint32_t seed = 123;
        for (int i = 1; i <= in; ++i) net.input[i] = signed_random(seed);
        for (float& w : net.weights) w = signed_random(seed) / std::sqrt((float)in);
        for (float& w : net.previous_weights) w = momentum ? signed_random(seed) * 0.2f : 0.0f;
        for (int j = 0; j <= WIDTH; ++j) {
            for (int k = 0; k <= 1; ++k) {
                net.hidden_weights[j][k] = signed_random(seed);
                net.hidden_previous[j][k] = momentum ? signed_random(seed) * 0.2f : 0.0f;
            }
        }
        net.target = momentum ? 0.9f : 0.1f;
    }
    BackpropNetwork initial = net;
    std::vector<double> hidden(WIDTH + 1, 1.0), delta(WIDTH + 1, 0.0);
    // Dense dot products, independent of the tiled GPU partial-sum algorithm.
    for (int j = 1; j <= WIDTH; ++j) {
        double dot = initial.weights[j];
        for (int i = 1; i <= in; ++i)
            dot += (double)initial.input[i] * initial.weights[(size_t)i * (WIDTH + 1) + j];
        hidden[j] = 1.0 / (1.0 + std::exp(-dot));
    }
    double output_dot = 0.0;
    for (int j = 0; j <= WIDTH; ++j) output_dot += hidden[j] * initial.hidden_weights[j][1];
    double output = 1.0 / (1.0 + std::exp(-output_dot));
    double output_delta = output * (1.0 - output) * ((double)initial.target - output);
    for (int j = 1; j <= WIDTH; ++j)
        delta[j] = hidden[j] * (1.0 - hidden[j]) * output_delta * initial.hidden_weights[j][1];

    BackpropLaps laps = run_backprop_epoch(net);
    if (!std::isfinite(laps.forward.gpu_ms) || !std::isfinite(laps.adjust.gpu_ms) ||
        laps.forward.gpu_ms < 0 || laps.adjust.gpu_ms < 0) return false;
    if (!close_value(net.output, output, "output", 0) ||
        !close_value(net.output_delta, output_delta, "output_delta", 0, 1e-7)) return false;
    for (int j = 0; j <= WIDTH; ++j) {
        if (!close_value(net.hidden[j], hidden[j], "hidden", j) ||
            !close_value(net.hidden_delta[j], delta[j], "hidden_delta", j, 1e-7)) return false;
        for (int k = 0; k <= 1; ++k) {
            double change = 0.3 * output_delta * hidden[j] + 0.3 * initial.hidden_previous[j][k];
            double expected_weight = initial.hidden_weights[j][k] + (k ? change : 0.0);
            double expected_previous = k ? change : initial.hidden_previous[j][k];
            if (!close_value(net.hidden_weights[j][k], expected_weight, "output_weight", 2*j+k) ||
                !close_value(net.hidden_previous[j][k], expected_previous, "output_momentum", 2*j+k, 1e-7)) return false;
            if (k && !close_value((double)net.hidden_weights[j][k] - initial.hidden_weights[j][k],
                                  change, "output_weight_change", j, 1e-7)) return false;
        }
    }
    for (int i = 0; i <= in; ++i) {
        if (net.input[i] != initial.input[i]) {
            fprintf(stderr, "input[%d]: got %.9g, expected unchanged %.9g\n",
                    i, net.input[i], initial.input[i]);
            return false;
        }
        for (int j = 0; j <= WIDTH; ++j) {
            size_t index = (size_t)i * (WIDTH + 1) + j;
            double change = 0.3 * delta[j] * (i ? initial.input[i] : 1.0) +
                            0.3 * initial.previous_weights[index];
            double expected_weight = initial.weights[index] + (j ? change : 0.0);
            double expected_previous = j ? change : initial.previous_weights[index];
            if (!close_value(net.weights[index], expected_weight, "input_weight", index) ||
                !close_value(net.previous_weights[index], expected_previous, "input_momentum", index, 1e-7)) return false;
            if (j && !close_value((double)net.weights[index] - initial.weights[index],
                                  change, "input_weight_change", index, 1e-7)) return false;
        }
    }
    printf("PASS: input=%d upstream_inputs=%d momentum=%d\n", in, upstream_inputs, momentum);
    return true;
}

int main(int argc, char** argv) {
    setenv("BENCH_PARAM_vector_add_warmup", "false", 1);
    if (argc == 3 && strcmp(argv[1], "--upstream") == 0) {
        int n = atoi(argv[2]);
        if (n < 16 || n > 1048560 || n % 16) return 2;
        return check_epoch(n, true, false) ? 0 : 1;
    }
    if (argc != 1) return 2;
    for (int n : {16, 32, 48, 128, 256, 1024, 4096})
        for (bool momentum : {false, true})
            if (!check_epoch(n, false, momentum)) return 1;
    for (int n : {16, 32, 65536, 1048560})
        if (!check_epoch(n, true, false)) return 1;
    return 0;
}
