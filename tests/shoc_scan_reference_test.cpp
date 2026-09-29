#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

static unsigned fixtures = 0;

static void check_period(int n, const std::vector<unsigned>& period) {
    const float guard = -1234.25f;
    std::vector<float> input(n + 2, guard), output(n + 2, guard);
    uint64_t period_sum = 0;
    for (unsigned x : period) period_sum += x;
    for (int i = 0; i < n; ++i) input[i + 1] = (float)period[i % period.size()];
    scan_cpu(output.data() + 1, input.data() + 1, n);
    assert(input.front() == guard && input.back() == guard);
    assert(output.front() == guard && output.back() == guard);
    for (int i = 0; i < n; ++i) {
        // Independent exact integer oracle: whole periods plus a short tail.
        uint64_t exact = (i / period.size()) * period_sum;
        for (size_t j = 0; j < i % period.size(); ++j) exact += period[j];
        assert(output[i + 1] == (float)exact);
        assert(input[i + 1] == (float)period[i % period.size()]);
    }
    ++fixtures;
}

int main() {
    const float known[] = {1, 2, 3, 4};
    float got[4];
    scan_cpu(got, known, 4);
    assert(got[0] == 0 && got[1] == 1 && got[2] == 3 && got[3] == 6);
    const std::vector<std::vector<unsigned>> periods = {
        {1}, {7}, {1, 2}, {7, 1, 6, 2}, {1, 2, 3, 4, 5, 6, 7}
    };
    for (const auto& period : periods) {
        for (int n = 0; n <= 64; ++n) check_period(n, period);
        for (int n : {511, 512, 513, 1023, 1024, 1025, 65535, 65536, 65537})
            check_period(n, period);
    }
    for (int n : {4194304, 4194305, 8388608}) check_period(n, periods.back());
    check_period(16777220, periods.front());
    std::printf("%u complete prefix fixtures plus known answer passed; input and output guards intact\n", fixtures);
}
