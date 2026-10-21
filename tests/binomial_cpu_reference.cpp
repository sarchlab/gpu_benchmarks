// Frozen pre-repair scalar recurrence from 8a4eb5c. Keep the per-node powf
// oracle independent of the production payoff cache and parallel scheduler.
#include "../tier2/tango_binomial_options/hip/cpu_reference.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static float original_price(OptionData opt, int numSteps) {
    float dt   = opt.T / (float)numSteps;
    float u    = expf(opt.sigma * sqrtf(dt));
    float d    = 1.0f / u;
    float R    = expf(opt.r * dt);
    float Rinv = 1.0f / R;
    float p    = (R - d) / (u - d);
    float q    = 1.0f - p;

    std::vector<float> vals((size_t)(numSteps + 1));

    // Terminal payoffs
    for (int j = 0; j <= numSteps; ++j) {
        float ST = opt.S * powf(u, (float)(2 * j - numSteps));
        vals[j] = fmaxf(opt.K - ST, 0.0f);
    }

    // Backward induction
    for (int step = numSteps; step > 0; --step) {
        for (int j = 0; j < step; ++j) {
            float cont = Rinv * (p * vals[j + 1] + q * vals[j]);
            float ST = opt.S * powf(u, (float)(2 * j - (step - 1)));
            float exercise = fmaxf(opt.K - ST, 0.0f);
            vals[j] = fmaxf(cont, exercise);
        }
    }

    return vals[0];
}


static void require(bool condition, const char* label) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
int main() {
    unsigned seed = 42;
    auto random = [&seed](float lo, float hi) {
        seed = seed * 1103515245u + 12345u;
        float t = (float)(seed & 0x7fffffffu) / (float)0x7fffffffu;
        return lo + t * (hi - lo);
    };
    std::vector<OptionData> options = {{5,300,10,.02f,1}, {200,1,.25f,.02f,.1f},
        {200,1,10,.02f,1}, {5,300,.25f,.02f,.1f}, {100,100,1,.02f,.2f}};
    for (int i=0;i<128;++i) {
        OptionData o;
        o.S=random(5,200); o.K=random(1,300); o.T=random(.25f,10);
        o.r=.02f; o.sigma=random(.1f,1);
        options.push_back(o);
    }
    const auto saved=options;
    std::vector<float> references;
    int comparisons=0;
    for (const auto& o: options) {
        for (int steps: {1,2,7,31,1024}) {
            float want=original_price(o,steps), got=binomial_reference::price(o,steps);
            require(std::isfinite(want) && !std::memcmp(&want,&got,sizeof(float)), "original recurrence bit equality");
            ++comparisons;
            if (steps==1024) references.push_back(want);
        }
    }
    // Large exponents may overflow, while the original put payoff stays zero.
    require(std::isinf(powf(expf(sqrtf(10.f/1024)),1024.f)), "overflowing payoff power control");
    int parallel=0;
    for (int count: {1,9,10,11,17,33,133}) {
        for (unsigned workers: {0u,1u,2u,3u,16u,128u}) {
            auto got=binomial_reference::prices(options.data(),count,workers);
            require(got.size()==static_cast<size_t>(count), "complete reference count");
            require(!std::memcmp(got.data(),references.data(),count*sizeof(float)), "parallel exact complete prices");
            ++parallel;
        }
    }
    require(!std::memcmp(options.data(),saved.data(),options.size()*sizeof(OptionData)), "input preservation");
    const float nan=std::numeric_limits<float>::quiet_NaN(), inf=std::numeric_limits<float>::infinity();
    int selections=0;
    for (float a: {nan,-inf,-1.f,-0.f,0.f,1.f,inf}) {
        for (float b: {nan,-inf,-1.f,-0.f,0.f,1.f,inf}) {
            float got=binomial_reference::maximum(a,b), want=std::fmax(a,b);
            require(std::isnan(want) ? std::isnan(got) : got==want, "maximum finite/nonfinite selection");
            ++selections;
        }
    }
    int rejections=0;
    for (int steps: {-1,0,1025,std::numeric_limits<int>::max()}) {
        try { binomial_reference::price(options[0],steps); require(false,"invalid step acceptance"); }
        catch (const std::invalid_argument&) { ++rejections; }
    }
    for (int count: {-1,0}) {
        try { binomial_reference::prices(options.data(),count); require(false,"invalid count acceptance"); }
        catch (const std::invalid_argument&) { ++rejections; }
    }
    std::printf("%d original-recurrence comparisons, %d complete parallel cases, %d maximum selections, %d invalid-reference rejections pass\n", comparisons, parallel, selections, rejections);
}
