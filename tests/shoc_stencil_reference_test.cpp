#include "../tier2/shoc_stencil2d/cpu_reference.h"
#include <cmath>
#include <cstdio>
#include <cstring>

// Independently indexed form of the existing qualified single-step formula.
// Keep its float grouping while validating the complete new ping-pong helper.
static bool check_case(int n, int steps, int pattern) {
    const size_t cells=(size_t)n*n;
    std::vector<float> input(cells,0.0f);
    for (int i=1;i<n-1;++i) for (int j=1;j<n-1;++j) {
        const size_t p=(size_t)i*n+j;
        input[p]=pattern==0?float((i*17+j*31)%101)/8.0f:
                 pattern==1?float((i*13+j*19)%67-33)/16.0f:
                 pattern==2?(i==n/2 && j==n/2?1.0f:0.0f):0.0f;
    }
    const auto original=input;
    std::vector<float> state=input,scratch=input;
    for (int t=0;t<steps;++t) {
        for (size_t p=0;p<cells;++p) {
            const int row=p/n,col=p%n;
            if (!row || !col || row==n-1 || col==n-1) continue;
            scratch[p]=0.5f*state[p]
                     + 0.1f*(state[p-n]+state[p+n]+state[p-1]+state[p+1])
                     + 0.025f*(state[p-n-1]+state[p-n+1]+state[p+n-1]+state[p+n+1]);
        }
        state.swap(scratch);
    }
    const auto actual=cpu_stencil2d_pass(input.data(),n,steps);
    if (input!=original || actual.size()!=cells) return false;
    if (std::memcmp(actual.data(),state.data(),cells*sizeof(float))) {
        fprintf(stderr,"FAIL N%d steps%d pattern%d: reference differs\n",n,steps,pattern);return false;
    }
    for (size_t p=0;p<cells;++p) {
        const int row=p/n,col=p%n;
        if (!std::isfinite(actual[p]) ||
            ((!row || !col || row==n-1 || col==n-1) && actual[p]!=0)) return false;
    }
    return true;
}

int main() {
    int cases=0;
    for (int n:{3,4,5,7,16,17,31,32,33,64,65,129})
        for (int steps:{1,2,5,50,51}) for (int pattern=0;pattern<4;++pattern) {
            if (!check_case(n,steps,pattern)) return 1;
            ++cases;
        }
    // Exact one-step coefficients: isolate all nine input positions in a3x3
    // grid. This step-only oracle also checks fixed output boundary ownership.
    for (int at=0;at<9;++at) {
        std::vector<float> input(9,0),output(9,0);input[at]=1;
        cpu_stencil2d(input.data(),output.data(),3);
        const float expected=at==4?0.5f:(at==1 || at==3 || at==5 || at==7)?0.1f:0.025f;
        if (output[4]!=expected) return 1;
        for (int p=0;p<9;++p) if (p!=4 && output[p]!=0) return 1;
    }
    printf("PASS: %d complete float-reference states bit-identical, original input and zero boundaries preserved; nine exact coefficient impulses\n",cases);
    return 0;
}
