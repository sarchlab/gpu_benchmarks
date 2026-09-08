#include "../tier2/parboil_stencil/cpu_reference.h"
#include <cstdio>
#include <limits>

// A dense transition matrix independently expresses the seven-point operator.
// Boundary rows are identity; each interior row has six equal neighbors and
// a negative diagonal. This oracle does not share the reference's stencil loop.
static bool check_case(int nx, int ny, int nz, int steps, int pattern) {
    const int cells = nx*ny*nz;
    std::vector<float> input(cells);
    for (int i=0; i<cells; ++i)
        input[i] = pattern >= 0 ? (i==pattern ? 1.0f : 0.0f) : float((i*37)%101-50)/16.0f;
    const auto original = input;
    std::vector<double> matrix((size_t)cells*cells,0), state(input.begin(),input.end());
    for (int i=0; i<cells; ++i) {
        const int x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
        if (!x || x==nx-1 || !y || y==ny-1 || !z || z==nz-1) {
            matrix[(size_t)i*cells+i]=1;
        } else {
            matrix[(size_t)i*cells+i]=-double(1.0f/6.0f);
            for (int d : {-nx*ny,-nx,-1,1,nx,nx*ny})
                matrix[(size_t)i*cells+i+d]=double(1.0f/6.0f/6.0f);
        }
    }
    for (int t=0; t<steps; ++t) {
        std::vector<double> next(cells,0);
        for (int i=0; i<cells; ++i)
            for (int j=0; j<cells; ++j) next[i]+=matrix[(size_t)i*cells+j]*state[j];
        state.swap(next);
    }
    const auto actual = cpu_stencil(input.data(),nx,ny,nz,steps);
    if (input != original || actual.size() != state.size()) return false;
    for (int i=0; i<cells; ++i) {
        const int x=i%nx, y=(i/nx)%ny, z=i/(nx*ny);
        const bool boundary=!x || x==nx-1 || !y || y==ny-1 || !z || z==nz-1;
        if (!std::isfinite(actual[i]) || std::fabs(actual[i]-state[i]) > 1e-14 ||
            (boundary && actual[i] != input[i])) {
            fprintf(stderr,"FAIL %dx%dx%d T%d pattern%d at%d: %.17g expected %.17g\n",
                    nx,ny,nz,steps,pattern,i,actual[i],state[i]); return false;
        }
    }
    return true;
}

int main() {
    int cases=0;
    const int shapes[][3]={{3,3,3},{3,5,4},{4,4,4},{5,4,3}};
    for (const auto& s : shapes)
        for (int t : {1,2,5,10,100})
            for (int pattern : {-1,0,s[0]*s[1]*s[2]/2,s[0]*s[1]*s[2]-1}) {
                if (!check_case(s[0],s[1],s[2],t,pattern)) return 1;
                ++cases;
            }
    // Every basis vector catches missing/misplaced stencil terms and boundaries.
    for (int i=0; i<27; ++i) {
        if (!check_case(3,3,3,1,i)) return 1;
        ++cases;
    }
    if (!stencil_value_matches(0,0,false) || !stencil_value_matches(1,1,true) ||
        !stencil_value_matches(5e-8f,0,false) || stencil_value_matches(2e-7f,0,false) ||
        !stencil_value_matches(1.00001f,1,false) || stencil_value_matches(1.00003f,1,false) ||
        stencil_value_matches(std::nextafter(1.0f,2.0f),1,true)) return 1;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})
        for (bool boundary : {false,true})
            if (stencil_value_matches(bad,0,boundary) || stencil_value_matches(0,bad,boundary) ||
                stencil_value_matches(bad,bad,boundary)) return 1;
    printf("PASS: %d complete dense-matrix references; input and exact boundaries preserved; finite/tolerance predicates checked\n",cases);
    return 0;
}
