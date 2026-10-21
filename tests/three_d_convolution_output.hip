// Check the production GPU operator with independent spatial basis weights.
#define main benchmark_main
#include "../tier2/polybench_3dconv/hip/polybench_3dconv.hip"
#undef main
#include <limits>
#include <vector>

static bool check_volume(int N, int B, int impulse) {
    const size_t cells = (size_t)N*N*N;
    const float guard = -12345.125f;
    std::vector<float> input(cells+2,guard), output(cells+2,guard);
    for (int i=0; i<N; ++i) for (int j=0; j<N; ++j) for (int k=0; k<N; ++k) {
        size_t n=1+((size_t)i*N+j)*N+k;
        input[n] = impulse >= 0 ? 0.0f : (float)((i*17+j*13+k*7)%101-50)/8.0f;
        if (i>0 && j>0 && k>0 && i<N-1 && j<N-1 && k<N-1)
            output[n] = std::numeric_limits<float>::quiet_NaN();
    }
    if (impulse >= 0) input[1+impulse]=1.0f;
    float *di,*dout;
    const size_t bytes=input.size()*sizeof(float);
    HIP_CHECK(hipMalloc(&di,bytes));
    HIP_CHECK(hipMalloc(&dout,bytes));
    HIP_CHECK(hipMemcpy(di,input.data(),bytes,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dout,output.data(),bytes,hipMemcpyHostToDevice));
    conv3d_kernel<<<dim3((N+B-1)/B,(N+B-1)/B,(N+B-1)/B),dim3(B,B,B)>>>(di+1,dout+1,N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(output.data(),dout,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(di));
    HIP_CHECK(hipFree(dout));
    if (output.front()!=guard || output.back()!=guard) {
        fprintf(stderr,"FAIL: allocation guard overwritten\n"); return false;
    }
    // Independent impulse response indexed by (di+1,dj+1,dk+1).
    // Combine upstream's repeated coordinates; all other spatial taps are zero.
    const int weights[27]={-1,0,2,0,0,5,0,0,-8,0,-3,0,0,6,0,0,-9,0,21,0,4,0,0,7,0,0,10};
    for (int i=0; i<N; ++i) for (int j=0; j<N; ++j) for (int k=0; k<N; ++k) {
        float actual=output[1+((size_t)i*N+j)*N+k];
        bool interior=i>0 && j>0 && k>0 && i<N-1 && j<N-1 && k<N-1;
        double expected=guard;
        if (interior) {
            expected=0;
            for (int a=-1; a<=1; ++a) for (int b=-1; b<=1; ++b) for (int c=-1; c<=1; ++c)
                expected += weights[(a+1)*9+(b+1)*3+c+1]*input[1+((size_t)(i+a)*N+j+b)*N+k+c];
        }
        // The basis and signed multiples of 1/8 give exactly representable
        // products and partial sums. No numerical tolerance is needed here.
        if (!std::isfinite(actual) || actual!=expected) {
            fprintf(stderr,"FAIL: N=%d B=%d impulse=%d at %d,%d,%d: %.9g expected %.12g\n",
                    N,B,impulse,i,j,k,actual,expected); return false;
        }
    }
    printf("PASS: N=%d B=%d impulse=%d\n",N,B,impulse);
    return true;
}

int main(int argc,char** argv) {
    if (argc==3 && strcmp(argv[1],"--impulse")==0) {
        char* end=nullptr; long t=strtol(argv[2],&end,10);
        if (end==argv[2] || *end || t<0 || t>26) return 2;
        return check_volume(3,8,(int)t)?0:1;
    }
    if (argc==4 && strcmp(argv[1],"--case")==0) {
        int n,b;
        if (!read_int("N",argv[2],&n) || !read_int("B",argv[3],&b) || n<3 || n>512 || b>10) return 2;
        return check_volume(n,b,-1)?0:1;
    }
    if (argc!=1) return 2;
    for (int t=0;t<27;++t) if (!check_volume(3,8,t)) return 1;
    const int shapes[]={3,5,7,8,9,15,17,31,33,65};
    const int blocks[]={1,2,4,8};
    for (int n:shapes) for (int b:blocks) if (!check_volume(n,b,-1)) return 1;
    return 0;
}
