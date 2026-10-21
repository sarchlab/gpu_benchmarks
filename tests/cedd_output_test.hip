// Test actual GPU stages and the production timed pass with independent host
// integer/double references, complete byte comparisons and allocation guards.
#define main benchmark_main
#include "../tier2/chai_cedd/hip/chai_cedd.hip"
#undef main

static unsigned char direction_ref(int gx,int gy) {
    double angle=std::atan2((double)gy,(double)gx);
    if (angle<0) angle+=2*std::acos(-1.0);
    return (unsigned char)(((int)std::floor(angle/(std::acos(-1.0)/4)+0.5)%4)*45);
}

static std::vector<unsigned char> stage_ref(int stage,const std::vector<unsigned char>& input,
                                           std::vector<unsigned char>& theta,int w,int h) {
    std::vector<unsigned char> out(input.size(),0);
    const int gx_weights[9]={-1,0,1,-2,0,2,-1,0,1};
    const int gy_weights[9]={-1,-2,-1,0,0,0,1,2,1};
    const int divisors[9]={16,8,16,8,4,8,16,8,16};
    for (int y=1;y<h-1;++y) for (int x=1;x<w-1;++x) {
        size_t p=(size_t)y*w+x;
        if (stage==0 || stage==1) {
            int sum=0,gx=0,gy=0,t=0;
            for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx,++t) {
                int v=input[(size_t)(y+dy)*w+x+dx];
                sum+=v/divisors[t]; gx+=v*gx_weights[t]; gy+=v*gy_weights[t];
            }
            out[p]=stage==0?sum:std::min(255,(int)std::hypot((double)gx,(double)gy));
            if (stage==1) theta[p]=direction_ref(gx,gy);
        } else if (stage==2) {
            int dy=theta[p]==0?0:1;
            int dx=theta[p]==0?1:theta[p]==45?-1:theta[p]==90?0:1;
            unsigned char a=input[(size_t)(y+dy)*w+x+dx];
            unsigned char b=input[(size_t)(y-dy)*w+x-dx];
            out[p]=(input[p]>a && input[p]>b)?input[p]:0;
        } else out[p]=input[p]>=40?255:0;
    }
    return out;
}

static bool equal_bytes(const std::vector<unsigned char>& actual,
                        const std::vector<unsigned char>& expected,const char* label) {
    for (size_t i=0;i<expected.size();++i) if (actual[i]!=expected[i]) {
        fprintf(stderr,"FAIL: %s[%zu] got %u expected %u\n",label,i,(unsigned)actual[i],(unsigned)expected[i]);
        return false;
    }
    return true;
}

static bool check_case(int w,int h,int pattern) {
    size_t n=(size_t)w*h;
    std::vector<unsigned char> input(n),theta(n);
    for (size_t i=0;i<n;++i) {
        input[i]=pattern==0?15:pattern==1?(unsigned char)i:pattern==2?(unsigned char)((i*17+i/w*73)%256):((i/w+i%w)%2?255:0);
        theta[i]=(unsigned char)((i%4)*45);
    }
    dim3 block(CEDD_TILE,CEDD_TILE),grid((w-2+CEDD_TILE-1)/CEDD_TILE,(h-2+CEDD_TILE-1)/CEDD_TILE);
    const size_t shared_bytes=cedd_shared_bytes(block);
    unsigned char *da,*db,*dt;
    HIP_CHECK(hipMalloc(&da,n+2));HIP_CHECK(hipMalloc(&db,n+2));HIP_CHECK(hipMalloc(&dt,n+2));
    bool ok=true;
    for (int stage=0;stage<4 && ok;++stage) {
        std::vector<unsigned char> expected_theta=theta;
        auto expected=stage_ref(stage,input,expected_theta,w,h);
        std::vector<unsigned char> a(n+2,165),b(n+2,165),t(n+2,165),expected_b(n+2,165),expected_t(n+2,165);
        std::copy(input.begin(),input.end(),a.begin()+1);
        std::copy(theta.begin(),theta.end(),t.begin()+1);
        std::copy(theta.begin(),theta.end(),expected_t.begin()+1);
        for (int y=1;y<h-1;++y) for (int x=1;x<w-1;++x) {
            size_t p=(size_t)y*w+x;
            b[p+1]=(unsigned char)~expected[p];expected_b[p+1]=expected[p];
            if (stage==1) {t[p+1]=(unsigned char)~expected_theta[p];expected_t[p+1]=expected_theta[p];}
        }
        HIP_CHECK(hipMemcpy(da,a.data(),n+2,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db,b.data(),n+2,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dt,t.data(),n+2,hipMemcpyHostToDevice));
        if (stage==0) gaussian_kernel<<<grid,block,shared_bytes>>>(da+1,db+1,h,w);
        else if (stage==1) sobel_kernel<<<grid,block,shared_bytes>>>(da+1,db+1,dt+1,h,w);
        else if (stage==2) non_max_supp_kernel<<<grid,block,shared_bytes>>>(da+1,db+1,dt+1,h,w);
        else hyst_kernel<<<grid,block>>>(da+1,db+1,h,w);
        HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(b.data(),db,n+2,hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(t.data(),dt,n+2,hipMemcpyDeviceToHost));
        ok=equal_bytes(b,expected_b,"stage output/guard") && equal_bytes(t,expected_t,"theta/guard");
    }
    if (ok) {
        // Independent stage references, preserving the original A/B boundaries.
        std::vector<unsigned char> cpu_a=input,cpu_b(n,0),cpu_t(n,0);
        for (int stage=0;stage<4;++stage) {
            auto next=stage_ref(stage,stage%2?cpu_b:cpu_a,cpu_t,w,h);
            auto& dest=stage%2?cpu_a:cpu_b;
            for (int y=1;y<h-1;++y) for (int x=1;x<w-1;++x) dest[(size_t)y*w+x]=next[(size_t)y*w+x];
        }
        std::vector<unsigned char> a(n+2,165),b(n+2,165),t(n+2,165);
        std::copy(input.begin(),input.end(),a.begin()+1);
        std::fill(b.begin()+1,b.end()-1,0);std::fill(t.begin()+1,t.end()-1,0);
        HIP_CHECK(hipMemcpy(da,a.data(),n+2,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db,b.data(),n+2,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dt,t.data(),n+2,hipMemcpyHostToDevice));
        run_cedd_pass(da+1,db+1,dt+1,w,h);
        HIP_CHECK(hipMemcpy(a.data(),da,n+2,hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(b.data(),db,n+2,hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(t.data(),dt,n+2,hipMemcpyDeviceToHost));
        std::vector<unsigned char> result(a.begin()+1,a.end()-1);
        ok=a.front()==165 && a.back()==165 && b.front()==165 && b.back()==165 && t.front()==165 && t.back()==165;
        if (!ok) fprintf(stderr,"FAIL: pipeline allocation guard\n");
        ok=ok && equal_bytes(result,cpu_a,"pipeline") && verify_cedd(input,result,w,h);
    }
    HIP_CHECK(hipFree(da));HIP_CHECK(hipFree(db));HIP_CHECK(hipFree(dt));
    if (ok) printf("PASS: four stages and production pass W=%d H=%d pattern=%d\n",w,h,pattern);
    return ok;
}

__global__ void gradient_domain(unsigned char* magnitude,unsigned char* theta,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) cedd_gradient((float)(i%2041-1020),(float)(i/2041-1020),magnitude+i,theta+i);
}

static bool check_gradient_domain() {
    const int count=2041*2041;
    std::vector<unsigned char> magnitude(count),theta(count);
    unsigned char *dm,*dt;HIP_CHECK(hipMalloc(&dm,count));HIP_CHECK(hipMalloc(&dt,count));
    gradient_domain<<<(count+255)/256,256>>>(dm,dt,count);
    HIP_CHECK(hipGetLastError());HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(magnitude.data(),dm,count,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(theta.data(),dt,count,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dm));HIP_CHECK(hipFree(dt));
    int checked=0;
    for (int i=0;i<count;++i) {
        int gx=i%2041-1020,gy=i/2041-1020;
        // Both Sobel sums have the same four corner pixels modulo two.
        // Opposite-parity pairs cannot arise from any byte-valued 3x3 patch.
        if ((gx-gy)%2) continue;
        ++checked;
        int expected=std::min(255,(int)std::hypot((double)gx,(double)gy));
        if (magnitude[i]!=expected || theta[i]!=direction_ref(gx,gy)) {
            fprintf(stderr,"FAIL: gradient (%d,%d): magnitude=%u direction=%u expected %d/%u\n",gx,gy,(unsigned)magnitude[i],(unsigned)theta[i],expected,(unsigned)direction_ref(gx,gy));return false;
        }
    }
    if (checked!=2082841) {
        fprintf(stderr,"FAIL: checked %d gradient pairs, expected 2082841\n",checked);return false;
    }
    printf("PASS: all %d same-parity Sobel gradient pairs\n",checked);return true;
}

int main(int argc,char** argv) {
    setenv("BENCH_PARAM_vector_add_warmup","false",1);
    if (argc==4 && !strcmp(argv[1],"--case")) {
        int w,h;
        if (!read_dimension("width",argv[2],&w) || !read_dimension("height",argv[3],&h) || w>16384 || h>4096) return 2;
        return check_case(w,h,2)?0:1;
    }
    if (argc!=1) return 2;
    const int shapes[][2]={{3,3},{4,19},{17,19},{18,18},{34,50},{33,65},{64,512},{127,131}};
    for (const auto& s:shapes) for (int p=0;p<4;++p) if (!check_case(s[0],s[1],p)) return 1;
    return check_gradient_domain()?0:1;
}
