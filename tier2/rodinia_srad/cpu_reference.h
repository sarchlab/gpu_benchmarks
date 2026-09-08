#pragma once
#include <algorithm>
#include <cmath>
#include <exception>
#include <thread>
#include <utility>
#include <vector>
// Independent host phases preserve the pinned Rodinia CPU float arithmetic.
struct SradRef { std::vector<float> J,C,E,W,S,N; };
template<class Function> static void srad_rows(int rows,unsigned workers,Function f) {
    // Reserve before starting any worker; row partitions are disjoint.
    std::vector<std::thread> threads;threads.reserve(workers-1);unsigned worker=1;
    auto part=[&](unsigned id){f((int)((size_t)rows*id/workers),(int)((size_t)rows*(id+1)/workers));};
    try {for(;worker<workers;++worker)threads.emplace_back(part,worker);}
    // Finish missing partitions on the caller and join previously started workers.
    catch(const std::exception&) {for(;worker<workers;++worker)part(worker);}
    part(0);for(auto& t:threads)t.join();
}
static SradRef cpu_srad(const float* initial,int rows,int cols,int steps,float lambda,unsigned requested=0) {
    const size_t count=(size_t)rows*cols;SradRef r;
    r.J.assign(initial,initial+count);r.C.resize(count);r.E.resize(count);r.W.resize(count);r.S.resize(count);r.N.resize(count);
    unsigned wanted=requested?requested:(count<512u*512u?1u:std::max(1u,std::thread::hardware_concurrency()));
    const unsigned workers=std::max(1u,std::min({16u,wanted,(unsigned)rows}));
    const int rr=std::min(128,rows),rc=std::min(128,cols),size_R=rr*rc;
    for(int iter=0;iter<steps;++iter) {
        float sum=0,sum2=0;
        for(int i=0;i<rr;++i)for(int j=0;j<rc;++j) {
            float tmp=r.J[(size_t)i*cols+j];sum+=tmp;sum2+=tmp*tmp;
        }
        float meanROI=sum/size_R;
        float varROI=(sum2/size_R)-meanROI*meanROI;
        float q0sqr=varROI/(meanROI*meanROI);
        srad_rows(rows,workers,[&](int begin,int end) {
            for(int i=begin;i<end;++i)for(int j=0;j<cols;++j) {
                size_t k=(size_t)i*cols+j;float Jc=r.J[k];
                r.N[k]=r.J[(size_t)std::max(0,i-1)*cols+j]-Jc;
                r.S[k]=r.J[(size_t)std::min(rows-1,i+1)*cols+j]-Jc;
                r.W[k]=r.J[(size_t)i*cols+std::max(0,j-1)]-Jc;
                r.E[k]=r.J[(size_t)i*cols+std::min(cols-1,j+1)]-Jc;
                float G2=(r.N[k]*r.N[k]+r.S[k]*r.S[k]+r.W[k]*r.W[k]+r.E[k]*r.E[k])/(Jc*Jc);
                float L=(r.N[k]+r.S[k]+r.W[k]+r.E[k])/Jc;
                float num=(0.5*G2)-((1.0/16.0)*(L*L));
                float den=1+(.25*L);
                float qsqr=num/(den*den);
                den=(qsqr-q0sqr)/(q0sqr*(1+q0sqr));
                r.C[k]=1.0/(1.0+den);
                if(r.C[k]<0)r.C[k]=0;else if(r.C[k]>1)r.C[k]=1;
            }
        });
        srad_rows(rows,workers,[&](int begin,int end) {
            for(int i=begin;i<end;++i)for(int j=0;j<cols;++j) {
                size_t k=(size_t)i*cols+j;
                float cN=r.C[k],cS=r.C[(size_t)std::min(rows-1,i+1)*cols+j];
                float cW=r.C[k],cE=r.C[(size_t)i*cols+std::min(cols-1,j+1)];
                float D=cN*r.N[k]+cS*r.S[k]+cW*r.W[k]+cE*r.E[k];
                r.J[k]=r.J[k]+0.25*lambda*D;
            }
        });
    }
    return r;
}

// Destruction releases the five unused fields before the final-step reference
// is allocated. The full trajectory always starts from the pristine image.
static std::vector<float> cpu_srad_image(const float* initial, int rows, int cols,
                                         int steps, float lambda) {
    auto reference = cpu_srad(initial, rows, cols, steps, lambda);
    return std::move(reference.J);
}

static bool srad_value_matches(float actual, float expected, double tolerance) {
    const double error = std::fabs((double)actual - expected);
    return std::isfinite(actual) && std::isfinite(expected) && std::isfinite(error)
        && error <= tolerance * std::max(1.0, std::fabs((double)expected));
}

static bool srad_coefficient_matches(float actual, float expected) {
    return actual >= 0 && actual <= 1 && expected >= 0 && expected <= 1
        && srad_value_matches(actual, expected, 1e-5);
}
