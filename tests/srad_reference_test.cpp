// Serial oracle phase body from Rodinia 3.1 srad_v2/srad.cu, commit
// 366b283456506ef2fe2c2b7dc5e83e941ffc5524; wrapper uses the suite ROI.
#include <cstdio>
#include <limits>
#include "../tier2/rodinia_srad/cpu_reference.h"
struct Original {std::vector<float> J,c,dN,dS,dW,dE;};
static Original original(const std::vector<float>& input,int rows,int cols,int niter,float lambda) {
    Original result;result.J=input;size_t n=input.size();result.c.resize(n);result.dN.resize(n);result.dS.resize(n);result.dW.resize(n);result.dE.resize(n);
    auto& J=result.J;auto& c=result.c;auto& dN=result.dN;auto& dS=result.dS;auto& dW=result.dW;auto& dE=result.dE;
    std::vector<int>iN(rows),iS(rows),jW(cols),jE(cols);
    for(int i=0;i<rows;++i){iN[i]=std::max(0,i-1);iS[i]=std::min(rows-1,i+1);}
    for(int j=0;j<cols;++j){jW[j]=std::max(0,j-1);jE[j]=std::min(cols-1,j+1);}
    float Jc,G2,L,num,den,qsqr,cN,cS,cW,cE,D,q0sqr,sum,sum2,tmp,meanROI,varROI;int k;
    const int r1=0,r2=std::min(127,rows-1),c1=0,c2=std::min(127,cols-1),size_R=(r2+1)*(c2+1);
    for(int iter=0;iter<niter;++iter) {
        sum=0;sum2=0;
        for(int i=r1;i<=r2;++i)for(int j=c1;j<=c2;++j){tmp=J[i*cols+j];sum+=tmp;sum2+=tmp*tmp;}
        meanROI=sum/size_R;varROI=(sum2/size_R)-meanROI*meanROI;q0sqr=varROI/(meanROI*meanROI);
        for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) {

                k = i * cols + j;
                Jc = J[k];

                // directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;

                G2 = (dN[k]*dN[k] + dS[k]*dS[k]
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);

                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;

                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
        }
    }
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                // current index
                k = i * cols + j;

                // diffusion coefficent
                    cN = c[k];
                    cS = c[iS[i] * cols + j];
                    cW = c[k];
                    cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
    }


    }
    return result;
}
int main() {
    int cases=0;
    for(int rows:{2,3,16,17,33})for(int cols:{2,5,16,17,33})for(int steps:{1,2,10}) for(float lambda:{.25f,.5f,.75f}) {
        std::vector<float>a((size_t)rows*cols);for(size_t q=0;q<a.size();++q)a[q]=1.0f+float((q*17+q/cols*3)%101)/64;
        const auto input=a;const auto expected=original(a,rows,cols,steps,lambda);
        for(unsigned workers:{1u,4u,16u}) {
            auto r=cpu_srad(a.data(),rows,cols,steps,lambda,workers);if(a!=input)return 1;
            const std::vector<float>* actual[]={&r.J,&r.C,&r.E,&r.W,&r.S,&r.N};
            const std::vector<float>* ref[]={&expected.J,&expected.c,&expected.dE,&expected.dW,&expected.dS,&expected.dN};
            for(int s=0;s<6;++s)for(size_t q=0;q<a.size();++q)
                if(!std::isfinite((*actual[s])[q])||(*actual[s])[q]!=(*ref[s])[q]){printf("FAIL %d %d %d %u stage%d at%zu %.9g %.9g\n",rows,cols,steps,workers,s,q,(*actual[s])[q],(*ref[s])[q]);return 1;}
        }
        if(cpu_srad_image(a.data(),rows,cols,steps,lambda)!=expected.J || a!=input)return 1;
        ++cases;
    }
    for (double tolerance : {1e-3,1e-5}) {
        if (!srad_value_matches(0,0,tolerance)
            || !srad_value_matches(float(tolerance*.5),0,tolerance)
            || srad_value_matches(float(tolerance*2),0,tolerance)
            || !srad_value_matches(float(100+100*tolerance*.5),100,tolerance)
            || srad_value_matches(float(100+100*tolerance*2),100,tolerance)
            || !srad_value_matches(float(-100-100*tolerance*.5),-100,tolerance)
            || srad_value_matches(float(-100-100*tolerance*2),-100,tolerance)) return 1;
    }
    for (float value : {0.0f,1.0f,-1.0f,100.0f}) {
        if (!srad_value_matches(value,value,0)
            || srad_value_matches(std::nextafter(value,std::numeric_limits<float>::infinity()),value,0)) return 1;
    }
    if (!srad_coefficient_matches(0,0) || !srad_coefficient_matches(1,1)
        || !srad_coefficient_matches(.500001f,.5f) || srad_coefficient_matches(.51f,.5f)
        || srad_coefficient_matches(std::nextafter(0.0f,-1.0f),0)
        || srad_coefficient_matches(std::nextafter(1.0f,2.0f),1)
        || srad_coefficient_matches(0,std::nextafter(0.0f,-1.0f))
        || srad_coefficient_matches(1,std::nextafter(1.0f,2.0f))) return 1;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}) {
        for (double tolerance : {0.0,1e-5,1e-3})
            if (srad_value_matches(bad,0,tolerance) || srad_value_matches(0,bad,tolerance)
                || srad_value_matches(bad,bad,tolerance)) return 1;
        if (srad_coefficient_matches(bad,0) || srad_coefficient_matches(0,bad)) return 1;
    }
    printf("PASS: %d complete rectangular pinned-CPU comparisons, identical 1/4/16-worker results, inputs preserved\n",cases);
}
