#pragma once
/*
 * ------------------------------------------------------------------------------
 *
 * MIT License
 *
 * Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
 *      GMAP website: https://gmap.pucrs.br
 *
 * Pontifical Catholic University of Rio Grande do Sul (PUCRS)
 * Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ------------------------------------------------------------------------------
 *
 * The original NPB 3.4 version was written in Fortran and belongs to:
 *      http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors of the Fortran code:
 *      E. Barszcz
 *      P. Frederickson
 *      A. Woo
 *      M. Yarrow
 *
 * ------------------------------------------------------------------------------
 *
 * The serial C++ version is a translation of the original NPB 3.4
 * Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
 *
 * Authors of the C++ code:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * ------------------------------------------------------------------------------
 *
 * The hip version is a parallel implementation of the serial C++ version
 * hip version: https://github.com/GMAP/NPB-GPU/tree/master/hip
 *
 * Authors of the hip code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */
// CPU stencil bodies from HIP/MG/mg.cpp at the README pin. Portable views and
// bounded independent plane execution replace runtime-sized C++ array casts.
// Per-cell expression order and phase barriers are preserved.
#include "nas_mg.h"
namespace nas_mg {
static void comm3(void* pointer_u,
		int n1,
		int n2,
		int n3,
		int kk){
    View u{static_cast<double*>(pointer_u),n1,n2};

	int i1, i2, i3;
	/* axis = 1 */
	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			u[i3][i2][0] = u[i3][i2][n1-2];
			u[i3][i2][n1-1] = u[i3][i2][1];
		}
	}
	/* axis = 2 */
	for(i3 = 1; i3 < n3-1; i3++){
		for(i1 = 0; i1 < n1; i1++){
			u[i3][0][i1] = u[i3][n2-2][i1];
			u[i3][n2-1][i1] = u[i3][1][i1];
		}
	}
	/* axis = 3 */
	for(i2 = 0; i2 < n2; i2++){
		for(i1 = 0; i1 < n1; i1++){
			u[0][i2][i1] = u[n3-2][i2][i1];
			u[n3-1][i2][i1] = u[1][i2][i1];
		}
	}
}

static void interp(void* pointer_z,
		int mm1,
		int mm2,
		int mm3,
		void* pointer_u,
		int n1,
		int n2,
		int n3,
		int k, PlanePool& pool){
    View z{static_cast<double*>(pointer_z),mm1,mm2};
    View u{static_cast<double*>(pointer_u),n1,n2};

    pool.run(0,mm3-1,[&](int begin,int end) {
        int i3,i2,i1;
        double z1[514],z2[514],z3[514];
		for(i3 = begin; i3 < end; i3++){
			for(i2 = 0; i2 < mm2-1; i2++){
				for(i1 = 0; i1 < mm1; i1++){
					z1[i1] = z[i3][i2+1][i1] + z[i3][i2][i1];
					z2[i1] = z[i3+1][i2][i1] + z[i3][i2][i1];
					z3[i1] = z[i3+1][i2+1][i1] + z[i3+1][i2][i1] + z1[i1];
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3][2*i2][2*i1] = u[2*i3][2*i2][2*i1]
						+z[i3][i2][i1];
					u[2*i3][2*i2][2*i1+1] = u[2*i3][2*i2][2*i1+1]
						+0.5*(z[i3][i2][i1+1]+z[i3][i2][i1]);
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3][2*i2+1][2*i1] = u[2*i3][2*i2+1][2*i1]
						+0.5 * z1[i1];
					u[2*i3][2*i2+1][2*i1+1] = u[2*i3][2*i2+1][2*i1+1]
						+0.25*( z1[i1] + z1[i1+1] );
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3+1][2*i2][2*i1] = u[2*i3+1][2*i2][2*i1]
						+0.5 * z2[i1];
					u[2*i3+1][2*i2][2*i1+1] = u[2*i3+1][2*i2][2*i1+1]
						+0.25*( z2[i1] + z2[i1+1] );
				}
				for(i1 = 0; i1 < mm1-1; i1++){
					u[2*i3+1][2*i2+1][2*i1] = u[2*i3+1][2*i2+1][2*i1]
						+0.25* z3[i1];
					u[2*i3+1][2*i2+1][2*i1+1] = u[2*i3+1][2*i2+1][2*i1+1]
						+0.125*( z3[i1] + z3[i1+1] );
				}
			}
		}
    });
}

static void norm2u3(void* pointer_r,
		int n1,
		int n2,
		int n3,
		double* rnm2,
		double* rnmu,
		int nx,
		int ny,
		int nz){
    View r{static_cast<double*>(pointer_r),n1,n2};

	double s, a;
	int i3, i2, i1;

	double dn;

	dn = 1.0*nx*ny*nz;

	s = 0.0;
	*rnmu = 0.0;
	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 1; i1 < n1-1; i1++){
				s = s + r[i3][i2][i1] * r[i3][i2][i1];
				a = fabs(r[i3][i2][i1]);
				if(a > *rnmu){*rnmu = a;}
			}
		}
	}

	*rnm2 = sqrt(s/dn);
}

static void psinv(void* pointer_r,
		void* pointer_u,
		int n1,
		int n2,
		int n3,
		double c[4],
		int k, PlanePool& pool){
    View r{static_cast<double*>(pointer_r),n1,n2};
    View u{static_cast<double*>(pointer_u),n1,n2};

    pool.run(1,n3-1,[&](int begin,int end) {

	int i3, i2, i1;
	double r1[514], r2[514];

	for(i3 = begin; i3 < end; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 0; i1 < n1; i1++){
				r1[i1] = r[i3][i2-1][i1] + r[i3][i2+1][i1]
					+ r[i3-1][i2][i1] + r[i3+1][i2][i1];
				r2[i1] = r[i3-1][i2-1][i1] + r[i3-1][i2+1][i1]
					+ r[i3+1][i2-1][i1] + r[i3+1][i2+1][i1];
			}
			for(i1 = 1; i1 < n1-1; i1++){
				u[i3][i2][i1] = u[i3][i2][i1]
					+ c[0] * r[i3][i2][i1]
					+ c[1] * ( r[i3][i2][i1-1] + r[i3][i2][i1+1]
							+ r1[i1] )
					+ c[2] * ( r2[i1] + r1[i1-1] + r1[i1+1] );
				/*
				 * --------------------------------------------------------------------
				 * assume c(3) = 0    (enable line below if c(3) not= 0)
				 * --------------------------------------------------------------------
				 * > + c(3) * ( r2(i1-1) + r2(i1+1) )
				 * --------------------------------------------------------------------
				 */
			}
		}
	}

    });

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary points
	 * --------------------------------------------------------------------
	 */
	comm3(u.data,n1,n2,n3,k);

}

static void resid(void* pointer_u,
		void* pointer_v,
		void* pointer_r,
		int n1,
		int n2,
		int n3,
		double a[4],
		int k, PlanePool& pool){
    View u{static_cast<double*>(pointer_u),n1,n2};
    View v{static_cast<double*>(pointer_v),n1,n2};
    View r{static_cast<double*>(pointer_r),n1,n2};

    pool.run(1,n3-1,[&](int begin,int end) {

	int i3, i2, i1;
	double u1[514], u2[514];

	for(i3 = begin; i3 < end; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 0; i1 < n1; i1++){
				u1[i1] = u[i3][i2-1][i1] + u[i3][i2+1][i1]
					+ u[i3-1][i2][i1] + u[i3+1][i2][i1];
				u2[i1] = u[i3-1][i2-1][i1] + u[i3-1][i2+1][i1]
					+ u[i3+1][i2-1][i1] + u[i3+1][i2+1][i1];
			}
			for(i1 = 1; i1 < n1-1; i1++){
				r[i3][i2][i1] = v[i3][i2][i1]
					- a[0] * u[i3][i2][i1]
					/*
					 * ---------------------------------------------------------------------
					 * assume a(1) = 0 (enable 2 lines below if a(1) not= 0)
					 * ---------------------------------------------------------------------
					 * > - a(1) * ( u(i1-1,i2,i3) + u(i1+1,i2,i3)
					 * > + u1(i1) )
					 * ---------------------------------------------------------------------
					 */
					- a[2] * ( u2[i1] + u1[i1-1] + u1[i1+1] )
					- a[3] * ( u2[i1-1] + u2[i1+1] );
			}
		}
	}

    });

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary data
	 * --------------------------------------------------------------------
	 */
	comm3(r.data,n1,n2,n3,k);

}

static void rprj3(void* pointer_r,
		int m1k,
		int m2k,
		int m3k,
		void* pointer_s,
		int m1j,
		int m2j,
		int m3j,
		int k, PlanePool& pool){
    View r{static_cast<double*>(pointer_r),m1k,m2k};
    View s{static_cast<double*>(pointer_s),m1j,m2j};

    pool.run(1,m3j-1,[&](int begin,int end) {

	int j3, j2, j1, i3, i2, i1, d1, d2, d3;

	double x1[514], y1[514], x2, y2;

	if(m1k == 3){
		d1 = 2;
	}else{
		d1 = 1;
	}
	if(m2k == 3){
		d2 = 2;
	}else{
		d2 = 1;
	}
	if(m3k == 3){
		d3 = 2;
	}else{
		d3 = 1;
	}
	for(j3 = begin; j3 < end; j3++){
		i3 = 2*j3-d3;
		for(j2 = 1; j2 < m2j-1; j2++){
			i2 = 2*j2-d2;
			for(j1 = 1; j1 < m1j; j1++){
				i1 = 2*j1-d1;
				x1[i1] = r[i3+1][i2][i1] + r[i3+1][i2+2][i1]
					+ r[i3][i2+1][i1] + r[i3+2][i2+1][i1];
				y1[i1] = r[i3][i2][i1] + r[i3+2][i2][i1]
					+ r[i3][i2+2][i1] + r[i3+2][i2+2][i1];
			}
			for(j1 = 1; j1 < m1j-1; j1++){
				i1 = 2*j1-d1;
				y2 = r[i3][i2][i1+1] + r[i3+2][i2][i1+1]
					+ r[i3][i2+2][i1+1] + r[i3+2][i2+2][i1+1];
				x2 = r[i3+1][i2][i1+1] + r[i3+1][i2+2][i1+1]
					+ r[i3][i2+1][i1+1] + r[i3+2][i2+1][i1+1];
				s[j3][j2][j1] =
					0.5 * r[i3+1][i2+1][i1+1]
					+ 0.25 * ( r[i3+1][i2+1][i1] + r[i3+1][i2+1][i1+2] + x2)
					+ 0.125 * ( x1[i1] + x1[i1+2] + y2)
					+ 0.0625 * ( y1[i1] + y1[i1+2] );
			}
		}
	}

    });

	comm3(s.data,m1j,m2j,m3j,k-1);

}
} // namespace nas_mg

namespace nas_mg {
inline Result reference(const Class& c,const Layout& l,unsigned workers) {
    Result result(l);result.v=input(c);
    PlanePool pool(workers);
    double a[]={-8.0/3,0.0,1.0/6,1.0/12};auto coefficients=smoother(c);
    auto up=[&](int k){return result.u.data()+l.offset[k];};
    auto rp=[&](int k){return result.r.data()+l.offset[k];};
    const int top=l.levels()-1,n=l.width[top];
    resid(up(top),result.v.data(),rp(top),n,n,n,a,top,pool);
    norm2u3(rp(top),n,n,n,&result.norms[0],&result.norms[1],c.n,c.n,c.n);
    for(int it=0;it<c.iterations;++it) {
        for(int k=top;k>0;--k) {
            const int fine=l.width[k],coarse=l.width[k-1];
            rprj3(rp(k),fine,fine,fine,rp(k-1),coarse,coarse,coarse,k,pool);
        }
        const int bottom=l.width[0];std::fill_n(up(0),Layout::cube(bottom),0.0);
        psinv(rp(0),up(0),bottom,bottom,bottom,coefficients.data(),0,pool);
        for(int k=1;k<top;++k) {
            const int fine=l.width[k],coarse=l.width[k-1];
            std::fill_n(up(k),Layout::cube(fine),0.0);
            interp(up(k-1),coarse,coarse,coarse,up(k),fine,fine,fine,k,pool);
            resid(up(k),rp(k),rp(k),fine,fine,fine,a,k,pool);
            psinv(rp(k),up(k),fine,fine,fine,coefficients.data(),k,pool);
        }
        const int coarse=l.width[top-1];
        interp(up(top-1),coarse,coarse,coarse,up(top),n,n,n,top,pool);
        resid(up(top),result.v.data(),rp(top),n,n,n,a,top,pool);
        psinv(rp(top),up(top),n,n,n,coefficients.data(),top,pool);
        resid(up(top),result.v.data(),rp(top),n,n,n,a,top,pool);
    }
    norm2u3(rp(top),n,n,n,&result.norms[2],&result.norms[3],c.n,c.n,c.n);
    return result;
}
} // namespace nas_mg
