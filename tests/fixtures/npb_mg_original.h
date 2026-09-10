#pragma once
// Frozen serial math from GMAP/NPB-GPU
// 3f12d84920ee315ab00ef283717c1e74b68f4d00 HIP/MG/mg.cpp and
// CUDA/common/c_randdp.cpp. Only runtime casts, qualifiers and driver/report
// scaffolding changed. No production stencil/input code is called by this oracle.
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


#include "../../tier2/npb_mg/nas_mg.h"
namespace pinned_mg {
struct Original {
    static constexpr int M=515,MM=10,MAXLEVEL=10,ONE=1;
    static constexpr double A=1220703125.0,X=314159265.0;
    static constexpr double r23=1.0/8388608.0,r46=1.0/70368744177664.0;
    static constexpr double t23=8388608.0,t46=70368744177664.0;
    int lt,lb=1,is1,is2,is3,ie1,ie2,ie3;
    int nx[11]{},ny[11]{},nz[11]{},m1[11]{},m2[11]{},m3[11]{},ir[11]{},debug_vec[8]{};
    void rep_nrm(...) {} void showall(...) {}
double randlc(double *x, double a){
	double t1,t2,t3,t4,a1,a2,x1,x2,z;

	/*
	 * ---------------------------------------------------------------------
	 * break A into two parts such that A = 2^23 * A1 + A2.
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * a;
	a1 = (int)t1;
	a2 = a - t23 * a1;

	/*
	 * ---------------------------------------------------------------------
	 * break X into two parts such that X = 2^23 * X1 + X2, compute
	 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
	 * X = 2^23 * Z + A2 * X2  (mod 2^46).
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * (*x);
	x1 = (int)t1;
	x2 = (*x) - t23 * x1;
	t1 = a1 * x2 + a2 * x1;
	t2 = (int)(r23 * t1);
	z = t1 - t23 * t2;
	t3 = t23 * z + a2 * x2;
	t4 = (int)(r46 * t3);
	(*x) = t3 - t46 * t4;

	return (r46 * (*x));
}
void vranlc(int n, double *x_seed, double a, double y[]){
	int i;
	double x,t1,t2,t3,t4,a1,a2,x1,x2,z;

	/*
	 * ---------------------------------------------------------------------
	 * break A into two parts such that A = 2^23 * A1 + A2.
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * a;
	a1 = (int)t1;
	a2 = a - t23 * a1;
	x = *x_seed;

	/*
	 * ---------------------------------------------------------------------
	 * generate N results. this loop is not vectorizable.
	 * ---------------------------------------------------------------------
	 */
	for(i=0; i<n; i++){
		/*
		 * ---------------------------------------------------------------------
		 * break X into two parts such that X = 2^23 * X1 + X2, compute
		 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
		 * X = 2^23 * Z + A2 * X2  (mod 2^46).
		 * ---------------------------------------------------------------------
		 */
		t1 = r23 * x;
		x1 = (int)t1;
		x2 = x - t23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int)(r23 * t1);
		z = t1 - t23 * t2;
		t3 = t23 * z + a2 * x2;
		t4 = (int)(r46 * t3);
		x = t3 - t46 * t4;
		y[i] = r46 * x;
	}
	*x_seed = x;
}
double power(double a,
		int n){
	double aj;
	int nj;
	double power;

	power = 1.0;
	nj = n;
	aj = a;

	while(nj != 0){
		if((nj%2)==1){randlc(&power, aj);}
		randlc(&aj, aj);
		nj = nj/2;
	}

	return power;
}
void bubble(double ten[][MM],
		int j1[][MM],
		int j2[][MM],
		int j3[][MM],
		int m,
		int ind){
	double temp;
	int i, j_temp;

	if(ind == 1){
		for(i = 0; i < m-1; i++){
			if(ten[ind][i] > ten[ind][i+1]){
				temp = ten[ind][i+1];
				ten[ind][i+1] = ten[ind][i];
				ten[ind][i] = temp;

				j_temp = j1[ind][i+1];
				j1[ind][i+1] = j1[ind][i];
				j1[ind][i] = j_temp;

				j_temp = j2[ind][i+1];
				j2[ind][i+1] = j2[ind][i];
				j2[ind][i] = j_temp;

				j_temp = j3[ind][i+1];
				j3[ind][i+1] = j3[ind][i];
				j3[ind][i] = j_temp;
			}else{
				return;
			}
		}
	}else{
		for(i = 0; i < m-1; i++){
			if(ten[ind][i] < ten[ind][i+1]){
				temp = ten[ind][i+1];
				ten[ind][i+1] = ten[ind][i];
				ten[ind][i] = temp;

				j_temp = j1[ind][i+1];
				j1[ind][i+1] = j1[ind][i];
				j1[ind][i] = j_temp;

				j_temp = j2[ind][i+1];
				j2[ind][i+1] = j2[ind][i];
				j2[ind][i] = j_temp;

				j_temp = j3[ind][i+1];
				j3[ind][i+1] = j3[ind][i];
				j3[ind][i] = j_temp;
			}else{
				return;
			}
		}
	}
}

void comm3(void* pointer_u,
		int n1,
		int n2,
		int n3,
		int kk){
    nas_mg::View u{static_cast<double*>(pointer_u),n1,n2};

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

void interp(void* pointer_z,
		int mm1,
		int mm2,
		int mm3,
		void* pointer_u,
		int n1,
		int n2,
		int n3,
		int k){
    nas_mg::View z{static_cast<double*>(pointer_z),mm1,mm2};
    nas_mg::View u{static_cast<double*>(pointer_u),n1,n2};

	int i3, i2, i1, d1, d2, d3, t1, t2, t3;

	/*
	 * --------------------------------------------------------------------
	 * note that m = 1037 in globals.h but for this only need to be
	 * 535 to handle up to 1024^3
	 * integer m
	 * parameter( m=535 )
	 * --------------------------------------------------------------------
	 */
	double z1[M], z2[M], z3[M];

	if(n1 != 3 && n2 != 3 && n3 != 3){
		for(i3 = 0; i3 < mm3-1; i3++){
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
	}else{
		if(n1 == 3){
			d1 = 2;
			t1 = 1;
		}else{
			d1 = 1;
			t1 = 0;
		}
		if(n2 == 3){
			d2 = 2;
			t2 = 1;
		}else{
			d2 = 1;
			t2 = 0;
		}
		if(n3 == 3){
			d3 = 2;
			t3 = 1;
		}else{
			d3 = 1;
			t3 = 0;
		}
		for(i3 = d3; i3 <= mm3-1; i3++){
			for(i2 = d2; i2 <= mm2-1; i2++){
				for(i1 = d1; i1 <= mm1-1; i1++){
					u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1] =
						u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1]
						+z[i3-1][i2-1][i1-1];
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1] =
						u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1]
						+0.5*(z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1]);
				}
			}
			for(i2 = 1; i2 <= mm2-1; i2++){
				for ( i1 = d1; i1 <= mm1-1; i1++) {
					u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1] =
						u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1]
						+0.5*(z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1] =
						u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1]
						+0.25*(z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
								+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
			}
		}
		for(i3 = 1; i3 <= mm3-1; i3++){
			for(i2 = d2; i2 <= mm2-1; i2++){
				for(i1 = d1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1] =
						u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1]
						+0.5*(z[i3][i2-1][i1-1]+z[i3-1][i2-1][i1-1]);
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1] =
						u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1]
						+0.25*(z[i3][i2-1][i1]+z[i3][i2-1][i1-1]
								+z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1]);
				}
			}
			for(i2 = 1; i2 <= mm2-1; i2++){
				for (i1 = d1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1] =
						u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1]
						+0.25*(z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
								+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
				for(i1 = 1; i1 <= mm1-1; i1++){
					u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1] =
						u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1]
						+0.125*(z[i3][i2][i1]+z[i3][i2-1][i1]
								+z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
								+z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
								+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1]);
				}
			}
		}
	}

	if(debug_vec[0] >= 1){
		rep_nrm(z.data,mm1,mm2,mm3,(char*)"z: inter",k-1);
		rep_nrm(u.data,n1,n2,n3,(char*)"u: inter",k);
	}
	if(debug_vec[5] >= k){
		showall(z.data,mm1,mm2,mm3);
		showall(u.data,n1,n2,n3);
	}
}

void mg3P(double u[],
		double v[],
		double r[],
		double a[4],
		double c[4],
		int n1,
		int n2,
		int n3,
		int k){
	int j;
	/*
	 * --------------------------------------------------------------------
	 * down cycle.
	 * restrict the residual from the find grid to the coarse
	 * -------------------------------------------------------------------
	 */
	for(k = lt; k >= lb+1; k--){
		j = k-1;
		rprj3(&r[ir[k]], m1[k], m2[k], m3[k], &r[ir[j]], m1[j], m2[j], m3[j], k);
	}
	k = lb;
	/*
	 * --------------------------------------------------------------------
	 * compute an approximate solution on the coarsest grid
	 * --------------------------------------------------------------------
	 */
	zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
	psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);
	for(k = lb+1; k <= lt-1; k++){
		j = k-1;
		/*
		 * --------------------------------------------------------------------
		 * prolongate from level k-1  to k
		 * -------------------------------------------------------------------
		 */
		zero3(&u[ir[k]], m1[k], m2[k], m3[k]);
		interp(&u[ir[j]], m1[j], m2[j], m3[j], &u[ir[k]], m1[k], m2[k], m3[k], k);
		/*
		 * --------------------------------------------------------------------
		 * compute residual for level k
		 * --------------------------------------------------------------------
		 */
		resid(&u[ir[k]], &r[ir[k]], &r[ir[k]], m1[k], m2[k], m3[k], a, k);
		/*
		 * --------------------------------------------------------------------
		 * apply smoother
		 * --------------------------------------------------------------------
		 */
		psinv(&r[ir[k]], &u[ir[k]], m1[k], m2[k], m3[k], c, k);
	}
	j = lt - 1;
	k = lt;
	interp(&u[ir[j]], m1[j], m2[j], m3[j], u, n1, n2, n3, k);
	resid(u, v, r, n1, n2, n3, a, k);
	psinv(r, u, n1, n2, n3, c, k);
}

void norm2u3(void* pointer_r,
		int n1,
		int n2,
		int n3,
		double* rnm2,
		double* rnmu,
		int nx,
		int ny,
		int nz){
    nas_mg::View r{static_cast<double*>(pointer_r),n1,n2};

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

void psinv(void* pointer_r,
		void* pointer_u,
		int n1,
		int n2,
		int n3,
		double c[4],
		int k){
    nas_mg::View r{static_cast<double*>(pointer_r),n1,n2};
    nas_mg::View u{static_cast<double*>(pointer_u),n1,n2};

	int i3, i2, i1;
	double r1[M], r2[M];

	for(i3 = 1; i3 < n3-1; i3++){
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

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary points
	 * --------------------------------------------------------------------
	 */
	comm3(u.data,n1,n2,n3,k);

	if(debug_vec[0] >= 1){
		rep_nrm(u.data,n1,n2,n3,(char*)"   psinv",k);
	}

	if(debug_vec[3] >= k){
		showall(u.data,n1,n2,n3);
	}
}

void resid(void* pointer_u,
		void* pointer_v,
		void* pointer_r,
		int n1,
		int n2,
		int n3,
		double a[4],
		int k){
    nas_mg::View u{static_cast<double*>(pointer_u),n1,n2};
    nas_mg::View v{static_cast<double*>(pointer_v),n1,n2};
    nas_mg::View r{static_cast<double*>(pointer_r),n1,n2};

	int i3, i2, i1;
	double u1[M], u2[M];

	for(i3 = 1; i3 < n3-1; i3++){
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

	/*
	 * --------------------------------------------------------------------
	 * exchange boundary data
	 * --------------------------------------------------------------------
	 */
	comm3(r.data,n1,n2,n3,k);

	if(debug_vec[0] >= 1){
		rep_nrm(r.data,n1,n2,n3,(char*)"   resid",k);
	}

	if(debug_vec[2] >= k){
		showall(r.data,n1,n2,n3);
	}
}

void rprj3(void* pointer_r,
		int m1k,
		int m2k,
		int m3k,
		void* pointer_s,
		int m1j,
		int m2j,
		int m3j,
		int k){
    nas_mg::View r{static_cast<double*>(pointer_r),m1k,m2k};
    nas_mg::View s{static_cast<double*>(pointer_s),m1j,m2j};

	int j3, j2, j1, i3, i2, i1, d1, d2, d3, j;

	double x1[M], y1[M], x2, y2;

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
	for(j3 = 1; j3 < m3j-1; j3++){
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

	j=k-1;
	comm3(s.data,m1j,m2j,m3j,j);

	if(debug_vec[0] >= 1){
		rep_nrm(s.data,m1j,m2j,m3j,(char*)"   rprj3",k-1);
	}

	if(debug_vec[4] >= k){
		showall(s.data,m1j,m2j,m3j);
	}
}

void setup(int* n1,
		int* n2,
		int* n3,
		int k){
	int j;

	int ax, mi[MAXLEVEL+1][3];
	int ng[MAXLEVEL+1][3];

	ng[lt][0] = nx[lt];
	ng[lt][1] = ny[lt];
	ng[lt][2] = nz[lt];
	for(ax = 0; ax < 3; ax++){
		for(k = lt-1; k >= 1; k--){
			ng[k][ax] = ng[k+1][ax]/2;
		}
	}
	for(k = lt; k >= 1; k--){
		nx[k] = ng[k][0];
		ny[k] = ng[k][1];
		nz[k] = ng[k][2];
	}

	for(k = lt; k >= 1; k--){
		for (ax = 0; ax < 3; ax++){
			mi[k][ax] = 2 + ng[k][ax];
		}

		m1[k] = mi[k][0];
		m2[k] = mi[k][1];
		m3[k] = mi[k][2];
	}

	k = lt;
	is1 = 2 + ng[k][0] - ng[lt][0];
	ie1 = 1 + ng[k][0];
	*n1 = 3 + ie1 - is1;
	is2 = 2 + ng[k][1] - ng[lt][1];
	ie2 = 1 + ng[k][1];
	*n2 = 3 + ie2 - is2;
	is3 = 2 + ng[k][2] - ng[lt][2];
	ie3 = 1 + ng[k][2];
	*n3 = 3 + ie3 - is3;

	ir[lt] = 0;
	for(j = lt-1; j >= 1; j--){
		ir[j] = ir[j+1]+ONE*m1[j+1]*m2[j+1]*m3[j+1];
	}

	if(debug_vec[1] >= 1){
		printf(" in setup, \n");
		printf("   k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3\n");
		printf("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d\n",
				k,lt,ng[k][0],ng[k][1],ng[k][2],*n1,*n2,*n3,is1,is2,is3,ie1,ie2,ie3);
	}
}

void zero3(void* pointer_z,
		int n1,
		int n2,
		int n3){
    nas_mg::View z{static_cast<double*>(pointer_z),n1,n2};

	int i1, i2, i3;
	for(i3 = 0;i3 < n3; i3++){
		for(i2 = 0; i2 < n2; i2++){
			for(i1 = 0; i1 < n1; i1++){
				z[i3][i2][i1] = 0.0;
			}
		}
	}
}

void zran3(void* pointer_z,
		int n1,
		int n2,
		int n3,
		int nx,
		int ny,
		int k){
    nas_mg::View z{static_cast<double*>(pointer_z),n1,n2};

	int i0, m0, m1;

	int i1, i2, i3, d1, e2, e3;
	double xx, x0, x1, a1, a2, ai;

	double ten[2][MM], best;
	int i, j1[2][MM], j2[2][MM], j3[2][MM];
	int jg[2][MM][4];

	a1 = power(A, nx);
	a2 = power(A, nx*ny);

	zero3(z.data, n1, n2, n3);

	i = is1-2+nx*(is2-2+ny*(is3-2));

	ai = power(A, i);
	d1 = ie1 - is1 + 1;
	e2 = ie2 - is2 + 2;
	e3 = ie3 - is3 + 2;
	x0 = X;
	randlc(&x0, ai);
	for(i3 = 1; i3 < e3; i3++){
		x1 = x0;
		for(i2 = 1; i2 < e2; i2++){
			xx = x1;
			vranlc(d1, &xx, A, &(z[i3][i2][1]));
			randlc(&x1,a1);
		}
		randlc(&x0, a2);
	}

	/*
	 * ---------------------------------------------------------------------
	 * each processor looks for twenty candidates
	 * ---------------------------------------------------------------------
	 */
	for(i = 0; i < MM; i++){
		ten[1][i] = 0.0;
		j1[1][i] = 0;
		j2[1][i] = 0;
		j3[1][i] = 0;
		ten[0][i] = 1.0;
		j1[0][i] = 0;
		j2[0][i] = 0;
		j3[0][i] = 0;
	}
	for(i3 = 1; i3 < n3-1; i3++){
		for(i2 = 1; i2 < n2-1; i2++){
			for(i1 = 1; i1 < n1-1; i1++){
				if(z[i3][i2][i1] > ten[1][0]){
					ten[1][0] = z[i3][i2][i1];
					j1[1][0] = i1;
					j2[1][0] = i2;
					j3[1][0] = i3;
					bubble(ten, j1, j2, j3, MM, 1);
				}
				if(z[i3][i2][i1] < ten[0][0]){
					ten[0][0] = z[i3][i2][i1];
					j1[0][0] = i1;
					j2[0][0] = i2;
					j3[0][0] = i3;
					bubble(ten, j1, j2, j3, MM, 0);
				}
			}
		}
	}

	/*
	 * ---------------------------------------------------------------------
	 * now which of these are globally best?
	 * ---------------------------------------------------------------------
	 */
	i1 = MM - 1;
	i0 = MM - 1;
	for(i = MM - 1; i >= 0; i--){
		best = 0.0;
		if(best < ten[1][i1]){
			jg[1][i][0] = 0;
			jg[1][i][1] = is1 - 2 + j1[1][i1];
			jg[1][i][2] = is2 - 2 + j2[1][i1];
			jg[1][i][3] = is3 - 2 + j3[1][i1];
			i1 = i1-1;
		}else{
			jg[1][i][0] = 0;
			jg[1][i][1] = 0;
			jg[1][i][2] = 0;
			jg[1][i][3] = 0;
		}
		best = 1.0;
		if(best > ten[0][i0]){
			jg[0][i][0] = 0;
			jg[0][i][1] = is1 - 2 + j1[0][i0];
			jg[0][i][2] = is2 - 2 + j2[0][i0];
			jg[0][i][3] = is3 - 2 + j3[0][i0];
			i0 = i0-1;
		}else{
			jg[0][i][0] = 0;
			jg[0][i][1] = 0;
			jg[0][i][2] = 0;
			jg[0][i][3] = 0;
		}
	}
	m1 = 0;
	m0 = 0;

	for(i3 = 0; i3 < n3; i3++){
		for(i2 = 0; i2 < n2; i2++){
			for(i1 = 0; i1 < n1; i1++){
				z[i3][i2][i1] = 0.0;
			}
		}
	}
	for (i = MM-1; i >= m0; i--){
		z[jg[0][i][3]][jg[0][i][2]][jg[0][i][1]] = -1.0;
	}
	for(i = MM-1; i >= m1; i--){
		z[jg[1][i][3]][jg[1][i][2]][jg[1][i][1]] = +1.0;
	}
	comm3(z.data, n1, n2, n3, k);
}
    nas_mg::Result solve(const nas_mg::Class& c,bool input_only=false) {
        lt=0;for(int n=c.n;n>1;n/=2)++lt;
        nx[lt]=ny[lt]=nz[lt]=c.n;int n1,n2,n3;setup(&n1,&n2,&n3,lt);
        nas_mg::Layout layout(c);nas_mg::Result result(layout);
        std::vector<double> u(layout.cells),r(layout.cells),v(nas_mg::Layout::cube(n1));
        zran3(v.data(),n1,n2,n3,c.n,c.n,lt);result.v=v;
        if(input_only)return result;
        double a[]={-8.0/3.0,0,1.0/6.0,1.0/12.0},coeff[4];
        if(c.name=='S'||c.name=='W'||c.name=='A'){coeff[0]=-3.0/8.0;coeff[1]=1.0/32.0;coeff[2]=-1.0/64.0;}
        else{coeff[0]=-3.0/17.0;coeff[1]=1.0/33.0;coeff[2]=-1.0/61.0;}coeff[3]=0;
        resid(u.data(),v.data(),r.data(),n1,n2,n3,a,lt);
        norm2u3(r.data(),n1,n2,n3,&result.norms[0],&result.norms[1],c.n,c.n,c.n);
        for(int it=0;it<c.iterations;++it){mg3P(u.data(),v.data(),r.data(),a,coeff,n1,n2,n3,lt);resid(u.data(),v.data(),r.data(),n1,n2,n3,a,lt);}
        norm2u3(r.data(),n1,n2,n3,&result.norms[2],&result.norms[3],c.n,c.n,c.n);
        for(int k=1;k<=lt;++k){const size_t count=nas_mg::Layout::cube(m1[k]);std::copy_n(u.data()+ir[k],count,result.u.data()+layout.offset[k-1]);std::copy_n(r.data()+ir[k],count,result.r.data()+layout.offset[k-1]);}
        return result;
    }
};
} // namespace pinned_mg
