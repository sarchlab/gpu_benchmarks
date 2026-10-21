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
// Numerical kernel bodies from the pinned HIP/MG/mg.cpp. Only shared-buffer
// names changed. Cubic S--C grids make the upstream psinv n2/n1 stride identical.
static constexpr int mg_max_width=514;
extern __shared__ double mg_shared[];

__global__ void comm3_gpu_kernel_1(double* u,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x*blockDim.x+threadIdx.x+1;

	if(i2>=n2-1){return;}

	u[i3*n2*n1+i2*n1+0]=u[i3*n2*n1+i2*n1+n1-2];
	u[i3*n2*n1+i2*n1+n1-1]=u[i3*n2*n1+i2*n1+1];
}

__global__ void comm3_gpu_kernel_2(double* u,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i1=blockIdx.x*blockDim.x+threadIdx.x;

	if(i1>=n1){return;}

	u[i3*n2*n1+0*n1+i1]=u[i3*n2*n1+(n2-2)*n1+i1];
	u[i3*n2*n1+(n2-1)*n1+i1]=u[i3*n2*n1+1*n1+i1];
}

__global__ void comm3_gpu_kernel_3(double* u,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int i2=blockIdx.y*blockDim.y+threadIdx.y;
	int i1=blockIdx.x*blockDim.x+threadIdx.x;

	if(i1>=n1){return;}

	u[0*n2*n1+i2*n1+i1]=u[(n3-2)*n2*n1+i2*n1+i1];
	u[(n3-1)*n2*n1+i2*n1+i1]=u[1*n2*n1+i2*n1+i1];
}

__global__ void interp_gpu_kernel(double* z_device,
		double* u_device,
		int mm1,
		int mm2,
		int mm3,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* z1 = (double*)(mg_shared);
	double* z2 = (double*)(z1+mg_max_width);
	double* z3 = (double*)(z2+mg_max_width);

	int i3=blockIdx.y*blockDim.y+threadIdx.y;
	int i2=blockIdx.x;
	int i1=threadIdx.x;

	z1[i1]=z_device[i3*mm2*mm1+(i2+1)*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1];
	z2[i1]=z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1];
	z3[i1]=z_device[(i3+1)*mm2*mm1+(i2+1)*mm1+i1]+z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z1[i1];

	__syncthreads();
	if(i1<mm1-1){
		double z321=z_device[i3*mm2*mm1+i2*mm1+i1];
		u_device[2*i3*n2*n1+2*i2*n1+2*i1]+=z321;
		u_device[2*i3*n2*n1+2*i2*n1+2*i1+1]+=0.5*(z_device[i3*mm2*mm1+i2*mm1+i1+1]+z321);
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1]+=0.5*z1[i1];
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.25*(z1[i1]+z1[i1+1]);
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1]+=0.5*z2[i1];
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1+1]+=0.25*(z2[i1]+z2[i1+1]);
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1]+=0.25*z3[i1];
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.125*(z3[i1]+z3[i1+1]);
	}
}

__global__ void norm2u3_gpu_kernel(double* r,
		const int n1,
		const int n2,
		const int n3,
		double* res_sum,
		double* res_max,
		int number_of_blocks_on_x_axis,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* scratch_sum = (double*)(mg_shared);
	double* scratch_max = (double*)(scratch_sum+blockDim.x);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int i1=threadIdx.x+1;

	double s=0.0;
	double my_rnmu=0.0;
	double a;

	while(i1<n1-1){
		double r321=r[i3*n2*n1+i2*n1+i1];
		s=s+r321*r321;
		a=fabs(r321);
		my_rnmu=(a>my_rnmu)?a:my_rnmu;
		i1+=blockDim.x;
	}

	int lid=threadIdx.x;
	scratch_sum[lid]=s;
	scratch_max[lid]=my_rnmu;

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(lid<i){
			scratch_sum[lid]+=scratch_sum[lid+i];
			scratch_max[lid]=(scratch_max[lid]>scratch_max[lid+i])?scratch_max[lid]:scratch_max[lid+i];
		}
		__syncthreads();
	}
	if(lid == 0){
		int idx=blockIdx.y*number_of_blocks_on_x_axis+blockIdx.x;
		res_sum[idx]=scratch_sum[0];
		res_max[idx]=scratch_max[0];
	}
}

__global__ void psinv_gpu_kernel(double* r,
		double* u,
		double* c,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* r1 = (double*)(mg_shared);
	double* r2 = (double*)(r1+mg_max_width);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int lid=threadIdx.x;
	int i1;

	for(i1=lid; i1<n1; i1+=blockDim.x){
		r1[i1]=r[i3*n2*n1+(i2-1)*n2+i1]
			+r[i3*n2*n1+(i2+1)*n1+i1]
			+r[(i3-1)*n2*n1+i2*n1+i1]
			+r[(i3+1)*n2*n1+i2*n1+i1];
		r2[i1]=r[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2+1)*n1+i1];
	} __syncthreads();
	for(i1=lid+1; i1<n1-1; i1+=blockDim.x){
		u[i3*n2*n1+i2*n1+i1]=u[i3*n2*n1+i2*n1+i1]
			+c[0]*r[i3*n2*n1+i2*n1+i1]
			+c[1]*(r[i3*n2*n1+i2*n1+i1-1]
					+r[i3*n2*n1+i2*n1+i1+1]
					+r1[i1])
			+c[2]*(r2[i1]+r1[i1-1]+r1[i1+1] );
	}
}

__global__ void resid_gpu_kernel(double* u,
		double* v,
		double* r,
		double* a,
		int n1,
		int n2,
		int n3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	double* u1=(double*)(mg_shared);
	double* u2=(double*)(u1+mg_max_width);

	int i3=blockIdx.y*blockDim.y+threadIdx.y+1;
	int i2=blockIdx.x+1;
	int lid=threadIdx.x;
	int i1;

	for(i1=lid; i1<n1; i1+=blockDim.x){
		u1[i1]=u[i3*n2*n1+(i2-1)*n1+i1]
			+u[i3*n2*n1+(i2+1)*n1+i1]
			+u[(i3-1)*n2*n1+i2*n1+i1]
			+u[(i3+1)*n2*n1+i2*n1+i1];
		u2[i1]=u[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2+1)*n1+i1];
	} __syncthreads();
	for(i1=lid+1; i1<n1-1; i1+=blockDim.x){
		r[i3*n2*n1+i2*n1+i1]=v[i3*n2*n1+i2*n1+i1]
			-a[0]*u[i3*n2*n1+i2*n1+i1]
			-a[2]*(u2[i1]+u1[i1-1]+u1[i1+1])
			-a[3]*(u2[i1-1]+u2[i1+1] );
	}
}

__global__ void rprj3_gpu_kernel(double* r_device,
		double* s_device,
		int m1k,
		int m2k,
		int m3k,
		int m1j,
		int m2j,
		int m3j,
		int d1,
		int d2,
		int d3,
		int amount_of_work){
	int check=blockIdx.x*blockDim.x+threadIdx.x;
	if(check>=amount_of_work){return;}

	int j3,j2,j1,i3,i2,i1;
	double x2,y2;

	double* x1 = (double*)(mg_shared);
	double* y1 = (double*)(x1+mg_max_width);

	j3=blockIdx.y*blockDim.y+threadIdx.y+1;
	j2=blockIdx.x+1;
	j1=threadIdx.x+1;

	i3=2*j3-d3;
	i2=2*j2-d2;
	i1=2*j1-d1;
	x1[i1]=r_device[(i3+1)*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+1)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1];
	y1[i1]=r_device[i3*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+i2*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1];
	__syncthreads();
	if(j1<m1j-1){
		i1=2*j1-d1;
		y2=r_device[i3*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+i2*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1+1];
		x2=r_device[(i3+1)*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+1)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1+1];
		s_device[j3*m2j*m1j+j2*m1j+j1]=
			0.5*r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+1]
			+0.25*(r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1]
					+r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+2]+x2)
			+0.125*(x1[i1]+x1[i1+2]+y2)
			+0.0625*(y1[i1]+y1[i1+2]);
	}
}

__global__ void coarse_zero(double* values, int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) values[i]=0.0;
}
