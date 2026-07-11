// rodinia_srad.hip — Rodinia SRAD (v2) benchmark (HIP, self-contained)
//
// Faithful port of Rodinia SRAD v2: the device kernels (srad_cuda_1,
// srad_cuda_2) are the ORIGINAL upstream kernels, copied verbatim from
// rodinia/cuda/srad/srad_v2/srad_kernel.cu, including their shared-memory
// neighborhood tiling (temp[][], north/south/east/west shared tiles) and the
// compile-time BLOCK_SIZE 16 thread block. Only the host driver is
// restructured for the single-pass timing harness and JSON output.
//
// Speckle-Reducing Anisotropic Diffusion iteratively smooths a 2D float
// image. Each SRAD iteration runs two kernel launches:
//   srad_cuda_1 — per pixel: load the 16x16 tile + halos into shared memory,
//                 compute the directional derivatives, the diffusion
//                 coefficient c (equ 33), and store c and the N/S/W/E deltas.
//   srad_cuda_2 — per pixel: reload c into shared memory, compute the
//                 divergence (equ 58) and update the image J (equ 61).
// The speckle coefficient q0sqr is recomputed each iteration on the CPU from
// an image-statistics (sum/sum2) reduction over the ROI of the current image
// — exactly as the original srad.cu host does — and passed to both kernels.
// This host reduction (and the host<->device copies the original performs
// each iteration) stay in the pass but OUTSIDE the timed brackets.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./rodinia_srad [--image_size N] [--num_iterations K] [--lambda L]
//
//   --image_size N      Image dimension (N×N, multiple of 16)  (default: 512)
//   --num_iterations K  SRAD iterations                        (default: 100)
//   --lambda L          Diffusion step size (0,1)              (default: 0.5)
//
// Environment variables (override argv):
//   BENCH_PARAM_image_size, BENCH_PARAM_num_iterations, BENCH_PARAM_lambda
//
// Output (stdout): one {"type":"kernel",...} JSON line per
// kernel (srad_cuda_1, srad_cuda_2) carrying that kernel's accumulated GPU
// event time (time_ms) and host time (host_time_ms) across the K SRAD
// iterations of the run, then a summary whose total_time_ms is the SUM
// of all recorded GPU event times.
// Output (stderr): human-readable results

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

static int parseIntParam(int argc, char** argv, const char* name, int defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return defaultVal;
}

// ---------------------------------------------------------------------------
// BLOCK_SIZE and kernels — verbatim from Rodinia srad_v2 (srad.h + srad_kernel.cu)
// ---------------------------------------------------------------------------
#define BLOCK_SIZE 16

__global__ void
srad_cuda_1(
		  float *E_C,
		  float *W_C,
		  float *N_C,
		  float *S_C,
		  float * J_cuda,
		  float * C_cuda,
		  int cols,
		  int rows,
		  float q0sqr
)
{

  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //indices
  int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];

  //load data to shared memory
  // NOTE: the original Rodinia srad_v2 issues these neighbour reads
  // unconditionally and then overwrites the border blocks afterwards, which
  // performs an out-of-bounds read of J_cuda at the image edges (index_n/_w
  // go negative on the top/left borders, index_s/_e past the end on the
  // bottom/right). NVIDIA tolerates the OOB read; AMD (MI300X) faults with an
  // illegal memory access. Selecting the index with a ternary avoids issuing
  // the out-of-bounds access while producing identical results.
  north[ty][tx] = ( by == 0 )
                  ? J_cuda[BLOCK_SIZE * bx + tx]
                  : J_cuda[index_n];
  south[ty][tx] = ( by == gridDim.y - 1 )
                  ? J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx]
                  : J_cuda[index_s];
   __syncthreads();

  west[ty][tx] = ( bx == 0 )
                 ? J_cuda[cols * BLOCK_SIZE * by + cols * ty]
                 : J_cuda[index_w];
  east[ty][tx] = ( bx == gridDim.x - 1 )
                 ? J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1]
                 : J_cuda[index_e];

  __syncthreads();



  temp[ty][tx]      = J_cuda[index];

  __syncthreads();

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx]  - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc;
    e  = temp[ty][tx+1] - jc;
   }

   else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc;
    e  = temp[ty][tx+1] - jc;
   }
   else{  //the data elements which are not on the borders
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc;
    e  = temp[ty][tx+1] - jc;
   }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	den  = 1 + (.25*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
	if (c < 0){temp_result[ty][tx] = 0;}
	else if (c > 1) {temp_result[ty][tx] = 1;}
	else {temp_result[ty][tx] = c;}

    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

}

__global__ void
srad_cuda_2(
		  float *E_C,
		  float *W_C,
		  float *N_C,
		  float *S_C,
		  float * J_cuda,
		  float * C_cuda,
		  int cols,
		  int rows,
		  float lambda,
		  float q0sqr
)
{
	//block id
	int bx = blockIdx.x;
    int by = blockIdx.y;

	//thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	//indices
    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation
	__shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    __syncthreads();

	// Same OOB-read avoidance as srad_cuda_1: index_s/index_e run past the end
	// of C_cuda on the bottom/right borders. Select the in-bounds index rather
	// than reading OOB and overwriting (which faults on AMD).
	south_c[ty][tx] = ( by == gridDim.y - 1 )
	                  ? C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx]
	                  : C_cuda[index_s];
	__syncthreads();


	east_c[ty][tx] = ( bx == gridDim.x - 1 )
	                 ? C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1]
	                 : C_cuda[index_e];

    __syncthreads();

    c_cuda_temp[ty][tx]      = C_cuda[index];

    __syncthreads();

	cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc;
    ce  = east_c[ty][tx];
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc;
    ce  = east_c[ty][tx];
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc;
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{ //the data elements which are not on the borders
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc;
    ce  = c_cuda_temp[ty][tx+1];
   }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

   __syncthreads();

   J_cuda[index] = c_cuda_result[ty][tx];

}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int image_size = parseIntParam(argc, argv, "--image_size",      512);
    int niter      = parseIntParam(argc, argv, "--num_iterations",  100);
    float lambda   = 0.5f;
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--lambda") == 0) { lambda = (float)atof(argv[i + 1]); break; }
    }

    const char* env_val;
    env_val = getenv("BENCH_PARAM_image_size");     if (env_val) image_size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_num_iterations"); if (env_val) niter = atoi(env_val);
    env_val = getenv("BENCH_PARAM_lambda");         if (env_val) lambda = (float)atof(env_val);

    // The original kernels index by exact BLOCK_SIZE tiles; rows/cols must be
    // multiples of 16 (upstream enforces the same constraint).
    if (image_size % BLOCK_SIZE != 0) {
        image_size = ((image_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    }
    int rows = image_size;
    int cols = image_size;
    int size_I = cols * rows;

    // ROI for the speckle-statistics reduction (original run: 0..127, 0..127).
    unsigned int r1 = 0, r2 = 127, c1 = 0, c2 = 127;
    if (r2 >= (unsigned)rows) r2 = rows - 1;
    if (c2 >= (unsigned)cols) c2 = cols - 1;
    int size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Image: %d×%d  |  SRAD iterations/pass: %d  |  lambda: %.3f\n",
            rows, cols, niter, lambda);
    fprintf(stderr, "WG size of kernel = %d X %d\n\n", BLOCK_SIZE, BLOCK_SIZE);

    // Host image. The original builds a random matrix I then J = exp(I); we
    // keep a pristine copy so every pass diffuses the identical starting image.
    float* h_I    = (float*)malloc(size_I * sizeof(float));
    float* h_Jini = (float*)malloc(size_I * sizeof(float));  // pristine J = exp(I)
    float* h_J    = (float*)malloc(size_I * sizeof(float));  // working image (host mirror)
    srand(7);
    for (int i = 0; i < size_I; ++i) h_I[i] = rand() / (float)RAND_MAX;
    for (int k = 0; k < size_I; ++k) h_Jini[k] = (float)exp(h_I[k]);

    // Device buffers (original set: J_cuda, C_cuda, E_C, W_C, S_C, N_C).
    float *J_cuda, *C_cuda, *E_C, *W_C, *S_C, *N_C;
    HIP_CHECK(hipMalloc(&J_cuda, size_I * sizeof(float)));
    HIP_CHECK(hipMalloc(&C_cuda, size_I * sizeof(float)));
    HIP_CHECK(hipMalloc(&E_C,    size_I * sizeof(float)));
    HIP_CHECK(hipMalloc(&W_C,    size_I * sizeof(float)));
    HIP_CHECK(hipMalloc(&S_C,    size_I * sizeof(float)));
    HIP_CHECK(hipMalloc(&N_C,    size_I * sizeof(float)));

    // Original host geometry: one thread per pixel, BLOCK_SIZE×BLOCK_SIZE tiles.
    int block_x = cols / BLOCK_SIZE;
    int block_y = rows / BLOCK_SIZE;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(block_x, block_y);

    char params[160];
    snprintf(params, sizeof(params),
             "\"image_size\":%d,\"num_iterations\":%d,\"block_size\":%d,\"lambda\":%.3f",
             image_size, niter, BLOCK_SIZE, lambda);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_srad1, acc_srad2;

    // One recorded pass = a full SRAD run of `niter` iterations. Each
    // iteration: recompute q0sqr on the CPU from the ROI statistics of the
    // current image (untimed), upload J (untimed), launch srad_cuda_1 then
    // srad_cuda_2 (timed), download the updated J so the next iteration's
    // statistics reduction sees it (untimed) — mirroring the original host
    // loop, which copies J to/from the device each SRAD iteration. J is reset
    // to the pristine exp(I) image at pass start so every pass is identical.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_srad1);
        bench_lap_reset(&acc_srad2);
        memcpy(h_J, h_Jini, size_I * sizeof(float));

        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int iter = 0; iter < niter; ++iter) {
            // Image-statistics reduction over the ROI -> q0sqr (CPU, untimed).
            float sum = 0.0f, sum2 = 0.0f;
            for (unsigned int i = r1; i <= r2; ++i) {
                for (unsigned int j = c1; j <= c2; ++j) {
                    float tmp = h_J[i * cols + j];
                    sum += tmp;
                    sum2 += tmp * tmp;
                }
            }
            float meanROI = sum / size_R;
            float varROI  = (sum2 / size_R) - meanROI * meanROI;
            float q0sqr   = varROI / (meanROI * meanROI);

            HIP_CHECK(hipMemcpy(J_cuda, h_J, size_I * sizeof(float), hipMemcpyHostToDevice));

            bench_lap_add(&acc_srad1, bench_time_op(&timer, [&]() {
                srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr);
            }));
            bench_lap_add(&acc_srad2, bench_time_op(&timer, [&]() {
                srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr);
            }));

            HIP_CHECK(hipMemcpy(h_J, J_cuda, size_I * sizeof(float), hipMemcpyDeviceToHost));
        }
        // Catch any launch-config/launch failure before it is recorded.
        HIP_CHECK(hipGetLastError());
    };

    double total_gpu_ms = 0.0;
    run_pass();
    bench_emit_iter("srad_cuda_1", 0, &acc_srad1, params);
    bench_emit_iter("srad_cuda_2", 0, &acc_srad2, params);
    total_gpu_ms += acc_srad1.gpu_ms + acc_srad2.gpu_ms;
    fprintf(stderr, "srad_cuda_1 gpu = %.3f ms, srad_cuda_2 gpu = %.3f ms\n",
            acc_srad1.gpu_ms, acc_srad2.gpu_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms (%d SRAD iters/pass)\n",
            total_gpu_ms, niter);

    HIP_CHECK(hipFree(J_cuda));
    HIP_CHECK(hipFree(C_cuda));
    HIP_CHECK(hipFree(E_C));
    HIP_CHECK(hipFree(W_C));
    HIP_CHECK(hipFree(S_C));
    HIP_CHECK(hipFree(N_C));
    free(h_I);
    free(h_Jini);
    free(h_J);

    return 0;
}
