// rodinia_lavamd.hip — Rodinia LavaMD benchmark (HIP, self-contained)
//
// Faithful port of Rodinia LavaMD: the device kernel (kernel_gpu_cuda) is the
// ORIGINAL upstream kernel, copied verbatim from
// rodinia/cuda/lavaMD/kernel/kernel_gpu_cuda.cu — including its staging of the
// home box and each neighbor box's particles into SHARED MEMORY
// (rA_shared[100], rB_shared[100], qB_shared[100]) before the all-pairs
// interaction, and the modified-Lennard-Jones / Gaussian potential
// (vij = exp(-a2*r2), fs = 2*a2*vij). The particle/box data structures
// (FOUR_VECTOR, box_str, par_str, dim_str, NUMBER_PAR_PER_BOX=100) and the
// host geometry (one block per home box, NUMBER_THREADS threads) are the
// original. Only the driver is restructured for the per-iteration timing
// harness and JSON output.
//
// Space is a boxes1d^3 grid of boxes; each box holds NUMBER_PAR_PER_BOX
// particles. For every home box the kernel walks its (up to 26) neighbor
// boxes plus itself, computing pairwise force/energy contributions.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./rodinia_lavamd [--boxes1d B]
//
//   --boxes1d B   Boxes per dimension (total boxes = B^3)   (default: 10)
//
// Environment variables (override argv):
//   BENCH_PARAM_boxes1d
//
// Output (stdout): one {"type":"kernel",...} JSON line per timed pass for
// kernel_gpu_cuda carrying the GPU event time (time_ms) and host wall-clock
// time (host_time_ms), then a summary whose total_time_ms is the SUM of all
// passes' GPU event times.
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
// DEFINES / STRUCTURES — verbatim from Rodinia lavaMD/main.h
// ---------------------------------------------------------------------------
#define fp double

#define NUMBER_PAR_PER_BOX 100          // keep low so more blocks share SM
#define NUMBER_THREADS 128              // ~ NUMBER_PAR_PER_BOX for best perf

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))  // STABLE

typedef struct { fp x, y, z; } THREE_VECTOR;
typedef struct { fp v, x, y, z; } FOUR_VECTOR;

typedef struct nei_str {
    int x, y, z;
    int number;
    long offset;
} nei_str;

typedef struct box_str {
    int x, y, z;
    int number;
    long offset;
    int nn;
    nei_str nei[26];
} box_str;

typedef struct par_str {
    fp alpha;
} par_str;

typedef struct dim_str {
    int cur_arg;
    int arch_arg;
    int cores_arg;
    int boxes1d_arg;
    long number_boxes;
    long box_mem;
    long space_elem;
    long space_mem;
    long space_mem2;
} dim_str;

// ---------------------------------------------------------------------------
// KERNEL — verbatim from Rodinia lavaMD/kernel/kernel_gpu_cuda.cu
// ---------------------------------------------------------------------------
__global__ void kernel_gpu_cuda(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str* d_box_gpu,
								FOUR_VECTOR* d_rv_gpu,
								fp* d_qv_gpu,
								FOUR_VECTOR* d_fv_gpu)
{

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int wtx = tx;

	if(bx<d_dim_gpu.number_boxes){

		// parameters
		fp a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

		// home box
		int first_i;
		FOUR_VECTOR* rA;
		FOUR_VECTOR* fA;
		__shared__ FOUR_VECTOR rA_shared[100];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR* rB;
		fp* qB;
		int j = 0;
		__shared__ FOUR_VECTOR rB_shared[100];
		__shared__ double qB_shared[100];

		// common
		fp r2;
		fp u2;
		fp vij;
		fp fs;
		fp fxij;
		fp fyij;
		fp fzij;
		THREE_VECTOR d;

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX){
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++){

			//----------------------------------------50
			//	nei box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = bx;													// set first box to be processed to home box
			}
			else{
				pointer = d_box_gpu[bx].nei[k-1].number;							// remaining boxes are nei boxes
			}

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			// nei box - shared memory
			while(wtx<NUMBER_PAR_PER_BOX){
				rB_shared[wtx] = rB[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			// loop for the number of particles in the home box
			while(wtx<NUMBER_PAR_PER_BOX){

				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX; j++){

					r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[j].v - DOT((fp)rA_shared[wtx],(fp)rB_shared[j]);
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2*vij;

					d.x = (fp)rA_shared[wtx].x  - (fp)rB_shared[j].x;
					fxij=fs*d.x;
					d.y = (fp)rA_shared[wtx].y  - (fp)rB_shared[j].y;
					fyij=fs*d.y;
					d.z = (fp)rA_shared[wtx].z  - (fp)rB_shared[j].z;
					fzij=fs*d.z;

					fA[wtx].v +=  (double)((fp)qB_shared[j]*vij);
					fA[wtx].x +=  (double)((fp)qB_shared[j]*fxij);
					fA[wtx].y +=  (double)((fp)qB_shared[j]*fyij);
					fA[wtx].z +=  (double)((fp)qB_shared[j]*fzij);

				}

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;

			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();

		}

	}

}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int boxes1d = parseIntParam(argc, argv, "--boxes1d", 10);
    const char* env_val = getenv("BENCH_PARAM_boxes1d");
    if (env_val) boxes1d = atoi(env_val);

    par_str par_cpu;
    dim_str dim_cpu;
    par_cpu.alpha = 0.5;                                   // original input
    dim_cpu.boxes1d_arg = boxes1d;

    // DIMENSIONS (verbatim from main.c)
    dim_cpu.number_boxes = (long)dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;
    dim_cpu.space_elem   = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
    dim_cpu.space_mem    = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
    dim_cpu.space_mem2   = dim_cpu.space_elem * sizeof(fp);
    dim_cpu.box_mem      = dim_cpu.number_boxes * sizeof(box_str);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "boxes1d = %d  |  boxes = %ld  |  particles/box = %d  |  particles = %ld\n",
            boxes1d, dim_cpu.number_boxes, NUMBER_PAR_PER_BOX, dim_cpu.space_elem);
    fprintf(stderr, "thread block size of kernel = %d\n\n", NUMBER_THREADS);

    // SYSTEM MEMORY — box neighbor lists (verbatim construction from main.c)
    box_str* box_cpu = (box_str*)malloc(dim_cpu.box_mem);
    int nh = 0;
    for (int i = 0; i < dim_cpu.boxes1d_arg; i++) {
        for (int j = 0; j < dim_cpu.boxes1d_arg; j++) {
            for (int k = 0; k < dim_cpu.boxes1d_arg; k++) {
                box_cpu[nh].x = k;
                box_cpu[nh].y = j;
                box_cpu[nh].z = i;
                box_cpu[nh].number = nh;
                box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;
                box_cpu[nh].nn = 0;
                for (int l = -1; l < 2; l++) {
                    for (int m = -1; m < 2; m++) {
                        for (int n = -1; n < 2; n++) {
                            if ((((i+l)>=0 && (j+m)>=0 && (k+n)>=0) &&
                                 ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)) &&
                                !(l==0 && m==0 && n==0)) {
                                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                                box_cpu[nh].nei[box_cpu[nh].nn].number =
                                    (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
                                    (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) +
                                     box_cpu[nh].nei[box_cpu[nh].nn].x;
                                box_cpu[nh].nei[box_cpu[nh].nn].offset =
                                    box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;
                                box_cpu[nh].nn = box_cpu[nh].nn + 1;
                            }
                        }
                    }
                }
                nh = nh + 1;
            }
        }
    }

    // PARAMETERS, DISTANCE, CHARGE, FORCE (as in main.c; fixed seed for repro)
    srand(42);
    FOUR_VECTOR* rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for (long i = 0; i < dim_cpu.space_elem; i++) {
        rv_cpu[i].v = (rand()%10 + 1) / 10.0;
        rv_cpu[i].x = (rand()%10 + 1) / 10.0;
        rv_cpu[i].y = (rand()%10 + 1) / 10.0;
        rv_cpu[i].z = (rand()%10 + 1) / 10.0;
    }
    fp* qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
    for (long i = 0; i < dim_cpu.space_elem; i++) {
        qv_cpu[i] = (rand()%10 + 1) / 10.0;
    }

    // Device buffers
    box_str* d_box_gpu;
    FOUR_VECTOR* d_rv_gpu;
    fp* d_qv_gpu;
    FOUR_VECTOR* d_fv_gpu;
    HIP_CHECK(hipMalloc((void**)&d_box_gpu, dim_cpu.box_mem));
    HIP_CHECK(hipMalloc((void**)&d_rv_gpu,  dim_cpu.space_mem));
    HIP_CHECK(hipMalloc((void**)&d_qv_gpu,  dim_cpu.space_mem2));
    HIP_CHECK(hipMalloc((void**)&d_fv_gpu,  dim_cpu.space_mem));
    HIP_CHECK(hipMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rv_gpu,  rv_cpu,  dim_cpu.space_mem, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qv_gpu,  qv_cpu,  dim_cpu.space_mem2, hipMemcpyHostToDevice));

    dim3 blocks(dim_cpu.number_boxes, 1);
    dim3 threads(NUMBER_THREADS, 1);

    char params[128];
    snprintf(params, sizeof(params),
             "\"boxes1d\":%d,\"particles_per_box\":%d,\"number_threads\":%d",
             boxes1d, NUMBER_PAR_PER_BOX, NUMBER_THREADS);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc;

    // One recorded pass = one kernel_gpu_cuda launch over all home boxes (the
    // original launches the force kernel exactly once). The kernel ACCUMULATES
    // into d_fv_gpu (fA[wtx].v += ...), so the force output is zeroed before
    // every launch (untimed) to keep each pass identical; the immutable
    // position/charge/box inputs need no reset.
    auto run_pass = [&]() {
        bench_lap_reset(&acc);
        HIP_CHECK(hipMemset(d_fv_gpu, 0, dim_cpu.space_mem));
        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        bench_lap_add(&acc, bench_time_op(&timer, [&]() {
            kernel_gpu_cuda<<<blocks, threads>>>(par_cpu, dim_cpu, d_box_gpu,
                                                 d_rv_gpu, d_qv_gpu, d_fv_gpu);
        }));
        HIP_CHECK(hipGetLastError());
    };

    run_pass();

    double total_gpu_ms = 0.0;
    bench_emit_iter("kernel_gpu_cuda", 0, &acc, params);
    total_gpu_ms += acc.gpu_ms;
    fprintf(stderr, "kernel_gpu_cuda gpu = %.3f ms, host = %.3f ms\n",
            acc.gpu_ms, acc.host_ms);

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over 1 pass\n", total_gpu_ms);

    HIP_CHECK(hipFree(d_box_gpu));
    HIP_CHECK(hipFree(d_rv_gpu));
    HIP_CHECK(hipFree(d_qv_gpu));
    HIP_CHECK(hipFree(d_fv_gpu));
    free(box_cpu);
    free(rv_cpu);
    free(qv_cpu);

    return 0;
}
