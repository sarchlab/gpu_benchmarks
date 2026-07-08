// rodinia_backprop.hip — Rodinia Backpropagation benchmark (HIP, self-contained)
//
// One training epoch of a 3-layer neural net (input -> hidden -> output). In the
// original Rodinia backprop the GPU runs exactly TWO kernels — the input->hidden
// forward pass (bpnn_layerforward_CUDA) and the input->hidden weight update
// (bpnn_adjust_weights_cuda); the small output layer and the error/delta math run
// on the CPU. This port keeps those two kernels VERBATIM from Rodinia 3.1
// cuda/backprop/backprop_cuda_kernel.cu, together with the original host launch
// geometry (num_blocks = in/16, grid(1,num_blocks), threads(16,16), HEIGHT=WIDTH=16
// shared-memory tiling with an in-block log-step reduction tree). Only the driver
// loop is restructured for the per-iteration timing harness and JSON output.
//
// The CPU-side partial-sum reduction and output-layer/delta computation from the
// original are NOT reproduced: they are host code outside the two timed GPU
// kernels and do not affect kernel timing. The hidden_delta vector consumed by
// bpnn_adjust_weights_cuda is synthesized instead.
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu (CUDA) via the HIP compat layer.
//
// Usage:
//   ./rodinia_backprop [--input N]
//
//   --input N   Input layer size (multiple of 16)   (default: 65536)
//
// The hidden layer is fixed at 16 units: the kernels tile with HEIGHT=WIDTH=16
// and index the weight matrix with a (hid+1) stride, so hid==16 is baked into
// the original algorithm and is emitted as a fixed descriptor, not a knob.
//
// Environment variables (override argv):
//   BENCH_PARAM_input
//   BENCH_PARAM_num_warmup   (warmup epochs, default 3)
//   BENCH_PARAM_num_iters    (timed epochs, each recorded, default 7)
//
// Output (stdout): per timed epoch, one {"type":"kernel",...} JSON line for each
// of the two kernels carrying that kernel's GPU event time and host time, then a
// summary whose total_time_ms is the SUM of all recorded GPU event times.
// Output (stderr): device info, per-epoch timing.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Constants — from Rodinia 3.1 cuda/backprop/backprop.h
// ---------------------------------------------------------------------------
#define WIDTH    16   // shared memory width (hidden units per tile)
#define HEIGHT   16   // shared memory height (input rows per block)
#define ETA      0.3f // learning rate
#define MOMENTUM 0.3f // momentum

static int parseIntParam(int argc, char** argv, const char* name, int def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            int v = atoi(argv[i + 1]);
            if (v > 0) return v;
            break;
        }
    }
    return def;
}

// ---------------------------------------------------------------------------
// Kernels — verbatim from Rodinia 3.1 cuda/backprop/backprop_cuda_kernel.cu
// ---------------------------------------------------------------------------
__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid)
{
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;

   int index_in = HEIGHT * by + ty + 1;

   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];


   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;

   __syncthreads();

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   __syncthreads();

   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads();

   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){

	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads();

   }

   //__syncthreads();

   input_hidden_cuda[index] = weight_matrix[ty][tx];

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}


__global__ void bpnn_adjust_weights_cuda(float * delta,
										 int hid,
										 float * ly,
										 int in,
										 float * w,
										 float * oldw)
{


   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int input_n = parseIntParam(argc, argv, "--input", 1048576);
    const char* env_val = getenv("BENCH_PARAM_input");
    if (env_val) input_n = atoi(env_val);

    // Original requires the input layer to be a multiple of 16 (one block row
    // per 16 input units).
    if (input_n % 16 != 0) {
        input_n = (input_n / 16) * 16;
        if (input_n < 16) input_n = 16;
        fprintf(stderr, "input rounded down to multiple of 16: %d\n", input_n);
    }
    const int hidden_n = 16;  // fixed by HEIGHT/WIDTH tiling (see header)

    const int num_warmup = bench_num_warmup();
    const int num_iters  = bench_num_iters();

    int in  = input_n;
    int hid = hidden_n;
    int num_blocks = in / 16;
    dim3 grid(1, num_blocks);
    dim3 threads(16, 16);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Network: %d -> %d -> 1  |  num_blocks: %d  |  %d warmup + %d timed epochs\n\n",
            in, hid, num_blocks, num_warmup, num_iters);

    // Host buffers (original one-dimensionalized layouts).
    size_t sz_w    = (size_t)(in + 1) * (hid + 1) * sizeof(float);
    size_t sz_in   = (size_t)(in + 1) * sizeof(float);
    size_t sz_hid1 = (size_t)(hid + 1) * sizeof(float);
    size_t sz_out1 = (size_t)(hid + 1) * sizeof(float);
    size_t sz_part = (size_t)num_blocks * WIDTH * sizeof(float);

    std::vector<float> h_input(in + 1);
    std::vector<float> h_weights((size_t)(in + 1) * (hid + 1));
    std::vector<float> h_prev_weights((size_t)(in + 1) * (hid + 1));
    std::vector<float> h_delta(hid + 1);

    srand(7);  // original bpnn seed
    for (int i = 0; i <= in; i++)
        h_input[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < h_weights.size(); i++) {
        h_weights[i]      = (float)rand() / RAND_MAX;
        h_prev_weights[i] = 0.0f;  // momentum term starts at zero
    }
    for (int i = 0; i <= hid; i++)
        h_delta[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;  // synthesized hidden deltas

    // Device buffers.
    float *d_input, *d_output_hidden, *d_input_hidden, *d_hidden_partial_sum;
    float *d_hidden_delta, *d_input_prev_weights;
    HIP_CHECK(hipMalloc(&d_input,              sz_in));
    HIP_CHECK(hipMalloc(&d_output_hidden,      sz_out1));
    HIP_CHECK(hipMalloc(&d_input_hidden,       sz_w));
    HIP_CHECK(hipMalloc(&d_hidden_partial_sum, sz_part));
    HIP_CHECK(hipMalloc(&d_hidden_delta,       sz_hid1));
    HIP_CHECK(hipMalloc(&d_input_prev_weights, sz_w));

    HIP_CHECK(hipMemcpy(d_input,        h_input.data(), sz_in,   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_hidden_delta, h_delta.data(), sz_hid1, hipMemcpyHostToDevice));

    char params[160];
    snprintf(params, sizeof(params),
             "\"input\":%d,\"hidden\":%d,\"height\":%d,\"width\":%d",
             in, hid, HEIGHT, WIDTH);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_fwd, acc_adj;

    // One recorded pass = one training epoch: the forward tiled reduction kernel
    // then the weight-adjust kernel. DATA RESET: bpnn_layerforward_CUDA overwrites
    // the weight matrix in place (input_hidden_cuda[index] = reduced product) and
    // bpnn_adjust_weights_cuda mutates both weight buffers. The original allocates
    // and re-uploads these buffers fresh on every bpnn_train_cuda call, so the
    // weight matrix and its previous-weights (momentum) buffer are re-uploaded
    // (untimed) at the start of every epoch — this both matches the original per
    // -call semantics and keeps every timed epoch identical. d_input and
    // d_hidden_delta are read-only and uploaded once.
    auto run_epoch = [&]() {
        bench_lap_reset(&acc_fwd);
        bench_lap_reset(&acc_adj);

        HIP_CHECK(hipMemcpy(d_input_hidden,       h_weights.data(),      sz_w, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_input_prev_weights, h_prev_weights.data(), sz_w, hipMemcpyHostToDevice));

        bench_lap_add(&acc_fwd, bench_time_op(&timer, [&]() {
            bpnn_layerforward_CUDA<<<grid, threads>>>(
                d_input, d_output_hidden, d_input_hidden,
                d_hidden_partial_sum, in, hid);
        }));

        // Original re-uploads the weight matrix before the adjust kernel (the
        // forward kernel clobbered it). Untimed.
        HIP_CHECK(hipMemcpy(d_input_hidden, h_weights.data(), sz_w, hipMemcpyHostToDevice));

        bench_lap_add(&acc_adj, bench_time_op(&timer, [&]() {
            bpnn_adjust_weights_cuda<<<grid, threads>>>(
                d_hidden_delta, hid, d_input, in,
                d_input_hidden, d_input_prev_weights);
        }));
    };

    for (int w = 0; w < num_warmup; ++w) {
        run_epoch();
    }

    double total_gpu_ms = 0.0;
    for (int it = 0; it < num_iters; ++it) {
        run_epoch();
        bench_emit_iter("bpnn_layerforward_CUDA",   it, &acc_fwd, params);
        bench_emit_iter("bpnn_adjust_weights_cuda", it, &acc_adj, params);
        double epoch_gpu_ms = acc_fwd.gpu_ms + acc_adj.gpu_ms;
        total_gpu_ms += epoch_gpu_ms;
        fprintf(stderr, "epoch %d: forward = %.3f ms, adjust = %.3f ms\n",
                it, acc_fwd.gpu_ms, acc_adj.gpu_ms);
    }

    bench_timer_destroy(&timer);
    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms over %d epochs\n", total_gpu_ms, num_iters);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output_hidden));
    HIP_CHECK(hipFree(d_input_hidden));
    HIP_CHECK(hipFree(d_hidden_partial_sum));
    HIP_CHECK(hipFree(d_hidden_delta));
    HIP_CHECK(hipFree(d_input_prev_weights));

    return 0;
}
