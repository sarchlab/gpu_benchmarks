// Rodinia 3.1 backprop: one complete hybrid training epoch, input -> 16 -> 1.
// The original two 16x16 GPU kernels are retained. CPU partial-sum reduction,
// sigmoid/output/error/hidden-delta stages and output-weight update connect them.
// Only the two GPU kernels contribute to recorded time; required hybrid host
// work and transfers are outside their event brackets. One entry warmer runs
// after initial setup; no reconditioning occurs between the dependent stages.
//
// Pinned source: HPC-FAIR/rodinia_3.1 at 366b283456506ef2fe2c2b7dc5e83e941ffc5524,
// cuda/backprop/{backprop_cuda.cu,backprop_cuda_kernel.cu,backprop.c,backprop.h,
// facetrain.c,imagenet.c}. See README for initialization and qualification limits.
// Usage: --input N or BENCH_PARAM_input (default 1000000); positive inputs are
// rounded down to a multiple of16, with a minimum of16. JSON output contains
// two accumulated kernel records and their summed GPU event time.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <climits>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// Constants — from Rodinia 3.1 cuda/backprop/backprop.h
// ---------------------------------------------------------------------------
#define WIDTH    16   // shared memory width (hidden units per tile)
#define HEIGHT   16   // shared memory height (input rows per block)
#define ETA      0.3 // learning rate
#define MOMENTUM 0.3 // momentum

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

// Host state for one input -> 16 -> 1 training epoch. The unused weight column
// zero is retained so initialization consumes the same rand() draws as Rodinia.
struct BackpropNetwork {
    int input_n;
    std::vector<float> input, weights, previous_weights;
    float hidden[WIDTH + 1] = {};
    float hidden_weights[WIDTH + 1][2] = {};
    float hidden_previous[WIDTH + 1][2] = {};
    float hidden_delta[WIDTH + 1] = {};
    float output = 0.0f, output_delta = 0.0f, target = 0.1f;

    explicit BackpropNetwork(int in)
        : input_n(in), input((size_t)in + 1),
          weights(((size_t)in + 1) * (WIDTH + 1)),
          previous_weights(weights.size()) {
        input[0] = 1.0f;
        hidden[0] = 1.0f;
    }
};

static void initialize_network(BackpropNetwork& net) {
    // bpnn_create initializes both weight matrices before imagenet.c loads the
    // non-bias inputs. Momentum starts at zero and the one output target is0.1.
    srand(7);
    for (float& weight : net.weights) weight = (float)rand() / RAND_MAX;
    for (int j = 0; j <= WIDTH; ++j)
        for (int k = 0; k <= 1; ++k)
            net.hidden_weights[j][k] = (float)rand() / RAND_MAX;
    for (int i = 1; i <= net.input_n; ++i)
        net.input[i] = (float)rand() / RAND_MAX;
}

static void finish_forward_and_backpropagate(BackpropNetwork& net,
                                             const std::vector<float>& partial) {
    const int blocks = net.input_n / HEIGHT;
    for (int j = 1; j <= WIDTH; ++j) {
        float sum = 0.0f;
        for (int b = 0; b < blocks; ++b) sum += partial[b * WIDTH + j - 1];
        sum += net.weights[j];  // input bias weight
        net.hidden[j] = (float)(1.0 / (1.0 + std::exp(-(double)sum)));
    }

    // Original CPU output-layer forward, then output and hidden error stages.
    net.hidden[0] = 1.0f;
    float sum = 0.0f;
    for (int j = 0; j <= WIDTH; ++j) sum += net.hidden_weights[j][1] * net.hidden[j];
    net.output = (float)(1.0 / (1.0 + std::exp(-(double)sum)));
    net.output_delta = net.output * (1.0 - net.output) * (net.target - net.output);
    for (int j = 1; j <= WIDTH; ++j) {
        float propagated = net.output_delta * net.hidden_weights[j][1];
        net.hidden_delta[j] = net.hidden[j] * (1.0 - net.hidden[j]) * propagated;
    }

    // Hidden deltas must use the pre-update output weights, as in Rodinia.
    for (int j = 0; j <= WIDTH; ++j) {
        float change = ETA * net.output_delta * net.hidden[j] +
                       MOMENTUM * net.hidden_previous[j][1];
        net.hidden_weights[j][1] += change;
        net.hidden_previous[j][1] = change;
    }
}

struct BackpropLaps { bench_lap_t forward, adjust; };

// Both main and the GPU reference test call this complete production epoch.
// Return the actual updated input weights/momentum in net for inspection.
static BackpropLaps run_backprop_epoch(BackpropNetwork& net) {
    const int in = net.input_n, hid = WIDTH;
    const int blocks = in / HEIGHT;
    const dim3 grid(1, blocks), threads(WIDTH, HEIGHT);
    const size_t input_bytes = net.input.size() * sizeof(float);
    const size_t weight_bytes = net.weights.size() * sizeof(float);
    const size_t hidden_bytes = (hid + 1) * sizeof(float);
    std::vector<float> partial((size_t)blocks * hid);
    const size_t partial_bytes = partial.size() * sizeof(float);
    float *d_input, *d_output_hidden, *d_weights, *d_partial, *d_delta, *d_previous;
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_output_hidden, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_weights, weight_bytes));
    HIP_CHECK(hipMalloc(&d_partial, partial_bytes));
    HIP_CHECK(hipMalloc(&d_delta, hidden_bytes));
    HIP_CHECK(hipMalloc(&d_previous, weight_bytes));
    HIP_CHECK(hipMemcpy(d_input, net.input.data(), input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, net.weights.data(), weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_previous, net.previous_weights.data(), weight_bytes, hipMemcpyHostToDevice));

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_pass_warm();
    BackpropLaps laps;
    laps.forward = bench_time_op(&timer, [&]() {
        bpnn_layerforward_CUDA<<<grid, threads>>>(d_input, d_output_hidden,
                                                d_weights, d_partial, in, hid);
    });

    // These are required stages of the original hybrid algorithm. Their host
    // time and transfers are excluded from the sum of the two GPU kernel times.
    HIP_CHECK(hipMemcpy(partial.data(), d_partial, partial_bytes, hipMemcpyDeviceToHost));
    finish_forward_and_backpropagate(net, partial);
    HIP_CHECK(hipMemcpy(d_delta, net.hidden_delta, hidden_bytes, hipMemcpyHostToDevice));
    // The original forward kernel clobbers its weight buffer; restore it.
    HIP_CHECK(hipMemcpy(d_weights, net.weights.data(), weight_bytes, hipMemcpyHostToDevice));
    laps.adjust = bench_time_op(&timer, [&]() {
        bpnn_adjust_weights_cuda<<<grid, threads>>>(d_delta, hid, d_input, in,
                                                   d_weights, d_previous);
    });
    bench_timer_destroy(&timer);

    HIP_CHECK(hipMemcpy(net.weights.data(), d_weights, weight_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(net.previous_weights.data(), d_previous, weight_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output_hidden));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_partial));
    HIP_CHECK(hipFree(d_delta));
    HIP_CHECK(hipFree(d_previous));
    return laps;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int in = parseIntParam(argc, argv, "--input", 1000000);
    if (const char* value = getenv("BENCH_PARAM_input")) in = atoi(value);
    if (in <= 0 || in > INT_MAX / (WIDTH + 1) - 1) {
        fprintf(stderr, "Input size must be positive and fit the kernel's weight indexing\n");
        return 1;
    }
    if (in % HEIGHT) {
        in = (in / HEIGHT) * HEIGHT;
        if (in < HEIGHT) in = HEIGHT;
        fprintf(stderr, "input rounded down to multiple of16: %d\n", in);
    }
    int device = 0;
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    if (in / HEIGHT > prop.maxGridSize[1]) {
        fprintf(stderr, "Input size exceeds the device's two-dimensional grid limit\n");
        return 1;
    }
    fprintf(stderr, "Device: %s (id=%d)\nNetwork: %d -> 16 -> 1 | single hybrid epoch\n",
            prop.name, device, in);
    BackpropNetwork net(in);
    initialize_network(net);
    BackpropLaps laps = run_backprop_epoch(net);
    char params[160];
    snprintf(params, sizeof(params), "\"input\":%d,\"hidden\":%d,\"height\":%d,\"width\":%d",
             in, WIDTH, HEIGHT, WIDTH);
    bench_emit_iter("bpnn_layerforward_CUDA", 0, &laps.forward, params);
    bench_emit_iter("bpnn_adjust_weights_cuda", 0, &laps.adjust, params);
    double total = laps.forward.gpu_ms + laps.adjust.gpu_ms;
    bench_emit_summary(total);
    fprintf(stderr, "forward = %.6f ms, adjust = %.6f ms, total GPU = %.6f ms\n",
            laps.forward.gpu_ms, laps.adjust.gpu_ms, total);
    return 0;
}
