// Pinned Rodinia 3.1 HotSpot pyramidal kernel; see README.md / UPSTREAM_LICENSE.
#include "../../../common_hip/bench_timing.h"
#include "hotspot_reference.h"
#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)
#define BLOCK_SIZE 16
#define IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               int border_cols,  // border offset
                               int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step,
                               float time_elapsed){

        __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;

    int bx = blockIdx.x;
        int by = blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;

        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the small block size
    int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
    int yidx = blkY+ty;
    int xidx = blkX+tx;

        // load data if it is within the valid input range
    int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;

    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
    }
    __syncthreads();

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;

        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                     (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 +
                     (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 +
                     (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)     //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }

      // update the global memory
      // after the last iteration, only threads coordinated within the
      // small block perform the calculation and switch on ``computed''
      if (computed){
          temp_dst[index]= temp_t[ty][tx];
      }
}

int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IONBF,0);
    const auto p=hotspot::parameters(argc,argv);const auto geometry=hotspot::geometry(p);const auto c=hotspot::coefficients(p.n);
    const unsigned workers=hotspot::reference_workers(p.n);const size_t cells=geometry.cells,bytes=cells*sizeof(float);
    int device=0;HIP_CHECK(hipGetDevice(&device));hipDeviceProp_t prop;HIP_CHECK(hipGetDeviceProperties(&prop,device));
    if(256>prop.maxThreadsPerBlock || 16>prop.maxThreadsDim[0] || 16>prop.maxThreadsDim[1]
       || geometry.blocks>prop.maxGridSize[0] || geometry.blocks>prop.maxGridSize[1] || 3*16*16*sizeof(float)>prop.sharedMemPerBlock)
        throw std::invalid_argument("HotSpot launch geometry exceeds device limits");
    fprintf(stderr,"Device: %s; grid=%d^2 steps=%d pyramid=%d reference_workers=%u\n",prop.name,p.n,p.steps,p.pyramid,workers);
    auto in=hotspot::input(p);const double bound=hotspot::error_bound(p,c,in);
    float *a=nullptr,*b=nullptr,*power=nullptr;
    HIP_CHECK(hipMalloc(&a,bytes));HIP_CHECK(hipMalloc(&b,bytes));HIP_CHECK(hipMalloc(&power,bytes));
    HIP_CHECK(hipMemcpy(a,in.temperature.data(),bytes,hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(power,in.power.data(),bytes,hipMemcpyHostToDevice));
    char params[1600];const int written=snprintf(params,sizeof(params),
        "\"grid_size\":%d,\"total_iterations\":%d,\"pyramid_height\":%d,\"block_size\":16,"
        "\"block_x\":16,\"block_y\":16,\"grid_x\":%d,\"grid_y\":%d,\"cells\":%llu,\"kernel_launches\":%llu,"
        "\"threads_per_launch\":%llu,\"launched_threads\":%llu,\"logical_cell_updates\":%llu,\"executed_cell_updates\":%llu,"
        "\"dt\":%.9g,\"upstream_dt\":%.9g,\"timestep_limited\":%s,\"simulated_time_seconds\":%.17g,"
        "\"absolute_error_bound\":%.17g,\"verification_workers\":%u,\"precision\":\"fp32_state_mixed_fp32_fp64_expression\","
        "\"boundary\":\"clamped_four_faces\",\"algorithm\":\"rodinia31_hotspot_pyramid_v1\","
        "\"input\":\"libc_rand42_interleaved_v1\",\"verification\":\"full_cpu_fp64_bound_v1\"",
        p.n,p.steps,p.pyramid,geometry.blocks,geometry.blocks,static_cast<unsigned long long>(geometry.cells),static_cast<unsigned long long>(geometry.launches),
        static_cast<unsigned long long>(geometry.threads_per_launch),static_cast<unsigned long long>(geometry.launched_threads),
        static_cast<unsigned long long>(geometry.logical_updates),static_cast<unsigned long long>(geometry.executed_updates),
        c.dt,c.upstream_dt,c.dt<c.upstream_dt?"true":"false",static_cast<double>(c.dt)*p.steps,bound,workers);
    if(written<0 || static_cast<size_t>(written)>=sizeof(params))throw std::runtime_error("HotSpot metadata overflow");
    const dim3 block(16,16),grid(geometry.blocks,geometry.blocks);
    bench_timer_t timer;bench_timer_create(&timer);bench_lap_t elapsed;bench_lap_reset(&elapsed);
    float* src=a;float* dst=b;bench_pass_warm();
    for(int completed=0;completed<p.steps;) {
        const int iteration=std::min(p.pyramid,p.steps-completed);
        bench_lap_add(&elapsed,bench_time_op(&timer,[&]() {
            calculate_temp<<<grid,block>>>(iteration,power,src,dst,p.n,p.n,p.pyramid,p.pyramid,c.cap,c.rx,c.ry,c.rz,c.dt,0.001f);
        }));
        std::swap(src,dst);completed+=iteration;
    }
    bench_timer_destroy(&timer);bench_emit_iter("calculate_temp",0,&elapsed,params);bench_emit_summary(elapsed.gpu_ms);
    // Download the actual final measured buffer after the last swap.
    std::vector<float> actual(cells);HIP_CHECK(hipMemcpy(actual.data(),src,bytes,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(a));HIP_CHECK(hipFree(b));HIP_CHECK(hipFree(power));
    const auto expected=hotspot::reference(p,c,in,workers);const size_t errors=hotspot::compare(actual,expected,bound);
    fprintf(stderr,"%s (verified %zu values in the complete measured final grid; %zu mismatches; bound=%.9g)\n",errors?"FAIL":"PASS",cells,errors,bound);
    return errors?1:0;
}catch(const std::exception& e){fprintf(stderr,"FAIL: HotSpot setup/reference unavailable: %s\n",e.what());return 1;}
