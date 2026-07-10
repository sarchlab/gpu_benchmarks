// shoc_sort.hip — SHOC Sort benchmark (HIP, self-contained)
//
// Faithful port of SHOC level1 "sort": the device kernels are the ORIGINAL
// SHOC/CUDPP 4-bit radix sort (radixSortBlocks, findRadixOffsets, scan,
// vectorAddUniform4, reorderData), taken verbatim from
//   shoc/src/cuda/level1/sort/sort_kernel.cu
// and driven with the original pipeline from Sort.cu (radixSortStep +
// scanArrayRecursive). A full sort of 32-bit keys is performed as 8 passes of
// 4 radix bits; each pass runs radixSortBlocks -> findRadixOffsets ->
// recursive block-sum scan -> reorderData. Only the host driver is
// restructured for the per-iteration timing harness and JSON output.
//
// Compiles with hipcc (AMD ROCm) or nvcc via HIP compatibility layer (CUDA).
//
// Usage:
//   ./shoc_sort [--size N] [--precision P]
//     --size N       Number of key/value pairs (multiple of 1024)  (default 4194304)
//     --precision P  Key data type label: uint|int|float           (default uint)
//
// Environment variables (override argv):
//   BENCH_PARAM_size, BENCH_PARAM_precision
//
// Output (stdout): per timed pass, one {"type":"kernel",...} JSON line per
// distinct kernel (radixSortBlocks, findRadixOffsets, scan, vectorAddUniform4,
// reorderData) carrying that kernel's GPU event time accumulated over all of
// its launches in the pass (8 radix passes plus the recursive scans), then a
// summary whose total_time_ms is the SUM of all recorded GPU event times.
// Output (stderr): human-readable diagnostics.

#include "../../common/bench_timing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)

// ---------------------------------------------------------------------------
// SHOC sort constants (sort_kernel.h / Sort.h)
// ---------------------------------------------------------------------------
typedef unsigned int uint;

#define WARP_SIZE       32
#define SORT_BLOCK_SIZE 128
#define SCAN_BLOCK_SIZE 256
#define SORT_BITS       32

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

static const char* parseStrParam(int argc, char** argv, const char* name,
                                 const char* defaultVal) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return defaultVal;
}

// Number of scan CTAs (each handles 4*SCAN_BLOCK_SIZE elems), min 1.
static inline unsigned int scanNumBlocks(unsigned int numElements) {
    unsigned int nb = (unsigned int)ceil((float)numElements / (4.f * SCAN_BLOCK_SIZE));
    return nb < 1u ? 1u : nb;
}

// ===========================================================================
// KERNELS — verbatim from SHOC sort_kernel.cu (based on CUDPP)
// ===========================================================================

__device__ uint scanLSB(const uint val, uint* s_data)
{
    // Shared mem is 256 uints long, set first half to 0's
    int idx = threadIdx.x;
    s_data[idx] = 0;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 128 in this case

    // Unrolled scan in local memory
    uint t;
    s_data[idx] = val;     __syncthreads();
    t = s_data[idx -  1];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();

    return s_data[idx] - val;  // convert inclusive -> exclusive
}

__device__ uint4 scan4(uint4 idata, uint* ptr)
{
    uint4 val4 = idata;
    uint4 sum;

    // Scan the 4 elements in idata within this thread
    sum.x = val4.x;
    sum.y = val4.y + sum.x;
    sum.z = val4.z + sum.y;
    uint val = val4.w + sum.z;

    // Now scan those sums across the local work group
    val = scanLSB(val, ptr);

    val4.x = val;
    val4.y = val + sum.x;
    val4.z = val + sum.y;
    val4.w = val + sum.z;

    return val4;
}

//----------------------------------------------------------------------------
// radixSortBlocks sorts all blocks of data independently in shared
// memory.  Each thread block (CTA) sorts one block of 4*CTA_SIZE elements
// on the basis of bits (startbit) -> (startbit + nbits)
//----------------------------------------------------------------------------
__global__ void radixSortBlocks(const uint nbits, const uint startbit,
                              uint4* keysOut, uint4* valuesOut,
                              uint4* keysIn,  uint4* valuesIn)
{
    __shared__ uint sMem[512];

    // Get Indexing information
    const uint i = threadIdx.x + (blockIdx.x * blockDim.x);
    const uint tid = threadIdx.x;
    const uint localSize = blockDim.x;

    // Load keys and vals from global memory
    uint4 key, value;
    key = keysIn[i];
    value = valuesIn[i];

    // For each of the 4 bits
    for(uint shift = startbit; shift < (startbit + nbits); ++shift)
    {
        // Check if the LSB is 0
        uint4 lsb;
        lsb.x = !((key.x >> shift) & 0x1);
        lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);

        // Do an exclusive scan of how many elems have 0's in the LSB
        uint4 address = scan4(lsb, sMem);

        __shared__ uint numtrue;

        // Store the total number of elems with an LSB of 0 to shared mem
        if (tid == localSize - 1)
        {
            numtrue = address.w + lsb.w;
        }
        __syncthreads();

        // Determine rank -- position in the block
        uint4 rank;
        const int idx = tid*4;
        rank.x = lsb.x ? address.x : numtrue + idx     - address.x;
        rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
        rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
        rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;

        // Scatter keys into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
        __syncthreads();

        // Read keys out of local mem into registers, in prep for write out
        key.x = sMem[tid];
        key.y = sMem[tid +     localSize];
        key.z = sMem[tid + 2 * localSize];
        key.w = sMem[tid + 3 * localSize];
        __syncthreads();

        // Scatter values into local mem
        sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
        sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
        sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
        sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
        __syncthreads();

        // Read values out of local mem into registers, in prep for write out
        value.x = sMem[tid];
        value.y = sMem[tid +     localSize];
        value.z = sMem[tid + 2 * localSize];
        value.w = sMem[tid + 3 * localSize];
        __syncthreads();
    }
    keysOut[i]   = key;
    valuesOut[i] = value;
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each
// block counts the number of keys that fall into each radix in the group, and
// finds the starting offset of each radix in the block.
//----------------------------------------------------------------------------
__global__ void findRadixOffsets(uint2* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
    __shared__ uint  sStartPointers[16];
    extern __shared__ uint sRadix1[];

    uint groupId = blockIdx.x;
    uint localId = threadIdx.x;
    uint groupSize = blockDim.x;

    uint2 radix2;
    radix2 = keys[threadIdx.x + (blockIdx.x * blockDim.x)];

    sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

    // Finds the position where the sRadix1 entries differ and stores start
    // index for each radix.
    if(localId < 16)
    {
        sStartPointers[localId] = 0;
    }
    __syncthreads();

    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId]] = localId;
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
    {
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
    }
    __syncthreads();

    if(localId < 16)
    {
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
    }
    __syncthreads();

    // Compute the sizes of each block.
    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
    {
        sStartPointers[sRadix1[localId - 1]] =
            localId - sStartPointers[sRadix1[localId - 1]];
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
    {
        sStartPointers[sRadix1[localId + groupSize - 1]] =
            localId + groupSize - sStartPointers[sRadix1[localId +
                                                         groupSize - 1]];
    }

    if(localId == groupSize - 1)
    {
        sStartPointers[sRadix1[2 * groupSize - 1]] =
            2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
    }
    __syncthreads();

    if(localId < 16)
    {
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets
// have been found.
//----------------------------------------------------------------------------
__global__ void reorderData(uint  startbit,
                            uint  *outKeys,
                            uint  *outValues,
                            uint2 *keys,
                            uint2 *values,
                            uint  *blockOffsets,
                            uint  *offsets,
                            uint  *sizes,
                            uint  totalBlocks)
{
    uint GROUP_SIZE = blockDim.x;
    __shared__ uint2 sKeys2[256];
    __shared__ uint2 sValues2[256];
    __shared__ uint  sOffsets[16];
    __shared__ uint  sBlockOffsets[16];
    uint* sKeys1   = (uint*) sKeys2;
    uint* sValues1 = (uint*) sValues2;

    uint blockId = blockIdx.x;

    uint i = blockId * blockDim.x + threadIdx.x;

    sKeys2[threadIdx.x]   = keys[i];
    sValues2[threadIdx.x] = values[i];

    if(threadIdx.x < 16)
    {
        sOffsets[threadIdx.x]      = offsets[threadIdx.x * totalBlocks +
                                             blockId];
        sBlockOffsets[threadIdx.x] = blockOffsets[blockId * 16 + threadIdx.x];
    }
    __syncthreads();

    uint radix = (sKeys1[threadIdx.x] >> startbit) & 0xF;
    uint globalOffset = sOffsets[radix] + threadIdx.x - sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[threadIdx.x];
    outValues[globalOffset] = sValues1[threadIdx.x];

    radix = (sKeys1[threadIdx.x + GROUP_SIZE] >> startbit) & 0xF;
    globalOffset = sOffsets[radix] + threadIdx.x + GROUP_SIZE -
                   sBlockOffsets[radix];

    outKeys[globalOffset]   = sKeys1[threadIdx.x + GROUP_SIZE];
    outValues[globalOffset] = sValues1[threadIdx.x + GROUP_SIZE];
}

__device__ uint scanLocalMem(const uint val, uint* s_data)
{
    // Shared mem is 512 uints long, set first half to 0
    int idx = threadIdx.x;
    s_data[idx] = 0.0f;
    __syncthreads();

    // Set 2nd half to thread local sum (sum of the 4 elems from global mem)
    idx += blockDim.x; // += 256

    uint t;
    s_data[idx] = val;     __syncthreads();
    t = s_data[idx -  1];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  2];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  4];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx -  8];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 16];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 32];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 64];  __syncthreads();
    s_data[idx] += t;      __syncthreads();
    t = s_data[idx - 128]; __syncthreads();
    s_data[idx] += t;      __syncthreads();

    return s_data[idx-1];
}

__global__ void
scan(uint *g_odata, uint* g_idata, uint* g_blockSums, const int n,
     const bool fullBlock, const bool storeSum)
{
    __shared__ uint s_data[512];

    // Load data into shared mem
    uint4 tempData;
    uint4 threadScanT;
    uint res;
    uint4* inData  = (uint4*) g_idata;

    const int gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;
    const int i = gid * 4;

    // If possible, read from global mem in a uint4 chunk
    if (fullBlock || i + 3 < n)
    {
        // scan the 4 elems read in from global
        tempData       = inData[gid];
        threadScanT.x = tempData.x;
        threadScanT.y = tempData.y + threadScanT.x;
        threadScanT.z = tempData.z + threadScanT.y;
        threadScanT.w = tempData.w + threadScanT.z;
        res = threadScanT.w;
    }
    else
    {   // if not, read individual uints, scan & store in lmem
        threadScanT.x = (i < n) ? g_idata[i] : 0.0f;
        threadScanT.y = ((i+1 < n) ? g_idata[i+1] : 0.0f) + threadScanT.x;
        threadScanT.z = ((i+2 < n) ? g_idata[i+2] : 0.0f) + threadScanT.y;
        threadScanT.w = ((i+3 < n) ? g_idata[i+3] : 0.0f) + threadScanT.z;
        res = threadScanT.w;
    }

    res = scanLocalMem(res, s_data);
    __syncthreads();

    // If we have to store the sum for the block, have the last work item
    // in the block write it out
    if (storeSum && tid == blockDim.x-1) {
        g_blockSums[blockIdx.x] = res + threadScanT.w;
    }

    // write results to global memory
    uint4* outData = (uint4*) g_odata;

    tempData.x = res;
    tempData.y = res + threadScanT.x;
    tempData.z = res + threadScanT.y;
    tempData.w = res + threadScanT.z;

    if (fullBlock || i + 3 < n)
    {
        outData[gid] = tempData;
    }
    else
    {
        if ( i    < n) { g_odata[i]   = tempData.x;
        if ((i+1) < n) { g_odata[i+1] = tempData.y;
        if ((i+2) < n) { g_odata[i+2] = tempData.z; } } }
    }
}

__global__ void
vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n)
{
    __shared__ uint uni[1];

    if (threadIdx.x == 0)
    {
        uni[0] = d_uniforms[blockIdx.x];
    }

    unsigned int address = threadIdx.x + (blockIdx.x *
            blockDim.x * 4);

    __syncthreads();

    // 4 elems per thread
    for (int i = 0; i < 4 && address < n; i++)
    {
        d_vector[address] += uni[0];
        address += blockDim.x;
    }
}

// ===========================================================================
// HOST DRIVER — adapted from SHOC Sort.cu (radixSortStep + scanArrayRecursive),
// with every kernel launch bracketed and accumulated into its per-kernel lap.
// ===========================================================================

// Per-kernel timing accumulators threaded through the (recursive) driver.
struct SortTiming {
    bench_timer_t* timer;
    bench_lap_t*   radix;   // radixSortBlocks
    bench_lap_t*   find;    // findRadixOffsets
    bench_lap_t*   scan;    // scan
    bench_lap_t*   vadd;    // vectorAddUniform4
    bench_lap_t*   reorder; // reorderData
};

static void scanArrayRecursive(uint* outArray, uint* inArray, int numElements,
        int level, uint** blockSums, SortTiming* T)
{
    // Kernels handle 4 elems per thread
    unsigned int numBlocks = scanNumBlocks((unsigned int)numElements);
    unsigned int sharedEltsPerBlock = SCAN_BLOCK_SIZE * 2;
    unsigned int sharedMemSize = sizeof(uint) * sharedEltsPerBlock;

    bool fullBlock = (numElements == (int)(numBlocks * 4 * SCAN_BLOCK_SIZE));

    dim3 grid(numBlocks, 1, 1);
    dim3 threads(SCAN_BLOCK_SIZE, 1, 1);

    // execute the scan
    if (numBlocks > 1)
    {
        bench_lap_add(T->scan, bench_time_op(T->timer, [&]() {
            scan<<<grid, threads, sharedMemSize>>>
               (outArray, inArray, blockSums[level], numElements, fullBlock, true);
        }));
    } else
    {
        bench_lap_add(T->scan, bench_time_op(T->timer, [&]() {
            scan<<<grid, threads, sharedMemSize>>>
               (outArray, inArray, blockSums[level], numElements, fullBlock, false);
        }));
    }
    if (numBlocks > 1)
    {
        scanArrayRecursive(blockSums[level], blockSums[level],
                numBlocks, level + 1, blockSums, T);
        bench_lap_add(T->vadd, bench_time_op(T->timer, [&]() {
            vectorAddUniform4<<< grid, threads >>>
                    (outArray, blockSums[level], numElements);
        }));
    }
}

static void radixSortStep(uint nbits, uint startbit, uint4* keys, uint4* values,
        uint4* tempKeys, uint4* tempValues, uint* counters,
        uint* countersSum, uint* blockOffsets, uint** scanBlockSums,
        uint numElements, SortTiming* T)
{
    // Threads handle either 4 or two elements each
    const size_t radixGlobalWorkSize   = numElements / 4;
    const size_t findGlobalWorkSize    = numElements / 2;
    const size_t reorderGlobalWorkSize = numElements / 2;

    // Radix kernel uses block size of 128, others use 256 (same as scan)
    const size_t radixBlocks   = radixGlobalWorkSize   / SORT_BLOCK_SIZE;
    const size_t findBlocks    = findGlobalWorkSize    / SCAN_BLOCK_SIZE;
    const size_t reorderBlocks = reorderGlobalWorkSize / SCAN_BLOCK_SIZE;

    bench_lap_add(T->radix, bench_time_op(T->timer, [&]() {
        radixSortBlocks
            <<<radixBlocks, SORT_BLOCK_SIZE, 4 * sizeof(uint)*SORT_BLOCK_SIZE>>>
            (nbits, startbit, tempKeys, tempValues, keys, values);
    }));

    bench_lap_add(T->find, bench_time_op(T->timer, [&]() {
        findRadixOffsets
            <<<findBlocks, SCAN_BLOCK_SIZE, 2 * SCAN_BLOCK_SIZE*sizeof(uint)>>>
            ((uint2*)tempKeys, counters, blockOffsets, startbit, numElements,
             findBlocks);
    }));

    scanArrayRecursive(countersSum, counters, 16*reorderBlocks, 0,
            scanBlockSums, T);

    bench_lap_add(T->reorder, bench_time_op(T->timer, [&]() {
        reorderData<<<reorderBlocks, SCAN_BLOCK_SIZE>>>
            (startbit, (uint*)keys, (uint*)values, (uint2*)tempKeys,
            (uint2*)tempValues, blockOffsets, countersSum, counters,
            reorderBlocks);
    }));
}

static bool verifySort(const uint* keys, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (keys[i] > keys[i + 1]) {
            fprintf(stderr, "Failure at idx %d: %u > %u\n",
                    i, keys[i], keys[i + 1]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int size = 1 << 22; // 4M key/value pairs (~16 MB keys, SHOC class ~2)
    const char* precision = "uint";

    size      = parseIntParam(argc, argv, "--size", size);
    precision = parseStrParam(argc, argv, "--precision", precision);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_size");      if (env_val) size = atoi(env_val);
    env_val = getenv("BENCH_PARAM_precision"); if (env_val) precision = env_val;

    // Validate string params so echoed JSON values are canonical literals.
    if (strcmp(precision, "uint") != 0 && strcmp(precision, "int") != 0 &&
        strcmp(precision, "float") != 0) {
        fprintf(stderr, "Invalid precision '%s' (use uint, int, float)\n", precision);
        return EXIT_FAILURE;
    }
    // The SHOC pipeline requires the element count to be a multiple of 1024
    // (4 * SCAN_BLOCK_SIZE) so every kernel's grid geometry divides evenly.
    if (size % 1024 != 0) {
        int rounded = (size / 1024) * 1024;
        if (rounded < 1024) rounded = 1024;
        fprintf(stderr, "Warning: size=%d not a multiple of 1024, using %d\n",
                size, rounded);
        size = rounded;
    }

    uint bytes = (uint)size * sizeof(uint);

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "Sorting %d key/value pairs (%.1f MB keys), 4-bit radix, %d passes\n\n",
            size, (double)bytes / (1024.0 * 1024.0), SORT_BITS / 4);

    // Host key/value buffers. SHOC uses key==val==i%1024; we use synthetic
    // random uint keys (values carry the source index) — radix sort work is
    // data-independent, so timing is unaffected.
    uint* hKeys = (uint*)malloc(bytes);
    uint* hVals = (uint*)malloc(bytes);
    if (!hKeys || !hVals) { fprintf(stderr, "Host malloc failed\n"); return 1; }
    srand(42);
    for (int i = 0; i < size; ++i) {
        hKeys[i] = (uint)rand();
        hVals[i] = (uint)i;
    }

    // Allocate space for block sums in the scan kernel (Sort.cu logic).
    uint numLevelsAllocated = 0;
    uint maxNumScanElements = size;
    uint numScanElts = maxNumScanElements;
    uint level = 0;
    do {
        uint numBlocks = scanNumBlocks(numScanElts);
        if (numBlocks > 1) level++;
        numScanElts = numBlocks;
    } while (numScanElts > 1);

    uint** scanBlockSums = (uint**)malloc((level + 1) * sizeof(uint*));
    numLevelsAllocated = level + 1;
    numScanElts = maxNumScanElements;
    level = 0;
    do {
        uint numBlocks = scanNumBlocks(numScanElts);
        if (numBlocks > 1) {
            HIP_CHECK(hipMalloc((void**)&(scanBlockSums[level]), numBlocks * sizeof(uint)));
            level++;
        }
        numScanElts = numBlocks;
    } while (numScanElts > 1);
    HIP_CHECK(hipMalloc((void**)&(scanBlockSums[level]), sizeof(uint)));

    // Device buffers for the sorting kernels.
    uint *dKeys, *dVals, *dTempKeys, *dTempVals;
    HIP_CHECK(hipMalloc((void**)&dKeys, bytes));
    HIP_CHECK(hipMalloc((void**)&dVals, bytes));
    HIP_CHECK(hipMalloc((void**)&dTempKeys, bytes));
    HIP_CHECK(hipMalloc((void**)&dTempVals, bytes));

    size_t numSortGroups = size / (4 * SORT_BLOCK_SIZE);
    uint *dCounters, *dCounterSums, *dBlockOffsets;
    HIP_CHECK(hipMalloc((void**)&dCounters,     WARP_SIZE * numSortGroups * sizeof(uint)));
    HIP_CHECK(hipMalloc((void**)&dCounterSums,  WARP_SIZE * numSortGroups * sizeof(uint)));
    HIP_CHECK(hipMalloc((void**)&dBlockOffsets, WARP_SIZE * numSortGroups * sizeof(uint)));

    char params[128];
    snprintf(params, sizeof(params),
             "\"size\":%d,\"precision\":\"%s\",\"sort_block_size\":%d,\"scan_block_size\":%d",
             size, precision, SORT_BLOCK_SIZE, SCAN_BLOCK_SIZE);

    bench_timer_t timer;
    bench_timer_create(&timer);
    bench_lap_t acc_radix, acc_find, acc_scan, acc_vadd, acc_reorder;
    SortTiming T{ &timer, &acc_radix, &acc_find, &acc_scan, &acc_vadd, &acc_reorder };

    // One recorded pass = a full sort of the array: 8 radix passes of 4 bits.
    // The sort is DESTRUCTIVE and returns its result in dKeys/dVals (each
    // radixSortStep ping-pongs keys->tempKeys->keys), so the pristine keys and
    // values are re-uploaded (untimed) at the start of every pass.
    auto run_pass = [&]() {
        bench_lap_reset(&acc_radix);
        bench_lap_reset(&acc_find);
        bench_lap_reset(&acc_scan);
        bench_lap_reset(&acc_vadd);
        bench_lap_reset(&acc_reorder);

        HIP_CHECK(hipMemcpy(dKeys, hKeys, bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dVals, hVals, bytes, hipMemcpyHostToDevice));

        /* Cold-cache calibration: evict the LLC and hold the clock high before
         * the pass's first timed launch, so it runs cold. No-op when
         * BENCH_PARAM_vector_add_warmup is disabled. */
        bench_pass_warm();

        for (int bit = 0; bit < SORT_BITS; bit += 4) {
            radixSortStep(4, bit, (uint4*)dKeys, (uint4*)dVals,
                    (uint4*)dTempKeys, (uint4*)dTempVals, dCounters,
                    dCounterSums, dBlockOffsets, scanBlockSums, size, &T);
        }
        // Surface any launch/config error before the times are recorded.
        HIP_CHECK(hipGetLastError());
    };

    double total_gpu_ms = 0.0;
    run_pass();
    bench_emit_iter("radixSortBlocks",   0, &acc_radix,   params);
    bench_emit_iter("findRadixOffsets",  0, &acc_find,    params);
    bench_emit_iter("scan",              0, &acc_scan,    params);
    bench_emit_iter("vectorAddUniform4", 0, &acc_vadd,    params);
    bench_emit_iter("reorderData",       0, &acc_reorder, params);
    double pass_ms = acc_radix.gpu_ms + acc_find.gpu_ms + acc_scan.gpu_ms +
                     acc_vadd.gpu_ms + acc_reorder.gpu_ms;
    total_gpu_ms += pass_ms;
    fprintf(stderr,
            "radixSortBlocks=%.3f findRadixOffsets=%.3f scan=%.3f "
            "vectorAddUniform4=%.3f reorderData=%.3f (total %.3f ms)\n",
            acc_radix.gpu_ms, acc_find.gpu_ms, acc_scan.gpu_ms,
            acc_vadd.gpu_ms, acc_reorder.gpu_ms, pass_ms);

    bench_timer_destroy(&timer);

    // Correctness check on the last pass's result.
    HIP_CHECK(hipMemcpy(hKeys, dKeys, bytes, hipMemcpyDeviceToHost));
    fprintf(stderr, "Correctness: %s\n", verifySort(hKeys, size) ? "PASS" : "FAIL");

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    for (uint i = 0; i < numLevelsAllocated; ++i) HIP_CHECK(hipFree(scanBlockSums[i]));
    free(scanBlockSums);
    HIP_CHECK(hipFree(dKeys));
    HIP_CHECK(hipFree(dVals));
    HIP_CHECK(hipFree(dTempKeys));
    HIP_CHECK(hipFree(dTempVals));
    HIP_CHECK(hipFree(dCounters));
    HIP_CHECK(hipFree(dCounterSums));
    HIP_CHECK(hipFree(dBlockOffsets));
    free(hKeys);
    free(hVals);

    return 0;
}
