// parboil_spmv.hip — faithful port of Parboil spmv:
// the device kernel (spmv_jds) is the original, and the host driver builds the
// original JDS (Jagged Diagonal Storage) matrix layout via a port of Parboil's
// coo_to_jds converter. Only the host driver is restructured for the
// per-iteration timing harness and JSON output.
//
// Provenance:
//   kernel  — parboil/benchmarks/spmv/src/cuda/jds_kernels.cu  (spmv_jds)
//   layout  — parboil/benchmarks/spmv/src/cuda/main.cu +
//             common_src/convert-dataset/convert_dataset.c (coo_to_jds)
//   grid    — gpu_info.cc (compute_active_thread)
//
// JDS layout (the faithful part): rows are sorted by descending non-zero count
// (jagged/Fubini transpose) and stored column-major so that thread `ix` walks
// the columns of its JDS row at stride-1 offsets jds_ptr[k]+ix. The number of
// packed columns per 32-row warp group is held in constant memory (sh_zcnt_int),
// the per-column base offsets in jds_ptr_int. One thread computes one row over
// the JDS permutation d_perm; the result is scattered back to the real row.
//
// TEXTURE SUBSTITUTION (spec rule 2): the canonical Parboil spmv fetches the
// dense vector x through a 1-D texture (tex1Dfetch) for the read-only cache.
// The compat layer does not map CUDA texture references, so the three x_vec
// reads are done through __ldg(&x_vec[i]) (read-only data cache) — the closest
// portable equivalent. The JDS data layout itself is preserved verbatim.
// Padding entries in the jagged columns carry column index 0 (instead of the
// converter's -1) with value 0, so the read-only gather never dereferences a
// negative index; the value is 0 so the product is unchanged.
//
// Compiles with hipcc (AMD ROCm) or nvcc -x cu via HIP compatibility layer.
//
// Environment variables (override argv):
//   BENCH_PARAM_num_rows, BENCH_PARAM_avg_nnz_per_row, BENCH_PARAM_block_size
//
// Output (stdout): one {"type":"kernel",...} JSON line for the single timed
// spmv_jds launch carrying the GPU event time and host wall-clock time, then
// a summary whose total_time_ms is the sum of all recorded GPU event times.
// Output (stderr): human-readable results + verification PASS/FAIL

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
// Constant memory + kernel — verbatim from Parboil spmv jds_kernels.cu / spmv_jds.h
// (x_vec reads routed through __ldg; see texture-substitution note above)
// ---------------------------------------------------------------------------
#define WARP_BITS 5
#define WARP_SIZE 32

// constant memory: JDS column base offsets and per-warp-group column counts
__constant__ int jds_ptr_int[5000];
__constant__ int sh_zcnt_int[5000];

__global__ void spmv_jds(float *dst_vector,
                         const float *d_data, const int *d_index, const int *d_perm,
                         const float *x_vec, const int *d_nzcnt, const int dim)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = ix >> WARP_BITS;
    if (ix < dim)
    {
        float sum = 0.0f;
        int bound = sh_zcnt_int[warp_id];
        // prefetch 0
        int j = jds_ptr_int[0] + ix;
        float d = d_data[j];
        int i = d_index[j];
        float t = __ldg(&x_vec[i]);   // texture fetch -> read-only cache load

        if (bound > 1)  // bound >= 2
        {
            // prefetch 1
            j = jds_ptr_int[1] + ix;
            i = d_index[j];
            int in;
            float dn;
            float tn;
            for (int k = 2; k < bound; k++)
            {
                // prefetch k-1
                dn = d_data[j];
                // prefetch k
                j = jds_ptr_int[k] + ix;
                in = d_index[j];
                // prefetch k-1
                tn = __ldg(&x_vec[i]);   // texture fetch -> read-only cache load

                // compute k-2
                sum += d * t;
                // sweep to k
                i = in;
                // sweep to k-1
                d = dn;
                t = tn;
            }

            // fetch last
            dn = d_data[j];
            tn = __ldg(&x_vec[i]);       // texture fetch -> read-only cache load

            // compute last-1
            sum += d * t;
            // sweep to last
            d = dn;
            t = tn;
        }
        // compute last
        sum += d * t;

        // write out data
        dst_vector[d_perm[ix]] = sum;
    }
}

// ---------------------------------------------------------------------------
// Launch geometry — verbatim port of Parboil gpu_info.cc compute_active_thread
// ---------------------------------------------------------------------------
static void compute_active_thread(unsigned int *thread, unsigned int *grid,
                                  int task, int pad, int major, int minor,
                                  int warp_size, int sm)
{
    int max_thread;
    int max_block = 8;
    (void)warp_size;
    if (major == 1) {
        if (minor >= 2) { max_thread = 1024; }
        else            { max_thread = 768;  }
    } else {
        // 2.x and newer: keep the original's 2.0 assumption
        max_thread = 1536;
    }
    int _grid, _thread;
    if (task * pad > sm * max_thread) {
        _thread = max_thread / max_block;
        _grid = (task * pad + _thread - 1) / _thread;
    } else {
        _thread = pad;
        _grid = task;
    }
    thread[0] = _thread;
    grid[0] = _grid;
}

// ---------------------------------------------------------------------------
// COO -> JDS conversion — port of convert_dataset.c coo_to_jds (in-memory,
// no file I/O / mirroring / binary path). Produces exactly the arrays the
// original main.cu uploads. Padding column index set to 0 (see note above).
// ---------------------------------------------------------------------------
typedef struct { int row, col; float val; } mat_entry;
typedef struct { int index, size, start, padding; } row_stats;

static int sort_rows(const void* a, const void* b) {
    return (((mat_entry*)a)->row - ((mat_entry*)b)->row);
}
static int sort_stats(const void* a, const void* b) {
    return (((row_stats*)b)->size - ((row_stats*)a)->size);
}

static void coo_to_jds_mem(mat_entry* entries, int nz, int rows_input,
                           int pad_rows, int warp_size, int pack_size,
                           float** data, int** data_row_ptr, int** nz_count,
                           int** data_col_index, int** data_row_map,
                           int* data_cols, int* dim, int* len,
                           int* nz_count_len, int* data_ptr_len)
{
    int i;
    *dim = rows_input;

    int total_size = 0;
    int irow, icol = 0, istart = 0;

    qsort(entries, nz, sizeof(mat_entry), sort_rows);
    int rows = entries[nz - 1].row + 1;      // last (greatest) row, zero-indexed
    if (rows % warp_size) rows += warp_size - rows % warp_size;

    row_stats* stats = (row_stats*)calloc(rows, sizeof(row_stats));
    *data_row_map = (int*)calloc(rows, sizeof(int));
    irow = entries[0].row;

    // Count nonzeros per row. Row-transition accounting and final-element
    // flushing must stay separate: folding them together (the old
    // `... || i == nz - 1` branch) double-counts the current nonzero into the
    // previous row and never records the final row when that final row holds
    // exactly one nonzero, corrupting the JDS structure (issue #63, C-01).
    for (i = 0; i < nz; i++) {
        if (entries[i].row != irow) {       // row transition: flush previous row
            stats[irow].size = icol;
            stats[irow].index = irow;
            stats[irow].start = istart;
            icol = 0;
            irow = entries[i].row;
            istart = i;
        }
        icol++;                             // this nonzero belongs to the current row
    }
    stats[irow].size = icol;                // flush the final row exactly once
    stats[irow].index = irow;
    stats[irow].start = istart;

    *nz_count_len = rows / warp_size + rows % warp_size;
    *nz_count = (int*)malloc(*nz_count_len * sizeof(int));

    qsort(stats, rows, sizeof(row_stats), sort_stats);   // largest row first

    int pad_to = 0, pack_to;
    pad_rows *= pack_size;
    for (i = 0; i < rows; i++) {
        (*data_row_map)[i] = stats[i].index;
        if (i % warp_size == 0) {          // warp-group leader (largest row)
            if (stats[i].size % pad_rows)
                stats[i].padding = pad_rows - (stats[i].size % pad_rows);
            else
                stats[i].padding = 0;
            if (stats[i].size % pack_size)
                pack_to = (int)ceil((float)stats[i].size / pack_size);
            else
                pack_to = stats[i].size / pack_size;
            pad_to = stats[i].size + stats[i].padding;
            (*nz_count)[i / warp_size] = pack_to;
            total_size += pad_to * warp_size;
        } else {
            stats[i].padding = pad_to - stats[i].size;
        }
    }

    *data = (float*)calloc(total_size, sizeof(float));
    *data_col_index = (int*)calloc(total_size, sizeof(int));
    *data_row_ptr = (int*)calloc(rows, sizeof(int));
    *len = total_size;

    irow = 0;
    int idata = 0, entry_index, ipack;
    mat_entry entry;
    while (1) {
        (*data_row_ptr)[irow] = idata;
        if (stats[0].size + stats[0].padding <= irow * pack_size) break;
        for (i = 0; i < rows; i++) {
            for (ipack = 0; ipack < pack_size; ipack++) {
                if (stats[i].size > irow * pack_size + ipack) {
                    entry_index = stats[i].start + irow * pack_size + ipack;
                    entry = entries[entry_index];
                    (*data)[idata] = entry.val;
                    (*data_col_index)[idata] = entry.col;
                } else if (stats[i].size + stats[i].padding > irow * pack_size + ipack) {
                    // padding: value stays 0, column index 0 (safe read-only gather)
                    (*data_col_index)[idata] = 0;
                } else {
                    goto endwrite;
                }
                idata += 1;
            }
        }
        endwrite:;
        irow += 1;
    }

    *data_cols = rows;
    *data_ptr_len = irow + 1;
    free(stats);
}

// ---------------------------------------------------------------------------
// Synthetic sparse-matrix generation (COO), sized to a Parboil-scale matrix.
// ---------------------------------------------------------------------------
static mat_entry* generate_coo(int num_rows, int avg_nnz, int seed, int* nz_out)
{
    srand(seed);
    int* nnz_per_row = (int*)malloc((size_t)num_rows * sizeof(int));
    long total = 0;
    for (int r = 0; r < num_rows; ++r) {
        int nnz = 1 + (rand() % (2 * avg_nnz - 1));
        if (nnz > num_rows) nnz = num_rows;
        nnz_per_row[r] = nnz;
        total += nnz;
    }
    mat_entry* entries = (mat_entry*)malloc((size_t)total * sizeof(mat_entry));
    long k = 0;
    for (int r = 0; r < num_rows; ++r) {
        int nnz = nnz_per_row[r];
        int step = (num_rows > nnz) ? (num_rows / nnz) : 1;
        for (int c = 0; c < nnz; ++c) {
            int col = (int)(((long long)c * step + (rand() % step)) % num_rows);
            entries[k].row = r;
            entries[k].col = col;
            entries[k].val = (float)(rand() % 100) / 100.0f + 0.01f;
            k++;
        }
    }
    free(nnz_per_row);
    *nz_out = (int)total;
    return entries;
}

// ---------------------------------------------------------------------------
// Host-only regression for coo_to_jds_mem (issue #63, C-01). Runs the COO->JDS
// conversion on hand-picked matrices, then reconstructs y = A*x from the JDS
// arrays using exactly the spmv_jds indexing (jds_ptr[k] + ix over the group's
// nz_count, permuted through data_row_map) and compares against a direct COO
// reference. No GPU is required. The pathological case is a final row holding
// exactly one nonzero, which the old combined transition/last-element branch
// mis-counted (over-counting the previous row and dropping the final row).
// Invoked with `--selftest`; returns 0 on success.
// ---------------------------------------------------------------------------
static int jds_reconstruct_matches(mat_entry* coo, int nz, int num_rows)
{
    // Sort a private copy first: coo_to_jds_mem sorts in place and the COO
    // reference below is order-independent, but we keep the caller's array intact.
    mat_entry* entries = (mat_entry*)malloc((size_t)nz * sizeof(mat_entry));
    memcpy(entries, coo, (size_t)nz * sizeof(mat_entry));

    float* data;   int* ptr;   int* nzcnt;   int* indices;   int* perm;
    int col_count, dim, len, nzcnt_len, depth;
    coo_to_jds_mem(entries, nz, num_rows,
                   1 /*pad_rows*/, WARP_SIZE /*warp_size*/, 1 /*pack_size*/,
                   &data, &ptr, &nzcnt, &indices, &perm,
                   &col_count, &dim, &len, &nzcnt_len, &depth);

    // Deterministic dense vector x.
    float* x = (float*)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; ++i) x[i] = 0.5f + 0.25f * (float)(i % 7);

    // Direct COO reference.
    float* ref = (float*)calloc((size_t)dim, sizeof(float));
    for (int k = 0; k < nz; ++k)
        ref[coo[k].row] += coo[k].val * x[coo[k].col];

    // Reconstruct exactly as spmv_jds does (host mirror of the kernel).
    float* got = (float*)calloc((size_t)dim, sizeof(float));
    for (int ix = 0; ix < dim; ++ix) {
        int bound = nzcnt[ix >> WARP_BITS];
        float sum = 0.f;
        for (int k = 0; k < bound; ++k) {
            int j = ptr[k] + ix;
            sum += data[j] * x[indices[j]];
        }
        got[perm[ix]] = sum;
    }

    int ok = 1;
    for (int r = 0; r < dim && ok; ++r) {
        float thresh = 1e-4f * fabsf(ref[r]) + 1e-5f;
        if (fabsf(got[r] - ref[r]) > thresh) {
            fprintf(stderr, "  row %d: got %.6f expected %.6f\n", r, got[r], ref[r]);
            ok = 0;
        }
    }

    free(entries); free(data); free(ptr); free(nzcnt); free(indices); free(perm);
    free(x); free(ref); free(got);
    return ok;
}

static int run_jds_selftest(void)
{
    int failures = 0;

    // Case 1 — the C-01 trigger: last row (3) has exactly one nonzero.
    {
        mat_entry m[] = {
            {0,0,1.f}, {0,2,2.f}, {0,3,3.f},   // row 0: 3 nnz
            {1,1,4.f}, {1,2,5.f},              // row 1: 2 nnz
            {2,0,6.f}, {2,3,7.f},              // row 2: 2 nnz
            {3,2,8.f},                          // row 3: 1 nnz  (final, single)
        };
        int ok = jds_reconstruct_matches(m, (int)(sizeof m / sizeof m[0]), 4);
        fprintf(stderr, "selftest final-single-nonzero row: %s\n", ok ? "PASS" : "FAIL");
        failures += !ok;
    }

    // Case 2 — every row a single nonzero (repeated final-single transitions).
    {
        mat_entry m[] = { {0,1,1.f}, {1,0,2.f}, {2,2,3.f} };
        int ok = jds_reconstruct_matches(m, 3, 3);
        fprintf(stderr, "selftest all-single-nonzero rows:  %s\n", ok ? "PASS" : "FAIL");
        failures += !ok;
    }

    // Case 3 — single total nonzero (old code indexed entries[-1] here).
    {
        mat_entry m[] = { {0,0,3.5f} };
        int ok = jds_reconstruct_matches(m, 1, 1);
        fprintf(stderr, "selftest single total nonzero:     %s\n", ok ? "PASS" : "FAIL");
        failures += !ok;
    }

    // Case 4 — final row multi-nonzero (guards against a regression in the
    // common path the old code handled correctly).
    {
        mat_entry m[] = {
            {0,0,1.f},
            {1,1,2.f}, {1,2,3.f},
            {2,0,4.f}, {2,1,5.f}, {2,3,6.f},   // final row: 3 nnz
        };
        int ok = jds_reconstruct_matches(m, 6, 3);
        fprintf(stderr, "selftest final-multi-nonzero row:  %s\n", ok ? "PASS" : "FAIL");
        failures += !ok;
    }

    if (failures)
        fprintf(stderr, "SELFTEST FAIL: %d case(s)\n", failures);
    else
        fprintf(stderr, "SELFTEST PASS\n");
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Host-only regression for the COO->JDS conversion (issue #63, C-01). Runs
    // without a GPU and exits; keep before any device initialization.
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--selftest") == 0)
            return run_jds_selftest();

    // Defaults sized to a Parboil-scale sparse matrix (Dubcova3/mixtank are tens
    // of thousands of rows). Constant-memory tables cap rows/32 and JDS columns
    // at 5000, so num_rows <= 160000 and column depth < 5000.
    int num_rows   = 131072;
    int avg_nnz    = 80;   // matches params.json default
    int block_size = 0;    // 0 => use compute_active_thread (original geometry)
    int seed       = 42;

    num_rows   = parseIntParam(argc, argv, "--num_rows",        num_rows);
    avg_nnz    = parseIntParam(argc, argv, "--avg_nnz_per_row", avg_nnz);
    block_size = parseIntParam(argc, argv, "--block_size",      block_size);
    seed       = parseIntParam(argc, argv, "--seed",            seed);

    const char* env_val;
    env_val = getenv("BENCH_PARAM_num_rows");        if (env_val) num_rows = atoi(env_val);
    env_val = getenv("BENCH_PARAM_avg_nnz_per_row"); if (env_val) avg_nnz = atoi(env_val);
    env_val = getenv("BENCH_PARAM_block_size");      if (env_val) block_size = atoi(env_val);
    const int pad = WARP_SIZE;  // warp size, must match kernel

    // ---- Build the matrix and convert to JDS (host, untimed) ----
    int nz = 0;
    mat_entry* entries = generate_coo(num_rows, avg_nnz, seed, &nz);

    float* h_data;   int* h_ptr;   int* h_nzcnt;   int* h_indices;   int* h_perm;
    int col_count, dim, len, nzcnt_len, depth;
    coo_to_jds_mem(entries, nz, num_rows,
                   1 /*pad_rows*/, pad /*warp_size*/, 1 /*pack_size*/,
                   &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
                   &col_count, &dim, &len, &nzcnt_len, &depth);

    if (depth > 5000 || nzcnt_len > 5000) {
        fprintf(stderr, "JDS tables exceed constant-memory capacity "
                "(depth=%d, nzcnt_len=%d, max 5000). Reduce num_rows/avg_nnz.\n",
                depth, nzcnt_len);
        return EXIT_FAILURE;
    }

    // dense vector x
    float* h_x = (float*)malloc((size_t)dim * sizeof(float));
    float* h_y = (float*)malloc((size_t)dim * sizeof(float));
    srand(seed + 1);
    for (int i = 0; i < dim; ++i)
        h_x[i] = (float)(rand() % 100) / 100.0f + 0.01f;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    fprintf(stderr, "Device: %s (id=%d)\n", prop.name, device_id);
    fprintf(stderr, "SpMV/JDS  |  rows: %d  |  nnz: %d  |  JDS entries(len): %d  |  "
            "cols(depth): %d  |  warp-groups: %d\n\n",
            dim, nz, len, depth, nzcnt_len);

    // ---- Device allocations + one-time uploads (untimed) ----
    float* d_data;   int* d_indices;   int* d_perm;   int* d_nzcnt;
    float* d_x;      float* d_Ax;
    HIP_CHECK(hipMalloc(&d_data,    (size_t)len * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_indices, (size_t)len * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_perm,    (size_t)dim * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_nzcnt,   (size_t)nzcnt_len * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_x,       (size_t)dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Ax,      (size_t)dim * sizeof(float)));
    HIP_CHECK(hipMemset(d_Ax, 0, (size_t)dim * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_data,    h_data,    (size_t)len * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_indices, h_indices, (size_t)len * sizeof(int),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_perm,    h_perm,    (size_t)dim * sizeof(int),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,       h_x,       (size_t)dim * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(jds_ptr_int), h_ptr,   (size_t)depth * sizeof(int)));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(sh_zcnt_int), h_nzcnt, (size_t)nzcnt_len * sizeof(int)));

    // ---- Launch geometry (original compute_active_thread, or override) ----
    unsigned int grid, block;
    if (block_size > 0) {
        block = block_size;
        grid = ((unsigned int)dim + block - 1) / block;
    } else {
        compute_active_thread(&block, &grid, nzcnt_len, pad,
                              prop.major, prop.minor, prop.warpSize,
                              prop.multiProcessorCount);
    }

    char params[192];
    snprintf(params, sizeof(params),
             "\"num_rows\":%d,\"avg_nnz_per_row\":%d,\"nnz\":%d,"
             "\"jds_len\":%d,\"jds_depth\":%d,\"block_size\":%u",
             num_rows, avg_nnz, nz, len, depth, block);

    // One pass = one spmv_jds launch. The kernel reads immutable JDS/vector data
    // and fully overwrites dst_vector (one write per real row), so repeated
    // launches do identical work — no per-pass reset needed.
    double total_gpu_ms =
        bench_run_once("spmv_jds", params, [&]() {
            spmv_jds<<<grid, block>>>(d_Ax, d_data, d_indices, d_perm,
                                      d_x, d_nzcnt, dim);
        });

    bench_emit_summary(total_gpu_ms);
    fprintf(stderr, "Total GPU time: %.3f ms\n", total_gpu_ms);

    // ---- Verification against CPU reference (from the original COO) ----
    HIP_CHECK(hipMemcpy(h_y, d_Ax, (size_t)dim * sizeof(float), hipMemcpyDeviceToHost));
    float* ref = (float*)calloc((size_t)dim, sizeof(float));
    for (int k = 0; k < nz; ++k)
        ref[entries[k].row] += entries[k].val * h_x[entries[k].col];

    int errors = 0;
    for (int r = 0; r < dim; ++r) {
        float thresh = 1e-3f * fabsf(ref[r]) + 1e-4f;
        if (fabsf(h_y[r] - ref[r]) > thresh) {
            if (errors < 5)
                fprintf(stderr, "Mismatch row %d: got %.6f expected %.6f\n",
                        r, h_y[r], ref[r]);
            ++errors;
        }
    }
    if (errors)
        fprintf(stderr, "FAIL: %d mismatches in %d rows\n", errors, dim);
    else
        fprintf(stderr, "PASS (%d/%d rows verified)\n", dim, dim);

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_perm));
    HIP_CHECK(hipFree(d_nzcnt));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_Ax));
    free(entries); free(ref);
    free(h_data); free(h_ptr); free(h_nzcnt); free(h_indices); free(h_perm);
    free(h_x); free(h_y);

    return (errors > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
