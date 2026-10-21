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
 *      M. Yarrow
 *      C. Kuszmaul
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
 * The CUDA version is a parallel implementation of the serial C++ version
 * CUDA version: https://github.com/GMAP/NPB-GPU/tree/master/CUDA
 *
 * Authors of the CUDA code:
 *      Gabriell Araujo <hexenoften@gmail.com>
 *
 * ------------------------------------------------------------------------------
 */


#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace nas_cg {
struct Class {
    int n, nonzer, iterations;
    double shift, zeta;
    char name;
};
inline const Class* problem(int n) {
    static const Class classes[] = {
        {1400,7,15,10.0,8.5971775078648,'S'},
        {7000,8,15,12.0,10.362595087124,'W'},
        {14000,11,15,20.0,17.130235054029,'A'},
        {75000,13,75,60.0,22.712745482631,'B'},
        {150000,15,75,110.0,28.973605592845,'C'}
    };
    for (const auto& c : classes) if (c.n == n) return &c;
    return nullptr;
}
constexpr int inner_iterations = 25;
constexpr double rcond = 0.1;
constexpr double tolerance = 1e-10;

// Exact 46-bit randlc state transition. Unsigned multiplication modulo 2^64
// followed by the low-46-bit mask equals multiplication modulo 2^46.
inline double random(std::uint64_t& state) {
    state = (state * 1220703125ULL) & ((1ULL << 46) - 1);
    return static_cast<double>(state) * (1.0 / 70368744177664.0);
}
struct Matrix {
    std::vector<int> offsets, columns;
    std::vector<double> values;
};

// Equivalent row assembly of pinned makea/sprnvc/vecset/sparse. Incidence lists
// preserve the original outer-vector order and every duplicate's addition
// order; sorting only the completed columns replaces insertion-sort moves.
inline Matrix make_matrix(const Class& c) {
    const int n = c.n, stride = c.nonzer + 1;
    std::vector<int> counts(n), columns(static_cast<size_t>(n)*stride);
    std::vector<double> elements(columns.size()), scales(n);
    std::uint64_t state = 314159265;
    random(state); // Pinned driver discards one draw before makea.
    int nn1 = 1;
    while (nn1 < n) nn1 *= 2;
    for (int i = 0; i < n; ++i) {
        int used = 0;
        while (used < c.nonzer) {
            const double value = random(state);
            const int column = static_cast<int>(nn1 * random(state));
            if (column >= n) continue;
            bool duplicate = false;
            for (int k = 0; k < used; ++k)
                if (columns[i*stride+k] == column) duplicate = true;
            if (duplicate) continue;
            columns[i*stride+used] = column;
            elements[i*stride+used++] = value;
        }
        int diagonal = 0;
        while (diagonal < used && columns[i*stride+diagonal] != i) ++diagonal;
        if (diagonal == used) columns[i*stride+used++] = i;
        elements[i*stride+diagonal] = 0.5;
        counts[i] = used;
    }
    std::vector<int> incidence_offsets(n+1,0);
    for (int i = 0; i < n; ++i) for (int k = 0; k < counts[i]; ++k)
        ++incidence_offsets[columns[i*stride+k]+1];
    for (int i = 0; i < n; ++i) incidence_offsets[i+1] += incidence_offsets[i];
    std::vector<int> cursor = incidence_offsets, incidence(incidence_offsets[n]);
    for (int i = 0; i < n; ++i) for (int k = 0; k < counts[i]; ++k)
        incidence[cursor[columns[i*stride+k]]++] = i*stride+k;
    const double ratio = std::pow(rcond,1.0/static_cast<double>(n));
    double size = 1.0;
    for (int i = 0; i < n; ++i) { scales[i] = size; size *= ratio; }
    Matrix a;
    a.offsets.resize(n+1);
    a.columns.reserve(static_cast<size_t>(n)*stride*stride);
    a.values.reserve(static_cast<size_t>(n)*stride*stride);
    std::vector<int> seen(n,-1), touched;
    std::vector<double> sums(n);
    for (int row = 0; row < n; ++row) {
        touched.clear();
        for (int p = incidence_offsets[row]; p < incidence_offsets[row+1]; ++p) {
            const int position = incidence[p], outer = position/stride;
            const double scale = scales[outer]*elements[position];
            for (int k = 0; k < counts[outer]; ++k) {
                const int col = columns[outer*stride+k];
                double value = elements[outer*stride+k]*scale;
                if (col == row && row == outer) value = value+rcond-c.shift;
                if (seen[col] != row) {
                    seen[col] = row; sums[col] = 0.0; touched.push_back(col);
                }
                sums[col] += value;
            }
        }
        std::sort(touched.begin(),touched.end());
        for (int col : touched) { a.columns.push_back(col); a.values.push_back(sums[col]); }
        a.offsets[row+1] = static_cast<int>(a.columns.size());
    }
    return a;
}

inline void spmv_rows(const Matrix& a, const double* x, double* y, int first, int last) {
    for (int row = first; row < last; ++row) {
        double sum = 0.0;
        for (int k = a.offsets[row]; k < a.offsets[row+1]; ++k)
            sum += a.values[k]*x[a.columns[k]];
        y[row] = sum;
    }
}
inline unsigned reference_workers(int n) {
    return n < 75000 ? 1u : std::max(1u,std::min(8u,std::thread::hardware_concurrency()));
}

// Persistent, bounded row workers. Each row retains serial summation order;
// scalar reductions below remain serial. Every run joins its work before any
// dependent vector/scalar operation. Resource failure tears down started workers.
class SpmvPool {
    const Matrix& matrix;
    const int n;
    const unsigned workers;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable start, done;
    unsigned generation = 0, pending = 0;
    bool stopping = false;
    const double* input = nullptr;
    double* output = nullptr;
    void rows(unsigned id, const double* x, double* y) {
        spmv_rows(matrix,x,y,static_cast<int>(static_cast<size_t>(n)*id/workers),
                  static_cast<int>(static_cast<size_t>(n)*(id+1)/workers));
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        start.notify_all();
        for (auto& thread : threads) thread.join();
    }
public:
    SpmvPool(const Matrix& a, unsigned count): matrix(a), n(static_cast<int>(a.offsets.size())-1),
        workers(std::max(1u,std::min(8u,count))) {
        try {
            for (unsigned id = 1; id < workers; ++id) {
                threads.emplace_back([this,id]() {
                    unsigned seen = 0;
                    std::unique_lock<std::mutex> lock(mutex);
                    while (true) {
                        start.wait(lock,[&]() { return stopping || seen != generation; });
                        if (stopping) return;
                        seen = generation;
                        const double* x = input; double* y = output;
                        lock.unlock(); rows(id,x,y); lock.lock();
                        if (--pending == 0) done.notify_one();
                    }
                });
            }
        } catch (...) { stop(); throw; }
    }
    ~SpmvPool() { stop(); }
    SpmvPool(const SpmvPool&) = delete;
    SpmvPool& operator=(const SpmvPool&) = delete;
    void run(const double* x, double* y) {
        if (workers == 1) { rows(0,x,y); return; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            input = x; output = y; pending = workers-1; ++generation;
        }
        start.notify_all(); rows(0,x,y);
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock,[&]() { return pending == 0; });
    }
};

struct Result {
    std::vector<double> x,z,p,q,r,zeta,residual;
    explicit Result(const Class& c): x(c.n,1.0),z(c.n),p(c.n),q(c.n),r(c.n),
        zeta(c.iterations),residual(c.iterations) {}
};
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i]*b[i];
    return sum;
}
inline Result reference(const Matrix& a, const Class& c, unsigned workers) {
    Result result(c);
    auto& x=result.x; auto& z=result.z; auto& p=result.p; auto& q=result.q; auto& r=result.r;
    SpmvPool multiply(a,workers);
    for (int outer = 0; outer < c.iterations; ++outer) {
        std::fill(z.begin(),z.end(),0.0); std::fill(q.begin(),q.end(),0.0);
        r=x; p=r;
        double rho=dot(r,r);
        for (int inner = 0; inner < inner_iterations; ++inner) {
            multiply.run(p.data(),q.data());
            const double alpha=rho/dot(p,q), previous=rho;
            for (int i = 0; i < c.n; ++i) { z[i] += alpha*p[i]; r[i] -= alpha*q[i]; }
            rho=dot(r,r);
            const double beta=rho/previous;
            for (int i = 0; i < c.n; ++i) p[i]=r[i]+beta*p[i];
        }
        multiply.run(z.data(),r.data());
        double residual=0.0;
        for (int i = 0; i < c.n; ++i) { const double d=x[i]-r[i]; residual+=d*d; }
        result.residual[outer]=std::sqrt(residual);
        result.zeta[outer]=c.shift+1.0/dot(x,z);
        const double inverse_norm=1.0/std::sqrt(dot(z,z));
        for (int i = 0; i < c.n; ++i) x[i]=inverse_norm*z[i];
    }
    return result;
}
inline bool close(double actual, double expected) {
    return std::isfinite(actual) && std::isfinite(expected) &&
        std::fabs(actual-expected) <= tolerance*(1.0+std::fabs(expected));
}
inline size_t compare(const Result& actual, const Result& expected) {
    const std::vector<double>* a[]={&actual.x,&actual.z,&actual.p,&actual.q,&actual.r,&actual.zeta,&actual.residual};
    const std::vector<double>* b[]={&expected.x,&expected.z,&expected.p,&expected.q,&expected.r,&expected.zeta,&expected.residual};
    const char* names[]={"x","z","p","q","r","zeta","residual"};
    size_t errors=0;
    for (int k=0;k<7;++k) {
        if (a[k]->size()!=b[k]->size()) throw std::invalid_argument("CG result dimensions differ");
        for (size_t i=0;i<a[k]->size();++i) if (!close((*a[k])[i],(*b[k])[i])) {
            if (errors<10) std::fprintf(stderr,"Mismatch %s[%zu]: GPU=%.17g CPU=%.17g\n",names[k],i,(*a[k])[i],(*b[k])[i]);
            ++errors;
        }
    }
    return errors;
}
inline bool published_zeta(double value, const Class& c) {
    return std::isfinite(value) && std::fabs(value-c.zeta)/c.zeta <= tolerance;
}
} // namespace nas_cg
