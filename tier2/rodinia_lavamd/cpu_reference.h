#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
// Complete independent reference for positive grid/particle extents. Derive
// box membership from coordinates instead of sharing the GPU's neighbor list.
// Every particle visits its home box first, then adjacent boxes in z/y/x order;
// both self interactions and every energy/force component are retained.
// Vector supplies double v/x/y/z fields (the benchmark's FOUR_VECTOR).
template<class Vector>
static std::vector<Vector> cpu_lavamd(const Vector* positions, const double* charges,
                                    int boxes1d, int particles_per_box, double alpha) {
    const size_t boxes=(size_t)boxes1d*boxes1d*boxes1d;
    std::vector<Vector> result(boxes*particles_per_box);
    const double a2=2.0*alpha*alpha;
    for (size_t home=0;home<boxes;++home) {
        const int x=home%boxes1d,y=(home/boxes1d)%boxes1d,z=home/((size_t)boxes1d*boxes1d);
        size_t neighbors[27]; int count=1; neighbors[0]=home;
        for (int dz=-1;dz<=1;++dz)
            for (int dy=-1;dy<=1;++dy)
                for (int dx=-1;dx<=1;++dx) {
                    const int xx=x+dx,yy=y+dy,zz=z+dz;
                    if ((!dx&&!dy&&!dz)||xx<0||yy<0||zz<0||xx>=boxes1d||yy>=boxes1d||zz>=boxes1d) continue;
                    neighbors[count++]=((size_t)zz*boxes1d+yy)*boxes1d+xx;
                }
        for (int i=0;i<particles_per_box;++i) {
            const size_t at=home*particles_per_box+i;
            const Vector& a=positions[at]; Vector total={};
            for (int neighbor=0;neighbor<count;++neighbor)
                for (int j=0;j<particles_per_box;++j) {
                    const size_t bt=neighbors[neighbor]*particles_per_box+j;
                    const Vector& b=positions[bt];
                    const double r2=a.v+b.v-(a.x*b.x+a.y*b.y+a.z*b.z);
                    const double value=std::exp(-a2*r2),force=2.0*value;
                    const double charge=charges[bt];
                    total.v+=charge*value;
                    total.x+=charge*(force*(a.x-b.x));
                    total.y+=charge*(force*(a.y-b.y));
                    total.z+=charge*(force*(a.z-b.z));
                }
            result[at]=total;
        }
    }
    return result;
}
// Conservative double-precision comparison, including a small absolute floor
// for force components near zero. Non-finite values always fail.
static bool lavamd_value_matches(double actual,double expected) {
    const double error=std::fabs(actual-expected);
    return std::isfinite(actual)&&std::isfinite(expected)&&std::isfinite(error)
        &&error<=1e-10*std::max(1.0,std::fabs(expected));
}
