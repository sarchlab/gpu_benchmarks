#pragma once
// Frozen numerical bodies from HPC-FAIR/rodinia_3.1 at
// 366b283456506ef2fe2c2b7dc5e83e941ffc5524, cuda/hotspot3D/3D.cu.
// Copyright (c)2008-2011 University of Virginia. See
// ../../tier2/rodinia_hotspot3d/hip/UPSTREAM_LICENSE for the complete license.
// Adaptations: namespace/inline scaffolding, physics returned as a struct,
// CPU final-buffer return (the upstream wrapper exposes the wrong parity).
namespace pinned_hotspot3d {
#define MAX_PD  (3.0e6)
/* required precision in degrees    */
#define PRECISION   0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor   */
#define FACTOR_CHIP 0.5


static float t_chip=0.0005,chip_height=0.016,chip_width=0.016,amb_temp=80.0;
struct Physics {float cap,rx,ry,rz,dt;};
inline Physics physics(int numRows,int numCols,int layers) {
    float dx = chip_height/numRows;
    float dy = chip_width/numCols;
    float dz = t_chip/layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;


    return {Cap,Rx,Ry,Rz,dt};
}
inline float* computeTempCPU(float *pIn, float* tIn, float *tOut,
        int nx, int ny, int nz, float Cap,
        float Rx, float Ry, float Rz,
        float dt, int numiter)
{   float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    int c,w,e,n,s,b,t;
    int x,y,z;
    int i = 0;
    do{
        for(z = 0; z < nz; z++)
            for(y = 0; y < ny; y++)
                for(x = 0; x < nx; x++)
                {
                    c = x + y * nx + z * nx * ny;

                    w = (x == 0) ? c      : c - 1;
                    e = (x == nx - 1) ? c : c + 1;
                    n = (y == 0) ? c      : c - nx;
                    s = (y == ny - 1) ? c : c + nx;
                    b = (z == 0) ? c      : c - nx * ny;
                    t = (z == nz - 1) ? c : c + nx * ny;

                    tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t]*ct + tIn[b]*cb + (dt/Cap) * pIn[c] + ct*amb_temp;
                }
        float *temp = tIn;
        tIn = tOut;
        tOut = temp;
        i++;
    }
    while(i < numiter);

    return tIn; // Return the actual final buffer after the last swap.
}

#undef MAX_PD
#undef PRECISION
#undef SPEC_HEAT_SI
#undef K_SI
#undef FACTOR_CHIP
} // namespace pinned_hotspot3d
