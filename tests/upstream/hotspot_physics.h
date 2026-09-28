#pragma once
// Frozen CUDA compute_tran_temp physics from HPC-FAIR/rodinia_3.1
// 366b283456506ef2fe2c2b7dc5e83e941ffc5524, cuda/hotspot/hotspot.cu.
// Copyright (c)2008-2011 University of Virginia. Complete license:
// ../../tier2/rodinia_hotspot/UPSTREAM_LICENSE. Only namespace/return scaffolding added.
namespace pinned_hotspot {
#define MAX_PD (3.0e6)
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP 0.5
static float t_chip=0.0005,chip_height=0.016,chip_width=0.016;
struct Physics {float cap,rx,ry,rz,dt;};
inline Physics physics(int row,int col) {
float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;

return {Cap,Rx,Ry,Rz,step};
}
#undef MAX_PD
#undef PRECISION
#undef SPEC_HEAT_SI
#undef K_SI
#undef FACTOR_CHIP
} // namespace pinned_hotspot
