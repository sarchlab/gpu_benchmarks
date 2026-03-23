# Eva — Workspace Notes

## Last completed: Issue #159

Fixed params.json for 15 Tier 2 batch C benchmarks on branch `ares/m21-milestone`.
Added non_scaling_params to reach minimum 3 per file.

### Files modified:
- cuda_nbody: +block_size
- cuda_scan_large: +block_size, +verify
- cuda_transpose: +block_size, +tile_dim
- graph_tc: +block_size, +verify
- heteromark_aes: +block_size, +key_length
- heteromark_fir: +num_taps, +block_size
- heteromark_pagerank: +damping_factor
- lonestar_sssp: +block_size
- npb_ep: +block_size, +seed
- npb_is: +block_size, +verify
- parboil_histogram: +num_bins, +block_size
- parboil_sgemm: +block_size, +precision
- parboil_stencil: +block_size
- tango_binomial_options: +precision
- tango_blackscholes: +block_size, +precision

Commit: b679381
