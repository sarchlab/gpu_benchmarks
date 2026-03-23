# Dan — Workspace Notes

## Last Completed (Issue #158)
- Added non_scaling_params to 11 Tier 2 batch B params.json files (shoc + chai)
- Branch: `ares/m21-milestone`
- Commit: 5f963b7
- Files updated:
  - shoc: devicememory (+transfer_mode), fft (+precision, +batch_count), gemm (+precision), reduction (+precision), scan (+precision), sort (+block_size, +precision), spmv (+precision), stencil2d (+block_size), triad (+precision)
  - chai: cedd (+threshold_low, +threshold_high), hsto (+num_bins, +block_size)
- All files now have exactly 3 non_scaling_params each
- All JSON validated successfully
