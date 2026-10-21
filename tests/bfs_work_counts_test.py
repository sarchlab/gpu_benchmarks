#!/usr/bin/env python3
"""Make source drift fail explicitly before compiling BFS test instrumentation."""
from pathlib import Path
import bfs_measured_result
import bfs_work_counts

source = (Path(__file__).resolve().parents[1]/'tier2/rodinia_bfs/hip/rodinia_bfs.hip').read_text()
# Representative setup, eligibility, shared-store, observation and launch anchors.
anchors = ['#define HIP_CHECK(cmd) BENCH_TIMING_CHECK(cmd)\n',
           '    // Host arrays for BFS state',
           '            int nb = col_indices[e];',
           '                atomicExch(&cost[nb], new_cost);',
           '            atomicExch(updated, 1);',
           '    // Device allocations',
           'd_frontier, d_visited, num_nodes);',
           'd_visited, d_updated, num_nodes);',
           '            ++bfs_levels;',
           '    // Cleanup',
           'int        num_nodes) {']
for anchor in anchors:
    for replacement in ('',anchor+anchor):
        changed = source.replace(anchor,replacement,1)
        try:
            bfs_work_counts.instrument(changed,bfs_measured_result.HOOK)
        except ValueError as exc:
            if repr(anchor) not in str(exc):raise AssertionError(str(exc))
        else:
            raise AssertionError('source drift was silently accepted: '+repr(anchor))
bfs_work_counts.instrument(source,bfs_measured_result.HOOK)
print('22 missing/duplicate anchor mutations rejected clearly; current source instruments successfully')
