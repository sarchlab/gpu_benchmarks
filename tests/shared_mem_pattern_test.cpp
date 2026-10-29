#include "../tier1/shared_mem_bandwidth/hip/shared_mem_pattern.h"
#include <array>
#include <cstdio>
#include <vector>

int main()
{
    // Include all supported 32-lane multiples, not only manifest defaults.
    for (int threads = 32; threads <= 1024; threads += 32) {
        std::vector<int> owner(SMEM_FLOATS, -1);
        for (int tid = 0; tid < threads; ++tid) {
            for (int i = tid; i < SMEM_FLOATS; i += threads) {
                const int address = smem_conflict_index(i);
                if (address < 0 || address >= SMEM_FLOATS || owner[address] != -1) {
                    fprintf(stderr, "invalid/aliased address: threads=%d logical=%d physical=%d\n",
                            threads, i, address);
                    return 1;
                }
                owner[address] = tid;
            }
        }
        for (int tid : owner) if (tid < 0) return 1;
    }
    // This validates address aliasing under explicit bank models. It does not
    // assert how a particular GPU divides an instruction into service phases.
    for (int first = 0; first < SMEM_FLOATS; first += SMEM_LANE_GROUP) {
        for (int banks : {32, 64}) {
            std::array<int, 64> contiguous{}, conflict{};
            for (int lane = 0; lane < SMEM_LANE_GROUP; ++lane) {
                const int address = smem_conflict_index(first + lane);
                if (address - smem_conflict_index(first) != lane * SMEM_LANE_GROUP) return 1;
                ++contiguous[(first + lane) % banks];
                ++conflict[address % banks];
            }
            int used = 0;
            for (int bank = 0; bank < banks; ++bank) {
                if (contiguous[bank] > 1) return 1;
                if (conflict[bank]) {
                    ++used;
                    if (conflict[bank] != (banks == 32 ? 32 : 16)) return 1;
                }
            }
            if (used != (banks == 32 ? 1 : 2)) return 1;
        }
    }
    printf("PASS: complete unique ownership for 32 block sizes; all 384 lane groups match 32/64-bank address models\n");
    return 0;
}
