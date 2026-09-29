// The driver prepends the production checker and a test-only calloc wrapper.
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

static size_t cases = 0;
static void check(const std::vector<uint>& input, const std::vector<uint>& keys,
                  const std::vector<uint>& values, int expected_errors = -1) {
    const size_t n = input.size();
    assert(n == keys.size() && n == values.size());
    // Independent oracle: compare complete multisets of key/value pairs and
    // key order. Sorting pairs here does not impose a tie order on the output.
    std::vector<std::pair<uint,uint>> wanted, actual;
    for (size_t i=0; i<n; ++i) {
        wanted.emplace_back(input[i], (uint)i);
        actual.emplace_back(keys[i], values[i]);
    }
    std::sort(wanted.begin(), wanted.end());
    std::sort(actual.begin(), actual.end());
    const bool expected = !fail_seen && wanted == actual && std::is_sorted(keys.begin(),keys.end());
    if (expected_errors >= 0) assert(expected == (expected_errors == 0));
    std::vector<uint> guarded_input(n+2,0x12345678u), guarded_keys(n+2,0x87654321u), guarded_values(n+2,0x13572468u);
    std::copy(input.begin(), input.end(), guarded_input.begin()+1);
    std::copy(keys.begin(), keys.end(), guarded_keys.begin()+1);
    std::copy(values.begin(), values.end(), guarded_values.begin()+1);
    const auto before_input=guarded_input, before_keys=guarded_keys, before_values=guarded_values;
    fprintf(stderr,"SORT_CASE %zu %zu %d\n",cases,n,expected_errors);
    assert(verifySort(guarded_keys.data()+1,guarded_values.data()+1,guarded_input.data()+1,n)==expected);
    assert(guarded_input==before_input && guarded_keys==before_keys && guarded_values==before_values);
    ++cases;
}

int main() {
    for (uint n=1; n<=6; ++n) for (uint pattern=0; pattern<3; ++pattern) {
        std::vector<uint> input(n), indices(n), keys(n);
        for (uint i=0;i<n;++i) input[i]=pattern==0 ? 7u : pattern==1 ? i*17u : (i%3==0 ? ~0u : i%3);
        std::iota(indices.begin(),indices.end(),0u);
        do {
            for (uint i=0;i<n;++i) keys[i]=input[indices[i]];
            check(input,keys,indices);
        } while (std::next_permutation(indices.begin(),indices.end()));
    }
    std::vector<uint> input(32), indices(32);
    std::iota(input.begin(),input.end(),1u); std::iota(indices.begin(),indices.end(),0u);
    check(input,input,indices,0);
    for (size_t at : {size_t(0),size_t(15),size_t(31)}) {
        for (uint invalid : {32u,~0u}) {
            auto values=indices; values[at]=invalid; check(input,input,values,1);
        }
        auto keys=input; keys[at]=~0u; check(input,keys,indices,at==31 ? 1 : 2);
        const size_t other=at==31 ? at-1 : at+1;
        auto values=indices; keys=input; keys[at]=keys[other]; values[at]=values[other];
        check(input,keys,values,1); // sorted duplicate pair with one missing pair
        values=indices; std::swap(values[at],values[other]); check(input,input,values,2);
    }
    check(input,std::vector<uint>(32,0),indices,32); // sorted, wrong multiset
    auto values=indices;
    for (size_t i=20;i<32;++i) values[i]=~0u;
    check(input,input,values,12); // complete count, only ten diagnostics
    fail_seen=true; check(input,input,indices,-2); fail_seen=false;
    printf("PASS: %zu independent complete sort-pair fixtures and immutable input/output guards\n",cases);
}
