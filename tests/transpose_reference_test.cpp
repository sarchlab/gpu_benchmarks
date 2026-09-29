// The driver prepends the actual production CPU transpose.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const float input[] = {1,2,3,4,5,6}, expected[] = {1,4,2,5,3,6};
    float output[6];
    transpose_cpu(input,output,3,2);
    assert(std::memcmp(output,expected,sizeof(output)) == 0);
    size_t cases=1;
    for (int width : {1,2,7,31,32,33,65,129})
        for (int height : {1,2,7,31,32,33,65,129}) {
            const size_t count=(size_t)width*height;
            std::vector<float> source(count+2,12345.0f), result(count+2,54321.0f);
            for (size_t i=0;i<count;++i) source[i+1]=0.125f*((int)i-8000);
            source[count]=-0.0f;
            const auto before=source;
            transpose_cpu(source.data()+1,result.data()+1,width,height);
            // Destination-flat index mapping is independent of the source's
            // nested source-row/source-column traversal.
            for (size_t dst=0;dst<count;++dst) {
                const size_t src=(dst%height)*width+dst/height;
                assert(std::memcmp(&result[dst+1],&source[src+1],sizeof(float)) == 0);
            }
            assert(result.front()==54321.0f && result.back()==54321.0f);
            assert(std::memcmp(source.data(),before.data(),source.size()*sizeof(float)) == 0);
            std::vector<float> round_trip(count);
            transpose_cpu(result.data()+1,round_trip.data(),height,width);
            assert(std::memcmp(round_trip.data(),source.data()+1,count*sizeof(float)) == 0);
            ++cases;
        }
    printf("PASS: %zu complete rectangular/square transpose fixtures, bit preservation and guards\n",cases);
}
