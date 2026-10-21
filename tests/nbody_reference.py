#!/usr/bin/env python3
"""Check N-body comparator boundaries and analytic CPU dynamics under sanitizers."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def host_source(source):
    for definition in ('#define TILE_SIZE 256', '#define SOFTENING 1e-5f', '#define DT        0.01f'):
        if source.count(definition)!=1:raise ValueError('N-body constants changed; review analytic fixtures')
    a='static void nbody_cpu(';b='// ---------------------------------------------------------------------------\n// Simple deterministic PRNG'
    if source.count(a)!=1 or source.count(b)!=1:raise ValueError('N-body reference/comparator anchors changed')
    return r'''#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <limits>
struct float3 {float x,y,z;};
struct float4 {float x,y,z,w;};
#define SOFTENING 1e-5f
#define DT 0.01f
''' + a + source.split(a,1)[1].split(b,1)[0] + r'''
static void require(bool condition) {if(!condition) {fprintf(stderr,"reference check failed\n");std::abort();}}
static float& component(float4& value,int c) {
    switch(c) {case 0:return value.x;case 1:return value.y;case 2:return value.z;default:return value.w;}
}
int main() {
    unsigned checks=0;
    // Every got/reference component must be finite, even matching infinities.
    const float4 ref={1.f,2.f,3.f,1.f};
    require(nbody_vector_matches(ref,ref));++checks;
    for(int side=0;side<2;++side) for(int c=0;c<4;++c)
    for(float bad : {NAN,INFINITY,-INFINITY}) {
        float4 got=ref,expected=ref;
        component(side ? expected : got,c)=bad;
        require(!nbody_vector_matches(got,expected));++checks;
    }
    // Relative XYZ allowance, absolute near-zero floor, and exact immutable w.
    for(int c=0;c<3;++c) {
        float4 got=ref;component(got,c)+=0.059f;require(nbody_vector_matches(got,ref));++checks;
        got=ref;component(got,c)+=0.061f;require(!nbody_vector_matches(got,ref));++checks;
        float4 zero={0,0,0,0};got=zero;component(got,c)=1e-5f;require(nbody_vector_matches(got,zero));++checks;
        component(got,c)=std::nextafter(1e-5f,INFINITY);require(!nbody_vector_matches(got,zero));++checks;
    }
    float4 got=ref;got.w=std::nextafter(ref.w,INFINITY);require(!nbody_vector_matches(got,ref));++checks;
    float max=std::numeric_limits<float>::max();float4 huge={max,max,max,1};require(!nbody_vector_matches(huge,huge));++checks;
    // One isolated particle has no force: velocity/mass stay fixed and position
    // advances by v*dt on each complete step (multiple nonzero velocity axes).
    float4 p={0,0,0,2},v={1,-2,3,0},expected=p;
    for(int step=0;step<10;++step) {
        expected.x+=DT;expected.y+=-2*DT;expected.z+=3*DT;
        nbody_cpu(&p,&v,1);
        require(p.x==expected.x && p.y==expected.y && p.z==expected.z && p.w==2);
        require(v.x==1 && v.y==-2 && v.z==3 && v.w==0);++checks;
    }
    // Equal masses at +/-1 remain symmetric; the first acceleration magnitude
    // is 2/(4+softening)^(3/2), independently evaluated in double precision.
    float4 pair[2]={{-1,0,0,1},{1,0,0,1}},vel[2]={{0,0,0,0},{0,0,0,0}};
    double speed=2.0/std::pow(4.0+double(SOFTENING),1.5)*double(DT);
    nbody_cpu(pair,vel,2);
    require(std::fabs(double(vel[0].x)-speed)<1e-8 && vel[0].x==-vel[1].x);
    require(std::fabs(double(pair[0].x)-(-1+speed*double(DT)))<1e-7 && pair[0].x==-pair[1].x);++checks;
    for(int step=0;step<9;++step) {
        nbody_cpu(pair,vel,2);
        require(pair[0].x==-pair[1].x && vel[0].x==-vel[1].x && pair[0].y==0 && pair[1].z==0);++checks;
    }
    printf("%u analytic dynamics/comparator checks pass\n",checks);
}
'''


def parameter_source(source):
    a='static int parseIntParam(';b='// ---------------------------------------------------------------------------\n// Constants'
    c='int main(';d='    // Print device info'
    for anchor in (a,b,c,d):
        if source.count(anchor)!=1:raise ValueError('N-body parameter anchor changed')
    return '#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <climits>\n#define TILE_SIZE 256\n'+a+source.split(a,1)[1].split(b,1)[0]+c+source.split(c,1)[1].split(d,1)[0]+'    printf("%d,%d\\n",N,timesteps);\n}\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compiler',default='c++');ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'tier2/cuda_nbody/hip/cuda_nbody.hip');args=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix='nbody-reference-') as tmp:
        cpp=Path(tmp)/'reference.cpp';binary=Path(tmp)/'reference';cpp.write_text(host_source(args.source.read_text()))
        subprocess.run([args.compiler,'-std=c++14','-O3','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
        subprocess.run([str(binary)],check=True,timeout=60)
        cpp.write_text(parameter_source(args.source.read_text()))
        subprocess.run([args.compiler,'-std=c++14','-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(cpp),'-o',str(binary)],check=True,timeout=60)
        clean={k:v for k,v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
        cases=[([],{},'16384,10'),(['--bodies','0','--timesteps','-1'],{},'16384,10'),
               (['--bodies','17','--timesteps','3'],{},'17,3'),
               (['--bodies','17'],{'bodies':'33','timesteps':'2'},'33,2')]
        for n in (-2147483648,-1,0,1,255,256,257,2147483392,2147483393,2147483647):
            for t in (-1,0,1,10,2147483647):
                cases.append(([],dict(bodies=str(n),timesteps=str(t)),f'{n},{t}' if 0<n<=2147483392 and t>0 else None))
        for argv,values,want in cases:
            env=dict(clean,**{'BENCH_PARAM_'+k:v for k,v in values.items()})
            p=subprocess.run([str(binary),*argv],env=env,text=True,capture_output=True,timeout=30)
            if want is None:
                if p.returncode!=1 or p.stdout or p.stderr!='Invalid bodies/timesteps or padded body index range\n':
                    raise AssertionError((argv,values,p.returncode,p.stdout,p.stderr))
            elif p.returncode or p.stderr or p.stdout.strip()!=want:
                raise AssertionError((argv,values,want,p.returncode,p.stdout,p.stderr))
        print(f'{len(cases)} production parameter/index cases pass without allocation or GPU access')


if __name__=='__main__':main()
