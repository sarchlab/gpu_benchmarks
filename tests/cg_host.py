#!/usr/bin/env python3
"""Run pinned CG host oracles, actual input admission, and worker cleanup checks."""
import argparse
import itertools
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    if not __debug__:
        raise RuntimeError('Run without Python optimization')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--all-matrices', action='store_true')
    ap.add_argument('--thread-sanitizer', action='store_true')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = (repo/'tier2/npb_cg/npb_cg.hip').read_text()
    header = (repo/'tier2/npb_cg/nas_cg.h').read_text()
    clean = {k: v for k, v in os.environ.items() if not k.startswith('BENCH_PARAM_')}
    flags = ['-std=c++14', '-O2', '-pthread', '-fno-sanitize-recover=all',
             '-fsanitize=thread' if args.thread_sanitizer else '-fsanitize=address,undefined']
    with tempfile.TemporaryDirectory(prefix='cg-host-') as folder:
        work = Path(folder)
        binary = work/'test'
        subprocess.run([args.compiler, *flags, str(repo/'tests/cg_reference.cpp'), '-o', str(binary)], check=True)
        subprocess.run([str(binary), *(['--all-matrices'] if args.all_matrices else [])], check=True,
                       timeout=600 if args.all_matrices else 120)
        # Faults after zero, one and six successful worker creations exercise
        # constructor cleanup, including waking and joining idle workers.
        anchor = '                threads.emplace_back([this,id]() {'
        assert header.count(anchor) == 1
        (work/'nas_cg.h').write_text(header.replace(anchor,
            '                if (static_cast<int>(id)-1==cg_test_fail_after) throw std::runtime_error("injected thread creation failure");\n'+anchor))
        cpp = work/'pool.cpp'
        cpp.write_text(r'''
static int cg_test_fail_after=-1;
#include "nas_cg.h"
int main() {
    const auto c=*nas_cg::problem(1400);
    const auto a=nas_cg::make_matrix(c);
    const auto baseline=nas_cg::reference(a,c,1);
    for(int round=0;round<4;++round) {
        for(int count : {0,1,6}) {
            cg_test_fail_after=count;
            bool threw=false;
            try { nas_cg::reference(a,c,8); }
            catch(const std::runtime_error& e) { threw=true; }
            if(!threw)return 1;
        }
        cg_test_fail_after=-1;
        if(nas_cg::compare(nas_cg::reference(a,c,8),baseline))return 1;
    }
    puts("12 partial worker-creation failures cleaned up; four subsequent complete references pass");
}
''')
        subprocess.run([args.compiler, *flags, str(cpp), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=120)
        if args.thread_sanitizer:
            return
        start = 'static int integer('
        end = 'static double ratio('
        assert source.count(start) == source.count(end) == 1
        cpp.write_text('#include <cerrno>\n#include <climits>\n#include <cstring>\n#include <cstdlib>\n'
            '#include "'+str(repo/'tier2/npb_cg/nas_cg.h')+'"\n'+
            start+source.split(start, 1)[1].split(end, 1)[0] + r'''
int main(int argc,char**argv) try {
    int b;const auto* c=parameters(argc,argv,&b);
    printf("%d,%c,%d,%d,%.0f,%d\n",c->n,c->name,c->nonzer,c->iterations,c->shift,b);
} catch(const std::exception& error) {fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
''')
        subprocess.run([args.compiler, *flags, str(cpp), '-o', str(binary)], check=True)
        classes = [(1400,'S',7,15,10),(7000,'W',8,15,12),(14000,'A',11,15,20),
                   (75000,'B',13,75,60),(150000,'C',15,75,110)]
        cases = []
        for c, b in itertools.product(classes, [1, 8, 32, 256, 1024]):
            cases.append((dict(matrix_size=str(c[0]), block_size=str(b)), [], ','.join(map(str, (*c,b)))))
        cases += [({}, [], '14000,A,11,15,20,256'),
                  (dict(matrix_size='7000'), ['--size','1400'], '1400,S,7,15,10,256')]
        for key in ['matrix_size', 'block_size']:
            for value in ['', 'junk', '1400x', '-2147483649', '2147483648', '999999999999999999999', '0', '-1', '3']:
                cases.append(({key: value}, [], None))
        cases += [(dict(matrix_size='50000'), [], None), (dict(block_size='2048'), [], None)]
        cases += [({key:value}, [], None) for key in ['nonzeros_per_row','max_cg_iterations'] for value in ['', '25']]
        cases += [({}, cli, None) for cli in [['--size'],['--other','1400'],['--size','bad'],['1400']]]
        for values, cli, want in cases:
            run = subprocess.run([str(binary), *cli], env=dict(clean, **{'BENCH_PARAM_'+k:v for k,v in values.items()}),
                                 text=True, capture_output=True, timeout=10)
            if want is None:
                assert run.returncode == 1 and not run.stdout and run.stderr.startswith('FAIL:'), (values, cli, run)
            else:
                assert run.returncode == 0 and not run.stderr and run.stdout.strip() == want, (values, cli, run)
        print(f'{len(cases)} actual NAS-class/input/CLI-precedence admission checks pass')


if __name__ == '__main__':
    main()
