from fractions import Fraction as F
from pathlib import Path

def value(bits):
    sign=-1 if bits>>31 else 1
    exponent=(bits>>23)&255; fraction=bits&0x7fffff
    assert exponent!=255
    return sign * F(fraction+(1<<23) if exponent else fraction) * F(2)**(exponent-150 if exponent else -149)

def rounded(x):
    if not x:return 0
    sign=0x80000000 if x<0 else 0;x=abs(x)
    exponent=x.numerator.bit_length()-x.denominator.bit_length()
    if x<F(2)**exponent:exponent-=1
    unit=F(2)**max(-149,exponent-23)
    scaled=x/unit; q,r=divmod(scaled.numerator,scaled.denominator)
    if 2*r>scaled.denominator or (2*r==scaled.denominator and q%2):q+=1
    if q==1<<24:q>>=1;exponent+=1
    if exponent < -126:
        return sign|q
    assert exponent<=127
    return sign|((exponent+127)<<23)|(q-(1<<23))

assert rounded(F(1)+F(1,2**24))==0x3f800000
assert rounded(F(1)+F(3,2**24))==0x3f800002
assert rounded(F(1,2**150))==0
assert rounded(F(3,2**150))==2
assert rounded(F(2)**-126-F(2)**-150)==0x800000

def distance(point,center,fused=True,reverse=False):
    acc=0
    if reverse:point=point[::-1];center=center[::-1]
    for p,c in zip(point,center):
        delta=value(rounded(value(p)-value(c)))
        acc=rounded(delta*delta+value(acc)) if fused else rounded(value(rounded(delta*delta))+value(acc))
    return value(acc)

def trajectory(bits,n,d,k,t,fused=True,reverse=False):
    points=[bits[p*d:(p+1)*d] for p in range(n)]
    centers=[p[:] for p in points[:k]];history=[]
    for iteration in range(t):
        labels=[min(range(k),key=lambda c:distance(point,centers[c],fused,reverse)) for point in points]
        history+=labels
        for c in range(k):
            selected=[p for p in range(n) if labels[p]==c]
            if not selected:continue
            for f in range(d):
                total=0
                for p in selected:total=rounded(value(total)+value(points[p][f]))
                centers[c][f]=rounded(value(total)/len(selected))
    return history,sum(centers,[])

seed=42
def random_bits():
    global seed
    seed=(seed*1664525+1013904223)&0xffffffff
    return rounded(F(seed>>8,1<<24))

fixtures=[]
for n,d,k,t in [(7,3,2,4),(17,7,3,10),(17,34,5,4),(33,3,5,10)]:
    bits=[random_bits() for _ in range(n*d)]
    labels,centers=trajectory(bits,n,d,k,t)
    fixtures.append((n,d,k,t,bits,labels,centers))
for trial in range(10000):
    d=3;a=[random_bits() for _ in range(d)];b=[random_bits() for _ in range(d)]
    midpoint=[rounded((value(x)+value(y))/2) for x,y in zip(a,b)]
    bits=a+b+midpoint
    fused=trajectory(bits,3,d,2,1,True);unfused=trajectory(bits,3,d,2,1,False)
    if fused[0]!=unfused[0]:
        fixtures.append((3,d,2,1,bits,*fused));break
else:raise RuntimeError('No contraction-sensitive fixture found')
for trial in range(10000):
    d=7;a=[random_bits() for _ in range(d)];b=[random_bits() for _ in range(d)]
    bits=a+b+[rounded((value(x)+value(y))/2) for x,y in zip(a,b)]
    forward=trajectory(bits,3,d,2,1)
    reverse=trajectory(bits,3,d,2,1,reverse=True)
    if forward[0]!=reverse[0]:
        fixtures.append((3,d,2,1,bits,*forward));break
else:raise RuntimeError('No feature-order-sensitive fixture found')
out=['// Generated using exact rational arithmetic and IEEE binary32 ties-to-even.\n',
     'struct Golden { int n,d,k,t; std::vector<unsigned> points; std::vector<int> labels; std::vector<unsigned> centers; };\n',
     'static const std::vector<Golden> golden = {\n']
for n,d,k,t,bits,labels,centers in fixtures:
    fmt=lambda values:','.join(hex(v)+'u' for v in values)
    out.append('{'+f'{n},{d},{k},{t},'+'{'+fmt(bits)+'},{'+','.join(map(str,labels))+'},{'+fmt(centers)+'}},\n')
out.append('};\n')
Path(__file__).with_suffix('.h').write_text(''.join(out))
print('Generated',len(fixtures),'exact-rational complete-trajectory fixtures; fused/unfused labels:',fused[0],unfused[0])
