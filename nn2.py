import torch
import numpy as np
import matplotlib.pyplot as plt
import triton
import triton.language as tl
from torch.nn import *
try:
    from flash_attn import flash_attn_func
except ModuleNotFoundError:
    pass
from IPython.display import clear_output
from functools import reduce
torch._dynamo.config.cache_size_limit = 64


class Plot(dict):
    t = 1
    w = 1

    def __call__(S, **a):
        if S.t//S.w == 2048:
            for s, c in S.values():
                s[:1024] = s.unflatten(0, (-1, 2)).sum(1)
                s[1024:].zero_()
                c.reshape(-1, 2).sum(1, out=c[:1024])
                c[1024:] = 0
            S.w *= 2
        for i, a in a.items():
            s, c = S.get(i, '  ')
            if s == ' ':
                S[i] = s, c = [torch.zeros(2048, len(a)if a.ndim else 1, device=a.device), np.zeros(2048, 'H')]
            s[S.t//S.w] += a
            c[S.t//S.w] += 1
        S.t += 1
        if not S.t % 256:
            S.show()

    def show(S):
        for i, (s, c) in enumerate(S.values(), 1):
            plt.subplot(1, len(S), i).plot(s.cpu()/c[:, None])
        clear_output(True)
        plt.show()


class Res(Sequential):
    def forward(s, _):
        return super().forward(_) + _


class Add(Sequential):
    def forward(s, _):
        return reduce(torch.add, (s(_)for s in s))


class Zip(Sequential):
    def forward(s, _):
        return reduce(torch.add, (s(_)for s, _ in zip(s, _ if isinstance(_, (list, tuple))else [_])if _ is not None))


class T(Sequential):
    def forward(s, _):
        return super().forward(_.mT).mT


class F(Module):
    def __init__(s, _):
        super().__init__()
        s._ = _

    def forward(s, _):
        return s._(_)


class Lora(Linear):
    def __init__(s, i, h, O, o, b=False):
        h, O, o = arg(i, h, O, o)
        s.__dict__ |= locals()
        super().__init__(i, min(O*o, (O+o)*h), b)

    def forward(s, _):
        _ = super().forward(_)
        return (lambda _, t: _@t.mT)(*_.unflatten(-1, (-1, s.h)).split((s.O, s.o), -2))if _.size(-1) < s.O*s.o else _.unflatten(-1, (s.O, -1))


class Cnv(Conv1d):
    def __init__(s, i, h=1, H=1, o=1., g=2., b=False, r=(3, 3)):
        h, H, o, g = arg(i, h, H, o, g)
        super().__init__(g*h or i, g*H, r[0]*2+1, padding=r, groups=g, bias=0)
        s.w = Linear(i, g*h, 0)if h else id
        s._ = Linear(g*H, o, b)if o else id

    def forward(s, _):
        return s._(super().forward(s.w(i)))


class Dycnv(Linear):
    def __init__(s, i, h=1, H=1, o=1., g=2., b=False, r=(3, 3)):
        h, H, o, g = arg(i, h, H, o, g)
        s.__dict__ |= locals()
        super().__init__(i, g*(h+H*(h or i)*(sum(r)+1)))
        s._ = Linear(g*H, o, b)if o else id

    def forward(s, _):
        x, t = super().forward(_).split((s.g*s.h, s.g*s.H*(s.h or s.i)*(sum(s.r)+1)), 2)
        return s._(torch.einsum('...gir,...goir->...go', torch.constant_pad_nd(x if s.h else _, (0, 0, *s.r)).unfold(1, sum(s.r)+1, 1).unflatten(2, (s.g, -1)), t.unflatten(2, (s.g, s.H, s.h or s.i, -1))).flatten(2))


class Cumavg(Module):
    def __init__(s, i, o=1., b=False):
        o, = arg(i, o)
        super().__init__()
        s._ = Linear(i, o, b)if o else id

    def forward(s, _):
        return s._(_.cumsum(-2)/torch.arange(1, _.size(-2)+1, dtype=_.dtype, device=_.device)[:, None])


class Cummax(Linear):
    def __init__(s, i, h=1., o=1., b=False):
        h, o = arg(i, h, o)
        super().__init__(i, h)
        s._ = Linear(h, o, b)if o else id

    def forward(s, _):
        return s._(super().forward(_).cummax(-2)[0])


class Cumdecay(Linear):
    def __init__(s, i, h=1., o=1., b=False, decay=[1], decay_=[4], norm=[-2], norm_=[4]):
        h, hr, o = arg(i, h, hr, o)
        super().__init__(i, h*len(norm)+hr*len(normr))
        s.d = Parameter(torch.zeros(len(norm), 1))
        s.n = Parameter(torch.zeros(len(norm), 1))
        s.decay = Buffer(torch.tensor(decay)[:, None])
        s.decay_ = Buffer(torch.tensor(decay_)[:, None])
        s.norm = Buffer(torch.tensor(norm)[:, None])
        s.norm_ = Buffer(torch.tensor(norm_)[:, None])
        s._ = Linear(h*len(norm), o, b)if o else id

    def forward(s, _):
        return s._(cum(super().forward(_).mT.unflatten(-2, [-1, len(s.norm)]), s.d.sigmoid()*s.decay_+s.decay, (s.n.sigmoid()*s.norm_+s.norm).exp()).flatten(-2).mT)


class At(Linear):
    def __init__(s, i, h=1., o=1., G=4, g=1, b=False, r=(8, 7), p='fro', d=-4):
        h, o, g, G = arg(i, h, o, g, G)
        s.__dict__ |= locals()
        del s.__dict__['d']
        super().__init__(i, G*(g+2)*h)
        s.d = d
        s._ = Linear((1 if p else G*g)*h, o, b)if o else id

    def forward(s, _):
        _, k, v = super().forward(_).unflatten(2, (-1, s.h)).split((s.G*s.g, s.G, s.G), 2)
        _ = flash_attn_func(rope(_), rope(k), v, window_size=s.r)
        return s._(_.norm(s.p, 2)if s.p else _.flatten(2))


class Fpol(Linear):
    def __init__(s, i, h=1., o=1., g=4, b=False):
        h, o, g = arg(i, h, o, g)
        s.__dict__ |= locals()
        super().__init__(i, (h+1)*g)
        s._ = Linear(h, o, b)if o else id

    def forward(s, _):
        _, t = super().forward(_).split((s.h*s.g, s.g), 2)
        return s._((_.unflatten(-1, (-1, s.g)) @ t[..., None])[..., 0])


class Swi(Linear):
    def __init__(s, i, h=4., o=1., b=False):
        h, o = arg(i, h, o)
        super().__init__(i, 2*h)
        s._ = Linear(h, o, b)if o else id

    def forward(s, _):
        return s._(functional.silu(torch.mul(*super().forward(_).chunk(2, -1))))


class Dl(torch.utils.data.DataLoader):
    def __init__(s, a='../../py.bin', n=4, l=8192, p=.9):
        s.__dict__ |= locals()
        s.a = torch.from_numpy(np.memmap(a, 'B', 'r'))
        s.a.as_strided_([len(s.a)-l+1, l], [1, 1])

    def __iter__(s):
        return s

    def __next__(s):
        _ = s.a[torch.randint(len(s.a), [s.n])].to('cuda', torch.long)
        return _.put(torch.randint(_.numel(), [int(_.numel()*s.p)], device='cuda'), torch.randint(256, [int(_.numel()*s.p)], device='cuda')), _


def rope(x, d=-4):
    _ = torch.logspace(0, 1, x.size(-1)//2, dtype=x.dtype, device=x.device)**d
    _ = _ * torch.arange(x.size(-2), dtype=x.dtype, device=x.device)[:, None]
    cos = _.cos()
    sin = _.sin()
    x, _ = x.chunk(2, -1)
    return torch.cat((x*cos-_*sin, x*sin+_*cos), -1)


def arg(i, *_):
    return (round(i*_)if isinstance(_, float)else _ for _ in _)


@triton.jit
def cumdecay_fuseker(A, B, l, an, al, ac, bn, bl, bc, decay, norm):
    abeg = tl.prograc_id(0)*an+tl.prograc_id(2)*ac
    bbeg = tl.prograc_id(0)*bn+tl.prograc_id(2)*bc
    len = (l+tl.num_progracs(1)-1)//tl.num_progracs(1)
    beg = tl.prograc_id(1)*len
    end = tl.minimum(beg+len, l)
    a_ = 0.
    a = -9e99
    decay = tl.log(decay)
    for i in range(tl.maximum(beg-tl.cast(tl.log(.02)/tl.log(decay), tl.int32), 0), beg):
        b = tl.load(A+abeg+i*al)
        a += decay
        b_ = tl.cast(b > 0, tl.float32)*2-1
        b = tl.log(tl.cast(tl.abs(b), tl.float32))+norm
        n = tl.maximum(a, b)
        a = a_*tl.exp(a-n)+b_*tl.exp(b-n)
        a_ = tl.cast(a > 0, tl.float32)*2-1
        a = tl.log(tl.abs(a))+n
    for i in range(beg, end):
        b = tl.load(A+abeg+i*al)
        a += decay
        b_ = tl.cast(b > 0, tl.float32)*2-1
        b = tl.log(tl.cast(tl.abs(b), tl.float32))+norm
        n = tl.maximum(a, b)
        a = a_*tl.exp(a-n)+b_*tl.exp(b-n)
        a_ = tl.cast(a > 0, tl.float32)*2-1
        a = tl.log(tl.abs(a))+n
        tl.store(B+bbeg+i*bl, a_*tl.exp(a-norm))


def cumdecay_fuse(A, decay=1., norm=1.):
    assert A.ndim == 3
    B = torch.empty_like(A)
    cumdecay_fuseker[A.size(0), 1, A.size(2)](A, B, A.size(1), *A.stride(), *B.stride(), float(decay), float(norm))
    return B


def bol(a):
    return a is not None or None


def add(a, b):
    return a+b if bol(b)else a


def mul(a, b):
    return a*b if bol(b)else a


def triu(a, b):
    return a.triu(1-b)if b else a


def cat(a, b):
    return torch.cat([a, b])if bol(b)else a


def cumdecay(a, logdecay=None, geom=None, wnd=None, sign=True, norm=True, type=torch.float32):
    otype = a.dtype
    a = a.to(type)
    logdecay = bol(logdecay) and torch.as_tensor(logdecay, dtype=type)
    if sign:
        sign = a.sign()
        a = a.abs()
    else:
        sign = None
    d = mul(torch.linspace(0, a.size(-1), a.size(-1)+1, dtype=type, device=a.device), logdecay)
    a = add(mul(a.log(), geom), bol(logdecay) and d[..., :-1])
    n = a.cummax(-1)[0]
    a = triu(mul((a[..., None, :]-n[..., None]).exp(), sign).tril(), wnd).sum(-1)
    if isinstance(sign, torch.Tensor):
        sign = a.sign()
        a = a.abs()
    wnd = wnd or 999999
    if not norm:
        d = None
    elif bol(logdecay):
        _ = (-(-d[..., 1:min(wnd, a.size(-1))+1]).exp()).log1p()
        d = logdecay.expm1().log() - d[..., 1:] - cat(_, _[..., -1:].expand(*[-1]*(d.ndim-1), a.size(-1)-wnd)if a.size(-1) > wnd else None)
    else:
        d = -torch.linspace(1, min(wnd, a.size(-1)), min(wnd, a.size(-1)), dtype=type, device=a.device).log()
        d = cat(d, d[-1].expand(a.size(-1)-wnd)if a.size(-1) > wnd else None)
    return mul(mul(add(a.log()+n, d), geom and 1/geom).exp(), sign).to(otype)


Seq = Sequential
def id(_): return _
