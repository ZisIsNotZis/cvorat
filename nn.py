from torch.nn import *
from flash_attn import flash_attn_func, flash_attn_kvpacked_func, flash_attn_qkvpacked_func
from IPython.display import clear_output
from inspect import getsource
import torch
import numpy as np
import matplotlib.pyplot as plt


class Seq(ModuleList):
    def __init__(s, *_):
        super().__init__(_)
    def forward(s, _):
        for s in s:
            _ = s(_)
        return _


class Res(Sequential):
    def forward(s, _):
        return super().forward(_) + _


class Add(Sequential):
    def forward(s, _):
        return sum(s(_)for s in s)


class Mul(Sequential):
    def forward(s, _):
        return s[0](_) * s[1](_)


class T(Sequential):
    def forward(s, _):
        return super().forward(_.mT).mT


class Attn(Linear):
    def __init__(s, i, qk=0, v=0, g=1, h=1, r=(4, 4), drop=0., alibi=1., imp='fa'):
        super().__init__(i, g*h*(qk or i)+g*qk+g*v)
        s.__dict__ |= locals()

    def forward(s, x):
        _ = super().forward(x)
        if s.imp == 'fa' and s.qk == s.v and s.h == 1 and s.qk:
            return flash_attn_qkvpacked_func(
                _.unflatten(-1, (3, s.g, -1)),
                dropout_p=s.drop,
                window_size=s.r,
                alibi_slopes=torch.arange(s.g*s.h*s.alibi, device='cuda') if s.alibi else None,
            ).flatten(-2)
        q = _[..., :s.g*s.h*(s.qk or s.i)].unflatten(-1, (s.g * s.h, -1))
        if s.imp == 'fa' and s.qk == s.v:
            return flash_attn_kvpacked_func(
                q,
                _[..., s.g*s.h*s.qk:].unflatten(-1, (2, s.g, -1))if s.qk else x[..., None, None, :].expand((*x.shape[:-1], 2, -1, -1)),
                dropout_p=s.drop,
                window_size=s.r,
                alibi_slopes=torch.arange(s.g*s.h*s.alibi, device='cuda') if s.alibi else None,
            ).flatten(-2)
        k = _[..., s.g*s.h*s.qk:-s.g*s.v].unflatten(-1, (s.g, -1)) if s.qk else x[..., None, :]
        v = _[..., -s.g*s.v:].unflatten(-1, (s.g, -1)) if s.v else x[..., None, :]
        if s.imp == 'fa':
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=s.drop,
                window_size=s.r,
                alibi_slopes=torch.arange(s.g*s.h*s.alibi, device='cuda') if s.alibi else None,
            ).flatten(-2)
        if s.imp == 'sdpa':
            return functional.scaled_dot_product_attention(
                q.transpose(-3, -2),
                k.transpose(-3, -2),
                v.transpose(-3, -2),
                dropout_p=s.drop,
                enable_gqa=True
            ).transpose(-3, -2).flatten(-2)
        q = q.unflatten(-2, (s.g, -1))
        if not s.r:
            return ((q @ k.permute(0, 2, 3, 1)[:, None]).softmax(-1) @ v.permute(0, 2, 1, 3)[:, None]).flatten(-3)
        k = functional.pad(k, (0, 0, 0, 0) + s.r)
        v = k if s.qk == s.v == 0 else functional.pad(v, (0, 0, 0, 0) + s.r)
        k = k.as_strided(q.shape[:-2]+(k.size(-1), sum(s.r)+1), k.stride()+(k.stride(-3),))
        v = v.as_strided(q.shape[:-2]+(sum(s.r)+1, v.size(-1)), v.stride()[:-1]+(v.stride(-3), v.stride(-1)))
        return ((q @ k).softmax(-1) @ v).flatten(-3)


class SwiGLU(Linear):
    def __init__(s, i, h):
        super().__init__(i, h)
        s._ = Linear(i, h)

    def forward(s, _):
        return functional.silu(super().forward(_)) * s._(_)


class F(Module):
    def __init__(s, f, *_, **__):
        super().__init__()
        s.__dict__ |= locals()

    def forward(s, _):
        return s.f(*s._, _, **s.__)

    def extra_repr(s):
        return getsource(s.f).strip()[2:-2] if getattr(s.f, '__name__', 0) == '<lambda>' else f'{s.f}{"".join(f", {s}"for s in s._)}{"".join(f", {s}={_}"for s, _ in s.__.items())}'


def rope(x, cosins={}):
    cosin = cosins.get((x.size(-2), x.size(-1)))
    if cosin is None:
        cosin = torch.logspace(0, -4, x.size(-1)//2, dtype=x.dtype, device=x.device)
        cosin = cosin * torch.arange(x.size(-2), dtype=x.dtype, device=x.device)[:, None]
        cosin = cosin.cos(), cosin.sin_()
        cosins[(x.size(-2), x.size(-1))] = cosin
    cos, sin = cosin
    x0, x1 = x.chunk(2, -1)
    return torch.cat((x0*cos-x1*sin, x0*sin+x1*cos), -1)


def myrope(x, l=1., k=1., cosins={}):
    l = l if isinstance(l, int) else l*x.size(-2)
    k = k if isinstance(k, int) else k*x.size(-1)
    cosin = cosins.get((l, k, x.shape[-2:]))
    if cosin is None:
        cosin = torch.linspace(0, torch.pi*2*l, x.size(-2), dtype=x.dtype, device=x.device)[:, None] * torch.linspace(1, k, x.size(-1)//2, dtype=x.dtype, device=x.device)
        cosin = cosin.cos(), cosin.sin_()
        cosins[l, k, x.shape[-2:]] = cosin
    cos, sin = cosin
    x0, x1 = x.chunk(2, -1)
    return torch.cat((x0*cos-x1*sin, x0*sin+x1*cos), -1)


class Plot(list):
    def __init__(s, _=1, skip=10, acc=50, show=1):
        super().__init__([np.empty(0, 'f'), [], 0]for _ in range(_))
        s.skip = s.skip_ = skip
        s.acc = s.acc_ = acc
        s.show = show

    def __call__(s, *_):
        s.skip -= 1
        if s.skip:
            return
        s.skip = s.skip_
        s.acc -= 1
        for j, (i, _) in enumerate(zip(s, _), 1):
            if hasattr(_, '__call__'):
                _ = _()
            i[2] += _.data.float()
            if s.acc:
                continue
            i[1].append(i[2])
            i[2] = 0
            if len(i[1]) != s.show:
                continue
            i[0] = np.concatenate((i[0], torch.stack(i[1]).cpu()/s.acc_))
            i[1].clear()
            plt.subplot(1, len(s), j).plot(i[0], label=f'{i[0][-1]:.3f}')
            plt.legend()
        if s.acc:
            return
        s.acc = s.acc_
        if s[0][1]:
            return
        clear_output(True)
        plt.show()


def remap(x, n, group=1, loc='order', span='lin'):
    x = x.unflatten(-1, (group, -1)).transpose(-3, -2)
    l = x[..., None, :, 0]
    if loc == 'order':
        l = l.softmax(-1)
        l = l.cumsum(-1) - l/2
    elif loc == 'end':
        l = l - loc.min()
        l = l / loc.max()
    elif loc == 'free':
        l = l.sigmoid()
    else:
        raise KeyError(loc)
    s = x[..., None, :, 1]
    if span == 'lin':
        s = s.exp()
        l = l * s
        l = functional.pad((l[..., :-1]+l[..., 1:])/(s[..., :-1]+s[..., 1:]), (1, 1))
        l[..., -1] = 1
        n = torch.linspace(0, 1, n+1, dtype=x.dtype, device=x.device)[:, None]
        weight = (l[..., 1:].minimum(n[1:]) - l[..., :-1].maximum(n[:-1])).relu()
    elif span == 'std':
        n = torch.linspace(0, 1, n, dtype=x.dtype, device=x.device)[:, None]
        weight = n.sub(loc).div(s).square().mul(-.5).softmax(-1)
        weight = weight / weight.sum(-2, True)
    else:
        raise KeyError(span)
    x = x[..., 2:]
    return (weight @ x).transpose(-3, -2).flatten(-2)
