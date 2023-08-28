from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
import yaml
from common import *


class Model(nn.Module):
    def __init__(self, cfg=None, ch=3, nc=None, auc=True) -> None:
        super().__init__()
        with open(cfg, 'r') as f:
            self.yaml = yaml.safe_load(f)

        self.model, self.save, self.outputs, self.ch = parse_model(self.yaml, [ch])
        self.name = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = True
        self.auc = auc
        self.nc = nc
        if self.auc:
            self.classifer1 = Classify(self.ch[self.outputs[0]], self.nc)
            self.classifer2 = Classify(self.ch[self.outputs[1]], self.nc)
            self.classifer3 = Classify(self.ch[self.outputs[2]], self.nc)
        else:
            self.classifer = Classify(self.ch[-1], self.nc)

    def forward(self, x):
        y = []  # outputs
        outputs = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer?
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            outputs.append(x if m.i in self.outputs else None)

        if self.auc:
            output = []
            output.append(self.classifer1(outputs[self.outputs[0]]))
            output.append(self.classifer2(outputs[self.outputs[1]]))
            output.append(self.classifer3(outputs[self.outputs[2]]))

            return output
        else:
            return self.classifer(x)


def parse_model(d, ch):
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")

    no, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']

    layers, save, outputs, c2 = [], [], [], ch[-1]
    outputs.extend([17, 20, 23])
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
            BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist

        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), sorted(outputs), ch


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

#
# x = torch.randn(8, 3, 224, 224).to('cuda')
# model = Model(cfg='s.yaml', ch=3, nc=25, auc=False).to('cuda')
# output = model(x)
# print(output.shape)
