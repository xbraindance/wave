#!/usr/bin/env python3
"""Derive the WVC voice-card input network transfer function (service manual p.47).

SIG_x -> R4000/C4004 -> R4001/C4002 -> TL062 inverting stage (R4002, R4003+C4003,
R4004||C4001+R4005, C4000) -> R4006/C4005 (+R4014||C4009) -> SIN via C4006.
Ideal op-amp, ideal source. The overall inversion is dropped (all voices share it).
Prints analog biquad sections for ReconstructionStage in Source/Dsp/WaldorfAsic.cpp.
"""
import numpy as np, sympy as sp

s = sp.symbols('s')
R4000, C4004, R4001, C4002 = 330, 15e-9, 10e3, 470e-12
R4002, R4003, C4003 = 220e3, 180e3, 220e-12
R4004, R4005, C4001, C4000 = 330e3, 220e3, 47e-9, 33e-12
R4006, C4005, R4014, C4009 = 1e3, 6.8e-9, 47e3, 220e-12

Yb = 1 / (R4003 + 1 / (s * C4003))
Yin = 1 / R4002 + Yb
Zf = 1 / (s * C4000 + 1 / (R4005 + 1 / (1 / R4004 + s * C4001)))
v1, v2 = sp.symbols('v1 v2')
sol = sp.solve([(1 - v1) / R4000 - v1 * s * C4004 - (v1 - v2) / R4001,
                (v1 - v2) / R4001 - v2 * (s * C4002 + Yin)], [v1, v2])
vout = sol[v2] * Yin * Zf                       # inverting stage (sign dropped)
Yl = s * C4005 + s * C4009 + 1 / R4014
h = sp.cancel(sp.together(vout / (1 + R4006 * Yl)))
num, den = [sp.Poly(p, s).all_coeffs() for p in sp.fraction(h)]
num = np.array([float(c) for c in num]); den = np.array([float(c) for c in den])
k = num[0] / den[0]
z, p = np.roots(num), np.roots(den)

def sections(roots):
    r = sorted(roots, key=lambda x: (abs(x.imag) < 1e-9 * abs(x), abs(x)))
    out, used = [], [False] * len(r)
    for i, a in enumerate(r):
        if used[i]: continue
        used[i] = True
        if abs(a.imag) > 1e-9 * abs(a):
            j = min((j for j in range(len(r)) if not used[j]), key=lambda j: abs(r[j] - a.conjugate()))
            used[j] = True; out.append((1.0, -2 * a.real, abs(a) ** 2))
        else:
            out.append(None if False else ('real', a.real))
    return out

print('gain(dc-normalised k)=%.6g' % k)
print('poles (Hz):', ['%.4g%+.4gj' % (x.real / 2 / np.pi, x.imag / 2 / np.pi) for x in p])
print('zeros (Hz):', ['%.4g%+.4gj' % (x.real / 2 / np.pi, x.imag / 2 / np.pi) for x in z])
for f in (50, 200, 1e3, 4e3, 8e3, 15e3, 20e3, 30e3, 60e3):
    hv = np.polyval(num, 2j * np.pi * f) / np.polyval(den, 2j * np.pi * f)
    print('%8.0f Hz %6.2f dB' % (f, 20 * np.log10(abs(hv))))
