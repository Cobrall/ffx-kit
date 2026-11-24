import numpy as np
F32 = np.float32
np.seterr(divide="ignore", invalid="ignore")  # inf/NaN handled by sat(), GPU-style

def _u(x): return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
def _f(u): return np.ascontiguousarray(u, dtype=np.uint32).view(np.float32)
def lo_rcp(x):  return _f(np.uint32(0x7EF07EBB) - _u(x))                  # APrxLoRcpF1
def med_rcp(x):                                                           # APrxMedRcpF1
    b = _f(np.uint32(0x7EF19FFF) - _u(x)); return (b*(-b*np.asarray(x,F32)+F32(2.0))).astype(F32)
def lo_sqrt(x): return _f((_u(x) >> np.uint32(1)) + np.uint32(0x1FBC4639)) # APrxLoSqrtF1
def sat(x):  # D3D rules: saturate(NaN)=0, saturate(+inf)=1, saturate(-inf)=0
    return np.where(np.isnan(x), F32(0.0), np.clip(x, F32(0.0), F32(1.0))).astype(F32)

def cas_golden(img, sharpness=0.0, better_diagonals=False, go_slower=False, slow=False):
    """Exact mirror of CasFilter from ffx_cas.h, fp32, all 8 variants."""
    x = np.asarray(img, F32)
    p = np.pad(x, ((1,1),(1,1),(0,0)), mode="edge")
    a,b,c = p[:-2,:-2], p[:-2,1:-1], p[:-2,2:]
    d,e,f = p[1:-1,:-2], p[1:-1,1:-1], p[1:-1,2:]
    g,h,i = p[2:,:-2], p[2:,1:-1], p[2:,2:]
    mn = np.minimum(np.minimum(d, np.minimum(e,f)), np.minimum(b,h))
    mx = np.maximum(np.maximum(d, np.maximum(e,f)), np.maximum(b,h))
    if better_diagonals:
        mn = mn + np.minimum(np.minimum(mn, np.minimum(a,c)), np.minimum(g,i))
        mx = mx + np.maximum(np.maximum(mx, np.maximum(a,c)), np.maximum(g,i))
    rcpM = (F32(1.0)/mx) if go_slower else lo_rcp(mx)
    limit = F32(2.0) if better_diagonals else F32(1.0)
    amp = sat(np.minimum(mn, limit-mx) * rcpM)
    amp = np.sqrt(amp).astype(F32) if go_slower else lo_sqrt(amp)
    peak = F32(-1.0)/(F32(8.0)+(F32(5.0)-F32(8.0))*sat(F32(sharpness)))   # CasSetup
    w = (amp*peak).astype(F32)
    out = np.empty_like(x)
    for ch in range(3):
        wch = w[...,ch] if slow else w[...,1]   # without CAS_SLOW weight comes from green
        rw = (F32(1.0)/(F32(1.0)+F32(4.0)*wch)) if go_slower else med_rcp(F32(1.0)+F32(4.0)*wch)
        out[...,ch] = sat((b[...,ch]*wch + d[...,ch]*wch + f[...,ch]*wch + h[...,ch]*wch + e[...,ch])*rw)
    return out

EXACT = dict(better_diagonals=True, go_slower=True, slow=True)
