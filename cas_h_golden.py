import numpy as np
F32 = np.float32
F16 = np.float16

def _u16(x): return np.ascontiguousarray(x, dtype=F16).view(np.uint16)
def _f16(u): return np.ascontiguousarray(u, dtype=np.uint16).view(F16)

def lo_rcp_h(x):  return _f16(np.uint16(0x7784) - _u16(x))
def med_rcp_h(x):
    b = _f16(np.uint16(0x778d) - _u16(x))
    return (b * (-b * x.astype(F32) + F32(2.0))).astype(F16)
def lo_sqrt_h(x): return _f16((_u16(x) >> np.uint16(1)) + np.uint16(0x1de2))
def sat_h(x): return np.clip(x, F16(0.0), F16(1.0))

def cas_h_golden(img, sharpness=0.0, better_diagonals=False, go_slower=False, slow=False):
    x = np.asarray(img, F16)
    p = np.pad(x, ((1,1),(1,1),(0,0)), mode="edge")
    a,b,c = p[:-2,:-2], p[:-2,1:-1], p[:-2,2:]
    d,e,f = p[1:-1,:-2], p[1:-1,1:-1], p[1:-1,2:]
    g,h,i = p[2:,:-2], p[2:,1:-1], p[2:,2:]
    mn = np.minimum(np.minimum(d, np.minimum(e,f)), np.minimum(b,h))
    mx = np.maximum(np.maximum(d, np.maximum(e,f)), np.maximum(b,h))
    if better_diagonals:
        mn = mn + np.minimum(np.minimum(mn, np.minimum(a,c)), np.minimum(g,i))
        mx = mx + np.maximum(np.maximum(mx, np.maximum(a,c)), np.maximum(g,i))
    rcpM = (F16(1.0)/mx.astype(F32)).astype(F16) if go_slower else lo_rcp_h(mx)
    limit = F16(2.0) if better_diagonals else F16(1.0)
    amp = sat_h(np.minimum(mn, limit-mx).astype(F32) * rcpM.astype(F32))
    amp = np.sqrt(amp.astype(F32)).astype(F16) if go_slower else lo_sqrt_h(amp)
    peak = F16(-1.0) / (F16(8.0) + (F16(5.0)-F16(8.0)) * sat_h(F16(sharpness)))
    w = (amp.astype(F32) * peak.astype(F32)).astype(F16)
    out = np.empty_like(x)
    for ch in range(3):
        wch = w[...,ch] if slow else w[...,1]
        rw = (F16(1.0) / (F16(1.0) + F16(4.0) * wch.astype(F32)).astype(F16)).astype(F16) if go_slower else med_rcp_h(F16(1.0) + F16(4.0) * wch)
        out[...,ch] = sat_h((b[...,ch].astype(F32)*wch.astype(F32) + d[...,ch].astype(F32)*wch.astype(F32) + 
                             f[...,ch].astype(F32)*wch.astype(F32) + h[...,ch].astype(F32)*wch.astype(F32) + 
                             e[...,ch].astype(F32)).astype(F16) * rw)
    return out

EXACT_H = dict(better_diagonals=True, go_slower=True, slow=True)

if __name__ == "__main__":
    flat = np.full((64,64,3), 0.5, F16)
    print("fp16 exact flat:", np.abs(cas_h_golden(flat,1.0,**EXACT_H).astype(F32)-flat.astype(F32)).max())
    print("fp16 fast  flat:", np.abs(cas_h_golden(flat,1.0).astype(F32)-flat.astype(F32)).max())
