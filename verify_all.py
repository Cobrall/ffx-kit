import hashlib, itertools, json, pathlib, subprocess, time
import numpy as np
import wgpu
from cas_golden import cas_golden
from run_gpu import find_dxc, build, run_gpu, test_image

HERE = pathlib.Path(__file__).resolve().parent

def images():
    rng = np.random.default_rng(7)
    yield "synthetic", test_image()
    yield "flat",  np.full((64,64,3), 0.5, np.float32)
    yield "black", np.zeros((64,64,3), np.float32)
    yield "white", np.ones((64,64,3), np.float32)
    imp = np.full((65,65,3), 0.2, np.float32); imp[32,32] = 0.9
    yield "impulse", imp
    yield "noise", rng.random((128,128,3), dtype=np.float32)

def big_image(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([xx/(W-1), ((xx//16+yy//16)%2)*0.8+0.1,
                    np.clip(0.5+0.4*np.sin(xx/7)*np.cos(yy/9), .02, .98)], -1)
    return np.clip(img*0.95+0.02, 0.01, 0.99).astype(np.float32)

def main():
    dxc = find_dxc()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    ver = subprocess.run([dxc, "--version"], capture_output=True, text=True).stdout.strip()
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "gpu": dict(adapter.info),
              "dxc": ver, "variants": {}}
    imgs = list(images()); ok = True; spv_fast = None
    for bd, gs, sl in itertools.product((0,1), repeat=3):
        defines = [n for n,v in (("CAS_BETTER_DIAGONALS",bd),("CAS_GO_SLOWER",gs),("CAS_SLOW",sl)) if v]
        tag = f"bd{bd}_gs{gs}_sl{sl}"
        spv = build(dxc, defines, tag)
        if tag == "bd0_gs0_sl0": spv_fast = spv
        kw = dict(better_diagonals=bool(bd), go_slower=bool(gs), slow=bool(sl))
        worst = 0.0; tests = []
        for name, img in imgs:
            for s in (0.0, 0.5, 1.0):
                m = float(np.abs(run_gpu(device, spv, img, s) - cas_golden(img, s, **kw)).max())
                worst = max(worst, m); tests.append({"image": name, "sharpness": s, "max_diff": m})
        good = worst < 1e-3; ok &= good
        report["variants"][tag] = {"defines": defines, "sha256": hashlib.sha256(spv).hexdigest(),
                                   "worst_max_diff": worst, "pass": good, "tests": tests}
        print(f"[{tag}] worst={worst:.2e}  {'PASS' if good else 'FAIL'}")
    big = big_image(2160, 3840)
    t0 = time.perf_counter(); g = run_gpu(device, spv_fast, big, 0.5)
    ms = 1000*(time.perf_counter()-t0)
    m4k = float(np.abs(g - cas_golden(big, 0.5)).max()); ok &= m4k < 1e-3
    report["4k"] = {"max_diff": m4k, "pass": m4k < 1e-3, "full_cycle_ms": round(ms,1)}
    print(f"[4K fast s=0.5] max={m4k:.2e}  {'PASS' if m4k<1e-3 else 'FAIL'}  (full cycle {ms:.0f} ms incl. readback)")
    report["all_pass"] = ok
    (HERE/"manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("manifest.json written.", "ALL PASS" if ok else "FAIL present - send full log")

if __name__ == "__main__":
    main()
