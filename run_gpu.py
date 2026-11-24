import pathlib, shutil, subprocess, sys
import numpy as np
import wgpu
from cas_golden import cas_golden, EXACT

HERE = pathlib.Path(__file__).resolve().parent
CAS_INC = HERE / "FidelityFX-CAS" / "ffx-cas"

def find_dxc():
    for c in (shutil.which("dxc"), HERE / "dxc" / "bin" / "x64" / "dxc.exe"):
        if c and pathlib.Path(c).exists(): return str(c)
    sys.exit("dxc не найден: winget install --id Microsoft.DirectX.ShaderCompiler -e, "
             "или распакуй dxc_*.zip в папку .\\dxc")

def build(dxc, defines, tag):
    out = HERE / f"cas_{tag}.spv"
    cmd = [dxc, "-T", "cs_6_2", "-E", "main", "-spirv", "-O3", "-I", str(CAS_INC)]
    for d in defines: cmd += ["-D", d]
    cmd += [str(HERE / "cas.hlsl"), "-Fo", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); sys.exit(f"DXC не собрал {tag} — пришли лог")
    return out.read_bytes()

def run_gpu(device, spv, img, sharpness):
    H, W = img.shape[:2]
    rgba = np.concatenate([img, np.ones((H,W,1), np.float32)], 2).astype(np.float32)
    params = np.array([W, H, sharpness, 0], np.float32); n = rgba.nbytes
    b_in  = device.create_buffer_with_data(data=rgba.tobytes(),   usage=wgpu.BufferUsage.STORAGE)
    b_par = device.create_buffer_with_data(data=params.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    b_out = device.create_buffer(size=n, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    b_rd  = device.create_buffer(size=n, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    module = device.create_shader_module(code=spv)
    bgl = device.create_bind_group_layout(entries=[
        {"binding": k, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}}
        for k, t in ((0,"read-only-storage"),(1,"storage"),(2,"read-only-storage"))])
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
        compute={"module": module, "entry_point": "main"})
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": k, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
        for k, buf in ((0,b_in),(1,b_out),(2,b_par))])
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline); cp.set_bind_group(0, bg)
    cp.dispatch_workgroups((W + 7)//8, (H + 7)//8, 1); cp.end()
    enc.copy_buffer_to_buffer(b_out, 0, b_rd, 0, n)
    device.queue.submit([enc.finish()])
    b_rd.map_sync("read")
    out = np.frombuffer(b_rd.read_mapped(), np.float32).copy().reshape(H,W,4)[...,:3]
    b_rd.unmap()
    return out

def test_image():
    rng = np.random.default_rng(7); H = W = 256
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([xx/(W-1), ((xx//16+yy//16)%2)*0.8+0.1,
                    np.clip(0.5+0.4*np.sin(xx/7)*np.cos(yy/9), .02, .98)], -1)
    return np.clip(img*0.9 + rng.random((H,W,3), dtype=np.float32)*0.1, 0.01, 0.99).astype(np.float32)

if __name__ == "__main__":
    if not CAS_INC.exists(): sys.exit("Склонируй FidelityFX-CAS в папку проекта")
    dxc = find_dxc()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    print("GPU:", adapter.info)
    img = test_image(); ok = True
    for tag, defines, kw in [("fast", [], {}),
                             ("exact", ["CAS_GO_SLOWER","CAS_BETTER_DIAGONALS","CAS_SLOW"], EXACT)]:
        spv = build(dxc, defines, tag)
        for s in (0.0, 0.5, 1.0):
            d = np.abs(run_gpu(device, spv, img, s) - cas_golden(img, s, **kw))
            good = d.max() < 1e-3; ok &= good
            print(f"[{tag}] s={s:.1f}  max={d.max():.2e}  mean={d.mean():.2e}  {'PASS' if good else 'FAIL'}")
    print("ИТОГ:", "ВСЁ PASS — первый кирпич готов" if ok else "есть FAIL — пришли весь вывод")