import pathlib, shutil, subprocess
import numpy as np
import wgpu

def find_dxc():
    for c in (shutil.which("dxc"), pathlib.Path(__file__).parent / "dxc" / "bin" / "x64" / "dxc.exe"):
        if c and pathlib.Path(c).exists(): return str(c)
    raise RuntimeError("dxc not found")

def build(dxc, defines, tag, hlsl_file="cas.hlsl"):
    out = pathlib.Path(__file__).parent / f"cas_{tag}.spv"
    cmd = [dxc, "-T", "cs_6_2", "-E", "main", "-spirv", "-O3", "-I", str(pathlib.Path(__file__).parent / "FidelityFX-CAS" / "ffx-cas")]
    for d in defines: cmd += ["-D", d]
    cmd += [str(pathlib.Path(__file__).parent / hlsl_file), "-Fo", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); raise RuntimeError(f"DXC failed on {tag}")
    return out.read_bytes()

def run_gpu(device, spv, img, sharpness, hlsl_file="cas.hlsl"):
    # ?????? ?????????? float32 ??? ???????. ?????? ??? ??????? cast ??????.
    H, W = img.shape[:2]
    rgba = np.concatenate([img, np.ones((H,W,1), np.float32)], 2).astype(np.float32)
    params = np.array([W, H, sharpness, 0], np.float32)
    n = rgba.nbytes

    b_in  = device.create_buffer_with_data(data=rgba.tobytes(), usage=wgpu.BufferUsage.STORAGE)
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
    
    if hlsl_file == "cas_h.hlsl":
        # Packed: 64 ?????? ?? ??????, ?????? ???????????? 2 ???????
        threads_needed = (W * H + 1) // 2
        workgroups = (threads_needed + 63) // 64
        cp.dispatch_workgroups(workgroups, 1, 1)
    else:
        # Standard: 8x8=64 ?????? ?? ??????, 1 ??????? ?? ?????
        cp.dispatch_workgroups((W + 7)//8, (H + 7)//8, 1)
    
    cp.end()
    enc.copy_buffer_to_buffer(b_out, 0, b_rd, 0, n)
    device.queue.submit([enc.finish()])

    b_rd.map_sync("read")
    # ?????? ?????? ??? float32
    out = np.frombuffer(b_rd.read_mapped(), np.float32).copy().reshape(H,W,4)[...,:3]
    b_rd.unmap()
    return out

def test_image():
    rng = np.random.default_rng(7); H = W = 256
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([xx/(W-1), ((xx//16+yy//16)%2)*0.8+0.1,
                    np.clip(0.5+0.4*np.sin(xx/7)*np.cos(yy/9), .02, .98)], -1)
    return np.clip(img*0.9 + rng.random((H,W,3), dtype=np.float32)*0.1, 0.01, 0.99).astype(np.float32)
