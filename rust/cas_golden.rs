// --- Битовые хаки AMD (один в один из ffx_a.h) ---
#[inline]
fn lo_rcp(x: f32) -> f32 {
    f32::from_bits(0x7EF07EBB_u32 - x.to_bits())
}

#[inline]
fn med_rcp(x: f32) -> f32 {
    let b = f32::from_bits(0x7EF19FFF_u32 - x.to_bits());
    b * (-b * x + 2.0)
}

#[inline]
fn lo_sqrt(x: f32) -> f32 {
    f32::from_bits((x.to_bits() >> 1) + 0x1FBC4639_u32)
}

// D3D saturate: saturate(NaN) = 0
#[inline]
fn sat(x: f32) -> f32 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

// --- Основной алгоритм CAS ---
pub fn cas_golden(
    img: &[f32],
    width: usize,
    height: usize,
    sharpness: f32,
    better_diagonals: bool,
    go_slower: bool,
    slow: bool,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; img.len()];
    let peak = -1.0_f32 / (8.0 + (5.0 - 8.0) * sat(sharpness));

    for y in 0..height {
        for x in 0..width {
            let get = |dx: isize, dy: isize| -> [f32; 3] {
                let nx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                let ny = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                let idx = (ny * width + nx) * 3;
                [img[idx], img[idx + 1], img[idx + 2]]
            };

            let a = get(-1, -1); let b = get(0, -1); let c = get(1, -1);
            let d = get(-1,  0); let e = get(0,  0); let f = get(1,  0);
            let g = get(-1,  1); let h = get(0,  1); let i = get(1,  1);

            for ch in 0..3 {
                let mn = [d[ch], e[ch], f[ch], b[ch], h[ch]].iter()
                    .cloned().fold(f32::INFINITY, f32::min);
                let mx = [d[ch], e[ch], f[ch], b[ch], h[ch]].iter()
                    .cloned().fold(f32::NEG_INFINITY, f32::max);

                let (mn, mx) = if better_diagonals {
                    (mn + [mn, a[ch], c[ch], g[ch], i[ch]].iter()
                        .cloned().fold(f32::INFINITY, f32::min),
                     mx + [mx, a[ch], c[ch], g[ch], i[ch]].iter()
                        .cloned().fold(f32::NEG_INFINITY, f32::max))
                } else { (mn, mx) };

                let rcp_m = if go_slower { 1.0 / mx } else { lo_rcp(mx) };
                let limit = if better_diagonals { 2.0 } else { 1.0 };

                let mut amp = sat(mn.min(limit - mx) * rcp_m);
                amp = if go_slower { amp.sqrt() } else { lo_sqrt(amp) };

                let w = amp * peak;
                let wch = if slow { w } else {
                    // В fast-режиме вес берётся от зелёного канала (ch=1)
                    let mn_g = [d[1], e[1], f[1], b[1], h[1]].iter()
                        .cloned().fold(f32::INFINITY, f32::min);
                    let mx_g = [d[1], e[1], f[1], b[1], h[1]].iter()
                        .cloned().fold(f32::NEG_INFINITY, f32::max);
                    let (mn_g, mx_g) = if better_diagonals {
                        (mn_g + [mn_g, a[1], c[1], g[1], i[1]].iter()
                            .cloned().fold(f32::INFINITY, f32::min),
                         mx_g + [mx_g, a[1], c[1], g[1], i[1]].iter()
                            .cloned().fold(f32::NEG_INFINITY, f32::max))
                    } else { (mn_g, mx_g) };
                    let rcp_m_g = if go_slower { 1.0 / mx_g } else { lo_rcp(mx_g) };
                    let limit_g = if better_diagonals { 2.0 } else { 1.0 };
                    let mut amp_g = sat(mn_g.min(limit_g - mx_g) * rcp_m_g);
                    amp_g = if go_slower { amp_g.sqrt() } else { lo_sqrt(amp_g) };
                    amp_g * peak
                };

                let rw = if go_slower { 1.0 / (1.0 + 4.0 * wch) }
                         else { med_rcp(1.0 + 4.0 * wch) };

                let val = sat((b[ch] * wch + d[ch] * wch + f[ch] * wch
                             + h[ch] * wch + e[ch]) * rw);
                out[(y * width + x) * 3 + ch] = val;
            }
        }
    }
    out
}

fn main() {
    let w = 64; let h = 64;
    let img = vec![0.5_f32; w * h * 3];

    // fast-режим (как в игре)
    let out_fast = cas_golden(&img, w, h, 1.0, false, false, false);
    let max_diff_fast = img.iter().zip(out_fast.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("Rust fast  flat max diff: {}", max_diff_fast);

    // exact-режим (точная математика)
    let out_exact = cas_golden(&img, w, h, 1.0, true, true, true);
    let max_diff_exact = img.iter().zip(out_exact.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("Rust exact flat max diff: {}", max_diff_exact);

    // Проверка: числа должны совпадать с Python
    // Python fast  flat: 0.0015581548
    // Python exact flat: 0.0
    let fast_ok = (max_diff_fast - 0.001558).abs() < 0.0001;
    let exact_ok = max_diff_exact < 1e-6;
    println!("\nFast  matches Python: {}", if fast_ok  { "PASS ✓" } else { "FAIL ✗" });
    println!("Exact matches Python: {}", if exact_ok { "PASS ✓" } else { "FAIL ✗" });
}