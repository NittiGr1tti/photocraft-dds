//! DirectDraw Surface (`.dds`), implemented here (no dependencies).
//!
//! Reading: the top mip level of a 2-D texture. Legacy headers and the DX10
//! extension header are both understood. Supported pixel formats are BC1/DXT1,
//! BC2/DXT3 (and DXT2), BC3/DXT5 (and DXT4), BC4/ATI1, BC5/ATI2, and
//! uncompressed 8/16/24/32-bit RGB(A), luminance and alpha-only data described
//! by channel masks, plus the matching DXGI formats. Cube maps, volume
//! textures and texture arrays are rejected as unsupported.
//!
//! Writing: the format chosen by [`EncodeOptions::dds_format`], optionally with a
//! full alpha-weighted mip chain ([`EncodeOptions::dds_mipmaps`]). The legacy
//! header is used (it is understood by every DDS consumer); alpha is straight.

use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::image::{ChannelLayout, Image, SampleType};
use crate::options::{DdsFormat, EncodeOptions, Limits};
use crate::Format;

const F: Format = Format::Dds;

fn err(msg: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(F, msg)
}

fn unsupported(msg: &str) -> CodecError {
    CodecError::unsupported(F, msg)
}

// ---------------------------------------------------------------------------
// Header constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"DDS ";
/// Magic + 124-byte header.
const HEADER_END: usize = 128;
/// DX10 extension header length.
const DX10_LEN: usize = 20;

const DDSD_CAPS: u32 = 0x1;
const DDSD_HEIGHT: u32 = 0x2;
const DDSD_WIDTH: u32 = 0x4;
const DDSD_PITCH: u32 = 0x8;
const DDSD_PIXELFORMAT: u32 = 0x1000;
const DDSD_MIPMAPCOUNT: u32 = 0x2_0000;
const DDSD_LINEARSIZE: u32 = 0x8_0000;

const DDPF_ALPHAPIXELS: u32 = 0x1;
const DDPF_ALPHA: u32 = 0x2;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x2_0000;

const DDSCAPS_COMPLEX: u32 = 0x8;
const DDSCAPS_TEXTURE: u32 = 0x1000;
const DDSCAPS_MIPMAP: u32 = 0x40_0000;
const DDSCAPS2_CUBEMAP: u32 = 0x200;
const DDSCAPS2_VOLUME: u32 = 0x20_0000;

const fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*s)
}

/// Alpha cutoff for DXT1 1-bit alpha.
const DXT1_CUTOFF: u8 = 128;

// ---------------------------------------------------------------------------
// Little helpers (all bounds-checked)
// ---------------------------------------------------------------------------

fn u32_at(b: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let s: [u8; 4] = b.get(off..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(s))
}

fn need(b: &[u8], off: usize) -> Result<u32, CodecError> {
    u32_at(b, off).ok_or_else(|| err("truncated header"))
}

fn to_usize(v: u32) -> Result<usize, CodecError> {
    usize::try_from(v).map_err(|_| err("value does not fit in memory"))
}

fn mul(a: usize, b: usize) -> Result<usize, CodecError> {
    a.checked_mul(b).ok_or_else(|| err("size overflow"))
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bc {
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
}

impl Bc {
    fn block_bytes(self) -> usize {
        match self {
            Bc::Bc1 | Bc::Bc4 => 8,
            _ => 16,
        }
    }
}

#[derive(Clone, Copy)]
struct Masks {
    bits: u32,
    r: u32,
    g: u32,
    b: u32,
    a: u32,
    /// Luminance data: `r` is the luminance mask, replicated into RGB.
    lum: bool,
}

#[derive(Clone, Copy)]
enum Pixels {
    Block(Bc),
    Raw(Masks),
}

fn dxgi_to_pixels(dxgi: u32) -> Option<Pixels> {
    let rgba = |r, g, b, a| Pixels::Raw(Masks { bits: 32, r, g, b, a, lum: false });
    Some(match dxgi {
        71 | 72 => Pixels::Block(Bc::Bc1),
        74 | 75 => Pixels::Block(Bc::Bc2),
        77 | 78 => Pixels::Block(Bc::Bc3),
        79..=81 => Pixels::Block(Bc::Bc4),
        82..=84 => Pixels::Block(Bc::Bc5),
        // R8G8B8A8_UNORM / _SRGB
        28 | 29 => rgba(0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000),
        // B8G8R8A8_UNORM / _SRGB
        87 | 91 => rgba(0xFF_0000, 0xFF00, 0xFF, 0xFF00_0000),
        // B8G8R8X8_UNORM / _SRGB
        88 | 93 => rgba(0xFF_0000, 0xFF00, 0xFF, 0),
        // R8_UNORM
        61 => Pixels::Raw(Masks { bits: 8, r: 0xFF, g: 0, b: 0, a: 0, lum: true }),
        // A8_UNORM
        65 => Pixels::Raw(Masks { bits: 8, r: 0, g: 0, b: 0, a: 0xFF, lum: false }),
        _ => return None,
    })
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    if bytes.get(..4) != Some(&MAGIC[..]) {
        return Err(err("missing DDS signature"));
    }
    if bytes.len() < HEADER_END {
        return Err(err("truncated header"));
    }
    if need(bytes, 4)? != 124 {
        return Err(err("bad header size"));
    }
    let height = need(bytes, 12)?;
    let width = need(bytes, 16)?;
    if need(bytes, 76)? != 32 {
        return Err(err("bad pixel-format size"));
    }
    let pf_flags = need(bytes, 80)?;
    let cc = need(bytes, 84)?;
    let bit_count = need(bytes, 88)?;
    let (rm, gm, bm, am) = (need(bytes, 92)?, need(bytes, 96)?, need(bytes, 100)?, need(bytes, 104)?);
    let caps2 = need(bytes, 112)?;
    if caps2 & DDSCAPS2_CUBEMAP != 0 {
        return Err(unsupported("cube maps"));
    }
    if caps2 & DDSCAPS2_VOLUME != 0 {
        return Err(unsupported("volume textures"));
    }

    let mut data_off = HEADER_END;
    let pixels = if pf_flags & DDPF_FOURCC != 0 {
        match cc {
            c if c == fourcc(b"DXT1") => Pixels::Block(Bc::Bc1),
            c if c == fourcc(b"DXT2") || c == fourcc(b"DXT3") => Pixels::Block(Bc::Bc2),
            c if c == fourcc(b"DXT4") || c == fourcc(b"DXT5") => Pixels::Block(Bc::Bc3),
            c if c == fourcc(b"ATI1") || c == fourcc(b"BC4U") => Pixels::Block(Bc::Bc4),
            c if c == fourcc(b"ATI2") || c == fourcc(b"BC5U") => Pixels::Block(Bc::Bc5),
            c if c == fourcc(b"DX10") => {
                let dxgi = need(bytes, HEADER_END)?;
                let dim = need(bytes, HEADER_END + 4)?;
                let array = need(bytes, HEADER_END + 12)?;
                if need(bytes, HEADER_END + 8)? & 0x4 != 0 {
                    return Err(unsupported("cube maps"));
                }
                if dim != 3 {
                    return Err(unsupported("non-2D textures"));
                }
                if array > 1 {
                    return Err(unsupported("texture arrays"));
                }
                data_off = HEADER_END + DX10_LEN;
                dxgi_to_pixels(dxgi).ok_or_else(|| unsupported("this DXGI format"))?
            }
            _ => return Err(unsupported("this FourCC pixel format")),
        }
    } else if pf_flags & (DDPF_RGB | DDPF_LUMINANCE | DDPF_ALPHA) != 0 {
        if !matches!(bit_count, 8 | 16 | 24 | 32) {
            return Err(unsupported("this bit depth"));
        }
        let lum = pf_flags & DDPF_LUMINANCE != 0;
        let has_a = pf_flags & (DDPF_ALPHAPIXELS | DDPF_ALPHA) != 0;
        Pixels::Raw(Masks { bits: bit_count, r: rm, g: if lum { 0 } else { gm }, b: if lum { 0 } else { bm }, a: if has_a { am } else { 0 }, lum })
    } else {
        return Err(unsupported("this pixel format"));
    };

    limits.check(width, height, ChannelLayout::Rgba, SampleType::U8)?;
    let (w, h) = (to_usize(width)?, to_usize(height)?);
    let data = bytes.get(data_off..).ok_or_else(|| err("truncated header"))?;
    let mut out = vec![0u8; mul(mul(w, h)?, 4)?];

    match pixels {
        Pixels::Block(bc) => decode_blocks(bc, data, w, h, &mut out)?,
        Pixels::Raw(m) => decode_raw(m, data, w, h, &mut out)?,
    }
    Image::from_u8(width, height, ChannelLayout::Rgba, out)
}

fn put(out: &mut [u8], w: usize, x: usize, y: usize, px: [u8; 4]) {
    let Some(i) = y.checked_mul(w).and_then(|v| v.checked_add(x)).and_then(|v| v.checked_mul(4)) else {
        return;
    };
    if let Some(dst) = out.get_mut(i..i.saturating_add(4)) {
        dst.copy_from_slice(&px);
    }
}

fn decode_blocks(bc: Bc, data: &[u8], w: usize, h: usize, out: &mut [u8]) -> Result<(), CodecError> {
    let bw = w.div_ceil(4);
    let bh = h.div_ceil(4);
    let bs = bc.block_bytes();
    let need_bytes = mul(mul(bw, bh)?, bs)?;
    let data = data.get(..need_bytes).ok_or_else(|| err("truncated pixel data"))?;
    for (bi, block) in data.chunks_exact(bs).enumerate() {
        let (bx, by) = (bi % bw, bi / bw);
        let px = decode_block(bc, block);
        for (i, p) in px.iter().enumerate() {
            let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
            if x < w && y < h {
                put(out, w, x, y, *p);
            }
        }
    }
    Ok(())
}

fn rgb565(c: u16) -> [u8; 3] {
    let r = u32::from((c >> 11) & 31);
    let g = u32::from((c >> 5) & 63);
    let b = u32::from(c & 31);
    [((r * 255 + 15) / 31) as u8, ((g * 255 + 31) / 63) as u8, ((b * 255 + 15) / 31) as u8]
}

fn le16(b: &[u8]) -> u16 {
    match b {
        [a, c, ..] => u16::from_le_bytes([*a, *c]),
        _ => 0,
    }
}

fn le32(b: &[u8]) -> u32 {
    match b {
        [a, c, d, e, ..] => u32::from_le_bytes([*a, *c, *d, *e]),
        _ => 0,
    }
}

fn le64_6(b: &[u8]) -> u64 {
    b.iter().take(6).enumerate().fold(0u64, |acc, (i, v)| acc | (u64::from(*v) << (8 * i)))
}

/// BC1 colour block (8 bytes) → 16 RGBA pixels.
fn bc1_colors(b: &[u8]) -> [[u8; 4]; 16] {
    let (c0, c1) = (le16(b), le16(b.get(2..).unwrap_or(&[])));
    let (e0, e1) = (rgb565(c0), rgb565(c1));
    let mix = |wa: u32, wb: u32, den: u32| -> [u8; 4] {
        let ch = |i: usize| -> u8 {
            let (x, y) = (u32::from(*e0.get(i).unwrap_or(&0)), u32::from(*e1.get(i).unwrap_or(&0)));
            ((x * wa + y * wb + den / 2) / den) as u8
        };
        [ch(0), ch(1), ch(2), 255]
    };
    let pal: [[u8; 4]; 4] = if c0 > c1 {
        [[e0[0], e0[1], e0[2], 255], [e1[0], e1[1], e1[2], 255], mix(2, 1, 3), mix(1, 2, 3)]
    } else {
        [[e0[0], e0[1], e0[2], 255], [e1[0], e1[1], e1[2], 255], mix(1, 1, 2), [0, 0, 0, 0]]
    };
    let idx = le32(b.get(4..).unwrap_or(&[]));
    let mut out = [[0u8; 4]; 16];
    for (i, o) in out.iter_mut().enumerate() {
        *o = pal.get(((idx >> (2 * i)) & 3) as usize).copied().unwrap_or([0; 4]);
    }
    out
}

/// BC4 single-channel block (8 bytes) → 16 values.
fn bc4_values(b: &[u8]) -> [u8; 16] {
    let a0 = u32::from(b.first().copied().unwrap_or(0));
    let a1 = u32::from(b.get(1).copied().unwrap_or(0));
    let mut pal = [0u8; 8];
    pal[0] = a0 as u8;
    pal[1] = a1 as u8;
    if a0 > a1 {
        for k in 0..6u32 {
            pal[2 + k as usize] = (((6 - k) * a0 + (1 + k) * a1 + 3) / 7) as u8;
        }
    } else {
        for k in 0..4u32 {
            pal[2 + k as usize] = (((4 - k) * a0 + (1 + k) * a1 + 2) / 5) as u8;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let bits = le64_6(b.get(2..).unwrap_or(&[]));
    let mut out = [0u8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        *o = pal.get(((bits >> (3 * i)) & 7) as usize).copied().unwrap_or(0);
    }
    out
}

fn decode_block(bc: Bc, b: &[u8]) -> [[u8; 4]; 16] {
    match bc {
        Bc::Bc1 => bc1_colors(b),
        Bc::Bc2 => {
            let mut px = bc1_colors(b.get(8..).unwrap_or(&[]));
            let alpha = b.get(..8).unwrap_or(&[]).iter().fold(0u64, |acc, v| acc >> 8 | (u64::from(*v) << 56));
            for (i, p) in px.iter_mut().enumerate() {
                p[3] = (((alpha >> (4 * i)) & 15) * 17) as u8;
            }
            px
        }
        Bc::Bc3 => {
            let mut px = bc1_colors(b.get(8..).unwrap_or(&[]));
            let a = bc4_values(b);
            for (p, v) in px.iter_mut().zip(a) {
                p[3] = v;
            }
            px
        }
        Bc::Bc4 => {
            let mut px = [[0u8; 4]; 16];
            for (p, v) in px.iter_mut().zip(bc4_values(b)) {
                *p = [v, v, v, 255];
            }
            px
        }
        Bc::Bc5 => {
            let r = bc4_values(b);
            let g = bc4_values(b.get(8..).unwrap_or(&[]));
            let mut px = [[0u8; 4]; 16];
            for ((p, rv), gv) in px.iter_mut().zip(r).zip(g) {
                *p = [rv, gv, 0, 255];
            }
            px
        }
    }
}

fn extract(val: u32, mask: u32, default: u8) -> u8 {
    if mask == 0 {
        return default;
    }
    let shift = mask.trailing_zeros();
    let bits = (mask >> shift).count_ones();
    let v = (val & mask) >> shift;
    if bits >= 8 {
        (v >> (bits - 8)) as u8
    } else {
        let max = (1u32 << bits) - 1;
        ((v * 255 + max / 2) / max) as u8
    }
}

fn decode_raw(m: Masks, data: &[u8], w: usize, h: usize, out: &mut [u8]) -> Result<(), CodecError> {
    let bpp = to_usize(m.bits / 8)?;
    let need_bytes = mul(mul(w, h)?, bpp)?;
    let data = data.get(..need_bytes).ok_or_else(|| err("truncated pixel data"))?;
    for (i, px) in data.chunks_exact(bpp).enumerate() {
        let val = px.iter().enumerate().fold(0u32, |acc, (k, v)| acc | (u32::from(*v) << (8 * k)));
        let a = extract(val, m.a, 255);
        let rgba = if m.lum {
            let l = extract(val, m.r, 0);
            [l, l, l, a]
        } else if m.r == 0 && m.g == 0 && m.b == 0 {
            [0, 0, 0, a]
        } else {
            [extract(val, m.r, 0), extract(val, m.g, 0), extract(val, m.b, 0), a]
        };
        put(out, w, i % w, i / w, rgba);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

fn pixel(rgba: &[u8], w: usize, h: usize, x: usize, y: usize) -> [u8; 4] {
    let (x, y) = (x.min(w.saturating_sub(1)), y.min(h.saturating_sub(1)));
    let i = y.saturating_mul(w).saturating_add(x).saturating_mul(4);
    match rgba.get(i..i.saturating_add(4)) {
        Some([r, g, b, a]) => [*r, *g, *b, *a],
        _ => [0; 4],
    }
}

fn p565(c: [f32; 3]) -> u16 {
    let q = |v: f32, max: f32| -> u16 { (v * max / 255.0).round().clamp(0.0, max) as u16 };
    (q(c[0], 31.0) << 11) | (q(c[1], 63.0) << 5) | q(c[2], 31.0)
}

fn u565(c: u16) -> [f32; 3] {
    [f32::from((c >> 11) & 31) * 255.0 / 31.0, f32::from((c >> 5) & 63) * 255.0 / 63.0, f32::from(c & 31) * 255.0 / 31.0]
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// BC1 colour block. `punch`: 3-colour mode with a transparent index for
/// pixels whose alpha is below the cutoff.
fn enc_bc1(px: &[[u8; 4]; 16], punch: bool) -> [u8; 8] {
    let opaque: [bool; 16] = std::array::from_fn(|i| !punch || px.get(i).is_some_and(|p| p[3] >= DXT1_CUTOFF));
    let (mut mn, mut mx) = ([255.0f32; 3], [0.0f32; 3]);
    let mut any = false;
    for (p, o) in px.iter().zip(opaque) {
        if o {
            any = true;
            for c in 0..3 {
                let v = f32::from(p[c]);
                mn[c] = mn[c].min(v);
                mx[c] = mx[c].max(v);
            }
        }
    }
    if !any {
        return [0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF];
    }
    for c in 0..3 {
        let inset = (mx[c] - mn[c]) / 16.0;
        mn[c] += inset;
        mx[c] -= inset;
    }
    let (hi, lo) = (p565(mx), p565(mn));
    let (c0, c1) = if punch { (hi.min(lo), hi.max(lo)) } else { (hi.max(lo), hi.min(lo)) };
    let (e0, e1) = (u565(c0), u565(c1));
    let mid = |wa: f32, wb: f32, den: f32| -> [f32; 3] { [(e0[0] * wa + e1[0] * wb) / den, (e0[1] * wa + e1[1] * wb) / den, (e0[2] * wa + e1[2] * wb) / den] };
    let pal: Vec<[f32; 3]> = if punch { vec![e0, e1, mid(1.0, 1.0, 2.0)] } else { vec![e0, e1, mid(2.0, 1.0, 3.0), mid(1.0, 2.0, 3.0)] };
    let mut bits = 0u32;
    if punch || c0 != c1 {
        for (i, (p, o)) in px.iter().zip(opaque).enumerate() {
            let idx = if !o {
                3
            } else {
                let c = [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])];
                let mut best = (0u32, f32::MAX);
                for (k, e) in pal.iter().enumerate() {
                    let d = dist(c, *e);
                    if d < best.1 {
                        best = (k as u32, d);
                    }
                }
                best.0
            };
            bits |= idx << (2 * i);
        }
    }
    let (b0, b1, bb) = (c0.to_le_bytes(), c1.to_le_bytes(), bits.to_le_bytes());
    [b0[0], b0[1], b1[0], b1[1], bb[0], bb[1], bb[2], bb[3]]
}

/// BC4 / DXT5-alpha block.
fn enc_bc4(v: &[u8; 16]) -> [u8; 8] {
    let e0 = v.iter().copied().max().unwrap_or(0);
    let e1 = v.iter().copied().min().unwrap_or(0);
    let mut bits = 0u64;
    if e0 != e1 {
        let (f0, f1) = (f32::from(e0), f32::from(e1));
        let mut pal = [f0, f1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for k in 0..6u32 {
            let (a, b) = ((6 - k) as f32, (1 + k) as f32);
            pal[2 + k as usize] = (a * f0 + b * f1) / 7.0;
        }
        for (i, x) in v.iter().enumerate() {
            let fx = f32::from(*x);
            let mut best = (0u64, f32::MAX);
            for (k, e) in pal.iter().enumerate() {
                let d = (fx - e).abs();
                if d < best.1 {
                    best = (k as u64, d);
                }
            }
            bits |= best.0 << (3 * i);
        }
    }
    let bb = bits.to_le_bytes();
    [e0, e1, bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]]
}

fn enc_alpha4(px: &[[u8; 4]; 16]) -> [u8; 8] {
    let mut bits = 0u64;
    for (i, p) in px.iter().enumerate() {
        bits |= u64::from((u32::from(p[3]) + 8) / 17) << (4 * i);
    }
    bits.to_le_bytes()
}

fn block_at(rgba: &[u8], w: usize, h: usize, bx: usize, by: usize) -> [[u8; 4]; 16] {
    std::array::from_fn(|i| pixel(rgba, w, h, bx * 4 + i % 4, by * 4 + i / 4))
}

fn encode_level_blocks(rgba: &[u8], w: usize, h: usize, fmt: DdsFormat, alpha: bool, out: &mut Vec<u8>) {
    for by in 0..h.div_ceil(4) {
        for bx in 0..w.div_ceil(4) {
            let mut px = block_at(rgba, w, h, bx, by);
            if !alpha {
                for p in &mut px {
                    p[3] = 255;
                }
            }
            match fmt {
                DdsFormat::Dxt1 => out.extend_from_slice(&enc_bc1(&px, alpha)),
                DdsFormat::Dxt3 => {
                    out.extend_from_slice(&enc_alpha4(&px));
                    out.extend_from_slice(&enc_bc1(&px, false));
                }
                DdsFormat::Dxt5 => {
                    out.extend_from_slice(&enc_bc4(&std::array::from_fn(|i| px.get(i).map_or(255, |p| p[3]))));
                    out.extend_from_slice(&enc_bc1(&px, false));
                }
                DdsFormat::Bc4 => out.extend_from_slice(&enc_bc4(&std::array::from_fn(|i| px.get(i).map_or(0, |p| p[0])))),
                DdsFormat::Bc5 => {
                    out.extend_from_slice(&enc_bc4(&std::array::from_fn(|i| px.get(i).map_or(0, |p| p[0]))));
                    out.extend_from_slice(&enc_bc4(&std::array::from_fn(|i| px.get(i).map_or(0, |p| p[1]))));
                }
                _ => {}
            }
        }
    }
}

fn encode_level_raw(rgba: &[u8], fmt: DdsFormat, alpha: bool, out: &mut Vec<u8>) {
    for p in rgba.chunks_exact(4) {
        let (r, g, b, a) = (p[0], p[1], p[2], if alpha { p[3] } else { 255 });
        match fmt {
            DdsFormat::Argb8 => out.extend_from_slice(&[b, g, r, a]),
            DdsFormat::Xrgb8 => out.extend_from_slice(&[b, g, r, 255]),
            DdsFormat::Rgb8 => out.extend_from_slice(&[b, g, r]),
            DdsFormat::L8 => out.push(((u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114 + 500) / 1000) as u8),
            DdsFormat::A8 => out.push(a),
            _ => {}
        }
    }
}

fn bytes_per_pixel(fmt: DdsFormat) -> usize {
    match fmt {
        DdsFormat::Argb8 | DdsFormat::Xrgb8 => 4,
        DdsFormat::Rgb8 => 3,
        _ => 1,
    }
}

fn is_block(fmt: DdsFormat) -> bool {
    matches!(fmt, DdsFormat::Dxt1 | DdsFormat::Dxt3 | DdsFormat::Dxt5 | DdsFormat::Bc4 | DdsFormat::Bc5)
}

fn block_size(fmt: DdsFormat) -> usize {
    if matches!(fmt, DdsFormat::Dxt1 | DdsFormat::Bc4) {
        8
    } else {
        16
    }
}

/// Alpha-weighted box filter down to `(nw, nh)`; handles odd sizes.
fn downsample(src: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(nw.saturating_mul(nh).saturating_mul(4));
    for y in 0..nh {
        let (y0, y1) = (y * h / nh, ((y + 1) * h).div_ceil(nh).max(y * h / nh + 1).min(h));
        for x in 0..nw {
            let (x0, x1) = (x * w / nw, ((x + 1) * w).div_ceil(nw).max(x * w / nw + 1).min(w));
            let (mut sa, mut sr, mut sg, mut sb, mut ur, mut ug, mut ub, mut n) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let [r, g, b, a] = pixel(src, w, h, xx, yy);
                    let (r, g, b, a) = (u64::from(r), u64::from(g), u64::from(b), u64::from(a));
                    sa += a;
                    sr += r * a;
                    sg += g * a;
                    sb += b * a;
                    ur += r;
                    ug += g;
                    ub += b;
                    n += 1;
                }
            }
            let n = n.max(1);
            let px = if sa > 0 {
                let d = sa;
                [((sr + d / 2) / d) as u8, ((sg + d / 2) / d) as u8, ((sb + d / 2) / d) as u8, ((sa + n / 2) / n) as u8]
            } else {
                [((ur + n / 2) / n) as u8, ((ug + n / 2) / n) as u8, ((ub + n / 2) / n) as u8, 0]
            };
            out.extend_from_slice(&px);
        }
    }
    out
}

fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn header(fmt: DdsFormat, w: u32, h: u32, mips: u32) -> Result<Vec<u8>, CodecError> {
    let (wu, hu) = (to_usize(w)?, to_usize(h)?);
    let mut flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT;
    if mips > 1 {
        flags |= DDSD_MIPMAPCOUNT;
    }
    let pf: u32;
    let (mut cc, mut bits) = (0u32, 0u32);
    let (mut rm, mut gm, mut bm, mut am) = (0u32, 0u32, 0u32, 0u32);
    let size = if is_block(fmt) {
        flags |= DDSD_LINEARSIZE;
        pf = DDPF_FOURCC;
        cc = fourcc(match fmt {
            DdsFormat::Dxt1 => b"DXT1",
            DdsFormat::Dxt3 => b"DXT3",
            DdsFormat::Dxt5 => b"DXT5",
            DdsFormat::Bc4 => b"ATI1",
            _ => b"ATI2",
        });
        mul(mul(wu.div_ceil(4), hu.div_ceil(4))?, block_size(fmt))?
    } else {
        flags |= DDSD_PITCH;
        bits = (bytes_per_pixel(fmt) * 8) as u32;
        match fmt {
            DdsFormat::Argb8 | DdsFormat::Xrgb8 | DdsFormat::Rgb8 => {
                (rm, gm, bm) = (0xFF_0000, 0xFF00, 0xFF);
                if fmt == DdsFormat::Argb8 {
                    pf = DDPF_RGB | DDPF_ALPHAPIXELS;
                    am = 0xFF00_0000;
                } else {
                    pf = DDPF_RGB;
                }
            }
            DdsFormat::L8 => {
                pf = DDPF_LUMINANCE;
                rm = 0xFF;
            }
            _ => {
                pf = DDPF_ALPHA;
                am = 0xFF;
            }
        }
        mul(wu, bytes_per_pixel(fmt))?
    };
    let size = u32::try_from(size).map_err(|_| CodecError::encode(F, "texture too large for a DDS header"))?;
    let mut caps = DDSCAPS_TEXTURE;
    if mips > 1 {
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP;
    }
    let mut v = Vec::with_capacity(HEADER_END);
    v.extend_from_slice(MAGIC);
    for x in [124, flags, h, w, size, 0, mips] {
        push_u32(&mut v, x);
    }
    v.extend_from_slice(&[0u8; 44]);
    for x in [32, pf, cc, bits, rm, gm, bm, am, caps, 0, 0, 0, 0] {
        push_u32(&mut v, x);
    }
    Ok(v)
}

pub(crate) fn encode(src: &Image, _plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let (w32, h32) = (src.width(), src.height());
    let (mut w, mut h) = (to_usize(w32)?, to_usize(h32)?);
    if w == 0 || h == 0 {
        return Err(CodecError::encode(F, "zero-sized image"));
    }
    let fmt = opts.dds_format;
    let alpha = opts.dds_alpha;
    let mut level = src.to_rgba8();

    let mut levels: u32 = 1;
    let mut body = Vec::new();
    loop {
        if is_block(fmt) {
            encode_level_blocks(&level, w, h, fmt, alpha, &mut body);
        } else {
            encode_level_raw(&level, fmt, alpha, &mut body);
        }
        if !opts.dds_mipmaps || (w == 1 && h == 1) {
            break;
        }
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        level = downsample(&level, w, h, nw, nh);
        (w, h) = (nw, nh);
        levels += 1;
    }
    let mut out = header(fmt, w32, h32, levels)?;
    out.extend_from_slice(&body);
    Ok(out)
}
