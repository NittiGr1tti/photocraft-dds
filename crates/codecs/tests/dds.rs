//! DirectDraw Surface: pixel formats, mip chains, alpha handling and malformed input.

use photocraft_codecs::*;

/// Smooth RGBA gradient with a horizontal alpha ramp.
fn gradient(w: u32, h: u32) -> Image {
    let mut d = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            d.extend_from_slice(&[(x * 255 / w) as u8, (y * 255 / h) as u8, ((x + y) * 255 / (w + h)) as u8, (x * 255 / w) as u8]);
        }
    }
    Image::from_u8(w, h, ChannelLayout::Rgba, d).unwrap()
}

fn opts(f: DdsFormat, mips: bool, alpha: bool) -> EncodeOptions {
    EncodeOptions { dds_format: f, dds_mipmaps: mips, dds_alpha: alpha, ..Default::default() }
}

fn tight() -> DecodeOptions {
    DecodeOptions { limits: Limits { max_width: 4096, max_height: 4096, max_pixels: 1 << 22, max_alloc: 64 << 20 }, ..Default::default() }
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

/// Mean absolute error over the given channel indices.
fn mae(a: &[u8], b: &[u8], channels: &[usize]) -> f64 {
    let mut sum = 0u64;
    let mut n = 0u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for &c in channels {
            sum += u64::from(pa[c].abs_diff(pb[c]));
            n += 1;
        }
    }
    sum as f64 / n as f64
}

#[test]
fn argb8_is_lossless() {
    let img = gradient(13, 7);
    let bytes = encode(&img, Format::Dds, &opts(DdsFormat::Argb8, true, true)).unwrap();
    assert_eq!(detect(&bytes), Some(Format::Dds));
    assert_eq!(decode(&bytes).unwrap().to_rgba8(), img.to_rgba8());
}

#[test]
fn xrgb8_and_rgb8_keep_colour_and_force_opaque() {
    let img = gradient(13, 7);
    for f in [DdsFormat::Xrgb8, DdsFormat::Rgb8] {
        let back = decode(&encode(&img, Format::Dds, &opts(f, false, true)).unwrap()).unwrap().to_rgba8();
        assert_eq!(mae(&back, &img.to_rgba8(), &[0, 1, 2]), 0.0, "{f:?}");
        assert!(back.chunks_exact(4).all(|p| p[3] == 255), "{f:?}");
    }
}

#[test]
fn single_channel_formats() {
    let img = gradient(20, 12);
    let src = img.to_rgba8();
    let a8 = decode(&encode(&img, Format::Dds, &opts(DdsFormat::A8, false, true)).unwrap()).unwrap().to_rgba8();
    assert_eq!(mae(&a8, &src, &[3]), 0.0);
    let l8 = decode(&encode(&img, Format::Dds, &opts(DdsFormat::L8, false, true)).unwrap()).unwrap().to_rgba8();
    assert!(l8.chunks_exact(4).all(|p| p[0] == p[1] && p[1] == p[2]));
}

#[test]
fn block_formats_are_close_on_smooth_images() {
    let img = gradient(97, 61); // not a multiple of 4
    let src = img.to_rgba8();
    for (f, rgb_limit, alpha_limit) in [(DdsFormat::Dxt1, 4.0, None), (DdsFormat::Dxt3, 4.0, Some(5.0)), (DdsFormat::Dxt5, 4.0, Some(1.0))] {
        let back = decode(&encode(&img, Format::Dds, &opts(f, false, f != DdsFormat::Dxt1)).unwrap()).unwrap();
        assert_eq!((back.width(), back.height()), (97, 61));
        let back = back.to_rgba8();
        assert!(mae(&back, &src, &[0, 1, 2]) < rgb_limit, "{f:?} colour error {}", mae(&back, &src, &[0, 1, 2]));
        if let Some(l) = alpha_limit {
            assert!(mae(&back, &src, &[3]) < l, "{f:?} alpha error {}", mae(&back, &src, &[3]));
        }
    }
    for (f, ch) in [(DdsFormat::Bc4, &[0usize][..]), (DdsFormat::Bc5, &[0usize, 1][..])] {
        let back = decode(&encode(&img, Format::Dds, &opts(f, false, true)).unwrap()).unwrap().to_rgba8();
        assert!(mae(&back, &src, ch) < 1.0, "{f:?}");
    }
}

#[test]
fn tiny_and_odd_sizes_round_trip_dimensions() {
    for (w, h) in [(1, 1), (1, 5), (3, 1), (2, 2), (4, 4), (5, 3), (17, 33)] {
        let img = gradient(w, h);
        for f in [DdsFormat::Dxt1, DdsFormat::Dxt3, DdsFormat::Dxt5, DdsFormat::Bc4, DdsFormat::Bc5, DdsFormat::Argb8, DdsFormat::L8, DdsFormat::A8] {
            let back = decode(&encode(&img, Format::Dds, &opts(f, true, true)).unwrap()).unwrap();
            assert_eq!((back.width(), back.height()), (w, h), "{f:?} {w}x{h}");
        }
    }
}

#[test]
fn mip_chain_layout() {
    let img = gradient(197, 130);
    let levels = [(197usize, 130usize), (98, 65), (49, 32), (24, 16), (12, 8), (6, 4), (3, 2), (1, 1)];
    for (f, bs) in [(DdsFormat::Dxt1, 8usize), (DdsFormat::Dxt3, 16), (DdsFormat::Dxt5, 16), (DdsFormat::Bc4, 8), (DdsFormat::Bc5, 16)] {
        let b = encode(&img, Format::Dds, &opts(f, true, true)).unwrap();
        let want: usize = 128 + levels.iter().map(|&(w, h)| w.div_ceil(4) * h.div_ceil(4) * bs).sum::<usize>();
        assert_eq!(b.len(), want, "{f:?}");
        assert_eq!(u32_at(&b, 28), 8, "{f:?} mip count");
        assert_eq!(u32_at(&b, 108) & 0x40_1008, 0x40_1008, "{f:?} caps");
        assert_eq!(u32_at(&b, 8) & 0x2_0000, 0x2_0000, "{f:?} MIPMAPCOUNT flag");
    }
    let single = encode(&img, Format::Dds, &opts(DdsFormat::Dxt5, false, true)).unwrap();
    assert_eq!(u32_at(&single, 8) & 0x2_0000, 0, "no MIPMAPCOUNT flag without mip-maps");
    assert_eq!(u32_at(&single, 108) & 0x40_0000, 0, "no MIPMAP cap without mip-maps");
    assert_eq!(single.len(), 128 + 50 * 33 * 16);
    let raw = encode(&img, Format::Dds, &opts(DdsFormat::Argb8, true, true)).unwrap();
    assert_eq!(raw.len(), 128 + levels.iter().map(|&(w, h)| w * h * 4).sum::<usize>());
}

#[test]
fn alpha_off_writes_opaque_texture() {
    let img = gradient(32, 32);
    for f in [DdsFormat::Dxt1, DdsFormat::Dxt3, DdsFormat::Dxt5, DdsFormat::Argb8] {
        let back = decode(&encode(&img, Format::Dds, &opts(f, true, false)).unwrap()).unwrap().to_rgba8();
        assert!(back.chunks_exact(4).all(|p| p[3] == 255), "{f:?}");
    }
}

#[test]
fn dxt1_alpha_is_one_bit() {
    let img = gradient(32, 32); // alpha ramps 0..255 left to right
    let back = decode(&encode(&img, Format::Dds, &opts(DdsFormat::Dxt1, false, true)).unwrap()).unwrap().to_rgba8();
    assert!(back.chunks_exact(4).all(|p| p[3] == 0 || p[3] == 255));
    let first = &back[..4];
    let last = &back[(31 * 4)..(31 * 4 + 4)];
    assert_eq!(first[3], 0);
    assert_eq!(last[3], 255);
}

#[test]
fn mips_do_not_bleed_colour_from_transparent_pixels() {
    // Left half: opaque white. Right half: fully transparent black. Alpha-weighted
    // filtering keeps the surviving colour white in the 1x1 mip.
    let (w, h) = (8u32, 8u32);
    let mut d = Vec::new();
    for _ in 0..h {
        for x in 0..w {
            d.extend_from_slice(if x < w / 2 { &[255, 255, 255, 255] } else { &[0, 0, 0, 0] });
        }
    }
    let img = Image::from_u8(w, h, ChannelLayout::Rgba, d).unwrap();
    let b = encode(&img, Format::Dds, &opts(DdsFormat::Argb8, true, true)).unwrap();
    let last = &b[b.len() - 4..]; // 1x1 level, BGRA
    assert_eq!(&last[..3], &[255, 255, 255]);
    assert!((120..=136).contains(&last[3]), "alpha {}", last[3]);
}

#[test]
fn decodes_dx10_header() {
    // Hand-built DX10 file: 4x4 BC1_UNORM, one solid-red block.
    let mut b = Vec::new();
    b.extend_from_slice(b"DDS ");
    let mut push = |v: u32| b.extend_from_slice(&v.to_le_bytes());
    for v in [124, 0x1007, 4, 4, 8, 0, 0] {
        push(v);
    }
    for _ in 0..11 {
        push(0);
    }
    for v in [32, 4, u32::from_le_bytes(*b"DX10"), 0, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0] {
        push(v);
    }
    for v in [71, 3, 0, 1, 0] {
        push(v); // DXGI_FORMAT_BC1_UNORM, Texture2D, no misc flags, 1 array slice
    }
    b.extend_from_slice(&[0x00, 0xF8, 0x00, 0xF8, 0, 0, 0, 0]); // c0 = c1 = pure red
    let img = decode(&b).unwrap();
    assert_eq!((img.width(), img.height()), (4, 4));
    assert!(img.to_rgba8().chunks_exact(4).all(|p| p == [255, 0, 0, 255]));
}

#[test]
fn unsupported_layouts_are_rejected_not_misread() {
    let img = gradient(8, 8);
    let mut b = encode(&img, Format::Dds, &opts(DdsFormat::Dxt5, false, true)).unwrap();
    b[112..116].copy_from_slice(&0x200u32.to_le_bytes()); // caps2: cube map
    assert!(matches!(decode(&b), Err(CodecError::Unsupported { .. })));
    b[112..116].copy_from_slice(&0x20_0000u32.to_le_bytes()); // caps2: volume
    assert!(matches!(decode(&b), Err(CodecError::Unsupported { .. })));
}

#[test]
fn truncated_and_corrupt_files_fail_cleanly() {
    let img = gradient(32, 16);
    for f in [DdsFormat::Dxt1, DdsFormat::Dxt5, DdsFormat::Bc5, DdsFormat::Argb8] {
        let b = encode(&img, Format::Dds, &opts(f, true, true)).unwrap();
        for n in 0..b.len().min(400) {
            let _ = decode_as_with(Format::Dds, &b[..n], &tight());
        }
        assert!(decode_as_with(Format::Dds, &b[..127], &tight()).is_err());
        let mut bad = b.clone();
        bad[0] = b'X';
        assert!(decode_as_with(Format::Dds, &bad, &tight()).is_err());
    }
}

#[test]
fn declared_size_beyond_limits_is_refused() {
    let mut b = encode(&gradient(16, 16), Format::Dds, &opts(DdsFormat::Dxt5, false, true)).unwrap();
    b[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    b[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(decode_as_with(Format::Dds, &b, &tight()), Err(CodecError::LimitExceeded(_))));
}

#[test]
fn capability_and_extension() {
    let c = caps(Format::Dds);
    assert!(c.read && c.write && c.lossy);
    assert_eq!(from_extension("DDS"), Some(Format::Dds));
}
