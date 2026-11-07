# libwebp (encoder-only subset)

Upstream: https://github.com/webmproject/libwebp — commit 8e7f95e00b8eb41d6c2c7659440cc179771d8f0a (2026-09-29), BSD-3-Clause.
Vendored 2026-09-29 for SOG export, which requires lossless WebP textures.

Subset: all of `src/enc` (lossless + lossy encoder), `src/dsp`, `src/utils`
(minus decoder-only `bit_reader_utils.c`, `huffman_utils.c`,
`quant_levels_dec_utils.c`), and `sharpyuv` (linked by `webp_enc.c` even on
the lossless path). No decoder/mux/demux sources. Five decoder *headers*
under `src/dec/` are kept because `dsp.h`, `yuv.h`, `lossless.c` and `dec.c`
include them; no `src/dec/*.c` is vendored and the encoder never calls them.

Layout preserves upstream's `src/` prefix so internal `"src/webp/..."` includes
resolve; the build adds this dir and its `src/` subdir to the include path.
SIMD variants are upstream-guarded per architecture and compile to empty
translation units where unsupported.
