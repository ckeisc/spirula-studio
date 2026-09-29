#pragma once

// Extra splat exports next to the checkpoint's splat.ply: SPZ (nianticlabs/spz,
// v3 gzip or v4 zstd), SOG (PlayCanvas self-organizing Gaussians v2, lossless
// WebP in a stored ZIP), RAD (flat Spark).
// Inputs use the write_splat_ply convention: log scales, logit opacities,
// (w,x,y,z) quats, DC as (colour - 0.5) / C0. features_sh is [srcN, K, 3]
// coefficient-major (RGB inner), K = (sh_degree+1)^2 - 1, null when degree 0.
// sh_decode covers a quantized SH store; map[k] is the source row of splat k.

#include <cstdint>
#include <functional>
#include <string>

namespace spirula {

struct SplatExportSource {
    int64_t num = 0;
    int sh_degree = 0;                    // 0..3
    const int64_t* map = nullptr;         // [num] source rows, null = identity
    const float* means = nullptr;         // [srcN,3]
    const float* quats = nullptr;         // [srcN,4] (w,x,y,z)
    const float* scales = nullptr;        // [srcN,3] log
    const float* opacities = nullptr;     // [srcN] logit
    const float* features_dc = nullptr;   // [srcN,3]
    const float* features_sh = nullptr;   // [srcN,K,3], null if sh_degree == 0
    // Fallback SH reader: sh_decode(src_row, coeff, channel).
    const std::function<float(int64_t, int, int)>* sh_decode = nullptr;
};

// Throws std::runtime_error naming the file on failure.
void write_splat_spz(const SplatExportSource& s, const std::string& path);
void write_splat_spz_v4(const SplatExportSource& s, const std::string& path);
// out_order optionally receives the Morton-order permutation (texel t <- src row).
void write_splat_sog(const SplatExportSource& s, const std::string& path,
                     std::vector<uint32_t>* out_order = nullptr);
void write_splat_rad(const SplatExportSource& s, const std::string& path);

}  // namespace spirula
