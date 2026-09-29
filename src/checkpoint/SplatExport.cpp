// SplatExport.cpp -- see SplatExport.h.

#include "checkpoint/SplatExport.h"

#include "external/miniz.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace spirula {
namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline uint8_t u8_sat(float v) {
    const long r = std::lround(v);
    return (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
}

inline int64_t src_row(const SplatExportSource& s, int64_t k) {
    return s.map ? s.map[k] : k;
}

inline int sh_k(int sh_degree) {  // rest coefficients per splat
    return (sh_degree + 1) * (sh_degree + 1) - 1;
}

inline float src_sh(const SplatExportSource& s, int64_t row, int coeff,
                    int ch) {
    if (s.features_sh) return s.features_sh[(row * sh_k(s.sh_degree) + coeff) * 3 + ch];
    return (*s.sh_decode)(row, coeff, ch);
}

void check_source(const SplatExportSource& s, const char* what) {
    if (s.num <= 0 || !s.means || !s.quats || !s.scales || !s.opacities ||
        !s.features_dc)
        throw std::runtime_error(std::string(what) + ": empty source");
    if (s.sh_degree < 0 || s.sh_degree > 3)
        throw std::runtime_error(std::string(what) + ": bad sh_degree");
    if (s.sh_degree > 0 && !s.features_sh && !s.sh_decode)
        throw std::runtime_error(std::string(what) + ": missing features_sh");
}

void write_file_bytes(const std::string& path,
                      const std::vector<uint8_t>& data) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open for writing: " + path);
    const bool ok =
        std::fwrite(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    if (!ok) {
        std::error_code ec;
        fs::remove(path, ec);
        throw std::runtime_error("write failed: " + path);
    }
}

// Raw deflate wrapped in a gzip container (this miniz build has no gzip mode).
void gzip_compress(const uint8_t* data, size_t len,
                   std::vector<uint8_t>& out) {
    static const uint8_t kGzipHeader[10] = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3};
    out.insert(out.end(), kGzipHeader, kGzipHeader + 10);
    mz_stream s = {};
    if (mz_deflateInit2(&s, MZ_DEFAULT_COMPRESSION, MZ_DEFLATED,
                        -MZ_DEFAULT_WINDOW_BITS, 9,
                        MZ_DEFAULT_STRATEGY) != MZ_OK)
        throw std::runtime_error("gzip deflate init failed");
    s.next_in = data;
    s.avail_in = (mz_uint32)len;
    uint8_t chunk[1 << 15];
    int rc;
    do {
        s.next_out = chunk;
        s.avail_out = sizeof chunk;
        rc = mz_deflate(&s, MZ_FINISH);
        if (rc != MZ_OK && rc != MZ_STREAM_END) {
            mz_deflateEnd(&s);
            throw std::runtime_error("gzip deflate failed");
        }
        out.insert(out.end(), chunk, chunk + (sizeof chunk - s.avail_out));
    } while (rc != MZ_STREAM_END);
    mz_deflateEnd(&s);
    const uint32_t crc = (uint32_t)mz_crc32(MZ_CRC32_INIT, data, len);
    const uint32_t isize = (uint32_t)(len & 0xffffffffu);
    for (int i = 0; i < 4; ++i) out.push_back((uint8_t)(crc >> (8 * i)));
    for (int i = 0; i < 4; ++i) out.push_back((uint8_t)(isize >> (8 * i)));
}

// zlib-wrapped deflate, matching miniz_oxide compress_to_vec level 6.
void zlib_compress(const uint8_t* data, size_t len,
                   std::vector<uint8_t>& out) {
    mz_ulong bound = mz_compressBound((mz_ulong)len);
    out.resize((size_t)bound);
    mz_ulong zlen = bound;
    if (mz_compress2(out.data(), &zlen, data, (mz_ulong)len, 6) != MZ_OK)
        throw std::runtime_error("zlib deflate failed");
    out.resize((size_t)zlen);
}

void zip_write_file(
    const std::string& path,
    const std::vector<std::pair<std::string, std::vector<uint8_t>>>& files) {
    mz_zip_archive zip = {};
    if (!mz_zip_writer_init_file(&zip, path.c_str(), 0))
        throw std::runtime_error("cannot open zip for writing: " + path);
    bool ok = true;
    for (const auto& f : files) {
        if (!mz_zip_writer_add_mem(&zip, f.first.c_str(), f.second.data(),
                                   f.second.size(), MZ_DEFAULT_COMPRESSION)) {
            ok = false;
            break;
        }
    }
    if (ok) ok = mz_zip_writer_finalize_archive(&zip) != 0;
    mz_zip_writer_end(&zip);
    if (!ok) {
        std::error_code ec;
        fs::remove(path, ec);
        throw std::runtime_error("zip write failed: " + path);
    }
}

// Minimal JSON value model with insertion-ordered objects.
struct JVal {
    enum class T { Null, Bool, Int, Float, Str, Arr, Obj } t = T::Null;
    bool b = false;
    int64_t i = 0;
    double f = 0;
    std::string s;
    std::vector<JVal> a;
    std::vector<std::pair<std::string, JVal>> o;
    static JVal Bool(bool v) { JVal j; j.t = T::Bool; j.b = v; return j; }
    static JVal Int(int64_t v) { JVal j; j.t = T::Int; j.i = v; return j; }
    static JVal Float(double v) { JVal j; j.t = T::Float; j.f = v; return j; }
    static JVal Str(const std::string& v) { JVal j; j.t = T::Str; j.s = v; return j; }
    static JVal Arr() { JVal j; j.t = T::Arr; return j; }
    static JVal Obj() { JVal j; j.t = T::Obj; return j; }
    void push(const JVal& v) { a.push_back(v); }
    void set(const std::string& k, const JVal& v) { o.emplace_back(k, v); }
};

// Shortest round-trip decimal for a double (ryu-style output not required,
// only that parsing yields the same value).
std::string json_float(double v) {
    if (!std::isfinite(v)) throw std::runtime_error("non-finite json float");
    char buf[32];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) return buf;
    }
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

void json_escape(const std::string& s, std::string& out) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[7];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    out += b;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

void json_write(const JVal& j, std::string& out, int indent) {
    const bool pretty = indent >= 0;
    switch (j.t) {
        case JVal::T::Null: out += "null"; break;
        case JVal::T::Bool: out += j.b ? "true" : "false"; break;
        case JVal::T::Int: out += std::to_string(j.i); break;
        case JVal::T::Float: out += json_float(j.f); break;
        case JVal::T::Str: json_escape(j.s, out); break;
        case JVal::T::Arr: {
            out.push_back('[');
            for (size_t k = 0; k < j.a.size(); ++k) {
                if (k) out.push_back(',');
                if (pretty) {
                    out.push_back('\n');
                    out.append((size_t)(indent + 1) * 2, ' ');
                }
                json_write(j.a[k], out, pretty ? indent + 1 : -1);
            }
            if (pretty && !j.a.empty()) {
                out.push_back('\n');
                out.append((size_t)indent * 2, ' ');
            }
            out.push_back(']');
            break;
        }
        case JVal::T::Obj: {
            out.push_back('{');
            for (size_t k = 0; k < j.o.size(); ++k) {
                if (k) out.push_back(',');
                if (pretty) {
                    out.push_back('\n');
                    out.append((size_t)(indent + 1) * 2, ' ');
                }
                json_escape(j.o[k].first, out);
                out.push_back(':');
                if (pretty) out.push_back(' ');
                json_write(j.o[k].second, out, pretty ? indent + 1 : -1);
            }
            if (pretty && !j.o.empty()) {
                out.push_back('\n');
                out.append((size_t)indent * 2, ' ');
            }
            out.push_back('}');
            break;
        }
    }
}

// IEEE-754 binary16 bits, round-to-nearest-even.
uint16_t f32_to_f16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 112;
    const uint32_t mant = x & 0x7fffffu;
    if (exp >= 31) return (uint16_t)(sign | 0x7bffu);  // inf/nan -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  // underflow -> 0
        const uint32_t m = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((m + 0xfffu + ((m >> 13) & 1u)) >> 13));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) |
                      ((mant + 0xfffu + ((mant >> 13) & 1u)) >> 13));
}

// q-th quantile via nth_element (q in [0,1]).
float quantile(std::vector<float>& v, double q) {
    if (v.empty()) return 0.0f;
    size_t k = (size_t)(q * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// SPZ v3 (legacy gzip single-stream). v3 is chosen over v4 because it needs
// no zstd dependency. Coordinates are written as-is, matching the official
// saveSpz default (no coordinate-system conversion).
constexpr float kSpzColorScale = 0.15f;
constexpr float kSpzSqrtHalf = 0.707106781186547524401f;

uint8_t spz_quant_sh(float x, int bucket) {
    long q = std::lround(x * 128.0f) + 128;
    q = (q + bucket / 2) / bucket * bucket;
    return (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
}

// Smallest-three packing: 2-bit largest-component index followed by three
// (sign bit, 9-bit magnitude) fields, little-endian. q is (w,x,y,z).
void spz_pack_quat(const float q[4], uint8_t r[4]) {
    float v[4] = {q[1], q[2], q[3], q[0]};
    float n = 0;
    for (float c : v) n += c * c;
    n = std::sqrt(n);
    if (!(n > 1e-12f)) {  // degenerate: identity rotation
        v[0] = v[1] = v[2] = 0;
        v[3] = 1;
        n = 1;
    }
    for (float& c : v) c /= n;
    unsigned i_largest = 0;
    for (unsigned i = 1; i < 4; ++i)
        if (std::fabs(v[i]) > std::fabs(v[i_largest])) i_largest = i;
    const unsigned negate = v[i_largest] < 0;
    uint32_t comp = i_largest;
    for (unsigned i = 0; i < 4; ++i) {
        if (i == i_largest) continue;
        const uint32_t negbit = (v[i] < 0) ^ negate;
        const uint32_t mag =
            (uint32_t)(511.0f * std::fabs(v[i]) / kSpzSqrtHalf + 0.5f);
        comp = (comp << 10) | (negbit << 9) | mag;
    }
    for (int i = 0; i < 4; ++i) r[i] = (uint8_t)(comp >> (8 * i));
}

void spz_push_u32le(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back((uint8_t)(x >> (8 * i)));
}

}  // namespace

void write_splat_spz(const SplatExportSource& s, const std::string& path) {
    check_source(s, "write_splat_spz");
    const int64_t n = s.num;
    const int sh_dim = sh_k(s.sh_degree);

    std::vector<uint8_t> raw;
    raw.reserve((size_t)n * (9 + 1 + 3 + 3 + 4 + sh_dim * 3) + 16);
    raw.push_back('N');
    raw.push_back('G');
    raw.push_back('S');
    raw.push_back('P');
    spz_push_u32le(raw, 3);            // version
    spz_push_u32le(raw, (uint32_t)n);  // numPoints
    raw.push_back((uint8_t)s.sh_degree);
    raw.push_back(12);  // fractionalBits
    raw.push_back(0);   // flags (not trained with antialiasing)
    raw.push_back(0);   // reserved

    for (int64_t k = 0; k < n; ++k) {  // positions: 24-bit fixed point
        const int64_t i = src_row(s, k);
        for (int j = 0; j < 3; ++j) {
            const int32_t f = (int32_t)std::lround(s.means[i * 3 + j] * 4096.0f);
            const uint32_t u = (uint32_t)f;
            raw.push_back((uint8_t)(u & 0xff));
            raw.push_back((uint8_t)((u >> 8) & 0xff));
            raw.push_back((uint8_t)((u >> 16) & 0xff));
        }
    }
    for (int64_t k = 0; k < n; ++k)  // alphas
        raw.push_back(u8_sat(std::round(sigmoidf(s.opacities[src_row(s, k)]) * 255.0f)));
    for (int64_t k = 0; k < n; ++k) {  // colors (DC)
        const int64_t i = src_row(s, k);
        for (int j = 0; j < 3; ++j)
            raw.push_back(u8_sat(std::round(s.features_dc[i * 3 + j] *
                                                kSpzColorScale * 255.0f +
                                            127.5f)));
    }
    for (int64_t k = 0; k < n; ++k) {  // scales
        const int64_t i = src_row(s, k);
        for (int j = 0; j < 3; ++j)
            raw.push_back(u8_sat(std::round((s.scales[i * 3 + j] + 10.0f) * 16.0f)));
    }
    for (int64_t k = 0; k < n; ++k) {  // rotations
        uint8_t r[4];
        spz_pack_quat(s.quats + src_row(s, k) * 4, r);
        raw.insert(raw.end(), r, r + 4);
    }
    for (int64_t k = 0; k < n; ++k)  // sh: coefficient outer, channel inner
        for (int c = 0; c < sh_dim; ++c) {
            const int bucket = c < 3 ? 8 : 16;  // 5 bits deg-1, 4 bits rest
            for (int j = 0; j < 3; ++j)
                raw.push_back(
                    spz_quant_sh(src_sh(s, src_row(s, k), c, j), bucket));
        }

    std::vector<uint8_t> gz;
    gzip_compress(raw.data(), raw.size(), gz);
    write_file_bytes(path, gz);
}

void write_splat_sog(const SplatExportSource& s, const std::string& path) {
    (void)s;
    (void)path;
    throw std::runtime_error(
        "SOG export requires a native lossless WebP encoder (not yet implemented). "
        "SPZ and RAD exports are available.");
}

// RAD v1 writer (flat, no level-of-detail tree). Matches the official Spark
// PLY->RAD conversion: raw byte encodings, each property zlib-compressed,
// chunk size 65536. All JSON numbers use the shortest round-trip decimal.

constexpr float kRadShC0 = 0.28209479177387814f;
constexpr float kRadLnScaleMin = -12.0f;
constexpr float kRadLnScaleMax = 9.0f;
constexpr int64_t kRadChunkSize = 65536;

void rad_le(std::vector<uint8_t>& v, uint64_t x, int n) {
    for (int i = 0; i < n; ++i) v.push_back((uint8_t)(x >> (8 * i)));
}

// Dim-major encodings, mirroring the official spark-lib layout.
std::vector<uint8_t> rad_f32_lebytes(const float* d, int dims, int64_t n) {
    std::vector<uint8_t> r;
    r.reserve((size_t)dims * n * 4);
    for (int b = 0; b < 4; ++b)
        for (int a = 0; a < dims; ++a)
            for (int64_t i = 0; i < n; ++i) {
                uint32_t x;
                std::memcpy(&x, d + i * dims + a, 4);
                r.push_back((uint8_t)(x >> (8 * b)));
            }
    return r;
}

std::vector<uint8_t> rad_r8(const float* d, int dims, int64_t n, float mn,
                            float mx) {
    std::vector<uint8_t> r;
    r.reserve((size_t)dims * n);
    const float sc = mx > mn ? 255.0f / (mx - mn) : 0.0f;
    for (int a = 0; a < dims; ++a)
        for (int64_t i = 0; i < n; ++i)
            r.push_back(u8_sat((d[i * dims + a] - mn) * sc));
    return r;
}

std::vector<uint8_t> rad_r8_delta(const float* d, int dims, int64_t n, float mn,
                                  float mx) {
    std::vector<uint8_t> r;
    r.reserve((size_t)dims * n);
    const float sc = mx > mn ? 255.0f / (mx - mn) : 0.0f;
    for (int a = 0; a < dims; ++a) {
        uint8_t last = 0;
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t v = u8_sat((d[i * dims + a] - mn) * sc);
            r.push_back((uint8_t)(v - last));
            last = v;
        }
    }
    return r;
}

std::vector<uint8_t> rad_ln_0r8(const float* d, int dims, int64_t n) {
    std::vector<uint8_t> r;
    r.reserve((size_t)dims * n);
    for (int a = 0; a < dims; ++a)
        for (int64_t i = 0; i < n; ++i) {
            const float v = d[i * dims + a];
            uint8_t q = 0;
            if (v > 0) {
                const float ln = std::log(v);
                if (ln > -30.0f) {
                    const float t =
                        (ln - kRadLnScaleMin) / (kRadLnScaleMax - kRadLnScaleMin) *
                        254.0f;
                    long c = std::lround(t < 0 ? 0 : t > 254 ? 254 : t);
                    q = (uint8_t)(c + 1);
                }
            }
            r.push_back(q);
        }
    return r;
}

// Octahedral axis-angle: [u, v, r] from an (x,y,z,w) quaternion.
void rad_oct888(const float q[4], uint8_t r[3]) {
    float v[4] = {q[0], q[1], q[2], q[3]};
    if (v[3] < 0)
        for (float& c : v) c = -c;
    const float theta =
        2.0f * std::acos(v[3] < 0 ? 0 : v[3] > 1 ? 1 : v[3]);
    const float sn = std::sin(theta * 0.5f);
    float ax[3];
    if (std::fabs(sn) < 1e-6f) {
        ax[0] = 1;
        ax[1] = ax[2] = 0;
    } else {
        for (int i = 0; i < 3; ++i) ax[i] = v[i] / sn;
    }
    const float sum = std::fabs(ax[0]) + std::fabs(ax[1]) + std::fabs(ax[2]);
    float p[2] = {ax[0] / sum, ax[1] / sum};
    if (ax[2] < 0) {
        const float p0 = (1.0f - std::fabs(p[1])) * (p[0] >= 0 ? 1.0f : -1.0f);
        const float p1 = (1.0f - std::fabs(p[0])) * (p[1] >= 0 ? 1.0f : -1.0f);
        p[0] = p0;
        p[1] = p1;
    }
    r[0] = u8_sat((p[0] + 1.0f) * 0.5f * 255.0f);
    r[1] = u8_sat((p[1] + 1.0f) * 0.5f * 255.0f);
    r[2] = u8_sat(theta / 3.14159265358979323846f * 255.0f);
}

// Robust per-band max like the official RAD encoder: max(|p5|, |p95|, 1).
float rad_band_max(const float* d, int64_t n) {
    std::vector<float> v(d, d + (size_t)n);
    const int64_t n5 = std::min<int64_t>((int64_t)(n * 0.05 + 0.5), n - 1);
    const int64_t n95 = std::min<int64_t>((int64_t)(n * 0.95 + 0.5), n - 1);
    std::nth_element(v.begin(), v.begin() + (size_t)n5, v.end());
    const float p5 = v[(size_t)n5];
    std::nth_element(v.begin(), v.begin() + (size_t)n95, v.end());
    const float p95 = v[(size_t)n95];
    return std::max(1.0f, std::max(std::fabs(p5), std::fabs(p95)));
}

std::vector<uint8_t> rad_s8(const float* d, int dims, int64_t n, float mx) {    std::vector<uint8_t> r;
    r.reserve((size_t)dims * n);
    for (int a = 0; a < dims; ++a)
        for (int64_t i = 0; i < n; ++i) {
            float t = d[i * dims + a] / mx * 127.0f;
            t = t < -127 ? -127 : t > 127 ? 127 : t;
            r.push_back((uint8_t)(int8_t)std::lround(t));
        }
    return r;
}

struct RadProp {
    const char* name;
    const char* encoding;
    std::vector<uint8_t> bytes;  // zlib-compressed
    double minv, maxv;
};

JVal rad_prop_json(const RadProp& p, uint64_t offset) {
    JVal j = JVal::Obj();
    j.set("offset", JVal::Int((int64_t)offset));
    j.set("bytes", JVal::Int((int64_t)p.bytes.size()));
    j.set("property", JVal::Str(p.name));
    j.set("encoding", JVal::Str(p.encoding));
    j.set("compression", JVal::Str("gz"));
    j.set("min", JVal::Float(p.minv));
    j.set("max", JVal::Float(p.maxv));
    return j;
}

// Builds one chunk: header, pretty meta JSON, pad, payloadBytes, payloads.
std::vector<uint8_t> rad_build_chunk(const SplatExportSource& s, int64_t base,
                                     int64_t count, int deg) {
    const int K = sh_k(deg);
    std::vector<float> buf((size_t)count * 3);
    std::vector<RadProp> props;

    for (int64_t i = 0; i < count; ++i) {  // center: f32_lebytes
        const int64_t r = src_row(s, base + i);
        for (int a = 0; a < 3; ++a) buf[(size_t)i * 3 + a] = s.means[r * 3 + a];
    }
    props.push_back({"center", "f32_lebytes", {}, 0, 0});  // min/max below
    {
        std::vector<uint8_t> raw = rad_f32_lebytes(buf.data(), 3, count);
        std::vector<uint8_t> z;
        zlib_compress(raw.data(), raw.size(), z);
        props.back().bytes = std::move(z);
    }
    {
        double mn = buf[0], mx = buf[0];
        for (float v : buf) {
            mn = std::min(mn, (double)v);
            mx = std::max(mx, (double)v);
        }
        props.back().minv = mn;
        props.back().maxv = mx;
    }

    std::vector<float> abuf((size_t)count);
    for (int64_t i = 0; i < count; ++i)  // alpha: r8
        abuf[(size_t)i] = sigmoidf(s.opacities[src_row(s, base + i)]);
    {
        double mn = 1, mx = 0;
        for (float v : abuf) {
            mn = std::min(mn, (double)v);
            mx = std::max(mx, (double)v);
        }
        std::vector<uint8_t> raw = rad_r8(abuf.data(), 1, count, (float)mn, (float)mx);
        std::vector<uint8_t> z;
        zlib_compress(raw.data(), raw.size(), z);
        props.push_back({"alpha", "r8", std::move(z), mn, mx});
    }

    for (int64_t i = 0; i < count; ++i) {  // rgb: r8_delta
        const int64_t r = src_row(s, base + i);
        for (int a = 0; a < 3; ++a)
            buf[(size_t)i * 3 + a] = 0.5f + s.features_dc[r * 3 + a] * kRadShC0;
    }
    {
        double mn = 1, mx = 0;
        for (int64_t i = 0; i < count; ++i)
            for (int a = 0; a < 3; ++a) {
                mn = std::min(mn, (double)buf[(size_t)i * 3 + a]);
                mx = std::max(mx, (double)buf[(size_t)i * 3 + a]);
            }
        std::vector<uint8_t> raw =
            rad_r8_delta(buf.data(), 3, count, (float)mn, (float)mx);
        std::vector<uint8_t> z;
        zlib_compress(raw.data(), raw.size(), z);
        props.push_back({"rgb", "r8_delta", std::move(z), mn, mx});
    }

    for (int64_t i = 0; i < count; ++i) {  // scales: ln_0r8
        const int64_t r = src_row(s, base + i);
        for (int a = 0; a < 3; ++a)
            buf[(size_t)i * 3 + a] = std::exp(s.scales[r * 3 + a]);
    }
    {
        std::vector<uint8_t> raw = rad_ln_0r8(buf.data(), 3, count);
        std::vector<uint8_t> z;
        zlib_compress(raw.data(), raw.size(), z);
        props.push_back(
            {"scales", "ln_0r8", std::move(z), kRadLnScaleMin, kRadLnScaleMax});
    }

    {  // orientation: oct88r8
        std::vector<uint8_t> raw;
        raw.reserve((size_t)count * 3);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t r = src_row(s, base + i);
            float q[4] = {s.quats[r * 4 + 1], s.quats[r * 4 + 2],
                          s.quats[r * 4 + 3], s.quats[r * 4 + 0]};  // xyzw
            float nrm = 0;
            for (float c : q) nrm += c * c;
            nrm = std::sqrt(nrm);
            if (!(nrm > 1e-12f)) {
                q[0] = q[1] = q[2] = 0;
                q[3] = 1;
            } else {
                for (float& c : q) c /= nrm;
            }
            uint8_t e[3];
            rad_oct888(q, e);
            raw.insert(raw.end(), e, e + 3);
        }
        std::vector<uint8_t> z;
        zlib_compress(raw.data(), raw.size(), z);
        props.push_back({"orientation", "oct88r8", std::move(z), -1, 1});
    }

    static const char* kShNames[3] = {"sh1", "sh2", "sh3"};
    static const int kShLens[3] = {3, 5, 7};
    for (int b = 0; b < deg; ++b) {  // sh1..sh3: s8
        const int band = kShLens[b], off = b == 0 ? 0 : b == 1 ? 3 : 8;
        std::vector<float> sh((size_t)count * band * 3);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t r = src_row(s, base + i);
            for (int c = 0; c < band; ++c)
                for (int j = 0; j < 3; ++j)
                    sh[(size_t)i * band * 3 + j * band + c] =
                        src_sh(s, r, off + c, j);
        }
        const float mx = rad_band_max(sh.data(), (int64_t)count * band * 3);
        std::vector<uint8_t> raw = rad_s8(sh.data(), band * 3, count, mx);
        std::vector<uint8_t> z;
        zlib_compress(raw.data(), raw.size(), z);
        props.push_back({kShNames[b], "s8", std::move(z), -mx, mx});
    }

    uint64_t payload_bytes = 0;
    for (const auto& p : props)
        payload_bytes += (p.bytes.size() + 7) & ~(uint64_t)7;  // 8-byte padded

    JVal m = JVal::Obj();
    m.set("version", JVal::Int(1));
    m.set("base", JVal::Int(base));
    m.set("count", JVal::Int(count));
    m.set("payloadBytes", JVal::Int((int64_t)payload_bytes));
    m.set("maxSh", JVal::Int(deg));
    JVal pa = JVal::Arr();
    uint64_t offset = 0;
    for (const auto& p : props) {
        pa.push(rad_prop_json(p, offset));
        offset += (p.bytes.size() + 7) & ~(uint64_t)7;  // 8-byte padded
    }
    m.set("properties", pa);
    std::string meta_json;
    json_write(m, meta_json, 0);
    meta_json.push_back('\n');

    std::vector<uint8_t> out;
    rad_le(out, 0x43444152u, 4);  // "RADC"
    rad_le(out, (uint32_t)meta_json.size(), 4);
    out.insert(out.end(), meta_json.begin(), meta_json.end());
    while (out.size() % 8) out.push_back(0);
    rad_le(out, payload_bytes, 8);
    for (const auto& p : props) {
        out.insert(out.end(), p.bytes.begin(), p.bytes.end());
        while (out.size() % 8) out.push_back(0);
    }
    return out;
}

void write_splat_rad(const SplatExportSource& s, const std::string& path) {
    check_source(s, "write_splat_rad");
    const int64_t n = s.num;
    const int deg = s.sh_degree;

    std::vector<std::vector<uint8_t>> chunks;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    uint64_t offset = 0;
    for (int64_t base = 0; base < n; base += kRadChunkSize) {
        const int64_t count = std::min(kRadChunkSize, n - base);
        chunks.push_back(rad_build_chunk(s, base, count, deg));
        ranges.emplace_back(offset, chunks.back().size());
        offset += chunks.back().size();
    }

    JVal m = JVal::Obj();
    m.set("version", JVal::Int(1));
    m.set("type", JVal::Str("gsplat"));
    m.set("count", JVal::Int(n));
    m.set("maxSh", JVal::Int(deg));
    m.set("chunkSize", JVal::Int(kRadChunkSize));
    m.set("allChunkBytes", JVal::Int((int64_t)offset));
    JVal ca = JVal::Arr();
    for (const auto& r : ranges) {
        JVal c = JVal::Obj();
        c.set("offset", JVal::Int((int64_t)r.first));
        c.set("bytes", JVal::Int((int64_t)r.second));
        ca.push(c);
    }
    m.set("chunks", ca);
    m.set("comment", JVal::Str("spirula-studio"));
    std::string meta_json;
    json_write(m, meta_json, 0);
    meta_json.push_back('\n');

    std::vector<uint8_t> out;
    rad_le(out, 0x30444152u, 4);  // "RAD0"
    rad_le(out, (uint32_t)meta_json.size(), 4);
    out.insert(out.end(), meta_json.begin(), meta_json.end());
    while (out.size() % 8) out.push_back(0);
    for (const auto& c : chunks) out.insert(out.end(), c.begin(), c.end());
    write_file_bytes(path, out);
}

}  // namespace spirula
