// SplatExport.cpp -- see SplatExport.h.

#include "checkpoint/SplatExport.h"

#include "external/miniz.h"
#include "external/zstd.h"
#include "webp/encode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

// SPZ packing, shared by the v3 and v4 writers. Coordinates are written
// as-is, matching the official saveSpz default (no coordinate conversion).
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

void spz_push_u64le(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back((uint8_t)(x >> (8 * i)));
}

// Attribute streams in official order: positions, alphas, colors, scales,
// rotations, sh.
struct SpzStreams {
    std::vector<uint8_t> positions, alphas, colors, scales, rotations, sh;
};

void spz_pack_streams(const SplatExportSource& s, SpzStreams& st) {
    const int64_t n = s.num;
    const int sh_dim = sh_k(s.sh_degree);
    st.positions.reserve((size_t)n * 9);
    st.alphas.reserve((size_t)n);
    st.colors.reserve((size_t)n * 3);
    st.scales.reserve((size_t)n * 3);
    st.rotations.reserve((size_t)n * 4);
    st.sh.reserve((size_t)n * (size_t)sh_dim * 3);
    for (int64_t k = 0; k < n; ++k) {  // positions: 24-bit fixed point
        const int64_t i = src_row(s, k);
        for (int j = 0; j < 3; ++j) {
            const int32_t f =
                (int32_t)std::lround(s.means[i * 3 + j] * 4096.0f);
            const uint32_t u = (uint32_t)f;
            st.positions.push_back((uint8_t)(u & 0xff));
            st.positions.push_back((uint8_t)((u >> 8) & 0xff));
            st.positions.push_back((uint8_t)((u >> 16) & 0xff));
        }
    }
    for (int64_t k = 0; k < n; ++k)  // alphas
        st.alphas.push_back(
            u8_sat(std::round(sigmoidf(s.opacities[src_row(s, k)]) * 255.0f)));
    for (int64_t k = 0; k < n; ++k) {  // colors (DC)
        const int64_t i = src_row(s, k);
        for (int j = 0; j < 3; ++j)
            st.colors.push_back(u8_sat(std::round(s.features_dc[i * 3 + j] *
                                                      kSpzColorScale * 255.0f +
                                                  127.5f)));
    }
    for (int64_t k = 0; k < n; ++k) {  // scales
        const int64_t i = src_row(s, k);
        for (int j = 0; j < 3; ++j)
            st.scales.push_back(
                u8_sat(std::round((s.scales[i * 3 + j] + 10.0f) * 16.0f)));
    }
    for (int64_t k = 0; k < n; ++k) {  // rotations
        uint8_t r[4];
        spz_pack_quat(s.quats + src_row(s, k) * 4, r);
        st.rotations.insert(st.rotations.end(), r, r + 4);
    }
    for (int64_t k = 0; k < n; ++k)  // sh: coefficient outer, channel inner
        for (int c = 0; c < sh_dim; ++c) {
            const int bucket = c < 3 ? 8 : 16;  // 5 bits deg-1, 4 bits rest
            for (int j = 0; j < 3; ++j)
                st.sh.push_back(
                    spz_quant_sh(src_sh(s, src_row(s, k), c, j), bucket));
        }
}

}  // namespace

void write_splat_spz(const SplatExportSource& s, const std::string& path) {
    check_source(s, "write_splat_spz");
    const int64_t n = s.num;
    const int sh_dim = sh_k(s.sh_degree);
    SpzStreams st;
    spz_pack_streams(s, st);

    std::vector<uint8_t> raw;
    raw.reserve(16 + (size_t)n * (9 + 1 + 3 + 3 + 4 + sh_dim * 3));
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
    const std::vector<uint8_t>* streams[6] = {
        &st.positions, &st.alphas, &st.colors,
        &st.scales, &st.rotations, &st.sh,
    };
    for (const auto* v : streams) raw.insert(raw.end(), v->begin(), v->end());

    std::vector<uint8_t> gz;
    gzip_compress(raw.data(), raw.size(), gz);
    write_file_bytes(path, gz);
}

// SPZ v4: 32-byte plaintext header, TOC, independent zstd streams.
void write_splat_spz_v4(const SplatExportSource& s, const std::string& path) {
    check_source(s, "write_splat_spz_v4");
    const int64_t n = s.num;
    SpzStreams st;
    spz_pack_streams(s, st);
    const std::vector<uint8_t>* streams[6] = {
        &st.positions, &st.alphas, &st.colors,
        &st.scales, &st.rotations, &st.sh,
    };

    std::vector<std::vector<uint8_t>> comp;
    std::vector<size_t> usize;
    for (int i = 0; i < 6; ++i) {
        if (streams[i]->empty()) continue;  // official writer skips these
        const size_t bound = ZSTD_compressBound(streams[i]->size());
        std::vector<uint8_t> c(bound);
        const size_t cn =
            ZSTD_compress(c.data(), bound, streams[i]->data(),
                          streams[i]->size(), 3);
        if (ZSTD_isError(cn))
            throw std::runtime_error(std::string("zstd compress failed: ") +
                                     ZSTD_getErrorName(cn));
        c.resize(cn);
        comp.push_back(std::move(c));
        usize.push_back(streams[i]->size());
    }
    const int nstreams = (int)comp.size();

    std::vector<uint8_t> out;
    out.reserve(32 + (size_t)nstreams * 16);
    out.push_back('N');
    out.push_back('G');
    out.push_back('S');
    out.push_back('P');
    spz_push_u32le(out, 4);            // version
    spz_push_u32le(out, (uint32_t)n);  // numPoints
    out.push_back((uint8_t)s.sh_degree);
    out.push_back(12);  // fractionalBits
    out.push_back(0);   // flags (not trained with antialiasing)
    out.push_back((uint8_t)nstreams);  // numStreams (empty ones skipped)
    spz_push_u32le(out, 32);  // tocByteOffset (no extensions)
    out.insert(out.end(), 12, 0);  // reserved
    for (int i = 0; i < nstreams; ++i) {  // TOC: (compressed, uncompressed)
        spz_push_u64le(out, comp[i].size());
        spz_push_u64le(out, usize[i]);
    }
    for (int i = 0; i < nstreams; ++i)
        out.insert(out.end(), comp[i].begin(), comp[i].end());
    write_file_bytes(path, out);
}

// ---- SOG v2 (PlayCanvas Self-Organizing Gaussians) -------------------------
// Morton-ordered splats packed into lossless WebP textures inside a stored
// (uncompressed) ZIP plus meta.json. Follows splat-transform's write-sog.ts.
namespace {

inline double sog_logt(double v) {  // sign(v) * log(|v| + 1), matches JS Math.sign
    const double s = v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
    return s * std::log(std::fabs(v) + 1.0);
}

// Optimal 1D quantization over pooled columns: 1024-bin histogram (uniform /
// quantile blend), dynamic programming to k centroids, nearest-centroid labels.
std::vector<float> sog_quantize_finite(const std::vector<float>& sorted, int kTarget,
                                       double alpha) {
    const int64_t N = (int64_t)sorted.size();
    const int64_t H = std::min<int64_t>(1024, N);
    const double vMin = sorted.front(), vMax = sorted.back();
    const double vRange = vMax - vMin;
    const double iqr = (double)sorted[(size_t)std::floor(N * 0.75)] -
                       (double)sorted[(size_t)std::floor(N * 0.25)];
    double beta = 1.0 - iqr / vRange;
    beta = std::max(0.5, std::min(0.999, beta));
    std::vector<double> counts((size_t)H, 0.0), sums((size_t)H, 0.0);
    for (int64_t i = 0; i < N; ++i) {
        const double v = sorted[(size_t)i];
        int64_t bin = (int64_t)(H * (beta * ((double)i / (double)N) +
                                     (1.0 - beta) * ((v - vMin) / vRange)));
        if (bin > H - 1) bin = H - 1;
        counts[(size_t)bin] += 1.0;
        sums[(size_t)bin] += v;
    }
    std::vector<double> centers((size_t)H), weights((size_t)H);
    for (int64_t i = 0; i < H; ++i) {
        if (counts[(size_t)i] > 0.0) {
            centers[(size_t)i] = sums[(size_t)i] / counts[(size_t)i];
            weights[(size_t)i] = std::pow(counts[(size_t)i], alpha);
        } else {
            centers[(size_t)i] = vMin + (double)(i + 0.5) / (double)H * vRange;
        }
    }
    std::vector<double> pW((size_t)H + 1, 0.0), pWX((size_t)H + 1, 0.0),
        pWXX((size_t)H + 1, 0.0);
    for (int64_t i = 0; i < H; ++i) {
        pW[(size_t)i + 1] = pW[(size_t)i] + weights[(size_t)i];
        pWX[(size_t)i + 1] = pWX[(size_t)i] + weights[(size_t)i] * centers[(size_t)i];
        pWXX[(size_t)i + 1] = pWXX[(size_t)i] +
                              weights[(size_t)i] * centers[(size_t)i] * centers[(size_t)i];
    }
    auto rangeCost = [&](int64_t a, int64_t b) {
        const double w = pW[(size_t)b + 1] - pW[(size_t)a];
        if (w <= 0.0) return 0.0;
        const double wx = pWX[(size_t)b + 1] - pWX[(size_t)a];
        const double wxx = pWXX[(size_t)b + 1] - pWXX[(size_t)a];
        return wxx - wx * wx / w;
    };
    auto rangeMean = [&](int64_t a, int64_t b) {
        const double w = pW[(size_t)b + 1] - pW[(size_t)a];
        if (w <= 0.0) return (centers[(size_t)a] + centers[(size_t)b]) * 0.5;
        return (pWX[(size_t)b + 1] - pWX[(size_t)a]) / w;
    };
    int64_t nonEmpty = 0;
    for (double c : counts) if (c > 0.0) ++nonEmpty;
    const int64_t eK = std::min<int64_t>(kTarget, nonEmpty);
    const double INF = 1e30;
    std::vector<double> dpP((size_t)H, INF), dpC((size_t)H, INF);
    std::vector<std::vector<int32_t>> split((size_t)eK + 1);
    split[1].assign((size_t)H, -1);
    for (int64_t j = 0; j < H; ++j) dpP[(size_t)j] = rangeCost(0, j);
    for (int64_t m = 2; m <= eK; ++m) {
        std::fill(dpC.begin(), dpC.end(), INF);
        split[(size_t)m].assign((size_t)H, 0);
        for (int64_t j = m - 1; j < H; ++j) {
            double best = INF;
            int32_t bestS = (int32_t)(m - 2);
            for (int64_t s = m - 2; s < j; ++s) {
                const double c = dpP[(size_t)s] + rangeCost(s + 1, j);
                if (c < best) { best = c; bestS = (int32_t)s; }
            }
            dpC[(size_t)j] = best;
            split[(size_t)m][(size_t)j] = bestS;
        }
        dpP.swap(dpC);
    }
    std::vector<float> cv((size_t)eK);
    int64_t j = H - 1;
    for (int64_t m = eK; m >= 1; --m) {
        const int64_t s = m > 1 ? split[(size_t)m][(size_t)j] : -1;
        cv[(size_t)m - 1] = (float)rangeMean(s + 1, j);
        j = s;
    }
    std::sort(cv.begin(), cv.end());
    return cv;
}

struct SogQuant {
    std::vector<float> centroids;               // k entries
    std::vector<std::vector<uint8_t>> labels;  // [numCols][numRows]
};

SogQuant sog_quantize1d(const std::vector<const float*>& cols, int64_t nrows, int k = 256,
                        double alpha = 0.5) {
    SogQuant out;
    const int64_t ncols = (int64_t)cols.size();
    out.labels.assign((size_t)ncols, std::vector<uint8_t>((size_t)nrows, 0));
    out.centroids.assign((size_t)k, 0.0f);
    const int64_t N = nrows * ncols;
    if (N == 0) return out;
    std::vector<float> data((size_t)N);
    for (int64_t c = 0; c < ncols; ++c)
        std::copy(cols[(size_t)c], cols[(size_t)c] + nrows, data.begin() + (size_t)(c * nrows));
    std::vector<float> finite;
    finite.reserve((size_t)N);
    bool hasNegInf = false, hasPosInf = false;
    for (float v : data) {
        if (std::isfinite(v)) finite.push_back(v);
        else if (v == -std::numeric_limits<float>::infinity()) hasNegInf = true;
        else if (v == std::numeric_limits<float>::infinity()) hasPosInf = true;
    }
    const int64_t nf = (int64_t)finite.size();
    if (nf == 0 && !hasNegInf && !hasPosInf) {  // all NaN: pad with +20, labels to 255
        std::fill(out.centroids.begin(), out.centroids.end(), 20.0f);
        for (auto& l : out.labels) std::fill(l.begin(), l.end(), (uint8_t)255);
        return out;
    }
    std::sort(finite.begin(), finite.end());
    const int loSlots = hasNegInf ? 1 : 0, hiSlots = hasPosInf ? 1 : 0;
    const float negInfC = (nf > 0 ? finite.front() : 0.0f) - 20.0f;
    const float posInfC = (nf > 0 ? finite.back() : 0.0f) + 20.0f;
    if (nf > 0) {
        const double vMin = finite.front(), vMax = finite.back();
        const bool flat = vMax - vMin < 1e-20;
        if (loSlots == 0 && hiSlots == 0 && flat) {
            std::fill(out.centroids.begin(), out.centroids.end(), (float)vMin);
            return out;
        }
        std::vector<float> cvals;
        if (flat) cvals.push_back((float)vMin);
        else if (k - loSlots - hiSlots > 0)
            cvals = sog_quantize_finite(finite, k - loSlots - hiSlots, alpha);
        const size_t ek = cvals.size();
        std::copy(cvals.begin(), cvals.end(), out.centroids.begin() + loSlots);
        const float pad = ek > 0 ? cvals[ek - 1] : (hasNegInf ? negInfC : posInfC);
        for (int i = loSlots + (int)ek; i < k - hiSlots; ++i) out.centroids[(size_t)i] = pad;
    } else {
        const float pad = hasNegInf ? negInfC : posInfC;
        for (int i = loSlots; i < k - hiSlots; ++i) out.centroids[(size_t)i] = pad;
    }
    if (loSlots) out.centroids[0] = negInfC;
    if (hiSlots) out.centroids[(size_t)k - 1] = posInfC;
    const float* fc = out.centroids.data();
    for (int64_t i = 0; i < N; ++i) {
        const double v = data[(size_t)i];
        int lo = 0, hi = k - 1;
        while (lo < hi) {  // nearest centroid by midpoint binary search
            const int mid = (lo + hi) >> 1;
            if (v < ((double)fc[mid] + (double)fc[mid + 1]) * 0.5) hi = mid;
            else lo = mid + 1;
        }
        out.labels[(size_t)(i / nrows)][(size_t)(i % nrows)] = (uint8_t)lo;
    }
    return out;
}

inline uint32_t sog_part1by2(uint32_t v) {
    v &= 0x000003ffu;
    v = (v ^ (v << 16)) & 0xff0000ffu;
    v = (v ^ (v << 8)) & 0x0300f00fu;
    v = (v ^ (v << 4)) & 0x030c30c3u;
    v = (v ^ (v << 2)) & 0x09249249u;
    return v;
}

// Recursively refined Morton (Z-order) sort over positions, 10 bits per axis.
void sog_morton_gen(uint32_t* idx, size_t n, const float* p) {
    if (n == 0) return;
    double mnx = 1e300, mxx = -1e300, mny = 1e300, mxy = -1e300, mnz = 1e300,
           mxz = -1e300;
    for (size_t i = 0; i < n; ++i) {
        const size_t b = (size_t)idx[i] * 3;
        const double x = p[b], y = p[b + 1], z = p[b + 2];
        if (x < mnx) mnx = x; if (x > mxx) mxx = x;
        if (y < mny) mny = y; if (y > mxy) mxy = y;
        if (z < mnz) mnz = z; if (z > mxz) mxz = z;
    }
    const double xlen = mxx - mnx, ylen = mxy - mny, zlen = mxz - mnz;
    if (!std::isfinite(xlen) || !std::isfinite(ylen) || !std::isfinite(zlen)) return;
    if (xlen == 0.0 && ylen == 0.0 && zlen == 0.0) return;
    const double xmul = xlen == 0.0 ? 0.0 : 1024.0 / xlen;
    const double ymul = ylen == 0.0 ? 0.0 : 1024.0 / ylen;
    const double zmul = zlen == 0.0 ? 0.0 : 1024.0 / zlen;
    std::vector<uint32_t> key(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t b = (size_t)idx[i] * 3;
        const uint32_t ix = (uint32_t)std::min(1023.0, ((double)p[b] - mnx) * xmul);
        const uint32_t iy = (uint32_t)std::min(1023.0, ((double)p[b + 1] - mny) * ymul);
        const uint32_t iz = (uint32_t)std::min(1023.0, ((double)p[b + 2] - mnz) * zmul);
        key[i] = (sog_part1by2(iz) << 2) + (sog_part1by2(iy) << 1) + sog_part1by2(ix);
    }
    std::vector<uint32_t> sI(n), sK(n), counts(1024);
    uint32_t *srcI = idx, *srcK = key.data(), *dstI = sI.data(), *dstK = sK.data();
    for (int shift = 0; shift < 30; shift += 10) {  // 3-pass 10-bit radix sort
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; ++i) ++counts[(srcK[i] >> shift) & 1023];
        uint32_t sum = 0;
        for (int d = 0; d < 1024; ++d) {
            const uint32_t c = counts[(size_t)d];
            counts[(size_t)d] = sum;
            sum += c;
        }
        for (size_t i = 0; i < n; ++i) {
            const uint32_t o = counts[(srcK[i] >> shift) & 1023]++;
            dstI[o] = srcI[i];
            dstK[o] = srcK[i];
        }
        std::swap(srcI, dstI);
        std::swap(srcK, dstK);
    }
    std::copy(sI.begin(), sI.end(), idx);
    size_t start = 0;  // buckets sharing a code get re-sorted over local bounds
    while (start < n) {
        size_t end = start + 1;
        while (end < n && sK[end] == sK[start]) ++end;
        if (end - start > 256) sog_morton_gen(idx + start, end - start, p);
        start = end;
    }
}

// Exact nearest-centroid kd-tree for the SH palette k-means.
struct SogKd {
    struct Node { int dim = -1, left = -1, right = -1, pt = -1; float cut = 0.0f; };
    int nc = 0;
    const float* pts = nullptr;
    std::vector<Node> nodes;
    std::vector<int> ids;
    void build(const float* p, int64_t k, int d) {
        pts = p; nc = d; nodes.clear(); ids.resize((size_t)k);
        for (int64_t i = 0; i < k; ++i) ids[(size_t)i] = (int)i;
        if (k > 0) build_rec(0, k);
    }
    int build_rec(int64_t l, int64_t r) {
        const int ni = (int)nodes.size();
        nodes.emplace_back();
        if (r - l == 1) { nodes[(size_t)ni].pt = ids[(size_t)l]; return ni; }
        int bd = 0; double bs = -1.0;
        for (int j = 0; j < nc; ++j) {
            double mn = 1e300, mx = -1e300;
            for (int64_t i = l; i < r; ++i) {
                const double v = pts[(size_t)ids[(size_t)i] * (size_t)nc + j];
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            if (mx - mn > bs) { bs = mx - mn; bd = j; }
        }
        const int64_t m = (l + r) / 2;
        std::nth_element(ids.begin() + l, ids.begin() + m, ids.begin() + r,
                         [&](int a, int b) {
                             return pts[(size_t)a * (size_t)nc + bd] <
                                    pts[(size_t)b * (size_t)nc + bd];
                         });
        nodes[(size_t)ni].dim = bd;
        nodes[(size_t)ni].cut = pts[(size_t)ids[(size_t)m] * (size_t)nc + bd];
        const int left = build_rec(l, m);
        const int right = build_rec(m, r);
        nodes[(size_t)ni].left = left;
        nodes[(size_t)ni].right = right;
        return ni;
    }
    int nearest(const float* q) const {
        double best = 1e300; int bi = -1;
        search(0, q, best, bi);
        return bi;
    }
    void search(int ni, const float* q, double& best, int& bi) const {
        const Node& nd = nodes[(size_t)ni];
        if (nd.pt >= 0) {
            double d2 = 0.0;
            for (int j = 0; j < nc; ++j) {
                const double d = (double)q[j] - (double)pts[(size_t)nd.pt * (size_t)nc + j];
                d2 += d * d;
            }
            if (d2 < best) { best = d2; bi = nd.pt; }
            return;
        }
        const double diff = (double)q[nd.dim] - (double)nd.cut;
        const int f = diff <= 0.0 ? nd.left : nd.right;
        const int s = diff <= 0.0 ? nd.right : nd.left;
        search(f, q, best, bi);
        if (diff * diff < best) search(s, q, best, bi);
    }
};

struct SogKm { std::vector<float> centroids; std::vector<uint32_t> labels; };

SogKm sog_kmeans_fit(const float* pts, int64_t n, int nc, int64_t k, int iters,
                     std::mt19937& rng) {
    SogKm r;
    r.labels.assign((size_t)n, 0);
    if (n < k) {  // fewer points than clusters: each point is its own centroid
        r.centroids.assign(pts, pts + (size_t)n * (size_t)nc);
        for (int64_t i = 0; i < n; ++i) r.labels[(size_t)i] = (uint32_t)i;
        return r;
    }
    std::vector<int64_t> picks;  // Floyd's algorithm: k unique random indices
    picks.reserve((size_t)k);
    std::unordered_set<int64_t> seen;
    for (int64_t j = n - k; j < n; ++j) {
        std::uniform_int_distribution<int64_t> d(0, j);
        const int64_t t = d(rng);
        if (seen.insert(t).second) picks.push_back(t);
        else { seen.insert(j); picks.push_back(j); }
    }
    r.centroids.assign((size_t)k * (size_t)nc, 0.0f);
    for (int64_t i = 0; i < k; ++i)
        std::copy(pts + picks[(size_t)i] * nc, pts + picks[(size_t)i] * nc + nc,
                  r.centroids.begin() + (size_t)(i * nc));
    std::vector<double> sums((size_t)k * (size_t)nc);
    std::vector<int64_t> counts((size_t)k);
    SogKd tree;
    for (int it = 0; it < iters; ++it) {
        tree.build(r.centroids.data(), k, nc);
        for (int64_t i = 0; i < n; ++i)
            r.labels[(size_t)i] = (uint32_t)tree.nearest(pts + i * nc);
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t c = r.labels[(size_t)i];
            ++counts[(size_t)c];
            for (int j = 0; j < nc; ++j)
                sums[(size_t)c * (size_t)nc + j] += (double)pts[(size_t)i * (size_t)nc + j];
        }
        for (int64_t i = 0; i < k; ++i) {
            if (counts[(size_t)i] == 0) {  // reseed empty clusters to a random point
                std::uniform_int_distribution<int64_t> d(0, n - 1);
                const int64_t src = d(rng);
                std::copy(pts + src * nc, pts + src * nc + nc,
                          r.centroids.begin() + (size_t)(i * nc));
            } else {
                const double inv = 1.0 / (double)counts[(size_t)i];
                for (int j = 0; j < nc; ++j)
                    r.centroids[(size_t)i * (size_t)nc + j] =
                        (float)(sums[(size_t)i * (size_t)nc + j] * inv);
            }
        }
    }
    return r;
}

// Lloyd k-means with a fixed seed (deterministic exports). Large scenes fit the
// palette on a stride subsample, then assign every splat to the nearest entry.
SogKm sog_kmeans(const float* pts, int64_t n, int nc, int64_t k, int iters) {
    std::mt19937 rng(0x51ab3u);
    const int64_t S = std::min<int64_t>(n, 65536);
    if (S == n) return sog_kmeans_fit(pts, n, nc, k, iters, rng);
    std::vector<float> sub((size_t)S * (size_t)nc);
    for (int64_t i = 0; i < S; ++i) {
        const int64_t src = i * n / S;
        std::copy(pts + src * nc, pts + src * nc + nc, sub.begin() + (size_t)(i * nc));
    }
    SogKm r = sog_kmeans_fit(sub.data(), S, nc, k, iters, rng);
    const int64_t nk = (int64_t)r.centroids.size() / nc;
    r.labels.assign((size_t)n, 0);
    SogKd tree;
    tree.build(r.centroids.data(), nk, nc);
    for (int64_t i = 0; i < n; ++i)
        r.labels[(size_t)i] = (uint32_t)tree.nearest(pts + i * nc);
    return r;
}

// Lossless WebP via the full API with exact=1, so transparent pixels keep RGB.
std::vector<uint8_t> sog_webp(const uint8_t* rgba, int w, int h) {
    WebPConfig config;
    if (!WebPConfigInit(&config) || !WebPConfigLosslessPreset(&config, 6))
        throw std::runtime_error("WebPConfigLosslessPreset failed");
    config.exact = 1;
    if (!WebPValidateConfig(&config))
        throw std::runtime_error("WebPValidateConfig failed");
    WebPPicture pic;
    if (!WebPPictureInit(&pic)) throw std::runtime_error("WebPPictureInit failed");
    pic.width = w; pic.height = h;
    pic.use_argb = 1;  // lossless RGB path; without this the encoder uses YUV
    WebPMemoryWriter writer;
    WebPMemoryWriterInit(&writer);
    pic.writer = WebPMemoryWrite;
    pic.custom_ptr = &writer;
    int ok = WebPPictureImportRGBA(&pic, rgba, w * 4) && WebPEncode(&config, &pic);
    WebPPictureFree(&pic);
    if (!ok || !writer.mem) {
        WebPMemoryWriterClear(&writer);
        throw std::runtime_error("WebP lossless encode failed");
    }
    std::vector<uint8_t> r(writer.mem, writer.mem + writer.size);
    WebPMemoryWriterClear(&writer);
    return r;
}

inline void sog_zip_u16(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)x); v.push_back((uint8_t)(x >> 8));
}
inline void sog_zip_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)x); v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)(x >> 16)); v.push_back((uint8_t)(x >> 24));
}

// Stored (method 0) ZIP with data descriptors, like the reference writer.
void sog_zip(const std::string& path,
             const std::vector<std::pair<std::string, std::vector<uint8_t>>>& files) {
    std::time_t now = std::time(nullptr);
    std::tm tmv = *std::localtime(&now);
    const uint16_t dosTime = (uint16_t)(((tmv.tm_hour) << 11) | ((tmv.tm_min) << 5) |
                                       ((tmv.tm_sec / 2)));
    const uint16_t dosDate = (uint16_t)(((tmv.tm_year - 80) << 9) | ((tmv.tm_mon + 1) << 5) |
                                       (tmv.tm_mday));
    struct Ent { std::string name; uint32_t crc, size, hdr; };
    std::vector<Ent> ents;
    std::vector<uint8_t> out;
    for (const auto& f : files) {
        if (out.size() >= 0xffffffffu)
            throw std::runtime_error("SOG zip: file too large for 32-bit offsets");
        const uint32_t hdr = (uint32_t)out.size();
        const uint32_t crc = (uint32_t)mz_crc32(MZ_CRC32_INIT, f.second.data(), f.second.size());
        const uint32_t sz = (uint32_t)f.second.size();
        sog_zip_u32(out, 0x04034b50); sog_zip_u16(out, 20); sog_zip_u16(out, 0x0808);
        sog_zip_u16(out, 0); sog_zip_u16(out, dosTime); sog_zip_u16(out, dosDate);
        sog_zip_u32(out, 0); sog_zip_u32(out, 0); sog_zip_u32(out, 0);
        sog_zip_u16(out, (uint32_t)f.first.size()); sog_zip_u16(out, 0);
        out.insert(out.end(), f.first.begin(), f.first.end());
        out.insert(out.end(), f.second.begin(), f.second.end());
        sog_zip_u32(out, 0x08074b50); sog_zip_u32(out, crc);
        sog_zip_u32(out, sz); sog_zip_u32(out, sz);
        ents.push_back({f.first, crc, sz, hdr});
    }
    const uint32_t cdir = (uint32_t)out.size();
    for (const auto& e : ents) {
        sog_zip_u32(out, 0x02014b50); sog_zip_u16(out, 20); sog_zip_u16(out, 20);
        sog_zip_u16(out, 0x0808); sog_zip_u16(out, 0);
        sog_zip_u16(out, dosTime); sog_zip_u16(out, dosDate);
        sog_zip_u32(out, e.crc); sog_zip_u32(out, e.size); sog_zip_u32(out, e.size);
        sog_zip_u16(out, (uint32_t)e.name.size());
        for (int i = 0; i < 4; ++i) sog_zip_u16(out, 0);  // extra/comment/disk/int attr
        sog_zip_u32(out, 0); sog_zip_u32(out, e.hdr);
        out.insert(out.end(), e.name.begin(), e.name.end());
    }
    const uint32_t cdirSize = (uint32_t)out.size() - cdir;
    sog_zip_u32(out, 0x06054b50);
    sog_zip_u16(out, 0); sog_zip_u16(out, 0);
    sog_zip_u16(out, (uint32_t)ents.size()); sog_zip_u16(out, (uint32_t)ents.size());
    sog_zip_u32(out, cdirSize); sog_zip_u32(out, cdir); sog_zip_u16(out, 0);
    write_file_bytes(path, out);
}

}  // namespace

void write_splat_sog(const SplatExportSource& s, const std::string& path,
                     std::vector<uint32_t>* out_order) {
    check_source(s, "write_splat_sog");
    const int64_t n = s.num;
    const int bands = s.sh_degree;
    const int restCount = bands == 0 ? 0 : bands == 1 ? 9 : bands == 2 ? 24 : 45;
    const int shCoeffs = bands == 0 ? 0 : bands == 1 ? 3 : bands == 2 ? 8 : 15;

    std::vector<float> pos((size_t)n * 3);
    for (int64_t k = 0; k < n; ++k) {
        const int64_t row = src_row(s, k);
        pos[(size_t)k * 3 + 0] = s.means[row * 3 + 0];
        pos[(size_t)k * 3 + 1] = s.means[row * 3 + 1];
        pos[(size_t)k * 3 + 2] = s.means[row * 3 + 2];
    }
    std::vector<uint32_t> order((size_t)n);
    for (int64_t k = 0; k < n; ++k) order[(size_t)k] = (uint32_t)k;
    sog_morton_gen(order.data(), (size_t)n, pos.data());
    if (out_order) *out_order = order;

    const int64_t W = (int64_t)std::ceil(std::sqrt((double)n) / 4.0) * 4;
    const int64_t H = (int64_t)std::ceil((double)n / (double)W / 4.0) * 4;
    if (W > 16383 || H > 16383)
        throw std::runtime_error("SOG export: texture dimensions exceed 16383 texels");
    const size_t texels = (size_t)W * (size_t)H;
    auto tex = [&]() { return std::vector<uint8_t>(texels * 4, 0); };

    // means: log-encoded, split across means_l (low bytes) and means_u (high).
    double lmn[3] = {1e300, 1e300, 1e300}, lmx[3] = {-1e300, -1e300, -1e300};
    for (int64_t k = 0; k < n; ++k)
        for (int j = 0; j < 3; ++j) {
            const double lt = sog_logt(pos[(size_t)k * 3 + j]);
            if (lt < lmn[j]) lmn[j] = lt;
            if (lt > lmx[j]) lmx[j] = lt;
        }
    std::vector<uint8_t> meansL = tex(), meansU = tex();
    for (int64_t t = 0; t < n; ++t) {
        const int64_t g = order[(size_t)t];
        for (int j = 0; j < 3; ++j) {
            const double lt = sog_logt(pos[(size_t)g * 3 + j]);
            double x = lmx[j] > lmn[j] ? 65535.0 * (lt - lmn[j]) / (lmx[j] - lmn[j]) : 0.0;
            if (!(x >= 0.0)) x = 0.0;  // NaN from infinite input becomes 0
            const uint32_t u = (uint32_t)x;
            meansL[(size_t)t * 4 + j] = (uint8_t)(u & 0xff);
            meansU[(size_t)t * 4 + j] = (uint8_t)((u >> 8) & 0xff);
        }
        meansL[(size_t)t * 4 + 3] = 255;
        meansU[(size_t)t * 4 + 3] = 255;
    }

    // quats: smallest-three packing, alpha tags the removed component.
    static const int qidx[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
    std::vector<uint8_t> quatsTex = tex();
    for (int64_t t = 0; t < n; ++t) {
        const int64_t row = src_row(s, order[(size_t)t]);
        double q[4] = {(double)s.quats[row * 4 + 0], (double)s.quats[row * 4 + 1],
                       (double)s.quats[row * 4 + 2], (double)s.quats[row * 4 + 3]};
        double l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        if (!(l > 1e-12)) { q[0] = 1.0; q[1] = q[2] = q[3] = 0.0; l = 1.0; }
        for (int i = 0; i < 4; ++i) q[i] /= l;
        int mc = 0;
        for (int i = 1; i < 4; ++i)
            if (std::fabs(q[i]) > std::fabs(q[mc])) mc = i;
        const double sgn = (q[mc] < 0.0 ? -1.0 : 1.0) * std::sqrt(2.0);
        for (int i = 0; i < 4; ++i) q[i] *= sgn;
        for (int c = 0; c < 3; ++c) {
            double b = 255.0 * (q[qidx[mc][c]] * 0.5 + 0.5);
            if (!(b >= 0.0)) b = 0.0; else if (b > 255.0) b = 255.0;
            quatsTex[(size_t)t * 4 + c] = (uint8_t)b;
        }
        quatsTex[(size_t)t * 4 + 3] = (uint8_t)(252 + mc);
    }

    // scales + DC: pooled 1D codebooks, labels scattered in Morton order.
    std::vector<float> sc0((size_t)n), sc1((size_t)n), sc2((size_t)n);
    std::vector<float> dc0((size_t)n), dc1((size_t)n), dc2((size_t)n);
    for (int64_t k = 0; k < n; ++k) {
        const int64_t row = src_row(s, k);
        sc0[(size_t)k] = s.scales[row * 3 + 0];
        sc1[(size_t)k] = s.scales[row * 3 + 1];
        sc2[(size_t)k] = s.scales[row * 3 + 2];
        dc0[(size_t)k] = s.features_dc[row * 3 + 0];
        dc1[(size_t)k] = s.features_dc[row * 3 + 1];
        dc2[(size_t)k] = s.features_dc[row * 3 + 2];
    }
    const SogQuant sq = sog_quantize1d({sc0.data(), sc1.data(), sc2.data()}, n);
    const SogQuant cq = sog_quantize1d({dc0.data(), dc1.data(), dc2.data()}, n);
    std::vector<uint8_t> scalesTex = tex(), sh0Tex = tex();
    for (int64_t t = 0; t < n; ++t) {
        const int64_t g = order[(size_t)t];
        const int64_t row = src_row(s, g);
        scalesTex[(size_t)t * 4 + 0] = sq.labels[0][(size_t)g];
        scalesTex[(size_t)t * 4 + 1] = sq.labels[1][(size_t)g];
        scalesTex[(size_t)t * 4 + 2] = sq.labels[2][(size_t)g];
        scalesTex[(size_t)t * 4 + 3] = 255;
        sh0Tex[(size_t)t * 4 + 0] = cq.labels[0][(size_t)g];
        sh0Tex[(size_t)t * 4 + 1] = cq.labels[1][(size_t)g];
        sh0Tex[(size_t)t * 4 + 2] = cq.labels[2][(size_t)g];
        const double op = 1.0 / (1.0 + std::exp(-(double)s.opacities[row]));
        const double b = op * 255.0;
        sh0Tex[(size_t)t * 4 + 3] = b >= 255.0 ? 255 : (b > 0.0 ? (uint8_t)b : 0);
    }

    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
    files.emplace_back("means_l.webp", sog_webp(meansL.data(), (int)W, (int)H));
    files.emplace_back("means_u.webp", sog_webp(meansU.data(), (int)W, (int)H));
    files.emplace_back("quats.webp", sog_webp(quatsTex.data(), (int)W, (int)H));
    files.emplace_back("scales.webp", sog_webp(scalesTex.data(), (int)W, (int)H));
    files.emplace_back("sh0.webp", sog_webp(sh0Tex.data(), (int)W, (int)H));

    JVal meta = JVal::Obj();
    meta.set("version", JVal::Int(2));
    JVal asset = JVal::Obj();
    asset.set("generator", JVal::Str("spirula-studio"));
    meta.set("asset", asset);
    meta.set("count", JVal::Int(n));
    JVal means = JVal::Obj(), mins = JVal::Arr(), maxs = JVal::Arr(), mfiles = JVal::Arr();
    for (int j = 0; j < 3; ++j) { mins.push(JVal::Float(lmn[j])); maxs.push(JVal::Float(lmx[j])); }
    mfiles.push(JVal::Str("means_l.webp")); mfiles.push(JVal::Str("means_u.webp"));
    means.set("mins", mins); means.set("maxs", maxs); means.set("files", mfiles);
    meta.set("means", means);
    auto codebookObj = [&](const char* name, const std::vector<float>& cb,
                           const std::vector<const char*>& fnames) {
        JVal o = JVal::Obj(), cba = JVal::Arr(), fa = JVal::Arr();
        for (float v : cb) cba.push(JVal::Float(v));
        for (const char* f : fnames) fa.push(JVal::Str(f));
        o.set("codebook", cba); o.set("files", fa);
        meta.set(name, o);
    };
    codebookObj("scales", sq.centroids, {"scales.webp"});
    JVal quats = JVal::Obj(), qfiles = JVal::Arr();
    qfiles.push(JVal::Str("quats.webp")); quats.set("files", qfiles);
    meta.set("quats", quats);
    codebookObj("sh0", cq.centroids, {"sh0.webp"});

    if (bands > 0) {  // higher SH: k-means palette + per-column codebook
        const int64_t paletteSize =
            (int64_t)(std::min(64.0, std::pow(2.0, std::floor(std::log2((double)n / 1024.0)))) *
                      1024.0);
        std::vector<float> rest((size_t)n * (size_t)restCount);
        for (int64_t k = 0; k < n; ++k) {
            const int64_t row = src_row(s, k);
            // Channel-major f_rest to match the reference: [r0..rN, g0..gN, b0..bN].
            for (int c = 0; c < restCount; ++c)
                rest[(size_t)k * (size_t)restCount + c] =
                    src_sh(s, row, c % shCoeffs, c / shCoeffs);
        }
        const SogKm km = sog_kmeans(rest.data(), n, restCount, paletteSize, 10);
        const int64_t nc = (int64_t)km.centroids.size() / restCount;
        std::vector<std::vector<float>> cbCols((size_t)restCount,
                                               std::vector<float>((size_t)nc));
        for (int64_t i = 0; i < nc; ++i)
            for (int j = 0; j < restCount; ++j)
                cbCols[(size_t)j][(size_t)i] =
                    km.centroids[(size_t)i * (size_t)restCount + j];
        std::vector<const float*> colPtrs((size_t)restCount);
        for (int j = 0; j < restCount; ++j) colPtrs[(size_t)j] = cbCols[(size_t)j].data();
        const SogQuant cbq = sog_quantize1d(colPtrs, nc);
        const int64_t cw = 64LL * shCoeffs, chh = (nc + 63) / 64;
        std::vector<uint8_t> cenTex((size_t)cw * (size_t)chh * 4, 0);
        for (int64_t i = 0; i < nc; ++i)
            for (int j = 0; j < shCoeffs; ++j) {
                const size_t o = ((size_t)i * (size_t)shCoeffs + j) * 4;
                cenTex[o + 0] = cbq.labels[(size_t)j][(size_t)i];
                cenTex[o + 1] = cbq.labels[(size_t)shCoeffs + j][(size_t)i];
                cenTex[o + 2] = cbq.labels[(size_t)shCoeffs * 2 + j][(size_t)i];
                cenTex[o + 3] = 255;
            }
        std::vector<uint8_t> labTex = tex();
        for (int64_t t = 0; t < n; ++t) {
            const uint32_t lab = km.labels[(size_t)order[(size_t)t]];
            labTex[(size_t)t * 4 + 0] = (uint8_t)(lab & 0xff);
            labTex[(size_t)t * 4 + 1] = (uint8_t)((lab >> 8) & 0xff);
            labTex[(size_t)t * 4 + 2] = 0;
            labTex[(size_t)t * 4 + 3] = 255;
        }
        files.emplace_back("shN_centroids.webp",
                           sog_webp(cenTex.data(), (int)cw, (int)chh));
        files.emplace_back("shN_labels.webp", sog_webp(labTex.data(), (int)W, (int)H));
        JVal shN = JVal::Obj(), cba = JVal::Arr(), fa = JVal::Arr();
        for (float v : cbq.centroids) cba.push(JVal::Float(v));
        fa.push(JVal::Str("shN_centroids.webp")); fa.push(JVal::Str("shN_labels.webp"));
        shN.set("count", JVal::Int(paletteSize));
        shN.set("bands", JVal::Int(bands));
        shN.set("codebook", cba); shN.set("files", fa);
        meta.set("shN", shN);
    }

    std::string metaJson;
    json_write(meta, metaJson, -1);
    files.emplace_back("meta.json",
                       std::vector<uint8_t>(metaJson.begin(), metaJson.end()));
    sog_zip(path, files);
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
