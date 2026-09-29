# zstd (single-file amalgamation)

Upstream: https://github.com/facebook/zstd — commit 01b7154f1172432f8abe9b3bb9909e14a1176b7d (2026-09-18), v1.6.0, BSD-3-Clause.
Vendored 2026-09-29 for SPZ v4 export, which compresses attribute streams with zstd.

`zstd.c` is the full-library amalgamation from `build/single_file_libs/`;
`zstd.h` + `zstd_errors.h` are the upstream public headers (the latter is
included by the former). Include as `"external/zstd.h"`; `src/external/*.c`
is already in `cmake/sources.txt`, so no build changes were needed.
