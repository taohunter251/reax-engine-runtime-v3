// KFX (kai/kx-k2, #6 KF-MX) two-segment tile map of kfx.cu. Plain C++ as well (no CUDA): tests/test_kfx.py builds it
// with g++ and checks it against its Python mirror mx.kfx_tiles for many (M, M1).
//   Segment 1 = rows [0, M1) (class 1: the D decisions), segment 2 = rows [M1, M) (class 2: the S decisions); M1 must
//   already be clamped to [0, M]. Row tile order: segment 1's full 128-row tiles, segment 2's full 128-row tiles, then
//   segment 1's row tail, then segment 2's. A tail of r rows (1..127) is ONE tile of h = ceil32(r) rows: 32 / 64 / 96
//   (KF's small tiles), 128 for 97..127 rows (a big tile with its rows past the segment end zero-filled, not stored).
//   A tile never holds rows of both segments: its rows past mend (the segment end) are neither loaded nor stored.
// The grid has kfx_row_tiles(M) row tiles for every M1: ceil(M1 / 128) + ceil((M - M1) / 128) <= ceil(M / 128) + 1.
#ifndef KFX_MAP_H
#define KFX_MAP_H

#ifdef __CUDACC__
#define KFX_HD __host__ __device__ __forceinline__
#else
#define KFX_HD static inline
#endif

#define KFX_BM 128

KFX_HD int kfx_row_tiles(int M) { return (M + KFX_BM - 1) / KFX_BM + 1; }

// row tile t at (M, M1) -> 0: past both segments (the CTA exits); 1: m0 (first row), mend (segment end), cls (1 / 2),
// h (tile config rows: 32 / 64 / 96 / 128)
KFX_HD int kfx_tile(int t, int M, int M1, int* m0, int* mend, int* cls, int* h) {
    const int L2 = M - M1;
    const int b1 = M1 / KFX_BM, b2 = L2 / KFX_BM;
    const int r1 = M1 - b1 * KFX_BM, r2 = L2 - b2 * KFX_BM;
    if (t < b1) {
        *m0 = t * KFX_BM; *mend = M1; *cls = 1; *h = KFX_BM;
        return 1;
    }
    t -= b1;
    if (t < b2) {
        *m0 = M1 + t * KFX_BM; *mend = M; *cls = 2; *h = KFX_BM;
        return 1;
    }
    t -= b2;
    if (r1 > 0) {
        if (t == 0) {
            *m0 = b1 * KFX_BM; *mend = M1; *cls = 1; *h = (r1 + 31) / 32 * 32;
            return 1;
        }
        --t;
    }
    if (r2 > 0 && t == 0) {
        *m0 = M1 + b2 * KFX_BM; *mend = M; *cls = 2; *h = (r2 + 31) / 32 * 32;
        return 1;
    }
    return 0;
}

#endif
