/* ftokc.c - FTOK (kai, S1ENG_FTOK): the exact ASCII fast path of the engine's tail tokenizer, loaded by ftok.py (named ftokc, not ftok: an ftok.so would shadow ftok.py on import).
 *
 * What it computes, for an ASCII string s (the prompt text after the template's '{"evidence":'), is the id list the
 * model's tokenizers.Tokenizer gives for encode(s, add_special_tokens=False) under the configuration ftok.py checks
 * at start-up (plumb-4b tokenizer.json):
 *   1. added tokens: s is split at its added tokens (leftmost-longest, as tokenizers' AddedVocabulary; every one of
 *      them has normalized / lstrip / rstrip / single_word false); each match is its own id;
 *   2. normalizer NFC on the other segments: the identity on ASCII (no ASCII code point decomposes or composes);
 *   3. pre-tokenizer Split(Isolated) with the Qwen regex, per segment:
 *        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *      (the transformers variant with [\p{L}\p{M}] is the same on ASCII: \p{M} has no ASCII member). On ASCII,
 *      \p{L} = [A-Za-z], \p{N} = [0-9], \s = [\t\n\v\f\r ] (Unicode White_Space); piece_end() below is the
 *      backtracking matcher's result for each alternative in order (leftmost-first, greedy);
 *   4. ByteLevel (use_regex false): every byte becomes its own byte-level symbol (byte_id[]);
 *   5. BPE (no dropout, no unk, no prefixes / suffixes, ignore_merges false) per pre-token: repeatedly merge the
 *      adjacent pair of lowest merge rank, the leftmost one among equal ranks (equal ranks are the same pair), which
 *      is the order tokenizers' Word::merge_all pops its heap in.
 * Anything else returns a negative code and ftok.py uses the tokenizers path for that tail: -1 a byte >= 0x80
 * (non-ASCII: NFC and the Unicode classes are left to tokenizers), -2 a pre-token longer than FTOK_MAXW bytes,
 * -3 the output would exceed cap (never with cap >= n: every token covers at least one byte).
 *
 * Freestanding: no libc, no allocation, no global state; every table is memory the caller owns (ftok.py keeps it
 * alive), scratch is on the stack, so concurrent calls are safe. The caller (ctypes.CDLL) releases the GIL.
 * Build (KERNELS.md): gcc -O2 -fPIC -shared -nostdlib -ffreestanding -fno-builtin -o ftokc.so ftokc.c
 */
#include <stdint.h>

#define FTOK_MAXW 256           /* bytes of one pre-token handled here */
#define NONE 0xFFFFFFFFu        /* no merge for this pair */

typedef struct {
    uint64_t *mkey;             /* merge table, open addressing: (a << 32 | b) + 1, 0 = free slot */
    uint64_t *mval;             /* (rank << 32) | merged id */
    uint64_t mask;              /* slots - 1, slots a power of two */
    const int32_t *byte_id;     /* [256]: id of each byte's byte-level symbol */
    const uint8_t *added;       /* the added tokens' contents, concatenated */
    const int32_t *added_off;   /* [n_added] */
    const int32_t *added_len;   /* [n_added] */
    const int32_t *added_id;    /* [n_added] */
    int32_t n_added;
    uint8_t first[256];         /* 1 when some added token starts with this byte */
} ftok_ctx;

int ftok_abi(void) { return 1; }   /* ftok.py checks this before it uses the context layout above */

static inline uint64_t slot_of(uint64_t k, uint64_t mask)
{
    k ^= k >> 31;
    k *= 0x9E3779B97F4A7C15ull;
    k ^= k >> 29;
    return k & mask;
}

/* Insert n merges (a[i], b[i]) -> (rank[i], id[i]); a later duplicate replaces the earlier one (tokenizers builds
 * its merge map with HashMap::insert in file order). Returns 0, or -1 if the table is full. */
int ftok_put(ftok_ctx *c, const int32_t *a, const int32_t *b, const int32_t *rank, const int32_t *id, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        uint64_t k = (((uint64_t)(uint32_t)a[i] << 32) | (uint32_t)b[i]) + 1;
        uint64_t s = slot_of(k, c->mask), probes = 0;
        while (c->mkey[s] != 0 && c->mkey[s] != k) {
            s = (s + 1) & c->mask;
            if (++probes > c->mask)
                return -1;
        }
        c->mkey[s] = k;
        c->mval[s] = ((uint64_t)(uint32_t)rank[i] << 32) | (uint32_t)id[i];
    }
    return 0;
}

static inline uint32_t get(const ftok_ctx *c, int32_t a, int32_t b, int32_t *id)
{
    uint64_t k = (((uint64_t)(uint32_t)a << 32) | (uint32_t)b) + 1;
    uint64_t s = slot_of(k, c->mask);
    for (;;) {
        uint64_t x = c->mkey[s];
        if (x == k) {
            uint64_t v = c->mval[s];
            *id = (int32_t)(uint32_t)v;
            return (uint32_t)(v >> 32);
        }
        if (x == 0)
            return NONE;
        s = (s + 1) & c->mask;
    }
}

/* Look up one pair (Python-visible, for tests): rank or -1, and the merged id. */
int64_t ftok_get(const ftok_ctx *c, int32_t a, int32_t b, int32_t *id)
{
    uint32_t r = get(c, a, b, id);
    return r == NONE ? -1 : (int64_t)r;
}

static inline int is_l(uint8_t x) { return (uint8_t)((x | 0x20) - 'a') < 26; }
static inline int is_n(uint8_t x) { return (uint8_t)(x - '0') < 10; }
static inline int is_s(uint8_t x) { return x == ' ' || (uint8_t)(x - 9) < 5; }     /* \t \n \v \f \r and space */
static inline int is_nl(uint8_t x) { return x == '\n' || x == '\r'; }
static inline int is_p(uint8_t x) { return !is_s(x) && !is_l(x) && !is_n(x); }     /* [^\s\p{L}\p{N}] */

/* End of the regex match that starts at i in the segment [.., e) (i < e). Every ASCII byte starts some match. */
static int piece_end(const uint8_t *s, int i, int e)
{
    uint8_t c = s[i];
    int j;
    /* 1. (?i:'s|'t|'re|'ve|'m|'ll|'d): the second letters differ, so at most one alternative matches */
    if (c == '\'' && i + 1 < e) {
        uint8_t d = s[i + 1] | 0x20;
        if (is_l(s[i + 1])) {
            if (d == 's' || d == 't' || d == 'm' || d == 'd')
                return i + 2;
            if (i + 2 < e && is_l(s[i + 2])) {
                uint8_t f = s[i + 2] | 0x20;
                if ((d == 'r' && f == 'e') || (d == 'v' && f == 'e') || (d == 'l' && f == 'l'))
                    return i + 3;
            }
        }
    }
    /* 2. [^\r\n\p{L}\p{N}]?\p{L}+ */
    if (is_l(c)) {
        for (j = i + 1; j < e && is_l(s[j]); j++)
            ;
        return j;
    }
    if (!is_nl(c) && !is_n(c) && i + 1 < e && is_l(s[i + 1])) {
        for (j = i + 2; j < e && is_l(s[j]); j++)
            ;
        return j;
    }
    /* 3. \p{N} */
    if (is_n(c))
        return i + 1;
    /* 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*' */
    j = (c == ' ' && i + 1 < e && is_p(s[i + 1])) ? i + 1 : i;
    if (is_p(s[j])) {
        for (j++; j < e && is_p(s[j]); j++)
            ;
        for (; j < e && is_nl(s[j]); j++)
            ;
        return j;
    }
    /* c is whitespace from here on. j = end of the whitespace run, last = its last \r or \n */
    int last = -1;
    for (j = i; j < e && is_s(s[j]); j++)
        if (is_nl(s[j]))
            last = j;
    /* 5. \s*[\r\n]+ : \s* backs off to the run's last newline, [\r\n]+ takes it */
    if (last >= 0)
        return last + 1;
    /* 6. \s+(?!\S): the whole run at the end of the segment, else the run but its last byte */
    if (j == e)
        return j;
    if (j - i >= 2)
        return j - 1;
    /* 7. \s+ */
    return j;
}

/* BPE of one pre-token w[0..n) (1 <= n <= FTOK_MAXW) into out; returns the number of ids. */
static int bpe(const ftok_ctx *c, const uint8_t *w, int n, int32_t *out)
{
    int32_t sym[FTOK_MAXW], nid[FTOK_MAXW];
    uint32_t rk[FTOK_MAXW];
    int k;
    for (k = 0; k < n; k++)
        sym[k] = c->byte_id[w[k]];
    for (k = 0; k + 1 < n; k++)
        rk[k] = get(c, sym[k], sym[k + 1], &nid[k]);
    while (n > 1) {
        int best = -1;
        uint32_t br = NONE;
        for (k = 0; k + 1 < n; k++)
            if (rk[k] < br) {
                br = rk[k];
                best = k;
            }
        if (best < 0)
            break;
        sym[best] = nid[best];
        for (k = best + 1; k + 1 < n; k++)          /* drop symbol best + 1 */
            sym[k] = sym[k + 1];
        for (k = best + 1; k + 2 < n; k++)          /* drop pair best + 1 (pairs best + 2.. move down) */
        {
            rk[k] = rk[k + 1];
            nid[k] = nid[k + 1];
        }
        n--;
        if (best > 0)
            rk[best - 1] = get(c, sym[best - 1], sym[best], &nid[best - 1]);
        if (best + 1 < n)
            rk[best] = get(c, sym[best], sym[best + 1], &nid[best]);
    }
    for (k = 0; k < n; k++)
        out[k] = sym[k];
    return n;
}

/* The ids of s[0..n) into out[0..cap); returns their number, or -1 / -2 / -3 (see the top of this file). */
int ftok_encode(const ftok_ctx *c, const uint8_t *s, int n, int32_t *out, int cap)
{
    int i, no = 0;
    for (i = 0; i < n; i++)
        if (s[i] & 0x80)
            return -1;
    i = 0;
    while (i < n) {
        /* the next added token: the leftmost start, the longest token at that start */
        int at = n, alen = 0, aid = 0;
        for (int j = i; j < n && at == n; j++) {
            if (!c->first[s[j]])
                continue;
            for (int t = 0; t < c->n_added; t++) {
                int L = c->added_len[t];
                if (L <= alen || j + L > n)
                    continue;
                const uint8_t *a = c->added + c->added_off[t];
                int q = 0;
                while (q < L && a[q] == s[j + q])
                    q++;
                if (q == L) {
                    at = j;
                    alen = L;
                    aid = c->added_id[t];
                }
            }
        }
        /* the segment [i, at): pre-tokens, each through BPE */
        int p = i;
        while (p < at) {
            int q = piece_end(s, p, at);
            if (q - p > FTOK_MAXW)
                return -2;
            if (no + (q - p) > cap)
                return -3;
            no += bpe(c, s + p, q - p, out + no);
            p = q;
        }
        if (at == n)
            break;
        if (no + 1 > cap)
            return -3;
        out[no++] = aid;
        i = at + alen;
    }
    return no;
}
