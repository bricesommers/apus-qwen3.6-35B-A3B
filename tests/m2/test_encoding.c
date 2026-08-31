/*
 * tests/m2/test_encoding.c — Qwen ChatML message-format conformance.
 *
 * The Qwen repo ships no conformance pairs, so golden/chat_case_N.txt is the
 * reference output of rendering reference/chat_template.jinja with jinja2 in
 * the same environment HF apply_chat_template uses (see gen_golden.py). For
 * every golden/chat_case_N.json spec:
 *   - run encoding.h twice, byte-identical (determinism)
 *   - byte-compare the rendered prompt against chat_case_N.txt exactly
 *   - compare the token-id sequence (encoding.h + tok.h) against
 *     chat_case_N.ids exactly
 *   - the ling_encode_ids convenience wrapper must agree
 * Coverage: thinking on/off, system/no-system, thinking stripped after the
 * last user query vs preserve_thinking, embedded <think> splits (incl.
 * non-string reasoning_content), tools preamble (with/without system, empty
 * system content), tool calls (all JSON value types, missing wrapper,
 * missing arguments), grouped tool responses, first-position tool message,
 * tool_response-wrapped user turns, content |trim (Unicode whitespace),
 * content as text-item lists, null content, no-generation-prompt.
 * Tokenizer path: reference/tokenizer.json, overridable via APUS_TOK_JSON.
 * Run from the repository root.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_TOK_IMPLEMENTATION
#define APUS_ENCODING_IMPLEMENTATION
#include "json.h"
#include "tok.h"
#include "encoding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[sz] = 0;
    *len = (size_t)sz;
    return buf;
}

static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

int main(void) {
    Tok *t = tok_load(getenv("APUS_TOK_JSON") ? getenv("APUS_TOK_JSON")
                                              : "reference/tokenizer.json");
    if (!t) { fprintf(stderr, "tok_load failed\n"); return 1; }

    int ncases = 0;
    for (int i = 1; ; i++) {
        char spath[256], tpath[256], idspath[256], what[64];
        snprintf(spath, sizeof spath, "tests/m2/golden/chat_case_%d.json", i);
        snprintf(tpath, sizeof tpath, "tests/m2/golden/chat_case_%d.txt", i);
        snprintf(idspath, sizeof idspath, "tests/m2/golden/chat_case_%d.ids", i);
        snprintf(what, sizeof what, "chat case %d", i);

        char err[256];
        JVal *spec = json_parse_file(spath, err, sizeof err);
        if (!spec) break;
        ncases++;
        JVal *messages = json_obj_get(spec, "messages");
        JVal *tools = json_obj_get(spec, "tools");
        CHECK(messages && json_type(messages) == J_ARR, "case %d: bad spec", i);
        LingEncOpts opts = LING_ENC_OPTS_DEFAULT;
        JVal *v;
        if ((v = json_obj_get(spec, "thinking")) && json_type(v) == J_NUM)
            opts.thinking = json_num(v) != 0.0;
        if ((v = json_obj_get(spec, "preserve_thinking")) && json_type(v) == J_NUM)
            opts.preserve_thinking = json_num(v) != 0.0;
        if ((v = json_obj_get(spec, "add_generation_prompt")) && json_type(v) == J_NUM)
            opts.add_generation_prompt = json_num(v) != 0.0;

        size_t elen, glen;
        char *expected = read_file(tpath, &elen);
        unsigned char *g = (unsigned char *)read_file(idspath, &glen);
        CHECK(expected != NULL, "case %d: missing %s", i, tpath);
        CHECK(g != NULL, "case %d: missing %s", i, idspath);

        char *p1 = ling_encode_messages(messages, tools, &opts);
        char *p2 = ling_encode_messages(messages, tools, &opts);
        CHECK(p1 != NULL, "case %d: encode failed: %s", i, ling_last_error());
        CHECK(p1 && p2 && strcmp(p1, p2) == 0, "case %d: non-deterministic encode", i);
        if (p1 && expected) {
            size_t plen = strlen(p1);
            if (!(plen == elen && memcmp(p1, expected, elen) == 0)) {
                /* locate first difference */
                size_t k = 0, m = plen < elen ? plen : elen;
                while (k < m && p1[k] == expected[k]) k++;
                CHECK(0, "case %d: prompt mismatch (len %zu vs %zu) first diff at byte %zu: "
                      "got %.40s | want %.40s", i, plen, elen, k,
                      k < plen ? p1 + k : "", k < elen ? expected + k : "");
            } else {
                CHECK(1, "case %d ok", i);
            }
        }
        /* token ids of the rendered prompt */
        if (p1 && g) {
            size_t n = 0;
            uint32_t *ids = tok_encode_str(t, p1, 1, &n);
            uint32_t gn = glen >= 4 ? rd_u32(g) : 0;
            int ok = ids && glen >= 4 && (size_t)gn * 4 + 4 == glen && gn == n;
            if (ok)
                for (uint32_t k = 0; k < gn; k++)
                    if (rd_u32(g + 4 + k * 4) != ids[k]) { ok = 0; break; }
            CHECK(ok, "case %d: token ids mismatch (golden %u, got %zu)", i, gn, n);
            /* ling_encode_ids convenience wrapper must agree */
            size_t n3 = 0;
            uint32_t *ids3 = ling_encode_ids(t, messages, tools, &opts, &n3);
            CHECK(ids3 && n3 == n && (n == 0 || memcmp(ids, ids3, n * 4) == 0),
                  "case %d: ling_encode_ids != tok_encode(prompt)", i);
            free(ids3);
            free(ids);
        }
        free(p1);
        free(p2);
        free(expected);
        free(g);
        json_free(spec);
    }
    CHECK(ncases == 27, "expected 27 chat cases, found %d", ncases);
    printf("chat: %d cases rendered byte-exact\n", ncases);

    /* error paths: malformed inputs must fail cleanly, not crash */
    {
        LingEncOpts o = LING_ENC_OPTS_DEFAULT;
        char *p;

        /* non-string scalar content (jinja: 'Unexpected content type.') */
        JVal *m = json_new_arr();
        JVal *u = json_new_obj();
        json_obj_set(u, "role", json_new_str("user"));
        json_obj_set(u, "content", json_new_int(42));
        json_arr_push(m, u);
        p = ling_encode_messages(m, NULL, &o);
        CHECK(p == NULL, "non-string user content should fail");
        json_free(m);

        p = ling_encode_messages(NULL, NULL, &o);
        CHECK(p == NULL, "NULL messages should fail");

        /* empty message list ('No messages provided.') */
        m = json_new_arr();
        p = ling_encode_messages(m, NULL, &o);
        CHECK(p == NULL, "empty messages should fail");
        json_free(m);

        /* unknown role ('Unexpected message role.') */
        m = json_new_arr();
        u = json_new_obj();
        json_obj_set(u, "role", json_new_str("user"));
        json_obj_set(u, "content", json_new_str("q"));
        json_arr_push(m, u);
        u = json_new_obj();
        json_obj_set(u, "role", json_new_str("developer"));
        json_obj_set(u, "content", json_new_str("invisible"));
        json_arr_push(m, u);
        p = ling_encode_messages(m, NULL, &o);
        CHECK(p == NULL, "unknown role should fail");
        json_free(m);

        /* system message not at the beginning */
        m = json_new_arr();
        u = json_new_obj();
        json_obj_set(u, "role", json_new_str("user"));
        json_obj_set(u, "content", json_new_str("q"));
        json_arr_push(m, u);
        u = json_new_obj();
        json_obj_set(u, "role", json_new_str("system"));
        json_obj_set(u, "content", json_new_str("late"));
        json_arr_push(m, u);
        p = ling_encode_messages(m, NULL, &o);
        CHECK(p == NULL, "late system message should fail");
        json_free(m);

        /* no real user query ('No user query found in messages.') */
        m = json_new_arr();
        u = json_new_obj();
        json_obj_set(u, "role", json_new_str("assistant"));
        json_obj_set(u, "content", json_new_str("a"));
        json_arr_push(m, u);
        p = ling_encode_messages(m, NULL, &o);
        CHECK(p == NULL, "assistant-only messages should fail");
        json_free(m);

        /* every user turn wrapped in <tool_response> -> same error */
        m = json_new_arr();
        u = json_new_obj();
        json_obj_set(u, "role", json_new_str("user"));
        json_obj_set(u, "content", json_new_str("<tool_response>\nx\n</tool_response>"));
        json_arr_push(m, u);
        p = ling_encode_messages(m, NULL, &o);
        CHECK(p == NULL, "tool_response-only user turns should fail");
        json_free(m);

        /* vision content item (text-only scope) */
        m = json_new_arr();
        u = json_new_obj();
        json_obj_set(u, "role", json_new_str("user"));
        JVal *cl = json_new_arr();
        JVal *item = json_new_obj();
        json_obj_set(item, "type", json_new_str("image"));
        json_obj_set(item, "image", json_new_str("img.png"));
        json_arr_push(cl, item);
        json_obj_set(u, "content", cl);
        json_arr_push(m, u);
        p = ling_encode_messages(m, NULL, &o);
        CHECK(p == NULL, "vision content should fail");
        json_free(m);
    }

    tok_free(t);
    printf("test_encoding: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
