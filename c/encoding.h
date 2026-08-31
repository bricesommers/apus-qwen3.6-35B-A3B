/*
 * apus encoding.h — Qwen3.6-35B-A3B chat/message format (ChatML), C11.
 *
 * Faithful C port of reference/chat_template.jinja (encode path), rendered
 * with the same settings HF apply_chat_template uses (jinja2 sandbox,
 * trim_blocks=True, lstrip_blocks=True, tojson = json.dumps with
 * ensure_ascii=False). Text-only scope (project-wide): message content may
 * be a string, null (renders empty), or a list of {"text": ...} items;
 * image/video items fail cleanly instead of rendering vision placeholders.
 *
 *   tools preamble (non-empty tools array):
 *       <|im_start|>system\n
 *       # Tools\n\nYou have access to the following functions:\n\n<tools>
 *       + "\n" + tojson(tool) per tool + "\n</tools>" + instruction text
 *       + ("\n\n" + trimmed system content, only if messages[0] is a system
 *          message whose content trims to non-empty) + <|im_end|>\n
 *   plain system:   <|im_start|>system\n + trimmed content + <|im_end|>\n
 *                   (only messages[0]; a later system message is an error)
 *   user turn:      <|im_start|>user\n + trimmed content + <|im_end|>\n
 *   assistant:      reasoning comes from reasoning_content (iff it is a
 *                   string) or is split out of content on </think>/<think>
 *                   with the template's exact Python split/rstrip/lstrip
 *                   semantics; then reasoning is |trim'd.
 *                   Preserved (thinking kept) iff opts.preserve_thinking or
 *                   the turn sits after the last real user query
 *                   (last_query_index machinery — see below):
 *                     <|im_start|>assistant\n<think>\nRC\n</think>\n\nCONTENT
 *                   stripped otherwise:
 *                     <|im_start|>assistant\nCONTENT
 *   tool calls:     per call, "\n\n" before the first when content|trim is
 *                   non-empty (nothing when empty), "\n" before later ones:
 *                     <tool_call>\n<function=NAME>\n
 *                     <parameter=k>\nv\n</parameter>\n ... per argument
 *                     </function>\n</tool_call>
 *                   (arguments must be a JSON object, in insertion order;
 *                   v verbatim if string, else json.dumps(v,
 *                   ensure_ascii=False); a "function" wrapper object is
 *                   unwrapped when present and truthy)
 *   tool results:   consecutive tool messages group under ONE
 *                   <|im_start|>user ... <|im_end|>\n turn, each wrapped in
 *                   \n<tool_response>\n...\n</tool_response>. A tool message
 *                   in first position opens no <|im_start|>user (faithful:
 *                   the template guards on loop.previtem).
 *   generation:     <|im_start|>assistant\n<think>\n        (thinking on, default)
 *                   <|im_start|>assistant\n<think>\n\n</think>\n\n (off)
 *   errors:         empty message list ("No messages provided."), no real
 *                   user query ("No user query found in messages."), a
 *                   non-first system message, an unknown role, non-string
 *                   scalar content, vision items — all fail cleanly.
 *
 * last_query_index: scanning messages in reverse, the index of the LAST
 * user message whose trimmed content is NOT wrapped in
 * <tool_response>...</tool_response>; thinking is stripped from assistant
 * turns at or before it (unless preserve_thinking). If every user message
 * is a tool_response wrapper (or there is no user message), the template
 * raises — so do we.
 *
 * All |trim operations are Python str.strip() semantics (CPython isspace:
 * 0x09-0x0D, 0x1C-0x1F, 0x20, 0x85, 0xA0, 0x1680, 0x2000-0x200A, 0x2028,
 * 0x2029, 0x202F, 0x205F, 0x3000), NOT the regex \s class.
 *
 * The message input is a json.h tree: a J_ARR of J_OBJ messages using the
 * OpenAI-style field names (role, content, reasoning_content, tool_calls
 * with function.name/function.arguments-as-object); tools is a separate
 * J_ARR (or NULL), like the template's `tools` variable.
 *
 * API note: the ling_* names are inherited from the base engine and kept
 * so the driver call sites (c/apus-qwen.c) stay put; the format is Qwen
 * ChatML, not Ling.
 *
 * Usage: #define APUS_ENCODING_IMPLEMENTATION in exactly one TU.
 */
#ifndef APUS_ENCODING_H
#define APUS_ENCODING_H

#include <stddef.h>
#include <stdint.h>

#include "json.h"
#include "tok.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int thinking;               /* 1 = on (default), 0 = off */
    int add_generation_prompt;  /* 1 (default) = append generation prompt */
    int preserve_thinking;      /* 1 = keep thinking in ALL assistant turns */
} LingEncOpts;

#define LING_ENC_OPTS_DEFAULT { 1, 1, 0 }

/* Render a message list to the prompt string (malloc'd; NULL on error,
 * see ling_last_error()). tools may be NULL. */
char *ling_encode_messages(const JVal *messages, const JVal *tools,
                           const LingEncOpts *opts);

/* Render straight to token ids (combines string assembly with tok.h).
 * Special tokens in the assembled prompt are recognized. */
uint32_t *ling_encode_ids(const Tok *t, const JVal *messages, const JVal *tools,
                          const LingEncOpts *opts, size_t *n_out);

const char *ling_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* APUS_ENCODING_H */

/* ================================================================== */
#if defined(APUS_ENCODING_IMPLEMENTATION) && !defined(APUS_ENCODING_IMPL_DONE)
#define APUS_ENCODING_IMPL_DONE

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* Exact literal blocks of the template's tools preamble */
static const char *QWEN_TOOLS_HEADER =
    "# Tools\n"
    "\n"
    "You have access to the following functions:\n"
    "\n"
    "<tools>";

static const char *QWEN_TOOLS_INSTRUCTIONS =
    "\n"
    "</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n"
    "\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

/* ---------------- error reporting ---------------- */

static char ling_err[256];

const char *ling_last_error(void) { return ling_err; }

static int ling_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ling_err, sizeof ling_err, fmt, ap);
    va_end(ap);
    return -1;
}

/* ---------------- helpers ---------------- */

static const char *jstr(const JVal *obj, const char *key) {
    JVal *v = json_obj_get((JVal *)obj, key);
    return json_type(v) == J_STR ? json_str(v) : NULL;
}

static const char *jrole(const JVal *msg) {
    const char *r = jstr(msg, "role");
    return r ? r : "";
}

/* CPython str.isspace() codepoint set (what jinja's |trim strips) */
static int py_isspace(uint32_t cp) {
    if (cp < 0x80)
        return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20);
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

/* decode one UTF-8 char; 0xFFFFFFFF on invalid (never py_isspace) */
static uint32_t dec1(const unsigned char *s, size_t n, size_t *adv) {
    if (!n) { *adv = 0; return 0xFFFFFFFFu; }
    unsigned char c = s[0];
    if (c < 0x80) { *adv = 1; return c; }
    if (c >= 0xC2 && c < 0xE0 && n >= 2 && (s[1] & 0xC0) == 0x80) {
        *adv = 2;
        return ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if (c >= 0xE0 && c < 0xF0 && n >= 3 &&
        (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *adv = 3;
        return ((uint32_t)(c & 0x0F) << 12) |
               ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if (c >= 0xF0 && c < 0xF5 && n >= 4 && (s[1] & 0xC0) == 0x80 &&
        (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *adv = 4;
        return ((uint32_t)(c & 0x07) << 18) |
               ((uint32_t)(s[1] & 0x3F) << 12) |
               ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    *adv = 1;
    return 0xFFFFFFFFu;
}

/* Python str.strip() on a byte span: returns offset of the first
 * non-whitespace char; *n_out becomes the trimmed length. */
static size_t pytrim(const char *s, size_t n, size_t *n_out) {
    const unsigned char *u = (const unsigned char *)s;
    size_t lo = 0;
    while (lo < n) {
        size_t adv;
        uint32_t cp = dec1(u + lo, n - lo, &adv);
        if (!py_isspace(cp)) break;
        lo += adv;
    }
    size_t hi = n;
    while (hi > lo) {
        size_t k = hi - 1;
        while (k > lo && (u[k] & 0xC0) == 0x80) k--;   /* lead byte of last char */
        size_t adv;
        uint32_t cp = dec1(u + k, hi - k, &adv);
        if (k + adv != hi || !py_isspace(cp)) break;
        hi = k;
    }
    *n_out = hi - lo;
    return lo;
}

/* Python str.lstrip('\n') / str.rstrip('\n') on a span */
static size_t lstrip_nl(const char *s, size_t n) {
    size_t i = 0;
    while (i < n && s[i] == '\n') i++;
    return i;
}
static size_t rstrip_nl(const char *s, size_t n) {
    while (n > 0 && s[n - 1] == '\n') n--;
    return n;
}

/* last occurrence of needle in s[0,n), -1 if none */
static long rfind_str(const char *s, size_t n, const char *needle, size_t nn) {
    if (nn > n) return -1;
    for (size_t i = n - nn + 1; i-- > 0;)
        if (memcmp(s + i, needle, nn) == 0) return (long)i;
    return -1;
}

static int starts_with(const char *s, size_t n, const char *prefix) {
    size_t pn = strlen(prefix);
    return n >= pn && memcmp(s, prefix, pn) == 0;
}
static int ends_with(const char *s, size_t n, const char *suffix) {
    size_t sn = strlen(suffix);
    return n >= sn && memcmp(s + n - sn, suffix, sn) == 0;
}

/* ---------------- render_content (template macro, text-only scope) ---------------- */

/* Appends the raw (untrimmed) content rendering to out. */
static int render_content(SBuf *out, const JVal *msg) {
    JVal *content = msg ? json_obj_get((JVal *)msg, "content") : NULL;
    if (!content || json_type(content) == J_NULL)
        return 0;                                  /* none/undefined -> '' */
    if (json_type(content) == J_STR) {
        sb_puts(out, json_str(content));
        return 0;
    }
    if (json_type(content) == J_ARR) {
        size_t m = json_arr_len(content);
        for (size_t i = 0; i < m; i++) {
            JVal *item = json_arr_get(content, i);
            if (json_type(item) != J_OBJ)
                return ling_fail("Unexpected item type in content.");
            const char *type = jstr(item, "type");
            if (json_obj_get(item, "image") || json_obj_get(item, "image_url") ||
                (type && strcmp(type, "image") == 0) ||
                json_obj_get(item, "video") ||
                (type && strcmp(type, "video") == 0))
                return ling_fail("vision content not supported (text-only)");
            JVal *text = json_obj_get(item, "text");
            if (!text || json_type(text) != J_STR)
                return ling_fail("Unexpected item type in content.");
            sb_puts(out, json_str(text));
        }
        return 0;
    }
    return ling_fail("Unexpected content type.");
}

/* render_content + |trim into a fresh malloc'd span (NUL-terminated;
 * *n_out = trimmed byte length). NULL on error. */
static char *render_trimmed(const JVal *msg, size_t *n_out) {
    SBuf b;
    sb_init(&b);
    if (render_content(&b, msg) < 0) {
        sb_free(&b);
        return NULL;
    }
    char *raw = sb_steal(&b);
    size_t n = 0;
    size_t off = pytrim(raw, strlen(raw), &n);
    memmove(raw, raw + off, n);
    raw[n] = '\0';
    *n_out = n;
    return raw;
}

/* ---------------- assistant turn ---------------- */

static int render_assistant(SBuf *out, const JVal *msg, int preserve,
                            int preserve_all) {
    size_t cn = 0;
    char *cbuf = render_trimmed(msg, &cn);
    if (!cbuf) return -1;
    const char *content = cbuf;

    const char *rc = "";
    size_t rcn = 0;
    JVal *rcf = json_obj_get((JVal *)msg, "reasoning_content");
    if (rcf && json_type(rcf) == J_STR) {
        rc = json_str(rcf);
        rcn = strlen(rc);
    } else {
        /* reasoning split: only when reasoning_content is not a string and
         * content contains </think>:
         *   rc      = content.split('</think>')[0].rstrip('\n')
         *                  .split('<think>')[-1].lstrip('\n')
         *   content = content.split('</think>')[-1].lstrip('\n') */
        const char *et = strstr(content, "</think>");
        if (et) {
            size_t p0n = rstrip_nl(content, (size_t)(et - content));
            long last = rfind_str(content, cn, "</think>", 8);
            const char *al = content + last + 8;
            size_t aln = cn - (size_t)last - 8;
            long lt = rfind_str(content, p0n, "<think>", 7);
            size_t roff = lt >= 0 ? (size_t)lt + 7 : 0;
            size_t n2 = p0n - roff;
            roff += lstrip_nl(content + roff, n2);
            rc = content + roff;
            rcn = p0n - roff;
            size_t skip = lstrip_nl(al, aln);
            content = al + skip;   /* span within cbuf */
            cn = aln - skip;
        }
    }
    /* reasoning_content|trim */
    {
        size_t n = 0;
        size_t off = pytrim(rc, rcn, &n);
        rc += off;
        rcn = n;
    }

    if (preserve_all || preserve) {
        sb_puts(out, "<|im_start|>assistant\n<think>\n");
        sb_write(out, rc, rcn);
        sb_puts(out, "\n</think>\n\n");
        sb_write(out, content, cn);
    } else {
        sb_puts(out, "<|im_start|>assistant\n");
        sb_write(out, content, cn);
    }

    JVal *tool_calls = json_obj_get((JVal *)msg, "tool_calls");
    if (tool_calls && json_truthy(tool_calls)) {
        if (json_type(tool_calls) != J_ARR) {
            free(cbuf);
            return ling_fail("tool_calls must be an array");
        }
        size_t m = json_arr_len(tool_calls);
        for (size_t i = 0; i < m; i++) {
            JVal *call = json_arr_get(tool_calls, i);
            JVal *fn = json_obj_get(call, "function");
            const JVal *tc = (fn && json_truthy(fn)) ? fn : call;
            if (i == 0) {
                size_t tn = 0;
                pytrim(content, cn, &tn);
                if (tn) sb_puts(out, "\n\n");
            } else {
                sb_putc(out, '\n');
            }
            const char *name = jstr(tc, "name");
            if (!name) {
                free(cbuf);
                return ling_fail("tool_call without function.name");
            }
            sb_puts(out, "<tool_call>\n<function=");
            sb_puts(out, name);
            sb_puts(out, ">\n");
            JVal *args = json_obj_get((JVal *)tc, "arguments");
            if (args) {
                /* template: arguments|items raises on non-mappings */
                if (json_type(args) != J_OBJ) {
                    free(cbuf);
                    return ling_fail("tool_call arguments must be an object");
                }
                size_t na = json_obj_len(args);
                for (size_t k = 0; k < na; k++) {
                    const char *key = json_obj_key(args, k);
                    JVal *v = json_obj_val(args, k);
                    sb_puts(out, "<parameter=");
                    sb_puts(out, key);
                    sb_puts(out, ">\n");
                    if (json_type(v) == J_STR) {
                        sb_puts(out, json_str(v));
                    } else {
                        char *j = json_dumps(v);
                        sb_puts(out, j);
                        free(j);
                    }
                    sb_puts(out, "\n</parameter>\n");
                }
            }
            sb_puts(out, "</function>\n</tool_call>");
        }
    }
    sb_puts(out, "<|im_end|>\n");
    free(cbuf);
    return 0;
}

/* ---------------- encode_messages ---------------- */

char *ling_encode_messages(const JVal *messages, const JVal *tools,
                           const LingEncOpts *opts) {
    ling_err[0] = '\0';
    if (!messages || json_type(messages) != J_ARR ||
        json_arr_len(messages) == 0) {
        ling_fail("No messages provided.");
        return NULL;
    }
    int thinking = opts ? opts->thinking : 1;
    int gen_prompt = opts ? opts->add_generation_prompt : 1;
    int preserve_all = opts ? opts->preserve_thinking : 0;
    /* template: tools and tools is iterable and tools is not mapping */
    int have_tools = tools && json_type(tools) == J_ARR &&
                     json_arr_len(tools) > 0;

    size_t n = json_arr_len(messages);
    const JVal *m0 = json_arr_get(messages, 0);
    int m0_system = strcmp(jrole(m0), "system") == 0;

    SBuf out;
    sb_init(&out);

    /* ---- system/tools preamble ---- */
    if (have_tools) {
        sb_puts(&out, "<|im_start|>system\n");
        sb_puts(&out, QWEN_TOOLS_HEADER);
        size_t nt = json_arr_len(tools);
        for (size_t i = 0; i < nt; i++) {
            char *j = json_dumps(json_arr_get(tools, i));
            sb_putc(&out, '\n');
            sb_puts(&out, j);
            free(j);
        }
        sb_puts(&out, QWEN_TOOLS_INSTRUCTIONS);
        if (m0_system) {
            size_t cn = 0;
            char *content = render_trimmed(m0, &cn);
            if (!content) goto fail;
            if (cn) {
                sb_puts(&out, "\n\n");
                sb_write(&out, content, cn);
            }
            free(content);
        }
        sb_puts(&out, "<|im_end|>\n");
    } else if (m0_system) {
        size_t cn = 0;
        char *content = render_trimmed(m0, &cn);
        if (!content) goto fail;
        sb_puts(&out, "<|im_start|>system\n");
        sb_write(&out, content, cn);
        sb_puts(&out, "<|im_end|>\n");
        free(content);
    }

    /* ---- last_query_index (template's ns machinery) ---- */
    size_t last_query = n - 1;
    {
        int multi_step_tool = 1;
        for (size_t ri = n; ri-- > 0 && multi_step_tool;) {
            const JVal *msg = json_arr_get(messages, ri);
            if (strcmp(jrole(msg), "user") != 0) continue;
            size_t cn = 0;
            char *content = render_trimmed(msg, &cn);
            if (!content) goto fail;
            int wrapped = starts_with(content, cn, "<tool_response>") &&
                          ends_with(content, cn, "</tool_response>");
            free(content);
            if (!wrapped) {
                multi_step_tool = 0;
                last_query = ri;
            }
        }
        if (multi_step_tool) {
            ling_fail("No user query found in messages.");
            goto fail;
        }
    }

    /* ---- message loop ---- */
    for (size_t i = 0; i < n; i++) {
        const JVal *msg = json_arr_get(messages, i);
        const char *role = jrole(msg);
        if (strcmp(role, "system") == 0) {
            if (i != 0) {
                ling_fail("System message must be at the beginning.");
                goto fail;
            }
            /* i == 0: already rendered in the preamble */
        } else if (strcmp(role, "user") == 0) {
            size_t cn = 0;
            char *content = render_trimmed(msg, &cn);
            if (!content) goto fail;
            sb_puts(&out, "<|im_start|>user\n");
            sb_write(&out, content, cn);
            sb_puts(&out, "<|im_end|>\n");
            free(content);
        } else if (strcmp(role, "assistant") == 0) {
            int preserve = i > last_query;
            if (render_assistant(&out, msg, preserve, preserve_all) < 0)
                goto fail;
        } else if (strcmp(role, "tool") == 0) {
            size_t cn = 0;
            char *content = render_trimmed(msg, &cn);
            if (!content) goto fail;
            if (i > 0 && strcmp(jrole(json_arr_get(messages, i - 1)), "tool") != 0)
                sb_puts(&out, "<|im_start|>user");
            sb_puts(&out, "\n<tool_response>\n");
            sb_write(&out, content, cn);
            sb_puts(&out, "\n</tool_response>");
            if (i + 1 == n || strcmp(jrole(json_arr_get(messages, i + 1)), "tool") != 0)
                sb_puts(&out, "<|im_end|>\n");
            free(content);
        } else {
            ling_fail("Unexpected message role.");
            goto fail;
        }
    }

    /* ---- generation prompt ---- */
    if (gen_prompt) {
        sb_puts(&out, "<|im_start|>assistant\n");
        sb_puts(&out, thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
    }
    return sb_steal(&out);

fail:
    sb_free(&out);
    return NULL;
}

uint32_t *ling_encode_ids(const Tok *t, const JVal *messages, const JVal *tools,
                          const LingEncOpts *opts, size_t *n_out) {
    char *prompt = ling_encode_messages(messages, tools, opts);
    if (!prompt) return NULL;
    uint32_t *ids = tok_encode_str(t, prompt, 1, n_out);
    free(prompt);
    return ids;
}

#endif /* APUS_ENCODING_IMPL_DONE */
