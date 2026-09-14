#!/usr/bin/env python
"""Assemble the Artifact page for the sagp narrative (content only: <title> + <style> + body content).

Reads NARRATIVE.md, captions.md, claims_ledger.csv and the small tables/ CSVs read-only; writes one HTML file.
Parsing of the markdown, of captions.md and the [L#] -> ledger link convention follow build_html.py.
"""
import csv, html, json, re, struct, sys
from pathlib import Path

D = Path("/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849")
OUT = Path(sys.argv[1])
FIGDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT.parent / "figures"
SISTER = sys.argv[3].rstrip("/") if len(sys.argv) > 3 else ""
sister_html = f'<p class="note sister">Companion page: <a href="{SISTER}">the same material sorted by research question</a>.</p>' if SISTER else ""

md_lines = (D / "NARRATIVE.md").read_text().splitlines()
ledger = list(csv.DictReader(open(D / "claims_ledger.csv")))
L = {r["id"]: r for r in ledger}
used_tags = []


def esc(s, quote=False):
    return html.escape(s, quote=quote)


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def sec_id(txt):
    m = re.match(r"(\d+[a-z]?)\.\s", txt)
    return "s" + m.group(1) if m else slug(txt)


KIND_CLASS = {"observation": "k-obs", "inference": "k-inf", "hypothesis": "k-hyp", "observation+inference": "k-mix"}


def kind_badge(kind):
    return f'<span class="badge {KIND_CLASS.get(kind, "k-obs")}">{esc(kind)}</span>'


def verdict_badge(cell):
    for word, cls in (("NOT SUPPORTED", "v-no"), ("SUPPORTED", "v-yes"), ("SUGGESTIVE", "v-maybe")):
        if cell.startswith(word):
            return f'<span class="badge {cls}">{word}</span>' + inline(cell[len(word):])
    return inline(cell)


def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    # *[observation ...]* / *[inference ...]* / *[hypothesis ...]* labels: text kept verbatim, class from the first word
    def lab(m):
        word = m.group(1)
        cls = {"observation": "lab-obs", "inference": "lab-inf", "hypothesis": "lab-hyp"}[word]
        return f'<em class="lab {cls}">[{word}{m.group(2)}]</em>'
    s = re.sub(r"\*\[(observation|inference|hypothesis)([^\]]*)\]\*", lab, s)
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)

    def tag(m):
        ids = re.findall(r"L\d+", m.group(0))
        parts = []
        for i in ids:
            used_tags.append(i)
            if i in L:
                title = esc(f'{L[i]["claim"]} — {L[i]["source_file"]} ({L[i]["locator"]})', quote=True)
                parts.append(f'<a class="tag" href="#{i}" title="{title}">{i}</a>')
            else:
                parts.append(f'<span class="tag tag-missing" title="no ledger row">{i}</span>')
        return '<span class="tags">[' + ", ".join(parts) + "]</span>"

    return re.sub(r"\[L\d+(?:,\s*L\d+)*\]", tag, s)


def table_html(hdr, body):
    verdict_col = next((k for k, h in enumerate(hdr) if h.strip().lower() == "verdict"), None)
    out = ['<div class="scroll"><table class="md"><thead><tr>' + "".join(f"<th>{inline(h)}</th>" for h in hdr) + "</tr></thead><tbody>"]
    for r in body:
        cells = []
        for k, c in enumerate(r):
            cells.append("<td>" + (verdict_badge(c) if k == verdict_col else inline(c)) + "</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def render_blocks(lines):
    """-> list of (kind, raw_start, html). kind in h1/h2/h3/p/table/list."""
    out, para, i, n = [], [], 0, len(lines)

    def flush():
        nonlocal para
        if para:
            raw = " ".join(para)
            p = inline(raw)
            out.append(("p", raw, f"<p>{p}</p>"))
            para = []

    while i < n:
        line = lines[i]
        if not line.strip():
            flush(); i += 1; continue
        if line.startswith("#"):
            flush()
            m = re.match(r"(#+)\s+(.*)", line)
            lvl, txt = len(m.group(1)), m.group(2)
            out.append((f"h{lvl}", txt, None))
            i += 1; continue
        if line.strip() == "---":
            flush(); i += 1; continue
        if line.startswith("|"):
            flush()
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            hdr = rows[0]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            out.append(("table", "|".join(hdr), table_html(hdr, body))); continue
        m = re.match(r"([-*]|\d+\.)\s+(.*)", line)
        if m:
            flush()
            ordered = m.group(1)[0].isdigit()
            items = []
            while i < n:
                m2 = re.match(r"([-*]|\d+\.)\s+(.*)", lines[i])
                if m2 and m2.group(1)[0].isdigit() == ordered:
                    items.append(m2.group(2)); i += 1
                elif lines[i].startswith("  ") and lines[i].strip():
                    items[-1] += " " + lines[i].strip(); i += 1
                elif not lines[i].strip():
                    # blank lines between items of one list (section 5c): continue if the next non-blank line is an item of the same type
                    j = i
                    while j < n and not lines[j].strip():
                        j += 1
                    m3 = re.match(r"([-*]|\d+\.)\s+(.*)", lines[j]) if j < n else None
                    if m3 and m3.group(1)[0].isdigit() == ordered:
                        i = j
                    else:
                        break
                else:
                    break
            t = "ol" if ordered else "ul"
            out.append(("list", items[0], f"<{t}>" + "".join(f"<li>{inline(it)}</li>" for it in items) + f"</{t}>"))
            continue
        para.append(line.strip()); i += 1
    flush()
    return out


blocks = render_blocks(md_lines)

# ---- split: header paragraphs (before the first h2), then sections keyed by h2
h1_text = next(raw for k, raw, _ in blocks if k == "h1")
first_h2 = next(idx for idx, b in enumerate(blocks) if b[0] == "h2")
header_blocks = [b for b in blocks[:first_h2] if b[0] == "p"]
assert len(header_blocks) == 3, [b[1][:30] for b in header_blocks]
snapshot_html, answer_html, labels_html = (b[2] for b in header_blocks)

sections = []  # (id, title, [blocks])
for b in blocks[first_h2:]:
    if b[0] == "h2":
        sections.append((sec_id(b[1]), b[1], []))
    else:
        sections[-1][2].append(b)
sec = {sid: (title, bl) for sid, title, bl in sections}
assert set(sec) == {f"s{i}" for i in range(1, 10)}, list(sec)

# ---- figures from captions.md (parsing as in build_html.py); placed after keyed paragraphs
cap_text = (D / "captions.md").read_text()


def png_size(p):
    with open(p, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


figs = {}
for m in re.finditer(r"\*\*Figure (\d+)\. `([^`]+)`\. (.+?)\*\*\n(.+?)(?=\n\n|\Z)", cap_text, re.S):
    num, fname, title, cap = m.group(1), m.group(2), m.group(3), m.group(4).strip()
    w, h = png_size(FIGDIR / fname)
    figs[int(num)] = (
        f'<figure class="fig" id="fig{num}" aria-labelledby="fig{num}-cap">'
        f'<a class="zoom plate" href="figures/{esc(fname, True)}" data-num="{num}">'
        f'<img src="figures/{esc(fname, True)}" alt="{esc(title, True)}" width="{w}" height="{h}" loading="lazy"></a>'
        f'<figcaption id="fig{num}-cap"><span class="fig-label">Figure {num}.</span> <code>{esc(fname)}</code>. '
        f'<span class="fig-title">{inline(title)}</span> {inline(cap)} '
        f'<a class="fig-open" href="figures/{esc(fname, True)}">Open the PNG</a></figcaption></figure>'
    )
assert sorted(figs) == [1, 2, 3, 4, 5, 6], sorted(figs)

PLACEMENT = {  # figure -> (section, paragraph the figure follows)
    1: ("s2", "Everything here is the paired final regret"),
    2: ("s2", "**interaction_g0.25.**"),
    3: ("s3", "**Ranking works in every cell**"),
    4: ("s3", "**When each cell locks on**"),
    5: ("s4", "**The amplitude cells did not converge"),
    6: ("s7", "Cost basis"),
}


def section_html(sid, with_figs=True):
    title, bl = sec[sid]
    parts = [f'<section id="{sid}"><h2>{inline(title)}</h2>']
    pending = {key: num for num, (s, key) in PLACEMENT.items() if s == sid} if with_figs else {}
    for kind, raw, h in bl:
        if kind == "h3":
            parts.append(f'<h3 id="{sec_id(raw)}">{inline(raw)}</h3>')
            continue
        parts.append(h)
        for key, num in list(pending.items()):
            if raw.startswith(key):
                parts.append(figs[num]); del pending[key]
    assert not pending, (sid, pending)
    parts.append("</section>")
    return "\n".join(parts)


# ---- ledger (section 8)
def code_list(s):
    return " ".join(f"<code>{esc(x.strip())}</code>" for x in s.split(";") if x.strip())


led = ['<div class="scroll"><table class="ledger" id="ledger-table"><thead><tr>']
for col, label in (("id", "id"), ("section", "section"), ("claim", "claim"), ("values", "values"),
                   ("source_file", "source file"), ("locator", "locator"), ("kind", "kind")):
    led.append(f'<th scope="col"><button type="button" class="sort" data-col="{col}">{label}</button></th>')
led.append("</tr></thead><tbody>")
for r in ledger:
    led.append(
        f'<tr id="{r["id"]}"><td class="mono id">{r["id"]}</td><td class="mono">{esc(r["section"])}</td>'
        f'<td class="claim">{esc(r["claim"])}</td><td class="values">{esc(r["values"])}</td>'
        f'<td class="src">{code_list(r["source_file"])}</td><td class="mono loc">{esc(r["locator"])}</td>'
        f'<td>{kind_badge(r["kind"])}</td></tr>')
led.append("</tbody></table></div>")
ledger_html = "\n".join(led)
kind_counts = {}
for r in ledger:
    kind_counts[r["kind"]] = kind_counts.get(r["kind"], 0) + 1

# ---- appendix: the small tables only
def fmt(v):
    try:
        f = float(v)
    except ValueError:
        return esc(v)
    if v.strip().lstrip("-").isdigit():
        return v
    return f"{f:.4g}"


def csv_table(path):
    rows = list(csv.reader(open(path)))
    hdr, body = rows[0], rows[1:]
    t = ['<div class="scroll"><table class="data"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in hdr) + "</tr></thead><tbody>"]
    for r in body:
        t.append("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in r) + "</tr>")
    t.append("</tbody></table></div>")
    return "\n".join(t), len(body)


INCLUDED = [  # (file, description from README.md)
    ("lockon_times.csv", "first t at which the median AP / F1 crosses 0.9 / 0.8, per family and cell"),
    ("decoupled_breakdown.csv", "which of the 8 decoupled coordinates each method finds, by (share, lengthscale) class: detected (p_active>0.5), top-8 by native / sobol"),
    ("native_on_true_S_summary.csv", "the native score and p_active on the true coordinates at t=199"),
    ("native_off_S_summary.csv", "off-S native maxima and false positives per cell and family"),
    ("ten_seed_bounds.csv", "sign-test p and exact 95% CI on the win probability for k of 10 paired seeds"),
    ("near_duplicate_queries_summary.csv", "fraction of late queries within 0.01 (max-norm) of an earlier query"),
    ("identical_regret_pairs.csv", "the product-lengthscale / oracle_S identical-regret seeds"),
    ("incumbent_boundary_summary.csv", "how often the incumbent sits on the box boundary in the active coordinates"),
    ("y_std_final_summary.csv", "the target standardization std at t=199 per cell run"),
    ("dsp_map_fallbacks.csv", "dsp_map's NotPSDError exception rows and Adam-fallback rows per run"),
]
app = []
for fname, desc in INCLUDED:
    t, nrows = csv_table(D / "tables" / fname)
    app.append(f'<details class="tbl" id="tbl-{slug(fname)}"><summary><code>tables/{esc(fname)}</code> <span class="desc">{esc(desc)}</span> <span class="rows">{nrows} rows</span></summary>{t}</details>')
eps_raw = (D / "tables" / "eps_constants.json").read_text()
json.loads(eps_raw)  # must be valid
app.append(f'<details class="tbl" id="tbl-eps-constants-json"><summary><code>tables/eps_constants.json</code> <span class="desc">ELL_EPS, RHO_EPS, ALPHA_AMPLITUDE reproduced with numpy from gp.py\'s definitions</span></summary><pre>{esc(eps_raw.strip())}</pre></details>')
appendix_html = "\n".join(app)
all_tables = sorted(p.name for p in (D / "tables").iterdir())
not_included = [f for f in all_tables if f not in {x for x, _ in INCLUDED} | {"eps_constants.json"}]
not_included_html = ", ".join(f"<code>{esc(f)}</code>" for f in not_included)

# ---- TOC
toc = []
for sid, title, bl in sections:
    toc.append(f'<li><a href="#{sid}">{inline(title)}</a>')
    subs = [(sec_id(raw), raw) for k, raw, _ in bl if k == "h3"]
    if subs:
        toc.append("<ul>" + "".join(f'<li><a href="#{hid}">{inline(t)}</a></li>' for hid, t in subs) + "</ul>")
    toc.append("</li>")
toc.append('<li><a href="#appendix">Appendix. Supporting tables</a></li>')
toc_html = "<ol class=\"toc-list\">" + "".join(toc) + "</ol>"
fig_links = " ".join(f'<a href="#fig{n}">{n}</a>' for n in range(1, 7))

CSS = r"""
:root{
  color-scheme:light;--bg:#f6f7f8;--surface:#ffffff;--ink:#1b232e;--muted:#5d6874;--line:#d7dce3;--line-strong:#b5bec9;
  --accent:#0f6b64;--accent-ink:#0b5a54;--tag:#0b5a54;--tag-bg:#e3f0ee;--tag-hover:#cfe6e2;
  --callout-bg:#edf3f2;--code-bg:#eceff2;--target:#fff1bf;--plate:#ffffff;--focus:#0f6b64;
  --obs-fg:#2c4a6e;--obs-bg:#e3ebf6;--inf-fg:#573b8c;--inf-bg:#ebe4f7;--hyp-fg:#874a09;--hyp-bg:#fbe9d1;
  --mix-fg:#3f4f7a;--mix-bg:#e6e7f6;
  --yes-fg:#1b6a3c;--yes-bg:#dcefe3;--maybe-fg:#865508;--maybe-bg:#fbeccc;--no-fg:#9c2d24;--no-bg:#f8dfdc;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;--bg:#12171e;--surface:#191f28;--ink:#e3e7ec;--muted:#98a3b1;--line:#2a333e;--line-strong:#3d4856;
    --accent:#6fcabf;--accent-ink:#8fd8cf;--tag:#8fd8cf;--tag-bg:#1a2f2c;--tag-hover:#24413d;
    --callout-bg:#182622;--code-bg:#222b35;--target:#40391a;--plate:#ffffff;--focus:#6fcabf;
    --obs-fg:#b7cbe8;--obs-bg:#213247;--inf-fg:#cfc0f0;--inf-bg:#33294d;--hyp-fg:#f1c78e;--hyp-bg:#443011;
    --mix-fg:#c3c8ee;--mix-bg:#2a2f4c;
    --yes-fg:#a3dcb6;--yes-bg:#183828;--maybe-fg:#f0ca8c;--maybe-bg:#423112;--no-fg:#f3a79f;--no-bg:#48231f;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;--bg:#12171e;--surface:#191f28;--ink:#e3e7ec;--muted:#98a3b1;--line:#2a333e;--line-strong:#3d4856;
  --accent:#6fcabf;--accent-ink:#8fd8cf;--tag:#8fd8cf;--tag-bg:#1a2f2c;--tag-hover:#24413d;
  --callout-bg:#182622;--code-bg:#222b35;--target:#40391a;--plate:#ffffff;--focus:#6fcabf;
  --obs-fg:#b7cbe8;--obs-bg:#213247;--inf-fg:#cfc0f0;--inf-bg:#33294d;--hyp-fg:#f1c78e;--hyp-bg:#443011;
  --mix-fg:#c3c8ee;--mix-bg:#2a2f4c;
  --yes-fg:#a3dcb6;--yes-bg:#183828;--maybe-fg:#f0ca8c;--maybe-bg:#423112;--no-fg:#f3a79f;--no-bg:#48231f;
}
@media (prefers-reduced-motion: no-preference){html{scroll-behavior:smooth}}
html{scroll-padding-top:12px}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;font-size:16.5px;line-height:1.6;-webkit-text-size-adjust:100%}
.page{max-width:1200px;margin:0 auto;padding-inline:16px;padding-block:0 72px}
.sans{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif}
code,pre,.mono,a.tag,.tags{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
h1,h2,h3{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;font-weight:600;line-height:1.2;text-wrap:balance;color:var(--ink)}
h1{font-size:clamp(1.6rem,1.2rem + 1.6vw,2.15rem);margin:0 0 .5rem;letter-spacing:-.01em}
h2{font-size:1.4rem;margin:2.6rem 0 .9rem;padding-top:1.2rem;border-top:1px solid var(--line)}
h3{font-size:1.12rem;margin:1.9rem 0 .6rem}
p{margin:.85rem 0}
ul,ol{padding-left:1.4rem}
li{margin:.45rem 0}
a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:2px}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
code{background:var(--code-bg);padding:.05em .35em;border-radius:3px;font-size:.86em}
pre{background:var(--code-bg);padding:12px;border-radius:4px;overflow-x:auto;font-size:.82rem;line-height:1.45}
strong{font-weight:600}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .6rem}
header.head{padding-block:32px 8px;border-bottom:1px solid var(--line);margin-bottom:8px}
header.head .snapshot{font-size:.95rem;color:var(--muted)}
header.head .snapshot strong{color:var(--ink)}
.answer{border-left:3px solid var(--accent);background:var(--callout-bg);padding:14px 18px;border-radius:0 6px 6px 0;margin:1.2rem 0}
.answer p{margin:0}
.legend{font-size:.95rem}
.prose p,.prose li,.answer,.legend,.snapshot,figcaption,.note{max-width:74ch}
/* layout */
@media (min-width:1040px){
  .page{display:grid;grid-template-columns:236px minmax(0,1fr);column-gap:44px;padding-inline:32px}
  header.head{grid-column:1 / -1}
  .rail{grid-column:1;position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;padding-right:8px;border-right:1px solid var(--line)}
  .rail summary{display:none}
  main{grid-column:2;min-width:0;max-width:900px}
}
@media (max-width:1039px){
  html{scroll-padding-top:52px}
  .rail{position:sticky;top:0;z-index:5;background:var(--bg);margin-inline:-16px;padding-inline:16px;border-bottom:1px solid var(--line)}
  .rail details[open] .toc-body{max-height:min(60vh,520px);overflow:auto;padding-bottom:10px}
}
main{min-width:0}
.rail summary{list-style:none;cursor:pointer;font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;font-size:.9rem;padding-block:12px;display:flex;align-items:center;gap:8px}
.rail summary::-webkit-details-marker{display:none}
.rail summary::after{content:"";width:.5em;height:.5em;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);margin-left:auto;transition:transform .15s}
.rail details[open] summary::after{transform:rotate(-135deg)}
.toc-title{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:14px 0 8px}
@media (max-width:1039px){.toc-title{display:none}}
.toc-list,.toc-list ul{list-style:none;margin:0;padding:0}
.toc-list>li{margin:0}
.toc-list a{display:block;font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.86rem;line-height:1.35;color:var(--ink);text-decoration:none;padding:5px 8px;border-radius:4px}
.toc-list ul a{padding-left:20px;color:var(--muted);font-size:.82rem}
.toc-list a:hover{background:var(--tag-bg);color:var(--accent-ink)}
.toc-figs{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.82rem;color:var(--muted);padding:8px 8px 4px;border-top:1px solid var(--line);margin-top:8px}
.toc-figs a{color:var(--accent-ink);text-decoration:none;margin-left:.35em;font-family:"IBM Plex Mono",Menlo,Consolas,monospace}
.toc-figs a:hover{text-decoration:underline}
/* tags and labels */
.tags{font-size:.78em;color:var(--muted);white-space:nowrap}
a.tag{color:var(--tag);background:var(--tag-bg);text-decoration:none;padding:.05em .3em;border-radius:3px;font-weight:500}
a.tag:hover{background:var(--tag-hover)}
.tag-missing{color:var(--no-fg);background:var(--no-bg);padding:.05em .3em;border-radius:3px}
em.lab{font-style:italic;font-size:.9em;padding:.02em .3em;border-radius:3px;white-space:normal}
.lab-obs{color:var(--obs-fg);background:var(--obs-bg)}
.lab-inf{color:var(--inf-fg);background:var(--inf-bg)}
.lab-hyp{color:var(--hyp-fg);background:var(--hyp-bg)}
.badge{display:inline-block;font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.7rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;padding:2px 7px;border-radius:3px;white-space:nowrap;vertical-align:middle;line-height:1.4}
.k-obs{color:var(--obs-fg);background:var(--obs-bg)}
.k-inf{color:var(--inf-fg);background:var(--inf-bg)}
.k-hyp{color:var(--hyp-fg);background:var(--hyp-bg)}
.k-mix{color:var(--mix-fg);background:var(--mix-bg)}
.v-yes{color:var(--yes-fg);background:var(--yes-bg)}
.v-maybe{color:var(--maybe-fg);background:var(--maybe-bg)}
.v-no{color:var(--no-fg);background:var(--no-bg)}
/* tables */
.scroll{overflow-x:auto;max-width:100%;margin:1rem 0;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem;line-height:1.45;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;font-size:.8rem;color:var(--muted);border-bottom:1px solid var(--line-strong);white-space:nowrap}
table.md td:first-child{white-space:nowrap}
table.data{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.76rem;width:auto;min-width:100%}
table.data td,table.data th{white-space:nowrap;padding:5px 10px}
table.data tbody tr:nth-child(even){background:color-mix(in srgb,var(--code-bg) 55%,transparent)}
table.ledger{min-width:1080px;font-size:.84rem}
table.ledger td.id{white-space:nowrap;font-weight:500}
table.ledger td.mono{font-size:.8rem}
table.ledger td.claim{min-width:18em}
table.ledger td.values{min-width:20em;overflow-wrap:anywhere}
table.ledger td.src{min-width:16em}
table.ledger td.src code{font-size:.78rem;display:inline-block;margin:1px 2px 1px 0;overflow-wrap:anywhere}
table.ledger td.loc{min-width:14em;font-size:.78rem;overflow-wrap:anywhere}
table.ledger tr:target{background:var(--target)}
table.ledger tr:target td.id{color:var(--accent-ink)}
button.sort{all:unset;cursor:pointer;font:inherit;color:inherit;padding-right:1.1em;position:relative}
button.sort::after{content:"\2195";position:absolute;right:0;opacity:.45}
th[aria-sort="ascending"] button.sort::after{content:"\2191";opacity:1}
th[aria-sort="descending"] button.sort::after{content:"\2193";opacity:1}
.ledger-legend{font-size:.9rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin:.6rem 0 0}
/* figures */
figure.fig{margin:1.8rem 0;padding:0}
.plate{display:block;background:var(--plate);border:1px solid var(--line);border-radius:4px;padding:6px;cursor:zoom-in}
.plate img{display:block;width:100%;height:auto;max-width:100%}
figcaption{font-size:.9rem;line-height:1.55;margin-top:.7rem;color:var(--ink)}
.fig-label{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600}
.fig-title{font-weight:600}
.fig-open{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.8rem;white-space:nowrap}
dialog.lb{border:0;padding:0;margin:0;width:100vw;height:100vh;max-width:100vw;max-height:100vh;background:transparent;color:#fff}
dialog.lb::backdrop{background:rgba(8,12,18,.88)}
.lb-inner{display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:100%;padding:12px;gap:10px}
.lb-inner img{max-width:100%;max-height:calc(100vh - 90px);width:auto;height:auto;object-fit:contain;background:#fff;border-radius:3px}
.lb-bar{display:flex;gap:16px;align-items:center;font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.9rem;color:#e6e9ee}
.lb-bar button{all:unset;cursor:pointer;border:1px solid rgba(255,255,255,.4);padding:4px 12px;border-radius:4px}
.lb-bar button:hover{background:rgba(255,255,255,.12)}
/* appendix */
details.tbl{border:1px solid var(--line);border-radius:4px;padding:6px 12px;margin:.6rem 0;background:var(--surface)}
details.tbl summary{cursor:pointer;font-size:.9rem;line-height:1.5;padding:4px 0}
details.tbl summary .desc{color:var(--muted)}
details.tbl summary .rows{font-family:"IBM Plex Mono",Menlo,Consolas,monospace;font-size:.75rem;color:var(--muted);white-space:nowrap}
details.tbl .scroll{margin:.4rem 0 .6rem}
.note{font-size:.9rem;color:var(--muted)}
footer.about{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.86rem;color:var(--muted)}
footer.about code{font-size:.8rem}
@media print{.rail{display:none}.page{display:block}figure.fig,details.tbl{break-inside:avoid}}
"""

JS = r"""
(function(){
  // contents: sidebar on wide screens, collapsible bar on narrow ones
  var toc=document.getElementById('toc'), mq=window.matchMedia('(max-width: 1039px)');
  function sync(){ if(mq.matches){toc.removeAttribute('open');} else {toc.setAttribute('open','');} }
  sync(); if(mq.addEventListener){mq.addEventListener('change',sync);} else {mq.addListener(sync);}
  toc.querySelectorAll('a').forEach(function(a){a.addEventListener('click',function(){ if(mq.matches){toc.removeAttribute('open');} });});
  // figures: click to enlarge
  var lb=document.getElementById('lightbox'), lbImg=lb.querySelector('img'), lbCap=lb.querySelector('.lb-cap'), lbOpen=lb.querySelector('.lb-open');
  if(typeof lb.showModal==='function'){
    document.querySelectorAll('a.zoom').forEach(function(a){
      a.addEventListener('click',function(e){
        e.preventDefault();
        var img=a.querySelector('img');
        lbImg.src=img.getAttribute('src'); lbImg.alt=img.alt; lbOpen.href=a.getAttribute('href');
        lbCap.textContent='Figure '+a.dataset.num+'. '+img.alt;
        lb.showModal();
      });
    });
    lb.addEventListener('click',function(e){ if(e.target===lb||e.target.classList.contains('lb-inner')){lb.close();} });
    lb.querySelector('.lb-close').addEventListener('click',function(){lb.close();});
  }
  // ledger: click a column header to sort
  var table=document.getElementById('ledger-table');
  table.querySelectorAll('button.sort').forEach(function(btn){
    btn.addEventListener('click',function(){
      var th=btn.parentNode, idx=Array.prototype.indexOf.call(th.parentNode.children,th), tbody=table.tBodies[0];
      var dir=th.getAttribute('aria-sort')==='ascending'?-1:1;
      var rows=Array.prototype.slice.call(tbody.rows);
      function key(r){return r.cells[idx].textContent.trim();}
      function num(s){var m=/^L(\d+)$/.exec(s); return m?+m[1]:null;}
      rows.sort(function(a,b){var ka=key(a),kb=key(b),na=num(ka),nb=num(kb);
        if(na!==null&&nb!==null){return (na-nb)*dir;}
        return ka.localeCompare(kb,undefined,{numeric:true,sensitivity:'base'})*dir;});
      rows.forEach(function(r){tbody.appendChild(r);});
      table.querySelectorAll('th').forEach(function(t){t.removeAttribute('aria-sort');});
      th.setAttribute('aria-sort',dir===1?'ascending':'descending');
    });
  });
})();
"""

kinds_legend = " ".join(f'{kind_badge(k)} <span>{v}</span>' for k, v in sorted(kind_counts.items(), key=lambda kv: -kv[1]))

page = f"""<title>sagp narrative</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{CSS}</style>
<div class="page">
<header class="head">
<p class="eyebrow">Narrative analysis · 2026-09-12-1849 · figures from the 2026-09-12-1428 snapshot</p>
{sister_html}
<h1>{inline(h1_text)}</h1>
<div class="snapshot">{snapshot_html}</div>
<div class="answer">{answer_html}</div>
<div class="legend">{labels_html}</div>
</header>
<nav class="rail" aria-label="Contents">
<details id="toc" open>
<summary>Contents</summary>
<div class="toc-body">
<div class="toc-title">Contents</div>
{toc_html}
<div class="toc-figs">Figures{fig_links}</div>
</div>
</details>
</nav>
<main class="prose">
{section_html("s1")}
{section_html("s2")}
{section_html("s3")}
{section_html("s4")}
{section_html("s5")}
{section_html("s6")}
{section_html("s7")}
<section id="s8"><h2>{inline(sec["s8"][0])}</h2>
{"".join(h for k, _, h in sec["s8"][1])}
<p class="ledger-legend"><span>{len(ledger)} rows, sorted by id; click a column header to sort. Kinds:</span> {kinds_legend}</p>
{ledger_html}
</section>
{section_html("s9")}
<section id="appendix"><h2>Appendix. Supporting tables</h2>
<p class="note">The small tables from this narrative's <code>tables/</code> directory, as computed by <code>compute.py</code> and <code>compute2.py</code> plus the two login-node reads. Floating-point values are shown to 4 significant digits; the CSV files hold full precision. Descriptions are from the narrative's <code>README.md</code>.</p>
{appendix_html}
<p class="note">Not included (per-run and per-coordinate files, under <code>/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/tables/</code>): {not_included_html}.</p>
</section>
<footer class="about">
<p>This page is the narrative's <code>NARRATIVE.md</code>, <code>captions.md</code> and <code>claims_ledger.csv</code> from <code>/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/</code>, with the six figures from <code>/scratch/work/tranq8/sagp_analysis/latest/figures/</code>. Prose, captions and ledger rows are verbatim; figures are placed in the section their caption belongs to. Tags such as <span class="tags">[<a class="tag" href="#L12">L12</a>]</span> link to the ledger row that decides the sentence; hover one to read the claim and its source.</p>
</footer>
</main>
</div>
<dialog class="lb" id="lightbox" aria-label="Enlarged figure">
<div class="lb-inner">
<img src="" alt="">
<div class="lb-bar"><span class="lb-cap"></span><a class="lb-open" href="#" style="color:inherit">Open the PNG</a><button type="button" class="lb-close">Close</button></div>
</div>
</dialog>
<script>{JS}</script>
"""

OUT.write_text(page)
used = sorted(set(used_tags), key=lambda t: int(t[1:]))
missing = [t for t in used if t not in L]
unused = [i for i in L if i not in set(used)]
print("wrote", OUT, len(page) // 1024, "KB")
print("figures:", len(figs), "ledger rows:", len(ledger), "sections:", [s for s, _, _ in sections])
print("distinct tags in prose:", len(used), "| missing from ledger:", missing, "| ledger rows unused:", unused)
print("kinds:", kind_counts)
print("appendix tables:", len(INCLUDED), "+ eps_constants.json; not included:", len(not_included))
body_only = re.sub(r"<style>.*?</style>", "", page, flags=re.S)
body_only = re.sub(r"<script>.*?</script>", "", body_only, flags=re.S)
stripped = re.sub(r"<[^>]+>", "", body_only)
print("leftover markdown markers:", re.findall(r"\*\*|(?<!\w)\[L\d+\]|`", stripped)[:10])
print("img srcs:", sorted(set(re.findall(r'<img src="([^"]+)"', page))))
