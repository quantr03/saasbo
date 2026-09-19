#!/usr/bin/env python
"""Build a self-contained NARRATIVE.html from NARRATIVE.md, captions.md, claims_ledger.csv, tables/ and latest/figures/."""
import base64, csv, html, re, sys
from pathlib import Path

D = Path(sys.argv[1])
LATEST = Path("/scratch/work/tranq8/sagp_analysis/latest")
md_lines = (D / "NARRATIVE.md").read_text().splitlines()
ledger = list(csv.DictReader(open(D / "claims_ledger.csv")))
L = {r["id"]: r for r in ledger}
heads = []


def esc(s, quote=False):
    return html.escape(s, quote=quote)


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def badge(cell):
    for word, cls in (("NOT SUPPORTED", "no"), ("SUPPORTED", "yes"), ("SUGGESTIVE", "maybe")):
        if cell.startswith(word):
            return f'<span class="badge {cls}">{word}</span>' + inline(cell[len(word):])
    return inline(cell)


def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)

    def tag(m):
        ids = re.findall(r"L\d+", m.group(0))
        parts = []
        for i in ids:
            title = esc(f'{L[i]["claim"]} — {L[i]["source_file"]} ({L[i]["locator"]})', quote=True) if i in L else ""
            parts.append(f'<a class="tag" href="#{i}" title="{title}">{i}</a>')
        return "[" + ", ".join(parts) + "]"

    return re.sub(r"\[L\d+(?:,\s*L\d+)*\]", tag, s)


def table_html(hdr, body, cls="md"):
    verdict_col = next((k for k, h in enumerate(hdr) if h.strip().lower() == "verdict"), None)
    out = [f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{inline(h)}</th>" for h in hdr) + "</tr></thead><tbody>"]
    for r in body:
        cells = []
        for k, c in enumerate(r):
            cells.append("<td>" + (badge(c) if k == verdict_col else inline(c)) + "</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render(lines):
    out, para, i, n = [], [], 0, len(lines)

    def flush():
        nonlocal para
        if para:
            p = inline(" ".join(para))
            cls = ' class="callout"' if p.startswith("<strong>The answer in three sentences") else ""
            out.append(f"<p{cls}>{p}</p>")
            para = []

    while i < n:
        line = lines[i]
        if not line.strip():
            flush(); i += 1; continue
        if line.startswith("#"):
            flush()
            m = re.match(r"(#+)\s+(.*)", line)
            lvl, txt = len(m.group(1)), m.group(2)
            hid = slug(txt)
            heads.append((lvl, txt, hid))
            out.append(f'<h{lvl} id="{hid}">{inline(txt)}</h{lvl}>')
            i += 1; continue
        if line.strip() == "---":
            flush(); out.append("<hr>"); i += 1; continue
        if line.startswith("|"):
            flush()
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            hdr = rows[0]
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            out.append(table_html(hdr, body)); continue
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
                else:
                    break
            t = "ol" if ordered else "ul"
            out.append(f"<{t}>" + "".join(f"<li>{inline(it)}</li>" for it in items) + f"</{t}>")
            continue
        para.append(line.strip()); i += 1
    flush()
    return "\n".join(out)


body_html = render(md_lines)

# ---- figures with captions
cap_text = (D / "captions.md").read_text()
figs = []
for m in re.finditer(r"\*\*Figure (\d+)\. `([^`]+)`\. (.+?)\*\*\n(.+?)(?=\n\n|\Z)", cap_text, re.S):
    num, fname, title, cap = m.group(1), m.group(2), m.group(3), m.group(4).strip()
    data = base64.b64encode((LATEST / "figures" / fname).read_bytes()).decode()
    figs.append(f'<figure id="fig{num}"><img src="data:image/png;base64,{data}" alt="{esc(title, True)}">'
                f'<figcaption><strong>Figure {num}. <code>{esc(fname)}</code>. {inline(title)}</strong> {inline(cap)}</figcaption></figure>')
figs_html = "\n".join(figs)

# ---- ledger
led = ['<table class="ledger"><thead><tr><th>id</th><th>section</th><th>claim</th><th>values</th><th>source file</th><th>locator</th><th>kind</th></tr></thead><tbody>']
for r in ledger:
    led.append(f'<tr id="{r["id"]}"><td class="idcell">{r["id"]}</td><td>{esc(r["section"])}</td><td>{esc(r["claim"])}</td>'
               f'<td>{esc(r["values"])}</td><td><code>{esc(r["source_file"])}</code></td><td>{esc(r["locator"])}</td><td>{esc(r["kind"])}</td></tr>')
led.append("</tbody></table>")
ledger_html = "\n".join(led)


# ---- supporting tables
def fmt(v):
    try:
        f = float(v)
    except ValueError:
        return esc(v)
    if v.strip().lstrip("-").isdigit():
        return v
    return f"{f:.4g}"


def csv_table(path, max_rows=None):
    rows = list(csv.reader(open(path)))
    hdr, body = rows[0], rows[1:]
    note = ""
    if max_rows and len(body) > max_rows:
        note = f"<p class='note'>first {max_rows} of {len(body)} rows shown; see the CSV.</p>"
        body = body[:max_rows]
    t = ['<table class="data"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in hdr) + "</tr></thead><tbody>"]
    for r in body:
        t.append("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in r) + "</tr>")
    t.append("</tbody></table>")
    return note + "\n".join(t)


own = [
    ("tables/ten_seed_bounds.csv", "k-of-10 paired seeds: sign-test p and exact 95 % CI on the win probability"),
    ("tables/decoupled_breakdown.csv", "decoupled: which coordinates each method finds, by (share, lengthscale) class"),
    ("tables/lockon_times.csv", "lock-on times: first t at which the median AP / F1 crosses 0.9 / 0.8"),
    ("tables/native_on_true_S_summary.csv", "native score and p_active on the true coordinates at t = 199"),
    ("tables/native_off_S_summary.csv", "off-S native maxima and false positives per cell and family"),
    ("tables/y_std_final_summary.csv", "target standardization std at t = 199 (mean over 10 runs)"),
    ("tables/degenerate_scan_flagged.csv", "runs with any readout of >= 50 active coordinates or AP < 0.3 at t >= 100"),
    ("tables/collapse_draws_aligned3_AL.csv", "retained NUTS draws in the sparse and the all-active mode, aligned3 / additive-lengthscale"),
    ("tables/identical_regret_pairs.csv", "runs ending with identical final regret: same queried point?"),
    ("tables/incumbent_boundary_summary.csv", "fraction of the incumbent's active coordinates at 0 or 1"),
    ("tables/near_duplicate_queries_summary.csv", "fraction of queries at t >= 100 within 0.01 (max-norm) of an earlier query"),
    ("tables/regret_vs_identification_spearman.csv", "Spearman correlation of identification metrics with final regret, per family and method"),
    ("tables/dsp_map_fallbacks.csv", "dsp_map exception rows and Adam fallbacks per run"),
    ("tables/final_regret_wide.csv", "final regret per (family, seed) across the seven methods"),
]
ref = [
    ("tables/final_regret_t199.csv", "final regret at t = 199, median [Q1, Q3] and ranks (numeric analysis)"),
    ("tables/paired_wilcoxon_final_regret.csv", "paired Wilcoxon tests over seeds (numeric analysis)"),
    ("tables/identification_by_cell.csv", "identification per cell, pooled over families (numeric analysis)"),
    ("tables/identification_by_cell_family.csv", "identification per cell and family (numeric analysis)"),
    ("tables/identification_by_gate_status.csv", "identification at t >= 100 split by gate status (numeric analysis)"),
    ("tables/diagnostics_by_method.csv", "NUTS diagnostics per method (numeric analysis)"),
    ("tables/gate_reason_breakdown.csv", "gate reason breakdown (numeric analysis)"),
    ("tables/cost_total_hours_by_method.csv", "cost totals per method (numeric analysis)"),
    ("tables/cost_by_method_device.csv", "cost per method and GPU (numeric analysis)"),
]
sup = []
for p, title in own:
    sup.append(f"<details><summary><code>{esc(p)}</code> — {esc(title)}</summary>{csv_table(D / p, 80)}</details>")
sup_ref = []
for p, title in ref:
    sup_ref.append(f"<details><summary><code>latest/{esc(p)}</code> — {esc(title)}</summary>{csv_table(LATEST / p, 80)}</details>")
eps = (D / "tables" / "eps_constants.json").read_text()

# ---- TOC
toc = ["<nav id='toc'><div class='toc-title'>Contents</div><ul>"]
for lvl, txt, hid in heads:
    if lvl in (2, 3):
        toc.append(f'<li class="l{lvl}"><a href="#{hid}">{inline(txt)}</a></li>')
for hid, txt in (("figures", "Figures and draft captions"), ("ledger", "Claims ledger"), ("supporting", "Supporting tables")):
    toc.append(f'<li class="l2"><a href="#{hid}">{txt}</a></li>')
toc.append("</ul></nav>")
toc_html = "\n".join(toc)

CSS = """
:root{--ink:#1d2330;--muted:#5b6472;--line:#d9dee6;--accent:#0b5cad;--bg:#ffffff;--soft:#f4f6f9;--yes:#1b7f3b;--maybe:#a35d00;--no:#b3261e}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);line-height:1.5;font-size:15.5px}
.wrap{display:grid;grid-template-columns:270px minmax(0,1fr);gap:28px;max-width:1380px;margin:0 auto;padding:24px}
@media(max-width:960px){.wrap{grid-template-columns:1fr}#toc{position:static}}
#toc{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow:auto;font-size:13.5px;border-right:1px solid var(--line);padding-right:12px}
#toc .toc-title{font-weight:600;margin-bottom:6px}
#toc ul{list-style:none;padding:0;margin:0}
#toc li{margin:3px 0}#toc li.l3{padding-left:14px;color:var(--muted)}
#toc a{color:var(--ink);text-decoration:none}#toc a:hover{color:var(--accent)}
main{min-width:0}
h1{font-size:1.7rem;margin:.2em 0 .6em}h2{font-size:1.32rem;margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.25em}h3{font-size:1.1rem;margin-top:1.6em}
p{margin:.7em 0}li{margin:.35em 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em;background:var(--soft);padding:1px 4px;border-radius:3px}
a.tag{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.82em;color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent)}
a.tag:hover{background:#e6f0fb}
.callout{border-left:4px solid var(--accent);background:var(--soft);padding:12px 16px;border-radius:0 6px 6px 0}
table{border-collapse:collapse;margin:1em 0;font-size:.9em;width:100%}
th,td{border:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
th{background:var(--soft);font-weight:600}
table.data{font-size:.8em;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;width:auto;max-width:100%;display:block;overflow-x:auto}
table.ledger{font-size:.82em}table.ledger td.idcell{font-family:ui-monospace,Menlo,Consolas,monospace;white-space:nowrap}
table.ledger tr:target{background:#fff6d6}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;color:#fff;font-size:.78em;font-weight:600;letter-spacing:.02em;margin-right:4px;white-space:nowrap}
.badge.yes{background:var(--yes)}.badge.maybe{background:var(--maybe)}.badge.no{background:var(--no)}
figure{margin:1.6em 0;padding:12px;border:1px solid var(--line);border-radius:6px;background:#fff}
figure img{max-width:100%;height:auto;display:block;margin:0 auto}
figcaption{margin-top:10px;font-size:.9em;color:var(--ink)}
details{margin:.6em 0;border:1px solid var(--line);border-radius:6px;padding:6px 10px;background:#fff}
summary{cursor:pointer;font-size:.92em}
.meta{color:var(--muted);font-size:.9em}
.note{color:var(--muted);font-size:.85em}
hr{border:0;border-top:1px solid var(--line);margin:1.6em 0}
pre{background:var(--soft);padding:10px;border-radius:6px;overflow-x:auto;font-size:.85em}
@media print{#toc{display:none}.wrap{display:block}details{page-break-inside:avoid}figure{page-break-inside:avoid}a.tag{border:0}}
"""

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sagp high-dimensional BO study: narrative analysis (2026-09-12)</title>
<style>{CSS}</style></head>
<body><div class="wrap">
{toc_html}
<main>
<p class="meta">Self-contained HTML build of <code>{esc(str(D))}/NARRATIVE.md</code>, <code>captions.md</code> and <code>claims_ledger.csv</code>; figures embedded from <code>{esc(str(LATEST))}/figures/</code>. Hover a tag like <span class="tag" style="font-family:monospace;color:var(--accent)">[L12]</span> to see its source; click it to jump to the ledger row.</p>
{body_html}
<hr>
<h2 id="figures">Figures and draft captions</h2>
{figs_html}
<hr>
<h2 id="ledger">Claims ledger</h2>
<p class="note">Every [L#] tag in the narrative, with the file, row and column that decides it. Files under <code>latest/tables/</code> belong to the numeric analysis snapshot; files under <code>tables/</code> were produced by <code>compute.py</code> / <code>compute2.py</code> for this narrative. "derived" marks arithmetic on cited numbers.</p>
{ledger_html}
<hr>
<h2 id="supporting">Supporting tables</h2>
<h3>This narrative's own tables</h3>
<details><summary><code>tables/eps_constants.json</code> — ELL_EPS, RHO_EPS, ALPHA_AMPLITUDE reproduced from gp.py's definitions</summary><pre>{esc(eps)}</pre></details>
{''.join(sup)}
<h3>Numeric-analysis tables cited by the ledger</h3>
{''.join(sup_ref)}
</main></div></body></html>
"""
(D / "NARRATIVE.html").write_text(page)
print("wrote", D / "NARRATIVE.html", len(page) // 1024, "KB;", len(figs), "figures;", len(ledger), "ledger rows;", len(heads), "headings")
# sanity: leftover markdown markers in the rendered body
left = re.findall(r"\*\*|(?<!\w)\[L\d", re.sub(r"<[^>]+>", "", body_html))
print("leftover markdown markers in body:", len(left))
