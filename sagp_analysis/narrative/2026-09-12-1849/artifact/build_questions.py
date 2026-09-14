#!/usr/bin/env python
"""Sister page to build_page.py: the same narrative re-sorted by research question.

Every paragraph, list, table row, caption and figure is taken verbatim from NARRATIVE.md, captions.md and
claims_ledger.csv (same parsing and [L#] convention as build_page.py). The questions and the grouping are this
page's own. Each narrative block is consumed exactly once; the script asserts that nothing substantive is left over.
Usage: build_questions.py OUT.html FIGDIR NARRATIVE_ARTIFACT_URL
"""
import csv, html, re, struct, sys
from pathlib import Path

D = Path("/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849")
OUT = Path(sys.argv[1])
FIGDIR = Path(sys.argv[2])
NARR = sys.argv[3].rstrip("/")

md_text = (D / "NARRATIVE.md").read_text()
md_lines = md_text.splitlines()
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
    """-> list of (kind, raw_start, html, extra). extra: list items, or (hdr, body) for a table."""
    out, para, i, n = [], [], 0, len(lines)

    def flush():
        nonlocal para
        if para:
            raw = " ".join(para)
            out.append(("p", raw, f"<p>{inline(raw)}</p>", None))
            para = []

    while i < n:
        line = lines[i]
        if not line.strip():
            flush(); i += 1; continue
        if line.startswith("#"):
            flush()
            m = re.match(r"(#+)\s+(.*)", line)
            out.append((f"h{len(m.group(1))}", m.group(2), None, None))
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
            out.append(("table", "|".join(hdr), table_html(hdr, body), (hdr, body))); continue
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
            out.append(("list", items[0], f"<{t}>" + "".join(f"<li>{inline(it)}</li>" for it in items) + f"</{t}>", items))
            continue
        para.append(line.strip()); i += 1
    flush()
    return out


blocks = render_blocks(md_lines)
first_h2 = next(idx for idx, b in enumerate(blocks) if b[0] == "h2")
header_blocks = [b for b in blocks[:first_h2] if b[0] == "p"]
assert len(header_blocks) == 3
snapshot_html, answer_html, labels_html = (b[2] for b in header_blocks)

sections = []
for b in blocks[first_h2:]:
    if b[0] == "h2":
        sections.append((sec_id(b[1]), b[1], []))
    else:
        sections[-1][2].append(b)
sec = {sid: (title, bl) for sid, title, bl in sections}
remaining = {sid: [b for b in bl if b[0] != "h3"] for sid, (t, bl) in sec.items()}
omitted = []


def take(sid, key):
    hits = [b for b in remaining[sid] if b[1].startswith(key)]
    assert len(hits) == 1, (sid, key, [h[1][:50] for h in hits])
    remaining[sid].remove(hits[0])
    return hits[0]


def para(sid, key):
    return take(sid, key)[2]


# ---- figures (captions.md; same parsing as build_html.py / build_page.py)
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
assert sorted(figs) == [1, 2, 3, 4, 5, 6]

# ---- section 6: the verdict rule and the claims table, split by question
verdict_rule_html = para("s6", "Rule (section 2)")
claims_hdr, claims_body = take("s6", "#|Claim")[3]
claims_used = set()


def claims_table(numbers):
    numbers = list(numbers)
    rows = [r for r in claims_body if int(r[0]) in numbers]
    assert len(rows) == len(numbers), (numbers, [r[0] for r in rows])
    claims_used.update(numbers)
    return table_html(claims_hdr, rows)


# ---- section 7: cost paragraph and the ordered next-run list (kept whole, with anchors per item)
cost_html = para("s7", "Cost basis")
next_items = take("s7", "**Re-score")[3]
assert len(next_items) == 6
next_list_html = '<ol class="next">' + "".join(f'<li id="next-{i + 1}">{inline(it)}</li>' for i, it in enumerate(next_items)) + "</ol>"


def from_link(sec_anchor, label):
    return f'<a class="from" href="{NARR}#{sec_anchor}">narrative {label}</a>'


def next_ptr(nums):
    links = ", ".join(f'<a href="#next-{n}">item {n}</a>' for n in nums)
    return f'<p class="next-ptr"><span class="ptr-label">Next runs for this question</span> ({from_link("s7", "§7")}, ordered by information per GPU-hour): {links}.</p>'


def short(*quotes):
    for q in quotes:
        assert q in md_text, "short answer is not a verbatim substring of NARRATIVE.md: " + q[:60]
    return '<div class="short"><p class="short-label">Short answer, in the narrative\'s words</p>' + "".join(f"<p>{inline(q)}</p>" for q in quotes) + "</div>"


def sub(qid, title, sec_anchor, label, parts):
    return f'<section class="subq" id="{qid}"><h3 class="q">{esc(title)} {from_link(sec_anchor, label)}</h3>\n' + "\n".join(parts) + "\n</section>"


def claims_block(nums):
    return f'<div class="claims"><h4 class="sub">Claims for the paper that this question decides {from_link("s6", "§6")}</h4>{claims_table(nums)}</div>'


# ---- the questions
QS = []  # (id, title, short_html, parts, sections_label, figures, claims_label, next_items)

# Q0
q0_parts = [f'<p class="from-note">{from_link("s1", "§1")}</p>', para("s1", "One paragraph, verified")]
omitted.append(take("s1", "The answer in three sentences is at the top")[1])
QS.append(("q0", "What was compared, on what, and how was it scored?", "", q0_parts, "§1", [], "", []))

# Q1
S1 = "The cell whose kernel structure matches the objective wins the optimization, by a paired margin that 10 seeds can establish: the additive cells on the two additive families, the product cells once a quarter of the variance sits in pairwise interactions, and every cell beats quasi-random search in every seed of every family [L1]."
q1_parts = [
    sub("q1-1", "What can a paired 10-seed comparison establish?", "s2", "§2", [para("s2", "**What a paired 10-seed")]),
    sub("q1-2", "Told by family: where does each cell win?", "s2", "§2", [
        para("s2", "Everything here is the paired"), figs[1],
        para("s2", "**aligned3.**"), para("s2", "**aligned10.**"), para("s2", "**decoupled.**"), para("s2", "**interaction_g0.25.**"), figs[2]]),
    sub("q1-3", "Why does product-lengthscale lose to additive-amplitude in every seed of both additive families?", "s2", "§2",
        [para("s2", "**Why product-lengthscale loses")]),
    sub("q1-4", "Why do the additive cells fall behind once a quarter of the variance is interaction, and only match oracle_S?", "s2", "§2",
        [para("s2", "**Why the additive cells fall behind")]),
    claims_block(range(1, 22)),
    next_ptr([4, 5]),
]
QS.append(("q1", "Which surrogate optimizes best, and does the answer depend on the objective's structure?", short(S1), q1_parts, "§2, §6", [1, 2], "1–21", [4, 5]))

# Q2
S2 = 'All four cells recover the true active set as a *ranking* (pooled average precision 0.89 to 0.97) [L2], but the thresholded rule "p_active > 0.5" is not a calibrated active-set decision in the amplitude cells: their NUTS chains never pass the convergence gate [L3], and the share threshold is applied in units that the BO design has deflated by a factor of about two [L4].'
q2_parts = [
    para("s3", "Scores are at the last readout"),
    sub("q2-1", "As a ranking?", "s3", "§3", [para("s3", "**Ranking works in every cell**"), figs[3]]),
    sub("q2-2", "As a thresholded decision? And if not, is it the prior, the chain, or the threshold?", "s3", "§3",
        [para("s3", "**The thresholded rule is precise"), take("s3", "*The threshold is applied")[2]]),
    sub("q2-3", "Does the neutral readout agree with the cell's own, and does the native score order by importance or by roughness?", "s3", "§3",
        [para("s3", "**Decoupled: importance versus roughness**"), take("s3", "class (share, ell)")[2], para("s3", "Three readings.")]),
    sub("q2-4", "When does each cell lock on?", "s3", "§3", [para("s3", "**When each cell locks on**"), figs[4]]),
    sub("q2-5", "Does identification predict optimization?", "s3", "§3", [para("s3", "**Does identification predict optimization?**")]),
    claims_block([22, 23, 24, 25, 26, 27, 29]),
    next_ptr([1]),
]
QS.append(("q2", "Do the cells recover the true active set?", short(S2), q2_parts, "§3, §6", [3, 4], "22–27, 29", [1]))

# Q3
S3 = "Gate exclusion is 100 % in both amplitude cells (7,200 of 7,200 iterations each), with r_hat_max median 1.49 and 1.51 (10th percentile 1.36 and 1.37) and n_eff_min median 6.1 and 5.9 (90th percentile 7.7 and 7.4), against thresholds of 1.1 and 16 [L64]."
q3_parts = [
    f'<p class="from-note">{from_link("s4", "§4")}</p>',
    para("s4", "**The amplitude cells did not converge"), figs[5],
    para("s4", "**What survives, for the amplitude cells.**"), take("s4", "*Regret survives*")[2],
    para("s4", "For the lengthscale cells the gate does its job"),
    claims_block([28]),
    next_ptr([2]),
]
QS.append(("q3", "Did the samplers converge, and what do the diagnostics license?", short(S3), q3_parts, "§4, §6", [5], "28", [2]))

# Q4
S4 = "Two things look wrong and should be fixed before a paper claims anything about the readouts: the additive-lengthscale cell flips intermittently into a mode that declares all 100 coordinates active on aligned3 (5 of 10 seeds, from about t = 75 on, invisible to single-chain R-hat) [L5], and oracle_S is a weak reference because a MAP ARD GP cannot exploit additivity and spends between 8 and 24 % of its late queries on near-duplicates of its incumbent [L6]."
q4_parts = [
    sub("q4-1", "Why does additive-lengthscale flip into an all-active mode on aligned3, and why this cell?", "s5a", "§5a", [
        para("s5", "*[observation]* The numeric report flags"), para("s5", "*[observation]* The retained draws"),
        para("s5", "*[inference]* This is a second posterior mode"), para("s5", "*[hypothesis; medium confidence]* Why this cell"),
        para("s5", "*[what to do]*")]),
    sub("q4-2", "Is oracle_S a fair oracle?", "s5b", "§5b", [
        para("s5", "*[observation]* oracle_S loses"), para("s5", "*[inference; confidence high]* The oracle knows S"),
        para("s5", "*[observation]* Two further weaknesses"), para("s5", "*[what a fair oracle would be]*")]),
    sub("q4-3", "What else looks wrong?", "s5c", "§5c", [take("s5", "**The share threshold is design-dependent**")[2]]),
    next_ptr([2, 3]),
]
QS.append(("q4", "What looks wrong, and must be fixed before a paper claims anything about the readouts?", short(S4), q4_parts, "§5", [], "", [2, 3]))

# Q5
S5a = "The four cells totalled 1,944 GPU-hours, of which the acquisition was 1,584 and additive-amplitude alone 788."
S5b = "Sampler repair and re-scoring precede further runs."
q5_parts = [f'<p class="from-note">{from_link("s7", "§7")}</p>', cost_html, figs[6],
            f'<h4 class="sub">What to run next, ordered by information per GPU-hour {from_link("s7", "§7")}</h4>', next_list_html]
QS.append(("q5", "What did the study cost, and what should run next?", short(S5a, S5b), q5_parts, "§7", [6], "", [1, 2, 3, 4, 5, 6]))

# ---- ledger (section 8) and summary (section 9)
ledger_intro = para("s8", "`claims_ledger.csv` lists every") + para("s8", "Where the numeric report and the tables disagree")
summary_html = para("s9", "On four synthetic 100-dimensional")

# everything consumed?
left = {sid: [b[1][:60] for b in bl] for sid, bl in remaining.items() if bl}
assert not left, left
assert claims_used == set(range(1, 30)), sorted(set(range(1, 30)) - claims_used)


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
kinds_legend = " ".join(f'{kind_badge(k)} <span>{v}</span>' for k, v in sorted(kind_counts.items(), key=lambda kv: -kv[1]))

# ---- question map (this page's own index of where each question draws from)
map_rows = []
for qid, title, _s, _p, secs, fg, cl, nx in QS:
    figl = ", ".join(f'<a href="#fig{n}">{n}</a>' for n in fg) or "—"
    nxl = ", ".join(f'<a href="#next-{n}">{n}</a>' for n in nx) or "—"
    map_rows.append(f'<tr><td class="qcell"><a href="#{qid}">{qid.upper()}</a></td><td>{esc(title)}</td><td class="mono">{secs}</td><td class="mono">{figl}</td><td class="mono">{cl or "—"}</td><td class="mono">{nxl}</td></tr>')
map_html = ('<div class="scroll"><table class="qmap"><thead><tr><th>Question</th><th></th><th>Narrative sections</th><th>Figures</th><th>Paper claims (§6)</th><th>Next runs (§7)</th></tr></thead><tbody>'
            + "".join(map_rows) + "</tbody></table></div>")

# ---- TOC
toc = ['<ol class="toc-list">']
for qid, title, _s, parts, *_ in QS:
    toc.append(f'<li><a href="#{qid}"><span class="qn">{qid.upper()}</span> {esc(title)}</a>')
    subs = re.findall(r'<section class="subq" id="([^"]+)"><h3 class="q">(.+?) <a class="from"', "\n".join(parts))
    if subs:
        toc.append("<ul>" + "".join(f'<li><a href="#{sid}">{t}</a></li>' for sid, t in subs) + "</ul>")
    toc.append("</li>")
toc.append('<li><a href="#ledger">Claims ledger</a></li><li><a href="#summary">Summary</a></li></ol>')
toc_html = "".join(toc)
fig_links = " ".join(f'<a href="#fig{n}">{n}</a>' for n in range(1, 7))

CSS = r"""
:root{
  color-scheme:light;--bg:#f6f7f8;--surface:#ffffff;--ink:#1b232e;--muted:#5d6874;--line:#d7dce3;--line-strong:#b5bec9;
  --accent:#0f6b64;--accent-ink:#0b5a54;--tag:#0b5a54;--tag-bg:#e3f0ee;--tag-hover:#cfe6e2;
  --callout-bg:#edf3f2;--code-bg:#eceff2;--target:#fff1bf;--plate:#ffffff;--focus:#0f6b64;
  --q:#7a3b12;--q-bg:#fbeee3;
  --obs-fg:#2c4a6e;--obs-bg:#e3ebf6;--inf-fg:#573b8c;--inf-bg:#ebe4f7;--hyp-fg:#874a09;--hyp-bg:#fbe9d1;
  --mix-fg:#3f4f7a;--mix-bg:#e6e7f6;
  --yes-fg:#1b6a3c;--yes-bg:#dcefe3;--maybe-fg:#865508;--maybe-bg:#fbeccc;--no-fg:#9c2d24;--no-bg:#f8dfdc;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;--bg:#12171e;--surface:#191f28;--ink:#e3e7ec;--muted:#98a3b1;--line:#2a333e;--line-strong:#3d4856;
    --accent:#6fcabf;--accent-ink:#8fd8cf;--tag:#8fd8cf;--tag-bg:#1a2f2c;--tag-hover:#24413d;
    --callout-bg:#182622;--code-bg:#222b35;--target:#40391a;--plate:#ffffff;--focus:#6fcabf;
    --q:#f0b98a;--q-bg:#3a2617;
    --obs-fg:#b7cbe8;--obs-bg:#213247;--inf-fg:#cfc0f0;--inf-bg:#33294d;--hyp-fg:#f1c78e;--hyp-bg:#443011;
    --mix-fg:#c3c8ee;--mix-bg:#2a2f4c;
    --yes-fg:#a3dcb6;--yes-bg:#183828;--maybe-fg:#f0ca8c;--maybe-bg:#423112;--no-fg:#f3a79f;--no-bg:#48231f;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;--bg:#12171e;--surface:#191f28;--ink:#e3e7ec;--muted:#98a3b1;--line:#2a333e;--line-strong:#3d4856;
  --accent:#6fcabf;--accent-ink:#8fd8cf;--tag:#8fd8cf;--tag-bg:#1a2f2c;--tag-hover:#24413d;
  --callout-bg:#182622;--code-bg:#222b35;--target:#40391a;--plate:#ffffff;--focus:#6fcabf;
  --q:#f0b98a;--q-bg:#3a2617;
  --obs-fg:#b7cbe8;--obs-bg:#213247;--inf-fg:#cfc0f0;--inf-bg:#33294d;--hyp-fg:#f1c78e;--hyp-bg:#443011;
  --mix-fg:#c3c8ee;--mix-bg:#2a2f4c;
  --yes-fg:#a3dcb6;--yes-bg:#183828;--maybe-fg:#f0ca8c;--maybe-bg:#423112;--no-fg:#f3a79f;--no-bg:#48231f;
}
@media (prefers-reduced-motion: no-preference){html{scroll-behavior:smooth}}
html{scroll-padding-top:12px}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;font-size:16.5px;line-height:1.6;-webkit-text-size-adjust:100%}
.page{max-width:1200px;margin:0 auto;padding-inline:16px;padding-block:0 72px}
code,pre,.mono,a.tag,.tags{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
h1,h2,h3,h4{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;font-weight:600;line-height:1.2;text-wrap:balance;color:var(--ink)}
h1{font-size:clamp(1.6rem,1.2rem + 1.6vw,2.15rem);margin:0 0 .5rem;letter-spacing:-.01em}
h2{font-size:1.45rem;margin:.2rem 0 .9rem}
h3{font-size:1.12rem;margin:1.9rem 0 .6rem}
h4.sub{font-size:.95rem;margin:1.6rem 0 .4rem;color:var(--muted)}
p{margin:.85rem 0}
ul,ol{padding-left:1.4rem}
li{margin:.45rem 0}
a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:2px}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
code{background:var(--code-bg);padding:.05em .35em;border-radius:3px;font-size:.86em}
strong{font-weight:600}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .6rem}
header.head{padding-block:32px 8px;border-bottom:1px solid var(--line);margin-bottom:8px}
header.head .snapshot{font-size:.95rem;color:var(--muted)}
header.head .snapshot strong{color:var(--ink)}
.framing{border:1px solid var(--line);background:var(--surface);padding:12px 16px;border-radius:6px;font-size:.95rem}
.framing p{margin:.3rem 0}
.answer{border-left:3px solid var(--accent);background:var(--callout-bg);padding:14px 18px;border-radius:0 6px 6px 0;margin:1.2rem 0}
.answer p{margin:0}
.legend,.rule{font-size:.95rem}
.rule{color:var(--muted)}
.rule p{margin:.4rem 0}
.prose p,.prose li,.answer,.legend,.rule,.snapshot,figcaption,.note,.framing,.short{max-width:74ch}
/* layout */
@media (min-width:1040px){
  .page{display:grid;grid-template-columns:250px minmax(0,1fr);column-gap:44px;padding-inline:32px}
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
.toc-list .qn{font-family:"IBM Plex Mono",Menlo,Consolas,monospace;font-size:.72rem;color:var(--q);font-weight:500;margin-right:2px}
.toc-list ul a{padding-left:20px;color:var(--muted);font-size:.8rem}
.toc-list a:hover{background:var(--tag-bg);color:var(--accent-ink)}
.toc-figs{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.82rem;color:var(--muted);padding:8px 8px 4px;border-top:1px solid var(--line);margin-top:8px}
.toc-figs a{color:var(--accent-ink);text-decoration:none;margin-left:.35em;font-family:"IBM Plex Mono",Menlo,Consolas,monospace}
.toc-figs a:hover{text-decoration:underline}
/* questions */
section.q{margin-top:3rem;padding-top:1.4rem;border-top:2px solid var(--line-strong)}
.q-eyebrow{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;color:var(--q);margin:0 0 .4rem}
.short{border-left:3px solid var(--q);background:var(--q-bg);padding:12px 16px;border-radius:0 6px 6px 0;margin:1rem 0 1.4rem}
.short p{margin:.3rem 0}
.short-label{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.74rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--q)}
h3.q{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline;justify-content:space-between}
a.from{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.72rem;letter-spacing:.04em;font-weight:500;color:var(--muted);text-decoration:none;white-space:nowrap;border:1px solid var(--line);padding:1px 7px;border-radius:10px}
a.from:hover{color:var(--accent-ink);border-color:var(--accent-ink)}
.from-note{margin:.2rem 0 0}
.claims{margin-top:1.4rem}
.next-ptr{font-size:.92rem;color:var(--muted);margin-top:1.2rem}
.ptr-label{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;color:var(--ink)}
ol.next li{padding:4px 6px;border-radius:4px}
ol.next li:target{background:var(--target)}
table.qmap{font-size:.88rem}
table.qmap td.qcell{font-family:"IBM Plex Mono",Menlo,Consolas,monospace;white-space:nowrap;font-weight:500}
table.qmap td.qcell a{color:var(--q);text-decoration:none}
table.qmap td.mono{font-size:.8rem;white-space:nowrap}
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
.note{font-size:.9rem;color:var(--muted)}
section.tail{margin-top:3rem;padding-top:1.4rem;border-top:2px solid var(--line-strong)}
section.tail h2{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline}
footer.about{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.86rem;color:var(--muted)}
footer.about code{font-size:.8rem}
@media print{.rail{display:none}.page{display:block}figure.fig{break-inside:avoid}}
"""

JS = r"""
(function(){
  var toc=document.getElementById('toc'), mq=window.matchMedia('(max-width: 1039px)');
  function sync(){ if(mq.matches){toc.removeAttribute('open');} else {toc.setAttribute('open','');} }
  sync(); if(mq.addEventListener){mq.addEventListener('change',sync);} else {mq.addListener(sync);}
  toc.querySelectorAll('a').forEach(function(a){a.addEventListener('click',function(){ if(mq.matches){toc.removeAttribute('open');} });});
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

q_sections = []
for qid, title, short_html, parts, *_ in QS:
    q_sections.append(f'<section class="q" id="{qid}"><p class="q-eyebrow">{qid.upper()}</p><h2>{esc(title)}</h2>\n{short_html}\n' + "\n".join(parts) + "\n</section>")

page = f"""<title>sagp research questions</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{CSS}</style>
<div class="page">
<header class="head">
<p class="eyebrow">Sister page · narrative analysis 2026-09-12-1849 · sorted by research question</p>
<h1>sagp high-dimensional BO study, by research question</h1>
<div class="framing"><p><strong>How to read this page.</strong> It is a companion to <a href="{NARR}">the narrative analysis</a> and re-sorts the same material by the research question it answers. The six questions and the grouping under them are this page's own organizing device; every answer, paragraph, number, table row, caption and figure is quoted verbatim from the narrative's <code>NARRATIVE.md</code>, <code>captions.md</code> and <code>claims_ledger.csv</code>, and each block carries a pill naming the narrative section it comes from. Section 6's claims table is split across the questions its rows decide, and section 7's next-run list is kept whole under Q5 with pointers from the other questions.</p></div>
<div class="snapshot">{snapshot_html}</div>
<div class="answer">{answer_html}</div>
<div class="legend">{labels_html}</div>
<div class="rule"><p><strong>Verdict rule for the claims tables</strong> ({from_link("s6", "§6")}).</p>{verdict_rule_html}</div>
<h4 class="sub">Question map</h4>
{map_html}
</header>
<nav class="rail" aria-label="Questions">
<details id="toc" open>
<summary>Questions</summary>
<div class="toc-body">
<div class="toc-title">Questions</div>
{toc_html}
<div class="toc-figs">Figures{fig_links}</div>
</div>
</details>
</nav>
<main class="prose">
{"".join(q_sections)}
<section class="tail" id="ledger"><h2>Claims ledger {from_link("s8", "§8")}</h2>
{ledger_intro}
<p class="ledger-legend"><span>{len(ledger)} rows, sorted by id; click a column header to sort. Kinds:</span> {kinds_legend}</p>
{ledger_html}
</section>
<section class="tail" id="summary"><h2>Summary {from_link("s9", "§9")}</h2>
{summary_html}
<p class="note">The narrative's supporting tables (lock-on times, decoupled breakdown, native-score summaries, 10-seed bounds, near-duplicate fractions, identical-regret pairs, incumbent boundary, y_std, dsp_map fallbacks, eps constants) are on <a href="{NARR}#appendix">the narrative page's appendix</a>, not repeated here.</p>
</section>
<footer class="about">
<p>Built from <code>/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/</code> (<code>NARRATIVE.md</code>, <code>captions.md</code>, <code>claims_ledger.csv</code>) and the six figures in <code>/scratch/work/tranq8/sagp_analysis/latest/figures/</code>. Every narrative paragraph, list and table appears exactly once on this page except one line of in-file navigation ("{esc(omitted[0])}"). Tags such as <span class="tags">[<a class="tag" href="#L12">L12</a>]</span> link to the ledger row that decides the sentence; hover one to read the claim and its source.</p>
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
print("questions:", [q[0] for q in QS], "| figures:", len(figs), "| ledger rows:", len(ledger))
print("distinct tags on page:", len(used), "| missing from ledger:", missing, "| ledger rows unused:", unused)
print("claims rows placed:", len(claims_used), "| omitted narrative lines:", omitted)
body_only = re.sub(r"<style>.*?</style>|<script>.*?</script>", "", page, flags=re.S)
stripped = re.sub(r"<[^>]+>", "", body_only)
print("leftover markdown markers:", re.findall(r"\*\*|`", stripped)[:10])
print("img srcs:", sorted(set(re.findall(r'<img src="([^"]+)"', page))))
