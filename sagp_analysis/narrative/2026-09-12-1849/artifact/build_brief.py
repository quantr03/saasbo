#!/usr/bin/env python
"""Concise page keyed to the research questions of the Design brief (Research Context/research direction/Design-brief.md).

For each brief item: the brief's text (transcribed to plain text; the build asserts the numbers match the brief), a status
badge (this page's classification), the narrative's own sentences as evidence (asserted verbatim substrings of NARRATIVE.md,
with their [L#] tags), the section-6 claim rows it decides, and what was not run.
Usage: build_brief.py OUT.html FIGDIR NARRATIVE_URL QUESTIONS_URL
"""
import csv, html, re, struct, sys
from pathlib import Path

D = Path("/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849")
BRIEF = Path("/scratch/work/tranq8/saasbo/Research Context/research direction/Design-brief.md")
OUT, FIGDIR, NARR, QPAGE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].rstrip("/"), sys.argv[4].rstrip("/")

md = (D / "NARRATIVE.md").read_text()
brief = BRIEF.read_text()
ledger = list(csv.DictReader(open(D / "claims_ledger.csv")))
L = {r["id"]: r for r in ledger}
used_tags = []
NUM = re.compile(r"\d+(?:\.\d+)?")


def esc(s, quote=False):
    return html.escape(s, quote=quote)


def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    def lab(m):
        cls = {"observation": "lab-obs", "inference": "lab-inf", "hypothesis": "lab-hyp"}[m.group(1)]
        return f'<em class="lab {cls}">[{m.group(1)}{m.group(2)}]</em>'
    s = re.sub(r"\*\[(observation|inference|hypothesis)([^\]]*)\]\*", lab, s)
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)

    def tag(m):
        parts = []
        for i in re.findall(r"L\d+", m.group(0)):
            used_tags.append(i)
            title = esc(f'{L[i]["claim"]} — {L[i]["source_file"]} ({L[i]["locator"]})', quote=True)
            parts.append(f'<a class="tag" href="#{i}" title="{title}">{i}</a>')
        return '<span class="tags">[' + ", ".join(parts) + "]</span>"
    return re.sub(r"\[L\d+(?:,\s*L\d+)*\]", tag, s)


# ---------- fidelity helpers
def N(q):
    """A verbatim narrative sentence: asserted substring of NARRATIVE.md."""
    assert q in md, "not verbatim in NARRATIVE.md: " + q[:70]
    return f'<p class="ev">{inline(q)}</p>'


def nums(s):
    return sorted(NUM.findall(s.replace("²", "2")))


def brief_block(start, end=None):
    i = brief.index(start)
    j = brief.index(end, i) if end else len(brief)
    return brief[i:j]


def B(text, block_start, block_end=None, exact=False):
    """A brief passage transcribed to plain text; its numbers must equal the brief block's numbers."""
    if exact:
        assert text in brief, "not verbatim in the brief: " + text[:60]
    else:
        assert nums(text) == nums(brief_block(block_start, block_end)), (block_start, nums(text), nums(brief_block(block_start, block_end)))
    return f'<blockquote class="brief">{esc(text)}</blockquote>'


def nums_in(text, corpus, what):
    for t in NUM.findall(text.replace("²", "2")):
        assert t in corpus, f"{what}: number {t} not found for cell: {text[:60]}"


STATUS = {"yes": "Answered", "part": "Partly answered", "no": "Not run", "fail": "Check failed", "cons": "Consistent", "nt": "Not testable here", "one": "One point only"}


def st(kind, note=""):
    return f'<p class="status"><span class="badge st-{kind}">{STATUS[kind]}</span> <span class="mapnote"><span class="maplabel">Mapping</span> {inline(note)}</span></p>' if note else f'<span class="badge st-{kind}">{STATUS[kind]}</span>'


# ---------- section-6 claims (#, claim, verdict) and figure titles
claims = {int(m.group(1)): (m.group(2).strip(), m.group(3).strip()) for m in re.finditer(r"^\| (\d+) \| (.+?) \| (.+?) \| .+? \|$", md, re.M)}
assert len(claims) == 29, len(claims)
claims_used = set()


def vbadge(cell):
    for word, cls in (("NOT SUPPORTED", "v-no"), ("SUPPORTED", "v-yes"), ("SUGGESTIVE", "v-maybe")):
        if cell.startswith(word):
            return f'<span class="badge {cls}">{word}</span>' + inline(cell[len(word):])
    return inline(cell)


def claim_rows(numbers):
    claims_used.update(numbers)
    rows = "".join(f'<tr><td class="n"><a href="{NARR}#s6">{n}</a></td><td>{inline(claims[n][0])}</td><td>{vbadge(claims[n][1])}</td></tr>' for n in numbers)
    return f'<div class="scroll claims"><table class="md"><thead><tr><th>#</th><th>Claim for the paper (narrative §6)</th><th>Verdict</th></tr></thead><tbody>{rows}</tbody></table></div>'


cap = (D / "captions.md").read_text()
figs = {}
for m in re.finditer(r"\*\*Figure (\d+)\. `([^`]+)`\. (.+?)\*\*\n", cap):
    figs[int(m.group(1))] = (m.group(2), m.group(3))


def fig(n):
    fname, title = figs[n]
    with open(FIGDIR / fname, "rb") as f:
        w, h = struct.unpack(">II", f.read(24)[16:24])
    return (f'<figure class="fig" id="fig{n}"><a class="zoom plate" href="figures/{fname}" data-num="{n}"><img src="figures/{fname}" alt="{esc(title, True)}" width="{w}" height="{h}" loading="lazy"></a>'
            f'<figcaption><span class="fig-label">Figure {n}.</span> <code>{fname}</code>. {esc(title)} <a class="fig-open" href="{NARR}#fig{n}">Full caption on the narrative page</a> · <a class="fig-open" href="figures/{fname}">Open the PNG</a></figcaption></figure>')


def nx(items):
    return '<p class="next-ptr"><span class="ptr-label">Next runs</span> (narrative §7, ordered by information per GPU-hour): ' + ", ".join(f'<a href="{QPAGE}#next-{i}">item {i}</a>' for i in items) + ".</p>"


# ---------- content
B_Q = "Holding kernel family (crossed), inference, acquisition, initialization and budget fixed, does moving the sparsity prior from inverse squared lengthscales to normalized additive amplitudes change which coordinates a fully Bayesian GP treats as relevant and how sample-efficiently BO optimizes sparse objectives — and where does any such difference end as first-order additivity fails?"
B_SQ1 = ("SQ1 — identification (offline, cheap, first). Fit all four cells to the same datasets: Sobol designs with n ∈ {30, 60, 100, 200}, D = 100, 10 seeds per family. Primary readout is parameterization-neutral: the first-order Sobol index of the posterior mean, Ŝ_i (exact per component in the additive cells; quasi-Monte Carlo in the product cells). Secondary: the native scores a_i² and ρ_i. Metrics: AUROC and precision/recall of the active set; Spearman of Ŝ_i against s_i and against g_i; calibration of P(a_i² > ε) against the active indicator. Manipulation checks that gate everything downstream: posterior a_i² tracks realized component variance across lengthscales; amplitude ranking agrees with the Sobol ranking; R̂ < 1.05, ESS and divergence bounds. "
         "Predictions. P1a: on the aligned family all four cells select the same set (Jaccard > 0.9 at n = 100). P1b: on the decoupled and anti-aligned families the amplitude column tracks s_i (Spearman > 0.8) and the lengthscale column tracks g_i better than s_i. If both columns track the same label even at ℓ = 0.08, the two priors differ in parameterization only, and that is the result.")
B_SQ2 = ("SQ2 — sample efficiency (in-loop, confirmatory). The 2 × 2 on aligned |S| ∈ {3, 10}, decoupled, and embedded Hartmann-6. Readout: simple regret r_t = f* − max_{s ≤ t} f(x_s), median and bootstrap interval over seeds, at t ∈ {50, 100, 200} plus area under the curve; analysed as a two-factor design with effect sizes against a preregistered smallest effect of interest (0.3 in log10 regret at T = 200). Gate before spending: oracle-S vs DSP-by-MAP headroom at D = 100 must exceed two standard errors, otherwise no selection method can matter at this budget and SQ1 becomes the core. "
         "Prediction P2: the structure main effect exceeds the parameterization main effect on aligned objectives; a parameterization effect appears, if at all, on decoupled.")
B_SQ3 = ("SQ3 — boundary (in-loop). Interaction sweep with the additive–amplitude cell, SAASBO-with-LogEI, DSP-by-MAP and Sobol; γ* = the share at which the additive advantage over the matched product cell vanishes, by interpolation with a bootstrap interval. Rotation sweep and dense-weak as one-factor sweeps. Lasso-DNA and MOPTA08 enter only here, as out-of-distribution tasks with a predicted direction (amplitude sparsity loses to DSP because of their many weak, smooth, boundary-seeking coordinates). "
         "Prediction P3: γ* ∈ (0.1, 0.5) at T = 200, the same for both parameterizations.")
B_ALPHA = "Prior scale. α for τ ~ HC(α) on the a_i² scale (SAASBO uses 0.1 on the ρ scale); I plan to fix it in the pilot from the prior-predictive number of coordinates with a_i² > ε = 0.02 (matched to the count SAASBO's own prior implies on the ρ scale), and would rather agree the rule now than tune it later."

sections = []

# --- primary question
sections.append(("pq", "The primary question", "Design brief §1", "".join([
    B(B_Q, None, exact=True),
    '<h3>(a) Does the parameterization change which coordinates the GP treats as relevant?</h3>',
    st("part", "Answered as a ranking and on decoupled as an ordering; the brief's Spearman-against-labels readout was not computed, and anti-aligned was not run."),
    N('All four cells recover the true active set as a *ranking* (pooled average precision 0.89 to 0.97) [L2], but the thresholded rule "p_active > 0.5" is not a calibrated active-set decision in the amplitude cells: their NUTS chains never pass the convergence gate [L3], and the share threshold is applied in units that the BO design has deflated by a factor of about two [L4].'),
    N('So on decoupled the answer to "does the neutral readout agree with the cell\'s own" is: on membership yes, on order no for the lengthscale cells, and sobol_hat is the readout to report as a share estimate.'),
    claim_rows([22, 23, 26, 27, 29]),
    '<h3>(b) Does it change how sample-efficiently BO optimizes?</h3>',
    st("yes", "The structure effect is the finding; the parameterization effect within a structure is not supported except on decoupled (claim 19)."),
    N("The cell whose kernel structure matches the objective wins the optimization, by a paired margin that 10 seeds can establish: the additive cells on the two additive families, the product cells once a quarter of the variance sits in pairwise interactions, and every cell beats quasi-random search in every seed of every family [L1]."),
    N("That the additive cells still beat dsp_map twenty-fold in median [L26] says that at D = 100 and this budget a wrong-but-sparse structural prior is worth more than a right-but-vague one."),
    claim_rows([16, 17, 18, 19, 21]),
    '<h3>(c) Where does the difference end as first-order additivity fails?</h3>',
    st("one", "Only gamma = 0.25 was run, so the boundary is bracketed from one side; the sweep is next-run item 5."),
    N("Cell against cell, product beats additive in 8/10 seeds in all four pairings (p from 0.0098 to 0.065) [L30]: SUGGESTIVE, not SUPPORTED."),
    N("Its posterior mean is the additive projection of f, whose maximizer is not f's; the regret floor of 0.017 to 0.021 [L26] is the price of that projection. The product cells represent the interaction natively and reach 0.0007 to 0.0013 [L26]."),
    claim_rows([13, 20]),
    nx([5]),
])))

# --- SQ1
sections.append(("sq1", "SQ1 — identification", "Design brief §4", "".join([
    B(B_SQ1, "**SQ1 — identification", "**SQ2 — sample efficiency"),
    st("part", "Run in-loop on the BO design (readouts t = 20 to 199, scored at t = 199) instead of offline on Sobol designs at n ∈ {30, 60, 100, 200}; AP by native median and by sobol_hat, and precision/recall/F1 at p_active > 0.5, stand in for AUROC, Spearman and calibration."),
    N("Identification is scored at the last readout, t = 199, by ranking the 100 coordinates by the cell's native median (rho_i or a_sq_i, higher = more active in every cell) or by sobol_hat (first-order Sobol indices of the posterior mean under the uniform measure), and by the thresholded rule p_active > 0.5, where p_active is the fraction of the 16 retained draws with share > 0.02 (amplitude cells) or rho > 0.152, i.e. ell < 2.56 (lengthscale cells) [L11]."),
    fig(3),
    '<h3>P1a — on the aligned family all four cells select the same set</h3>',
    st("part", "Consistent on aligned3. On aligned10 the rankings agree (median AP 1.0) but the thresholded sets differ by cell (recall 0.25 to 0.90); Jaccard was not computed."),
    N("aligned3: all four cells reach AP 1.0 and F1 1.0 at t = 50 and stay there."),
    N("The median AP over runs is 1.0 in all four cells [L41]: the typical run ranks S perfectly and the means are pulled down by a few seeds."),
    N("Precision at p_active > 0.5 is 0.92 to 1.00 in every cell [L45], but recall is 0.25 (product-amplitude) and 0.51 (additive-amplitude) on aligned10, and 0.39 and 0.54 on decoupled [L46]; the lengthscale cells sit at 0.53 / 0.90 (aligned10) and 0.56 / 0.79 (decoupled) [L47]."),
    N("aligned10: additive-lengthscale reaches AP >= 0.9 at t = 125 and F1 >= 0.8 at 125 (F1 1.0 at 199); additive-amplitude and product-amplitude reach AP 0.9 at 175, product-lengthscale at 199; no amplitude cell and not product-lengthscale ever reaches median F1 0.8."),
    claim_rows([22, 23, 24]),
    '<h3>P1b — the amplitude column tracks the share, the lengthscale column tracks the slope</h3>',
    st("part", "Direction consistent on decoupled, descriptively: the lengthscale cells' native score orders by roughness, the amplitude cells' by importance with roughness leaking in, and sobol_hat recovers the shares. Spearman against s_i and g_i was not computed; anti-aligned was not run (next-run item 5)."),
    N("(ii) The lengthscale cells' native score orders by roughness, as its units dictate: additive-lengthscale's median rho is 120 on the high-share rough class (ell 0.09; the true value is 0.08) against 2.2 on the high-share smooth class (ell 0.67; true 1.5), and its median native rank is 1.5 for high-rough against 5.0 for high-smooth at the same true share [L60]. (iii) The amplitude cells' native score orders by importance but roughness leaks in: additive-amplitude's median a_sq is 0.079 on high-rough against 0.038 on high-smooth, the same true share, and product-amplitude detects high-smooth coordinates only 45 % of the time against 95 % for high-rough [L61]."),
    N("The neutral readout does what it was built for: sobol_hat's medians are 0.20 and 0.26 on the two high-share classes and 0.040 and 0.037 on the low-share ones for additive-lengthscale, and 0.23 / 0.20 and 0.037 / 0.032 for additive-amplitude, against true shares 0.21 and 0.04 [L62]."),
    N("anti_aligned (6 active; shares 0.35 to 0.03 anti-correlated with lengthscales 3.0 to 0.06, per `families.py`) is the sharp test of section 3's roughness-versus-importance finding; the two lengthscale cells plus additive-amplitude at 10 seeds cost about 320 GPU-hours."),
    claim_rows([26, 27]),
    '<h3>Manipulation checks — convergence, calibration, amplitude tracks variance</h3>',
    st("fail", "The R-hat check fails in every amplitude fit and in about half of the lengthscale fits; p_active is not a calibrated probability; posterior a_sq on the true coordinates sits below even the deflated share."),
    N("Gate exclusion is 100 % in both amplitude cells (7,200 of 7,200 iterations each), with r_hat_max median 1.49 and 1.51 (10th percentile 1.36 and 1.37) and n_eff_min median 6.1 and 5.9 (90th percentile 7.7 and 7.4), against thresholds of 1.1 and 16 [L64]."),
    N("The lengthscale cells fail marginally: exclusion 52.8 % and 59.3 %, r_hat_max median 1.10 and 1.11, n_eff_min median 23 and 19, with the exclusion rate peaking at 70 to 75 % around t = 60 to 80 and falling to 34 to 47 % by t = 180 [L67]."),
    fig(5),
    N("*p_active as a probability does not survive* *[inference]*: 16 retained draws from about 6 effective ones, at a threshold the design has moved onto the true values (section 3)."),
    N("In the model's units a true share s_i is therefore about s_i / y_std^2: a 0.10 share on aligned10 becomes 0.042, a 0.04 share on decoupled becomes 0.020, which is the threshold itself, and a 0.21 share becomes 0.106 [L49]."),
    N("The median a_sq on the true coordinates is 0.021 (additive-amplitude) and 0.011 (product-amplitude) on aligned10 against the deflated 0.042 [L50]; on decoupled the additive-amplitude medians are 0.038 (high-share smooth), 0.079 (high-share rough), 0.010 and 0.012 (low-share) against deflated 0.106 and 0.020 [L51]."),
    N("Scanning every readout of every cell run (200 runs x 180 readouts) [L73]: the same mode appears in five of the ten aligned3 additive-lengthscale seeds, 01, 02, 05, 08 and 09, in 25, 23, 27, 70 and 54 of their 180 readouts, first at t = 83, 87, 73, 80 and 85."),
    claim_rows([25, 28]),
    nx([1, 2]),
])))

# --- SQ2
sections.append(("sq2", "SQ2 — sample efficiency", "Design brief §4", "".join([
    B(B_SQ2, "**SQ2 — sample efficiency", "**SQ3 — boundary"),
    st("part", "Run on aligned3, aligned10 and decoupled; embedded Hartmann-6 was not run. The readout is the paired Wilcoxon test on final regret at t = 199 with medians at t = 100, not regret at t ∈ {50, 100, 200} with area under the curve and two-factor effect sizes."),
    N("Everything here is the paired final regret at t = 199 (regret = f_star − best_f, lower is better), the median over the 10 seeds of a family, and the two-sided Wilcoxon signed-rank test over seeds from `paired_wilcoxon_final_regret.csv` [analysis: `analyze.py`, `paired_tests`]."),
    N("Paired tests also say nothing about the *size* of a difference on a scale that transfers to other problems: the medians below are on the regret scale of these particular draws."),
    fig(2),
    '<h3>P2 — structure main effect exceeds parameterization main effect on aligned; a parameterization effect, if at all, on decoupled</h3>',
    st("cons", "On aligned10 the structure contrasts are SUPPORTED and the parameterization contrast within a structure is NOT SUPPORTED; on decoupled a parameterization effect is SUGGESTIVE (claim 19); aligned3 does not discriminate."),
    N("**aligned10.** The additive cells lead: additive-amplitude 0.027, additive-lengthscale 0.032, product-amplitude 0.065, then dsp_map 0.28, product-lengthscale 0.29 and oracle_S 0.43 [L17]. Additive-amplitude beats dsp_map and oracle_S in 10/10 seeds (p = 0.002) [L18] and product-lengthscale in 10/10 [L19]."),
    N("**decoupled.** The same ordering with wider spread: additive-amplitude 0.058, product-amplitude 0.13, additive-lengthscale 0.16, dsp_map 0.29, oracle_S 0.44, product-lengthscale 0.47 [L22]. Additive-amplitude beats oracle_S 10/10 and product-lengthscale 10/10, and additive-lengthscale beats product-lengthscale 10/10 [L23]."),
    N("**aligned3.** Every model-based method reaches a median regret below 0.004 and no cell is distinguishable from dsp_map (p >= 0.084) [L15]; all four cells beat sobol 10/10, p = 0.002 [L16], the pattern in every family. A three-coordinate problem at this budget does not discriminate surrogate classes."),
    N("The additive lead is visible from the middle of the run: at t = 100 the additive-amplitude median is 0.18 against 0.48 for product-lengthscale and 0.47 for dsp_map [L21]."),
    claim_rows([1, 2, 14, 16, 17, 18, 19, 21]),
    '<h3>Gate — oracle-S versus DSP-by-MAP headroom must exceed two standard errors</h3>',
    st("fail", "As written, the gate is not met: oracle_S trails dsp_map in median on all three SQ2 families. The narrative reads this as a weak oracle rather than absent headroom (§5b) and asks for a structure-matched oracle (next-run item 3)."),
    N("*[observation]* oracle_S loses to additive-amplitude in 10/10 seeds on aligned10 (median 0.43 against 0.027) and on decoupled (0.44 against 0.058), and to additive-lengthscale 9/10 on both [L85]. On aligned3 it is the worst model-based method (median 0.0033, beaten 9/10 by each amplitude cell) [L86]."),
    N("*[inference; confidence high]* The oracle knows S but fits one ARD Matern-5/2 GP at the MAP on those coordinates. On an additive objective of 10 coordinates that is a 10-dimensional interpolation problem with 200 points; it cannot pool information across coordinates the way an additive kernel does, so it is a full-interaction model with the wrong inductive bias on exactly the families where the additive cells have the right one."),
    N("*[what a fair oracle would be]* The same structural class as the cell under test, fitted on S: for the additive families an additive GP on the true coordinates (additive Matern factors, MAP or NUTS, cheap at |S| <= 10), and for the interaction families a product kernel on S."),
    claim_rows([3, 4, 7, 15]),
    nx([3, 4]),
])))

# --- SQ3
sections.append(("sq3", "SQ3 — boundary", "Design brief §4", "".join([
    B(B_SQ3, "**SQ3 — boundary", "## 5. Cost"),
    st("one", "One interaction share, gamma = 0.25, with all four cells and the three references. The sweep, the rotation sweep, dense-weak, Lasso-DNA and MOPTA08 were not run."),
    N("**interaction_g0.25.** The product cells lead: product-lengthscale 0.0007, product-amplitude 0.0013, oracle_S 0.012, additive-amplitude 0.017, additive-lengthscale 0.021, dsp_map 0.42 [L26]. Product-lengthscale beats dsp_map 10/10 [L27]. The additive cells are indistinguishable from oracle_S (4 seeds better, 6 worse, p = 0.56) [L28] and beat dsp_map 9/10 and 8/10 (p = 0.049 and 0.084) [L29]. Cell against cell, product beats additive in 8/10 seeds in all four pairings (p from 0.0098 to 0.065) [L30]: SUGGESTIVE, not SUPPORTED."),
    N("At t = 199, 6 of 10 additive-lengthscale runs and 5 of 10 additive-amplitude runs on interaction_g0.25 declare one inactive coordinate active (7 and 5 false positives in total; the largest off-S native scores are rho 1.5e5 and a_sq 0.59); the product cells have no false positive on that family, and the additive cells have at most one on any other family [L39]."),
    '<h3>P3 — γ* in (0.1, 0.5) at T = 200, the same for both parameterizations</h3>',
    st("nt", "γ* cannot be interpolated from one gamma. At 0.25 the product cells already lead (SUGGESTIVE), with no gamma = 0 point to anchor the additive advantage."),
    N("The interaction sweep (gamma = 0, 0.10, 0.50, 0.75) locates the gamma at which additive cells stop being competitive; product-lengthscale and additive-amplitude alone at 5 seeds cost about 100 GPU-hours per gamma."),
    N("rotated_t0 / t15 / t45 test non-axis-aligned structure, a different question; aligned10_sin and dense_weak are lower priority for the identification question."),
    claim_rows([10, 11, 12, 13, 20]),
    nx([5]),
])))

# --- prior scale (brief §6.3)
sections.append(("alpha", "Also in the brief: the prior scale α", "Design brief §6, item 3", "".join([
    B(B_ALPHA, "**Prior scale.**"),
    st("yes", "Fixed by the rule the brief proposed. The narrative names the resulting funnel as the likely geometric reason the amplitude chains do not mix."),
    N("the amplitude cells put the same mixture on the component variances a_sq_i with alpha = 0.0131, chosen so that both priors imply the same prior-predictive count of active coordinates, plus a LogNormal(0, 1.5) prior on each ell_i, and a_sq_i -> 0 removes coordinate i [L8]."),
    N("The amplitude cells sample 2D + 3 = 203 sites against the lengthscale cells' D + 4 = 104 [L69], which is the structural reason they are worse; the funnel of the a_sq scale mixture at alpha = 0.013 [L8] is the likely geometric one *[hypothesis]*."),
    nx([2]),
])))

# --- brief vs study table
ROWS = [
    ("Design", "D = 100; T = 200; 20-point Sobol initialization per seed; 10 seeds", "D = 100 and T = 200 evaluations, of which the first 20 are a scrambled-Sobol initial design shared by the seven methods of a seed; 10 seeds [L7]"),
    ("Acquisition", "EI averaged over the 16 retained samples; 5000 Sobol candidates plus the incumbent; top-5 L-BFGS-B", "BoTorch's analytic LogEI maximized by `optimize_acqf` (512 raw samples, 5 restarts, perturbations around the best 5 % at sigma 0.001) [L7]"),
    ("NUTS", "512 warm-up + 256 samples thinned to 16 retained, tree depth 6, one chain, a fresh chain at every refit", "one NUTS chain, 512 warmup, 256 draws thinned to 16 retained, max_tree_depth 6, dense mass matrix [L9]"),
    ("Convergence check", "R̂ < 1.05, ESS and divergence bounds", "excluded when split-R-hat > 1.1 or ESS < 16 on any sampled site or more than 5 divergences; excluded draws are still used to choose the next point [L9]"),
    ("Prior scale on a_i²", "fixed in the pilot from the prior-predictive number of coordinates with a_i² > ε = 0.02, matched to SAASBO's count on the ρ scale", "alpha = 0.0131, chosen so that both priors imply the same prior-predictive count of active coordinates [L8]"),
    ("Identification readout", "offline fits on Sobol designs, n ∈ {30, 60, 100, 200}; Ŝ_i primary; AUROC, precision/recall, Spearman against s_i and g_i, calibration of P(a_i² > ε)", "in-loop readouts at t = 20, 25, 50, ..., 175, 199, scored at t = 199; AP by native median and by sobol_hat; precision, recall and F1 at p_active > 0.5 with share > 0.02 or rho > 0.152 [L11]"),
    ("Regret analysis", "regret at t ∈ {50, 100, 200} plus area under the curve; two-factor effect sizes against a smallest effect of interest of 0.3 in log10 regret at T = 200", "paired final regret at t = 199, medians over 10 seeds, two-sided Wilcoxon signed-rank test; medians at t = 100 [L21, L32]; no effect sizes"),
    ("References", "Sobol search; DSP by MAP; an oracle GP given the true active set", "sobol; dsp_map (ARD Matern-5/2 at the MAP under the dimension-scaled prior); oracle_S (the same MAP GP restricted to the true active set through a FilterFeatures transform)"),
    ("Families run", "aligned |S| ∈ {3, 10}; decoupled; anti-aligned; interaction sweep γ ∈ {0, .1, .25, .5, .75}; rotated θ ∈ {0°, 15°, 45°}; dense-weak; embedded Hartmann-6 and Levy-4; a sinusoid family; Lasso-DNA and MOPTA08", "aligned3, aligned10, decoupled, interaction_g0.25: 4 families x 7 methods x 10 seeds, 280/280 runs; everything else not run"),
    ("Anti-aligned family", "|S| = 6, s = (0.35, 0.25, 0.18, 0.12, 0.07, 0.03), ℓ = (2, 1.2, 0.7, 0.4, 0.2, 0.1)", "anti_aligned (6 active; shares 0.35 to 0.03 anti-correlated with lengthscales 3.0 to 0.06, per `families.py`); not run"),
]
trs = []
for what, b, s in ROWS:
    nums_in(b, brief, "brief cell")
    nums_in(s, md, "study cell")
    trs.append(f"<tr><th scope=\"row\">{esc(what)}</th><td>{esc(b)}</td><td>{inline(s)}</td></tr>")
diff_table = '<div class="scroll"><table class="diff"><thead><tr><th></th><th>Design brief (2 September 2026)</th><th>The study as run (narrative, 2026-09-12)</th></tr></thead><tbody>' + "".join(trs) + "</tbody></table></div>"

# --- ledger rows cited on this page
cited = sorted(set(used_tags), key=lambda t: int(t[1:]))
led = ['<div class="scroll"><table class="ledger"><thead><tr><th>id</th><th>section</th><th>claim</th><th>values</th><th>source file</th><th>locator</th><th>kind</th></tr></thead><tbody>']
KC = {"observation": "k-obs", "inference": "k-inf", "hypothesis": "k-hyp", "observation+inference": "k-mix"}
for i in cited:
    r = L[i]
    src = " ".join(f"<code>{esc(x.strip())}</code>" for x in r["source_file"].split(";") if x.strip())
    led.append(f'<tr id="{i}"><td class="mono id">{i}</td><td class="mono">{esc(r["section"])}</td><td>{esc(r["claim"])}</td><td class="values">{esc(r["values"])}</td><td class="src">{src}</td><td class="mono loc">{esc(r["locator"])}</td><td><span class="badge {KC[r["kind"]]}">{esc(r["kind"])}</span></td></tr>')
led.append("</tbody></table></div>")
ledger_html = "".join(led)

toc = "".join(f'<li><a href="#{sid}">{esc(title)}</a></li>' for sid, title, _, _ in sections) + '<li><a href="#diff">Brief versus study</a></li><li><a href="#ledger">Ledger rows cited</a></li>'
body = "".join(f'<section class="item" id="{sid}"><p class="eyebrow">{esc(src)}</p><h2>{esc(title)}</h2>{content}</section>' for sid, title, src, content in sections)

CSS = r"""
:root{color-scheme:light;--bg:#f6f7f8;--surface:#ffffff;--ink:#1b232e;--muted:#5d6874;--line:#d7dce3;--line-strong:#b5bec9;
  --accent:#0f6b64;--accent-ink:#0b5a54;--tag:#0b5a54;--tag-bg:#e3f0ee;--tag-hover:#cfe6e2;--code-bg:#eceff2;--target:#fff1bf;--plate:#ffffff;--focus:#0f6b64;
  --brief:#4a3c8c;--brief-bg:#eeeaf8;--map:#7a3b12;--map-bg:#fbeee3;
  --obs-fg:#2c4a6e;--obs-bg:#e3ebf6;--inf-fg:#573b8c;--inf-bg:#ebe4f7;--hyp-fg:#874a09;--hyp-bg:#fbe9d1;--mix-fg:#3f4f7a;--mix-bg:#e6e7f6;
  --yes-fg:#1b6a3c;--yes-bg:#dcefe3;--maybe-fg:#865508;--maybe-bg:#fbeccc;--no-fg:#9c2d24;--no-bg:#f8dfdc;--part-fg:#2f5a8a;--part-bg:#e0ebf7}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#12171e;--surface:#191f28;--ink:#e3e7ec;--muted:#98a3b1;--line:#2a333e;--line-strong:#3d4856;
  --accent:#6fcabf;--accent-ink:#8fd8cf;--tag:#8fd8cf;--tag-bg:#1a2f2c;--tag-hover:#24413d;--code-bg:#222b35;--target:#40391a;--plate:#ffffff;--focus:#6fcabf;
  --brief:#c9bcf2;--brief-bg:#2a2544;--map:#f0b98a;--map-bg:#3a2617;
  --obs-fg:#b7cbe8;--obs-bg:#213247;--inf-fg:#cfc0f0;--inf-bg:#33294d;--hyp-fg:#f1c78e;--hyp-bg:#443011;--mix-fg:#c3c8ee;--mix-bg:#2a2f4c;
  --yes-fg:#a3dcb6;--yes-bg:#183828;--maybe-fg:#f0ca8c;--maybe-bg:#423112;--no-fg:#f3a79f;--no-bg:#48231f;--part-fg:#a9c8ee;--part-bg:#1f3350}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#12171e;--surface:#191f28;--ink:#e3e7ec;--muted:#98a3b1;--line:#2a333e;--line-strong:#3d4856;
  --accent:#6fcabf;--accent-ink:#8fd8cf;--tag:#8fd8cf;--tag-bg:#1a2f2c;--tag-hover:#24413d;--code-bg:#222b35;--target:#40391a;--plate:#ffffff;--focus:#6fcabf;
  --brief:#c9bcf2;--brief-bg:#2a2544;--map:#f0b98a;--map-bg:#3a2617;
  --obs-fg:#b7cbe8;--obs-bg:#213247;--inf-fg:#cfc0f0;--inf-bg:#33294d;--hyp-fg:#f1c78e;--hyp-bg:#443011;--mix-fg:#c3c8ee;--mix-bg:#2a2f4c;
  --yes-fg:#a3dcb6;--yes-bg:#183828;--maybe-fg:#f0ca8c;--maybe-bg:#423112;--no-fg:#f3a79f;--no-bg:#48231f;--part-fg:#a9c8ee;--part-bg:#1f3350}
@media (prefers-reduced-motion: no-preference){html{scroll-behavior:smooth}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;font-size:16px;line-height:1.55;-webkit-text-size-adjust:100%}
.page{max-width:1120px;margin:0 auto;padding-inline:16px;padding-block:0 64px}
code,.mono,a.tag,.tags{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
h1,h2,h3{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;font-weight:600;line-height:1.2;text-wrap:balance}
h1{font-size:clamp(1.5rem,1.15rem + 1.5vw,2rem);margin:0 0 .5rem;letter-spacing:-.01em}
h2{font-size:1.35rem;margin:.1rem 0 .8rem}
h3{font-size:1.02rem;margin:1.6rem 0 .4rem}
p{margin:.7rem 0}
a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:2px}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
code{background:var(--code-bg);padding:.05em .35em;border-radius:3px;font-size:.86em}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 .4rem}
header.head{padding-block:28px 6px;border-bottom:1px solid var(--line);margin-bottom:6px}
.framing{border:1px solid var(--line);background:var(--surface);padding:10px 14px;border-radius:6px;font-size:.93rem}
.framing p{margin:.3rem 0}
.keys{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;font-size:.86rem;color:var(--muted);margin:.6rem 0 0}
.prose p,.prose li,.framing,.brief,figcaption,.status{max-width:76ch}
@media (min-width:1000px){.page{display:grid;grid-template-columns:230px minmax(0,1fr);column-gap:40px;padding-inline:32px}header.head{grid-column:1 / -1}
  .rail{grid-column:1;position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;padding-right:8px;border-right:1px solid var(--line)}.rail summary{display:none}main{grid-column:2;min-width:0;max-width:860px}}
@media (max-width:999px){html{scroll-padding-top:52px}.rail{position:sticky;top:0;z-index:5;background:var(--bg);margin-inline:-16px;padding-inline:16px;border-bottom:1px solid var(--line)}.rail details[open] .toc-body{max-height:min(60vh,520px);overflow:auto;padding-bottom:10px}}
main{min-width:0}
.rail summary{list-style:none;cursor:pointer;font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;font-size:.9rem;padding-block:12px;display:flex;align-items:center}
.rail summary::-webkit-details-marker{display:none}
.rail summary::after{content:"";width:.5em;height:.5em;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);margin-left:auto}
.rail details[open] summary::after{transform:rotate(-135deg)}
.toc-title{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:14px 0 8px}
@media (max-width:999px){.toc-title{display:none}}
.toc-list{list-style:none;margin:0;padding:0}
.toc-list a{display:block;font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.86rem;line-height:1.35;color:var(--ink);text-decoration:none;padding:5px 8px;border-radius:4px}
.toc-list a:hover{background:var(--tag-bg);color:var(--accent-ink)}
.toc-links{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.8rem;color:var(--muted);padding:8px;border-top:1px solid var(--line);margin-top:8px}
section.item{margin-top:2.6rem;padding-top:1.2rem;border-top:2px solid var(--line-strong)}
blockquote.brief{margin:.6rem 0 1rem;padding:10px 14px;border-left:3px solid var(--brief);background:var(--brief-bg);color:var(--ink);font-size:.93rem;line-height:1.5;border-radius:0 6px 6px 0}
blockquote.brief::before{content:"From the brief";display:block;font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--brief);margin-bottom:.3rem}
.status{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap;margin:.4rem 0 .8rem;font-size:.93rem}
.mapnote{color:var(--ink)}
.maplabel{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--map);background:var(--map-bg);padding:1px 6px;border-radius:3px;margin-right:4px}
p.ev{margin:.55rem 0;padding-left:12px;border-left:2px solid var(--line)}
.badge{display:inline-block;font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.7rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;padding:2px 7px;border-radius:3px;white-space:nowrap;vertical-align:middle;line-height:1.4}
.st-yes,.st-cons{color:var(--yes-fg);background:var(--yes-bg)}.st-part,.st-one{color:var(--part-fg);background:var(--part-bg)}.st-no,.st-nt{color:var(--maybe-fg);background:var(--maybe-bg)}.st-fail{color:var(--no-fg);background:var(--no-bg)}
.k-obs{color:var(--obs-fg);background:var(--obs-bg)}.k-inf{color:var(--inf-fg);background:var(--inf-bg)}.k-hyp{color:var(--hyp-fg);background:var(--hyp-bg)}.k-mix{color:var(--mix-fg);background:var(--mix-bg)}
.v-yes{color:var(--yes-fg);background:var(--yes-bg)}.v-maybe{color:var(--maybe-fg);background:var(--maybe-bg)}.v-no{color:var(--no-fg);background:var(--no-bg)}
.tags{font-size:.78em;color:var(--muted);white-space:nowrap}
a.tag{color:var(--tag);background:var(--tag-bg);text-decoration:none;padding:.05em .3em;border-radius:3px;font-weight:500}a.tag:hover{background:var(--tag-hover)}
em.lab{font-style:italic;font-size:.9em;padding:.02em .3em;border-radius:3px}.lab-obs{color:var(--obs-fg);background:var(--obs-bg)}.lab-inf{color:var(--inf-fg);background:var(--inf-bg)}.lab-hyp{color:var(--hyp-fg);background:var(--hyp-bg)}
.scroll{overflow-x:auto;max-width:100%;margin:.8rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem;line-height:1.45;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;font-size:.78rem;color:var(--muted);border-bottom:1px solid var(--line-strong);white-space:nowrap}
table.md td.n{white-space:nowrap}table.md td.n a{text-decoration:none;font-family:"IBM Plex Mono",Menlo,Consolas,monospace}
table.diff{min-width:760px}table.diff th[scope="row"]{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;font-size:.82rem;white-space:nowrap;color:var(--muted)}
table.ledger{min-width:1040px;font-size:.82rem}table.ledger td.id{white-space:nowrap;font-weight:500}table.ledger td.mono{font-size:.78rem}table.ledger td.values{min-width:18em;overflow-wrap:anywhere}
table.ledger td.src code{font-size:.76rem;display:inline-block;margin:1px 2px 1px 0;overflow-wrap:anywhere}table.ledger td.loc{min-width:12em;font-size:.76rem;overflow-wrap:anywhere}table.ledger tr:target{background:var(--target)}
.next-ptr{font-size:.9rem;color:var(--muted);margin-top:1rem}.ptr-label{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600;color:var(--ink)}
figure.fig{margin:1.2rem 0;padding:0}.plate{display:block;background:var(--plate);border:1px solid var(--line);border-radius:4px;padding:6px;cursor:zoom-in}.plate img{display:block;width:100%;height:auto;max-width:100%}
figcaption{font-size:.86rem;line-height:1.5;margin-top:.5rem}.fig-label{font-family:"IBM Plex Sans",Arial,sans-serif;font-weight:600}.fig-open{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.8rem;white-space:nowrap}
dialog.lb{border:0;padding:0;margin:0;width:100vw;height:100vh;max-width:100vw;max-height:100vh;background:transparent;color:#fff}dialog.lb::backdrop{background:rgba(8,12,18,.88)}
.lb-inner{display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:100%;padding:12px;gap:10px}
.lb-inner img{max-width:100%;max-height:calc(100vh - 90px);width:auto;height:auto;object-fit:contain;background:#fff;border-radius:3px}
.lb-bar{display:flex;gap:16px;align-items:center;font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.9rem;color:#e6e9ee}.lb-bar button{all:unset;cursor:pointer;border:1px solid rgba(255,255,255,.4);padding:4px 12px;border-radius:4px}
.note{font-size:.88rem;color:var(--muted)}
footer.about{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.84rem;color:var(--muted)}
@media print{.rail{display:none}.page{display:block}figure.fig{break-inside:avoid}}
"""
JS = r"""
(function(){var toc=document.getElementById('toc'),mq=window.matchMedia('(max-width: 999px)');
function sync(){if(mq.matches){toc.removeAttribute('open');}else{toc.setAttribute('open','');}}sync();if(mq.addEventListener){mq.addEventListener('change',sync);}else{mq.addListener(sync);}
toc.querySelectorAll('a').forEach(function(a){a.addEventListener('click',function(){if(mq.matches){toc.removeAttribute('open');}});});
var lb=document.getElementById('lightbox'),lbImg=lb.querySelector('img'),lbCap=lb.querySelector('.lb-cap'),lbOpen=lb.querySelector('.lb-open');
if(typeof lb.showModal==='function'){document.querySelectorAll('a.zoom').forEach(function(a){a.addEventListener('click',function(e){e.preventDefault();var img=a.querySelector('img');lbImg.src=img.getAttribute('src');lbImg.alt=img.alt;lbOpen.href=a.getAttribute('href');lbCap.textContent='Figure '+a.dataset.num+'. '+img.alt;lb.showModal();});});
lb.addEventListener('click',function(e){if(e.target===lb||e.target.classList.contains('lb-inner')){lb.close();}});lb.querySelector('.lb-close').addEventListener('click',function(){lb.close();});}})();
"""

page = f"""<title>sagp brief questions</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{CSS}</style>
<div class="page">
<header class="head">
<p class="eyebrow">Design brief, 2 September 2026 → sagp study snapshot 2026-09-12-1428 (280/280 runs) → narrative 2026-09-12-1849</p>
<h1>The brief's research questions, answered by the sagp study</h1>
<div class="framing"><p><strong>How to read this page.</strong> Each item below is a question, prediction or check from <code>Research Context/research direction/Design-brief.md</code>, quoted in the purple block (math set in plain text; the build checks that every number matches the brief). Under it, a status badge and a <span class="maplabel">Mapping</span> note are this page's own classification. Everything else is quoted verbatim from the narrative of 2026-09-12-1849 with its <span class="tags">[<a class="tag" href="#L1">L#</a>]</span> tags, and the claim rows are the narrative's section 6 verdicts. Fuller treatments: <a href="{NARR}">the narrative page</a> and <a href="{QPAGE}">the same material by research question</a>.</p>
<p class="keys"><span>Status badges:</span> <span class="badge st-yes">Answered</span> <span class="badge st-cons">Consistent</span> <span class="badge st-part">Partly answered</span> <span class="badge st-one">One point only</span> <span class="badge st-nt">Not testable here</span> <span class="badge st-no">Not run</span> <span class="badge st-fail">Check failed</span></p></div>
</header>
<nav class="rail" aria-label="Contents"><details id="toc" open><summary>Contents</summary><div class="toc-body"><div class="toc-title">Contents</div><ul class="toc-list">{toc}</ul>
<div class="toc-links"><a href="{NARR}">Narrative page</a> · <a href="{QPAGE}">By research question</a></div></div></details></nav>
<main class="prose">
{body}
<section class="item" id="diff"><p class="eyebrow">Design brief §2 to §4 against narrative §1</p><h2>Brief versus study, where they differ</h2>
<p class="note">Left column transcribed from the brief; right column quoted or condensed from the narrative's section 1 and section 7 (the build checks that every number in each cell occurs in its source document).</p>
{diff_table}
</section>
<section class="item" id="ledger"><p class="eyebrow">Claims ledger, subset</p><h2>Ledger rows cited on this page</h2>
<p class="note">{len(cited)} of the 97 rows of <code>claims_ledger.csv</code>; the full ledger is on <a href="{NARR}#s8">the narrative page</a>. Hover a tag above to read a row without scrolling.</p>
{ledger_html}
</section>
<footer class="about"><p>Built from <code>Design-brief.md</code> (repo, read-only), <code>NARRATIVE.md</code>, <code>captions.md</code> and <code>claims_ledger.csv</code> of <code>sagp_analysis/narrative/2026-09-12-1849/</code>, and three of the six figures in <code>sagp_analysis/latest/figures/</code>. The brief items and the status classification are this page's framing; the study says only what its quoted sentences say.</p></footer>
</main></div>
<dialog class="lb" id="lightbox" aria-label="Enlarged figure"><div class="lb-inner"><img src="" alt=""><div class="lb-bar"><span class="lb-cap"></span><a class="lb-open" href="#" style="color:inherit">Open the PNG</a><button type="button" class="lb-close">Close</button></div></div></dialog>
<script>{JS}</script>
"""
OUT.write_text(page)
print("wrote", OUT, len(page) // 1024, "KB; sections:", [s[0] for s in sections])
print("tags cited:", len(cited), "| claims rows used:", sorted(claims_used), "| not used:", sorted(set(range(1, 30)) - claims_used))
print("figures:", sorted(set(re.findall(r'<img src="figures/([^"]+)"', page))))
