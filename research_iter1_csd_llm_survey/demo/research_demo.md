# CSD-LLM Survey

## Summary

Comprehensive literature survey covering six blocks: (1) ecological CSD indicator formulas (variance, lag-1 autocorrelation, skewness) from Scheffer/Dakos with ewstools implementation; (2) fold bifurcation variance scaling Var ~ (d*-d)^{-1/2} via OU linearization; (3) bimodality detection methods (Hartigan dip, Z-Dip, BC, Ashman D, silhouette, Silverman) with N=50 power analysis; (4) LLM uncertainty baselines (SPUQ, ProSA, Gnosis, self-consistency) with cost profiles; (5) embedding model selection (all-MiniLM-L6-v2 primary, Qwen3-Embedding-0.6B secondary) and within-chain confidence extraction; (6) related work framing and leading-indicator statistical tests. Provides unified methodology checklist for building an early-warning system for LLM reasoning collapse.

## Research Findings

## Comprehensive Survey: Ecological CSD Signals, Fold Bifurcation Theory, Bimodality Detection, and LLM Uncertainty Methods

### Executive Summary

This survey synthesizes six domains of literature to provide a concrete methodological foundation for building an early-warning system that detects approaching reasoning collapse in large language models. The core hypothesis draws on ecological critical slowing down (CSD) theory: as task difficulty approaches an LLM's capability boundary, statistical signatures analogous to those before ecological tipping points should emerge in sampled response distributions. The survey establishes: (a) precise CSD indicator definitions with validated `ewstools` implementations [3, 4]; (b) the fold bifurcation variance scaling law Var ~ (d* - d)^{-1/2} from OU linearization [7, 8, 9]; (c) a recommended bimodality test battery (Z-Dip + silhouette + BC, flag when ≥2/3 agree) validated at N=50 [10, 11, 12]; (d) structured comparison of four LLM uncertainty baselines [14, 15, 16, 17]; (e) concrete embedding pipeline (all-MiniLM-L6-v2 primary, PC1 projection) [20, 21]; and (f) statistical protocols for testing leading indicators [25, 26, 27].

---

### BLOCK 1: CSD Early Warning Signals

Scheffer et al. (2009) established that systems approaching fold bifurcations exhibit CSD: the dominant eigenvalue approaches zero, causing increasingly slow recovery from perturbations [1]. This produces three generic indicators: rising variance, rising lag-1 autocorrelation, and rising skewness [1]. Scheffer et al. (2012) expanded this to include flickering — noise-induced jumping between alternative states where frequency distributions approximate basin shapes, producing detectable bimodality [2].

Dakos et al. (2012) provided the definitive computational methodology [3]: rolling window variance/autocorrelation (50% windows), Gaussian detrending (10% bandwidth), Kendall tau significance via 1000 ARMA surrogates (P < 0.1), spectral reddening analysis, BDS test, and conditional heteroskedasticity. The `ewstools` Python package (Bury, 2023) implements these with `TimeSeries.compute_var(0.25)`, `.compute_auto(0.25, lag=1)`, `.compute_skew(0.25)`, `.compute_ktau()`, plus deep learning bifurcation classifiers [4, 5].

### BLOCK 2: Fold Bifurcation Variance Scaling

The stochastic normal form dx = (μ + x²)dt + σdW has stable fixed point x* = -√(-μ) [7]. Linearizing yields eigenvalue λ = -2√(-μ), and the OU approximation gives stationary variance Var = σ²/(2|λ|) = σ²/(4√(-μ)) [7, 9]. Therefore **Var ~ (d* - d)^{-1/2}** with exponent α = -0.5, universal by center manifold reduction [7, 8]. For discrete-time maps (relevant to LLMs), the AR(1) coefficient approaches unity and the same scaling applies [28]. The mapping: state variable = response quality embedding, control parameter = task difficulty, d* = capability boundary, noise = stochastic sampling.

### BLOCK 3: Bimodality Detection

**Hartigan's Dip Test** [10]: max|F_emp - F_uni|; Python `diptest.diptest(x)` returns (stat, p-value). Limited by sample-size dependent critical values.

**Z-Dip** [11]: Z = (Dip - μ_N)/σ_N; at N=50, weak bimodal Z=7.16 (100% detection), strong bimodal Z=9.64 (100%); universal threshold z=1.975.

**Bimodality Coefficient** [12]: BC = (skew² + 1)/kurtosis_adj; threshold > 5/9 ≈ 0.555. Limitation: sensitive to skewness (false positives on skewed unimodal).

**Ashman's D** [13]: D = √2·|μ₁-μ₂|/√(σ₁²+σ₂²) from fitted 2-component GMM; D > 2 = clean separation.

**Recommendation:** Use ≥3 tests in parallel (dip/Z-Dip + silhouette + BC); flag bimodality when ≥2/3 agree [29, 30].

### BLOCK 4: LLM Uncertainty Baselines

**Self-consistency** [17]: Sample N reasoning paths, majority vote; disagreement = 1 - max_fraction. Zero-cost (reuses voting samples). +17.9% on GSM8K.

**SPUQ** [14]: Perturbation-based (paraphrasing, dummy tokens, system messages); 50% ECE reduction; 6x API cost. Black-box.

**ProSA** [15]: PromptSensiScore = avg|Y(Pi)-Y(Pj)| across prompt pairs. Higher confidence = more robust. Black-box.

**Gnosis** [16]: White-box probe from hidden states + attention → 5M param classifier. AUROC 0.95 (math). Upper-bound baseline only.

### BLOCK 5: Embedding & Confidence

**Primary:** all-MiniLM-L6-v2 (22M params, 384 dims, 14.7ms/1K tokens) — fast enough for 16K+ embeddings [20]. **Secondary:** Qwen3-Embedding-0.6B (0.6B params, 1024 dims, SOTA for size) [21]. Pipeline: embed N=50 responses → PC1 projection for 1D tests → k-means(k=2) + silhouette for high-dimensional test. Within-chain: extract step-by-step logprobs, compute lag-1 autocorrelation [22, 23, 24].

### BLOCK 6: Related Work & Leading Indicators

**Mode-Conditioning** [18]: Defines reasoning modes via gradient clustering; diversity collapse detected. Our distinction: we use mode structure as diagnostic, not optimization target.

**Reasoning Boundary Framework** [6]: Static capability limits via harmonic mean combination law. Our distinction: we detect dynamic approach to those limits prospectively.

**Pres et al.** [32]: Phase transitions in training via order parameters and f-divergence. Our distinction: we analyze inference dynamics, not training dynamics.

**Leading indicator test protocol:** (1) Kendall tau with modified Mann-Kendall for serial correlation [25]; (2) Cross-correlation with ARIMA prewhitening [27]; (3) CSD indicator significant at d₁ where Acc(d₁) > 80% proves leading signal; lead time = d* - d₁ [25, 26].

## Sources

[1] [Scheffer et al. (2009) - Early-warning signals for critical transitions, Nature](https://www.nature.com/articles/nature08227) — Foundational review: CSD theory with rising variance, autocorrelation, skewness as generic early warning signals near fold bifurcations.

[2] [Scheffer et al. (2012) - Anticipating Critical Transitions, Science](https://www.science.org/doi/abs/10.1126/science.1225244) — Expanded CSD to include flickering where frequency distributions approximate basin shapes, plus network-level indicators.

[3] [Dakos et al. (2012) - Methods for Detecting Early Warnings, PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010) — Definitive computational methodology: rolling windows, Kendall tau with ARMA surrogates, spectral reddening, BDS test.

[4] [ewstools Python package (Bury 2023, JOSS)](https://github.com/ThomasMBury/ewstools) — Python TimeSeries class with compute_var/auto/skew/ktau methods and deep learning bifurcation classifiers.

[5] [ewstools API Documentation](https://ewstools.readthedocs.io/en/latest/ewstools.html) — Full API: detrend(Gaussian/Lowess), compute methods with 0.25 default rolling window, classifier methods.

[6] [Chen et al. (2024) - Reasoning Boundary Framework, NeurIPS 2024](https://arxiv.org/html/2410.05695v2) — Reasoning Boundary definition, three categories (CFRB/PFRB/CIRB), weighted harmonic mean combination law.

[7] [Kuehn (2011) - Mathematical Framework for Critical Transitions](https://arxiv.org/pdf/1101.2908) — Formal derivation of Var ~ (d*-d)^{-1/2} scaling via center manifold + OU approximation. Universality proof.

[8] [Stochastic Early Warning Signals and Multi-Dimensional Fold Bifurcations (PIK)](https://publications.pik-potsdam.de/rest/items/item_30137_8/component/file_33153/content) — Center manifold reduction to 1D near fold; noise in other dimensions may interfere with CSD indicators.

[9] [Ornstein-Uhlenbeck Process - PlanetMath](https://planetmath.org/ornsteinuhlenbeckprocess) — OU SDE, stationary variance σ²/(2κ), conditional variance formula. Key for CSD variance divergence.

[10] [diptest - Python Hartigan's Dip Test](https://github.com/RUrlus/diptest) — diptest.diptest(x) returns (stat, p-value); critical table interpolation or bootstrap. Wheels for Py 3.8-3.12.

[11] [Z-Dip: Validated Generalization of the Dip Test (2025)](https://arxiv.org/html/2511.01705v1) — Z-Dip standardization for sample-size independence. At N=50: 100% detection for bimodal signals. Threshold z=1.975.

[12] [Bimodality Coefficient - PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3791391/) — BC formula, 5/9 threshold. Sensitive to skewness with documented false positive/negative cases.

[13] [Ashman's D Statistic (1994)](https://rdrr.io/cran/modes/man/Ashmans_D.html) — D = √2|μ₁-μ₂|/√(σ₁²+σ₂²) for GMM components. D > 2 = clean separation.

[14] [SPUQ: Perturbation-Based Uncertainty for LLMs (EACL 2024)](https://arxiv.org/html/2403.02509v1) — Black-box perturbation method. 50% ECE reduction. 6x API cost. Three perturbation types.

[15] [ProSA: Prompt Sensitivity Analysis (EMNLP 2024)](https://arxiv.org/abs/2410.12405) — PromptSensiScore metric. Higher confidence = more robust. Larger models more robust.

[16] [Gnosis: LLM Self-Awareness via Internal Circuits](https://arxiv.org/html/2512.20578v1) — White-box 5M-param probe. AUROC 0.95 math. Requires model internals. Upper-bound baseline.

[17] [Wang et al. (2023) - Self-Consistency, ICLR 2023](https://arxiv.org/abs/2203.11171) — Sample N paths, majority vote. +17.9% GSM8K. Disagreement as zero-cost uncertainty.

[18] [Mode-Conditioning for Test-Time Scaling](https://arxiv.org/html/2512.01127v1) — Gradient clustering defines modes. 4x efficiency. 98.7% F1 recovering teacher identity.

[19] [Silverman's Multimodality Test - Python](https://github.com/lberaldoesilva/silverman-test) — KDE-based test for k modes. No distributional assumptions. Bootstrap p-values.

[20] [Best Open-Source Embedding Models 2026](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) — all-MiniLM-L6-v2: 22M params, 384 dims, 14.7ms/1K tokens. Fast with moderate accuracy.

[21] [Qwen3-Embedding-0.6B - Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — 0.6B params, configurable dims, SOTA for size, Apache 2.0, sentence-transformers compatible.

[22] [Yang et al. (2024) - Verbalized Confidence Scores for LLMs](https://arxiv.org/pdf/2412.14737) — Reliability depends on model capacity and prompt design. 10 datasets, 11 LLMs tested.

[23] [How do LLMs Compute Verbal Confidence? (2025)](https://arxiv.org/html/2603.17839) — Verbal confidence via cached retrieval, not just-in-time. Doesn't merely reflect logprobs.

[24] [Tanneru et al. (2024) - Uncertainty in CoT Explanations](https://proceedings.mlr.press/v238/harsha-tanneru24a/harsha-tanneru24a.pdf) — LLMs exhibit high overconfidence in verbalized uncertainty. Near 100% with little std deviation.

[25] [Ghadami & Epureanu (2022) - Kendall's tau for Critical Transitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC9326300/) — Modified Mann-Kendall test for EWS. Handles serial correlation from sliding windows efficiently.

[26] [Evaluating EWIs in Natural Aquatic Ecosystems - PNAS](https://www.pnas.org/doi/10.1073/pnas.1608242113) — EWIs preceded transitions by years in some cases but failed in many. Composite indicators recommended.

[27] [Cross-Correlation Dangers in Time Series - Behavior Research Methods](https://link.springer.com/article/10.3758/s13428-015-0611-2) — Prewhitening essential. ARIMA filtering recommended. Granger causality as complement.

[28] [Bury et al. (2023) - Discrete-Time Bifurcations with Deep Learning](https://arxiv.org/html/2303.09669) — At fold, Jacobian eigenvalue → +1, AR(1) → unity, same variance scaling exponent.

[29] [Kang et al. (2019) - Combined HDS+BC Method](https://onlinelibrary.wiley.com/doi/10.1155/2019/4819475) — Combined approach more accurate/robust than either alone at small N including N=50.

[30] [Freeman & Dale (2013) - Bimodality Detection Comparison](https://link.springer.com/article/10.3758/s13428-012-0225-x) — Compared BC, dip, AIC. No single test dominates; depends on separation, proportion, skewness.

[31] [Cycles of Thought: LLM Confidence via Explanation Stability (2024)](https://arxiv.org/html/2406.03441v1) — Entailment-weighted marginalization over explanations. AUROC 0.852 GPT-4-turbo.

[32] [Pres et al. (2025) - Phase Transitions in LLMs](https://arxiv.org/html/2508.20015v1) — Order parameters with f-divergence for training phase transitions. Gradient norm as early warning.

## Follow-up Questions

- What is the minimum mode separation detectable by the dip test at N=50, and how does this translate to minimum cosine distance between response embedding clusters for practical LLM experiments?
- Do token-level logprobs correlate with verbalized confidence for within-chain autocorrelation measurement, or are they measuring fundamentally different internal processes as suggested by recent mechanistic interpretability work?
- Should spectral reddening (power spectrum shift to lower frequencies) be added as a fourth CSD indicator alongside variance, autocorrelation, and skewness, and does it provide independent predictive power in the LLM reasoning context?

---
*Generated by AI Inventor Pipeline*
