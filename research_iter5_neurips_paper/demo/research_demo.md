# NeurIPS Paper

## Summary

Complete NeurIPS-format manuscript (~7942 words) for 'Flickering Before Failing: Ecological Early Warning Signals Predict LLM Reasoning Collapse' with verified BibTeX bibliography (38 entries). Follows 6-beat honest discovery narrative: sharp boundaries exist, ecology predicts flickering, fold bifurcation hypothesized, flickering confirmed but scaling failed (α=-0.0005 vs -0.5), mixture model explains observations, CSD classifier works regardless (LOPO F1=0.814, AUROC=0.897, 16.38% over best baseline, zero API cost). All sections: Abstract, Introduction, Background, Methods, Results (Table 1), Theoretical Analysis, Related Work (12+ papers), Discussion, Limitations (8 items), Broader Impact, Conclusion. Bibliography verified via Semantic Scholar. Freshness scan (Jan 2025–Mar 2026) found no direct competitor in ecological-CSD-for-LLMs space.

## Research Findings

Complete NeurIPS-format manuscript and verified bibliography generated for "Flickering Before Failing: Ecological Early Warning Signals Predict LLM Reasoning Collapse."

**Manuscript Structure (~7942 words):**
- **Abstract** (~200 words): Summarizes CSD approach, key results (F1=0.814, AUROC=0.897), fold bifurcation failure, and mixture-switching model
- **Introduction** (~1200 words): Motivates problem, reviews existing UQ methods (Gnosis, SPUQ), introduces ecological CSD analogy, states 6-beat narrative arc, lists 3 contributions [1, 2, 3]
- **Background** (~900 words): CSD in ecology [1, 14], flickering as EWS [4, 12], fold bifurcation formalism [10], formal hypotheses SC1-SC3
- **Methods** (~1500 words): 4 task families (arithmetic, graph coloring, syllogistic logic, multi-hop), 4 models, 6 CSD indicators, classifier design with LOPO/LOMO/LOTO protocols, variance scaling analysis
- **Results** (~1600 words): Accuracy profiles, SC1 flickering analysis (lead times 13.7/11.7), SC2 scaling failure (α≈-0.0005, R²=0.066), SC3 classifier comparison (Table 1), temperature ablation, negative controls [6, 31]
- **Theoretical Analysis** (~750 words): Why fold fails, mixture-switching model (Eq. 1), cusp catastrophe connection, DDM mechanism, three-layer theoretical framework [10, 13, 14]
- **Related Work** (~900 words): Gnosis [3], SPUQ [16], ProSA [17], self-consistency [15], logical phase transitions [19], mode structure [22, 25], ecological CSD [1, 4, 6], UQ surveys [26]
- **Discussion** (~800 words): What transferred/didn't, ecology's own EWS failures, 4 practical deployment applications, cross-task gap analysis, test-time compute connection
- **Limitations** (~400 words): 8 concrete items including scaling failure, SC1 partial, LOTO gap, autocorrelation absence
- **Broader Impact** (~150 words): Safety toolkit contribution, economic accessibility, potential misuse
- **Conclusion** (~450 words): Summary of contributions, honest negative result context, unique design-space position, 4 future directions

**Bibliography:** 38 BibTeX entries verified via Semantic Scholar with DOIs [7-38]

**Freshness Scan (Jan 2025–Mar 2026):** No direct competitor found in ecological-CSD-for-LLMs intersection. Closest parallel: PNAS 2026 "Interpretable early warnings using ML in online game-experiment" (different domain) [39]. Also identified: "UQ in LLM Agents" framework paper (Feb 2026) as complementary [40].

**Key Experimental Numbers:** LOPO F1=0.814, AUROC=0.897; LOMO F1=0.798, AUROC=0.855; LOTO F1=0.355; 16.38% improvement over best baseline; fold α≈-0.0005 vs predicted -0.5; R²=0.066; bifurcation AIC=-117.23 vs Gaussian AIC=-132.03; zero additional API cost vs ~$360K/month for SPUQ.

## Sources

[1] [Scheffer et al. (2009) - Early-warning signals for critical transitions, Nature 461](https://www.semanticscholar.org/paper/Early-warning-signals-for-critical-transitions-Scheffer-Bascompte/d7aa74690ce0bb9317236b7035df4042419509c7) — Foundational CSD review. DOI verified via Semantic Scholar.

[2] [Scheffer et al. (2012) - Anticipating Critical Transitions, Science 338](https://www.science.org/doi/abs/10.1126/science.1225244) — Expanded CSD framework including flickering mechanism.

[3] [Ghasemabadi & Niu (2025) - Gnosis: Self-Awareness via Internal Circuits](https://arxiv.org/abs/2512.20578) — White-box LLM failure prediction, AUROC 0.95-0.96, key comparison baseline.

[4] [Wang et al. (2012) - Flickering gives early warning signals, Nature 492](https://www.nature.com/articles/nature11655) — Empirical flickering in lake system, 20-year lead time before transition.

[5] [Dakos et al. (2012) - Methods for Detecting Early Warnings, PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010) — Definitive CSD computational methodology for ecological time series.

[6] [O'Brien et al. (2023) - EWS limited applicability, Nature Communications](https://www.nature.com/articles/s41467-023-43744-8) — CSD indicators no better than chance on empirical lake data. Contextualizes our partial success.

[7] [Hartigan & Hartigan (1985) - Dip Test, Annals of Statistics 13](https://www.semanticscholar.org/paper/The-Dip-Test-of-Unimodality-Hartigan-Hartigan/9ea6542eba9f632308e26a50e2dcf47fa77dd432) — Nonparametric unimodality test used as key bimodality indicator.

[8] [Ashman, Bird & Zepf (1994) - Bimodality in astronomical datasets, AJ 108](https://ui.adsabs.harvard.edu/abs/1994AJ....108.2348A) — Ashman D statistic for bimodality detection.

[9] [Freeman & Dale (2013) - Assessing bimodality, Behavior Research Methods](https://www.semanticscholar.org/paper/Assessing-bimodality-to-detect-the-presence-of-a-Freeman-Dale/ec32aa05dba30a0ef4272f1b3e4ef0037a9878ee) — Bimodality coefficient methodology and comparison of detection methods.

[10] [Kuehn (2011) - Mathematical framework for critical transitions, Physica D 240](https://www.semanticscholar.org/paper/A-mathematical-framework-for-critical-transitions:-Kuehn/c722438db3bff6931b2788f44ff05bb3a7fb7ff4) — Variance scaling law derivation via center manifold reduction.

[11] [Dakos et al. (2012) - Robustness of variance and autocorrelation, Ecology](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/11-0889.1) — Variance can decrease near transitions under specific conditions.

[12] [Dakos et al. (2013) - Flickering as early warning, Theoretical Ecology 6](https://link.springer.com/article/10.1007/s12080-013-0186-4) — Formal theory of flickering via Kramers escape formula.

[13] [Chen et al. (2022) - Stochastic cusp catastrophe model](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041743/) — Cusp SDE and Cardan discriminant for bimodality.

[14] [Ratcliff & McKoon (2008) - Diffusion Decision Model, Neural Computation 20](https://www.semanticscholar.org/paper/The-Diffusion-Decision-Model-Theory-and-Data-for-Ratcliff-McKoon/1dccc2d2207d25b4d7bde33d74f33b9ec97f0eaf) — DDM providing computational mechanism for mixture-switching.

[15] [Wang et al. (2023) - Self-Consistency, ICLR 2023](https://www.semanticscholar.org/paper/Self-Consistency-Improves-Chain-of-Thought-in-Wang-Wei/5f19ae1135a9500940978104ec15a5b8751bc7d2) — Majority voting and disagreement rate baseline.

[16] [Gao et al. (2024) - SPUQ, EACL 2024](https://www.semanticscholar.org/paper/SPUQ-Perturbation-Based-Uncertainty-Quantification-Gao-Zhang/9f02a3fa885aebaf322ea8e4475939495dea70f7) — Perturbation-based UQ requiring 6x API cost, key cost comparison.

[17] [Zhuo et al. (2024) - ProSA, EMNLP 2024 Findings](https://aclanthology.org/2024.findings-emnlp.108/) — Prompt sensitivity measurement, different from failure prediction.

[18] [Zhang et al. (2026) - Logical Phase Transitions](https://arxiv.org/abs/2601.02902) — Sharp LLM reasoning collapse at critical complexity; no early warning proposed.

[19] [Chen et al. (2024) - Reasoning Boundary Framework, NeurIPS 2024](https://arxiv.org/abs/2410.05695) — Static capability bounds, not runtime detection.

[20] [Arnold & Lorch (2025) - Behavioral Phase Transitions in LLMs](https://arxiv.org/abs/2508.20015) — Training-time retrospective analysis of phase transitions.

[21] [Wu et al. (2025) - Mode-Conditioning for Test-Time Scaling](https://arxiv.org/abs/2512.01127) — ModC framework, complementary use of mode structure.

[22] [Hazra et al. (2025) - LLMs on 3-SAT Phase Transition, COLM 2025](https://arxiv.org/abs/2504.03930) — LLM reasoning at computational hardness transition.

[23] [Ding et al. (2025) - BEST-Route, ICML 2025](https://arxiv.org/abs/2506.22716) — Adaptive LLM routing, CSD could serve as difficulty proxy.

[24] [Zhang et al. (2025) - Verbalized Sampling](https://arxiv.org/abs/2510.01171) — Mode collapse mitigation, complementary to our diagnostic.

[25] [Huang et al. (2025) - UQ Survey, ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3744238) — Comprehensive LLM uncertainty quantification taxonomy.

[26] [Liu et al. (2025) - Superposition, NeurIPS 2025 Best Paper Runner-Up](https://arxiv.org/abs/2505.10465) — Ecology/physics-to-ML cross-domain transfer precedent.

[27] [Silverman (1981) - KDE multimodality, JRSS-B 43](https://www.semanticscholar.org/paper/Using-Kernel-Density-Estimates-to-Investigate-Silverman/42271fed96cf15ec512d8a4bd03002f041ca07ff) — Kernel density estimation for multimodality testing.

[28] [Reimers & Gurevych (2019) - Sentence-BERT, EMNLP](https://www.semanticscholar.org/paper/Sentence-BERT-Reimers-Gurevych/93d63ec754f29fa22572615320afe0521f7ec66d) — Sentence embedding model used for response encoding.

[29] [Carpenter & Brock (2006) - Rising variance, Ecology Letters 9](https://www.semanticscholar.org/paper/Rising-variance-a-leading-indicator-of-ecological-Carpenter-Brock/66ece56fad0abf74bc3d4299caff0399fc6ee926) — Variance as leading indicator of ecological transition.

[30] [Boettiger & Hastings (2012) - Limits to detection, J R Soc Interface](https://pmc.ncbi.nlm.nih.gov/articles/PMC3427498/) — Warning about severe error rates in CSD detection.

[31] [Chen, Ghadami & Epureanu (2022) - Kendall's tau guide](https://pmc.ncbi.nlm.nih.gov/articles/PMC9326300/) — Practical guide for trend significance testing in EWS.

[32] [Dean & Dunsmuir (2016) - Cross-correlation dangers](https://link.springer.com/article/10.3758/s13428-015-0611-2) — Autocorrelation analysis methodology.

[33] [Kang et al. (2019) - HDS+BC combined bimodality method](https://onlinelibrary.wiley.com/doi/10.1155/2019/4819475) — Combined bimodality detection methodology.

[34] [Yang et al. (2024) - Verbalized Confidence Scores for LLMs](https://arxiv.org/abs/2412.14737) — Shows verbalized confidence is unreliable, motivates distributional approach.

[35] [Pres et al. (2025) - Phase Transitions in LLM Outputs, ICLR 2025](https://openreview.net/forum?id=dq3keisMjT) — Statistical physics methods for LLM output distribution analysis.

[36] [Belem et al. (2024) - Cycles of Thought](https://arxiv.org/abs/2406.03441) — Uncertainty via explanation stability, AUROC 0.852.

[37] [Sam et al. (2025) - Predicting Black-box LLM Performance](https://arxiv.org/abs/2501.01558) — Follow-up queries with token probabilities for performance prediction.

[38] [Interpretable early warnings using ML (PNAS, Jan 2026)](https://www.pnas.org/doi/10.1073/pnas.2503493122) — Freshness scan: ML-based EWS in online games, closest methodological parallel.

[39] [UQ in LLM Agents (Feb 2026)](https://arxiv.org/html/2602.05073v2) — Freshness scan: Framework for agent-level uncertainty quantification.

[40] [Bury et al. (2021) - Deep learning for EWS, PNAS](https://www.pnas.org/doi/10.1073/pnas.2106140118) — Deep learning for ecological early warning signals across domains.

## Follow-up Questions

- The PNAS 2026 paper on 'Interpretable early warnings using ML in an online game-experiment' applies ML-based EWS in a social system with gradient-boosted trees -- how does their approach compare to our CSD-LogReg on feature importance and generalization, and should we cite it as a methodological parallel?
- Can the cross-task generalization gap (LOTO F1=0.355) be addressed by normalizing CSD features relative to task-specific embedding geometry (e.g., z-scoring within task family), and would a meta-learning approach across diverse task families improve transfer?
- How would frontier reasoning models (o3, Claude Opus 4.6, DeepSeek-R1) -- which use extended chain-of-thought with potentially different mode structures -- respond to the same CSD analysis, and do their longer reasoning chains provide richer within-chain autocorrelation signals that were absent (0.0) in our mid-tier models?

---
*Generated by AI Inventor Pipeline*
