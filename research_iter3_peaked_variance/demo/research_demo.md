# Peaked Variance

## Summary

Comprehensive research identifying five theoretical frameworks that predict non-monotonic (peaked) variance near critical transitions, synthesized into a recommended three-layer model to replace the failed fold bifurcation prediction for LLM capability boundaries. Layer 1: Mixture model with between-component variance p(1-p)||Delta_mu||^2 peaking at 50% accuracy. Layer 2: Stochastic cusp catastrophe traversal where asymmetry parameter alpha(d) sweeps through the bimodal region. Layer 3: Finite-size scaling corrections for N=50 samples. Supported by ecological precedent (Dakos et al. 2012 showing variance can decrease; Scheffer's flickering theory) and decision-theoretic mechanism (drift-diffusion model with P(correct)=1/(1+exp(-2va/s^2))). Includes 27 sources and all key equations needed for the paper revision.

## Research Findings

The fold bifurcation model predicted Var proportional to (d*-d)^{-1/2}, a monotonic divergence as difficulty approaches d*. Empirical data contradicts this: variance PEAKS near d* and then DECREASES. Five theoretical frameworks explain this non-monotonic behavior, and a three-layer synthesis provides the recommended replacement model.

## Why the Fold Bifurcation Model Failed

The TIPMOC framework formalizes the fold prediction as V(u) = a(u_c - u)^{-0.5} + b for saddle-node bifurcations [1]. This model assumes unidirectional approach to the tipping point and is undefined beyond it [1]. LLM difficulty sweeps traverse THROUGH the transition (from easy to beyond-capability), violating this fundamental assumption.

## Framework 1: Mixture Model / Law of Total Variance (RECOMMENDED PRIMARY MODEL)

The law of total variance states Var(X) = E[Var(X|Theta)] + Var[E(X|Theta)] [2, 3]. For a two-component mixture of correct and incorrect LLM responses:

Var_total(d) = p(d)*sigma^2_c + (1-p(d))*sigma^2_i + p(d)*(1-p(d))*(mu_c - mu_i)^2

The between-component term p(1-p)*(Delta_mu)^2 is a parabola in p, MAXIMIZED at p = 0.5, with maximum value (Delta_mu)^2/4 [2, 3]. Since p(d) = accuracy(d) decreases from ~1 to ~0 across the capability boundary, variance traces an inverted-U that peaks at d* where accuracy is approximately 50%. For binary scoring, this reduces to Bernoulli variance p(1-p), confirmed in LLM evaluation contexts [4]. This is the simplest and most direct explanation for peaked variance.

## Framework 2: Stochastic Cusp Catastrophe Traversal (MECHANISTIC GROUNDING)

The stochastic cusp catastrophe is described by dX = (alpha + beta*X - X^3)dt + sigma*dW [5, 6]. The stationary density is pi(x) = C*exp{(2/sigma^2)(alpha*x + beta*x^2/2 - x^4/4)} [5]. Bimodality occurs when Cardan's discriminant Delta = 27*alpha^2 - 4*beta^3 is less than 0 [5, 6, 7]. Task difficulty maps to the asymmetry parameter alpha(d), which sweeps from positive (correct-mode dominates) through zero (symmetric bistability, maximum flickering) to negative (incorrect-mode dominates) [5, 6]. This traversal THROUGH the cusp region naturally produces peaked variance. Gilmore's catastrophe flags explicitly include "anomalous variance" as a recognized phenomenon near the bifurcation set [6, 8].

## Framework 3: Kramers Escape / Double-Well Switching

In a double-well potential, the Kramers escape rate r is proportional to exp(-2*Delta_V/sigma^2) [9, 10]. High barrier (easy tasks) traps the system in the correct well (low variance); low barrier (boundary tasks) produces frequent switching/flickering (high variance); no barrier (beyond capability) drops the system into the incorrect well (low variance). This connects to stochastic resonance where variance peaks at moderate barrier heights [11].

## Framework 4: Drift-Diffusion Model (DECISION-THEORETIC MECHANISM)

The DDM describes evidence accumulation as dX = v*dt + s*dW [12, 13]. The accuracy formula P(correct) = 1/(1+exp(-2va/s^2)) is a logistic sigmoid derived via the exponential martingale and optional stopping theorem [14]. At high drift rate v (easy tasks), almost all trajectories reach the correct boundary (low variance). At v near 0 (boundary), trajectories hit either boundary with approximately equal probability (maximum variance). At negative v (beyond capability), trajectories reach incorrect boundary (low variance). The EZ-diffusion equations also give MRT = a/v and VRT = a^2/v^3 [14]. The Leaky Competing Accumulator adds neural plausibility through mutual inhibition and attractor dynamics [15].

## Framework 5: Finite-Size Scaling (CORRECTION LAYER)

In finite systems, divergences are replaced by finite rounded peaks [16, 17]. Pseudocritical point shift: T*(L) - T_c ~ L^{-1/nu}. Peak height: chi_max ~ L^{gamma/nu}. Neural network output variance shows the same FSS behavior with peak width sigma(L) ~ 1/L^{1/nu} [17]. For N=50 LLM samples, variance peaks are finite and shifted from the true capability boundary.

## Ecological Precedent: Non-Monotonic Variance Is Known

Dakos et al. (2012) showed ANALYTICALLY that variance can decrease near transitions under three conditions: (a) reduced sensitivity to environmental factors near threshold, (b) CSD filtering of high-frequency fluctuations, (c) systematic underestimation from limited data [18, 19]. Autocorrelation ALWAYS increases, making it more robust [18]. Scheffer et al. (2012) distinguished CSD-based from flickering-based warnings, with flickering providing more direct evidence of bistability [20]. Dakos et al. (2013) showed flickering occurs in the bistable region under strong noise, where CSD may not be relevant [21]. Wang et al. (2012) demonstrated empirical flickering detection in lake ecosystems with sparse data [22].

## Additional Context: LLM Phase Transitions and Rate Effects

Sun &amp; Haghighat (2025) mapped Transformers to an O(N) model, identifying phase transitions at critical temperature Tc~1.2 and parameter count ~7B [23]. Bonciolini et al. (2018) showed finite ramping rates cause rate-dependent delays in tipping points [24]. Nair et al. (2016) found monotonic variance increase before subcritical Hopf transitions in thermoacoustic systems [25]. Non-equilibrium EWS research suggests multi-dimensional indicators beyond 1D CSD for complex systems [26].

## Recommended Three-Layer Model

**Layer 1 (Phenomenological):** Mixture model: Var(d) = p(d)*(1-p(d))*||Delta_mu||^2 + constant. Peaks at accuracy = 50%.

**Layer 2 (Dynamical Systems):** Cusp catastrophe: dX = (alpha(d) + beta*X - X^3)dt + sigma*dW with alpha(d) sweeping through bimodal region.

**Layer 3 (Finite-Size):** N=50 samples round sharp features; peak height and width scale with N.

## Revised Quantitative Predictions
1. Variance follows inverted-U: Var(d) ~ p(d)*(1-p(d))*||Delta_mu||^2
2. Variance peaks at d* where accuracy ~ 50%
3. Bimodality (flickering) is the PRIMARY leading indicator, detectable before variance peaks
4. Variance profile approximately symmetric around d* if accuracy follows sigmoid
5. Autocorrelation increases monotonically even when variance peaks
6. Peak variance approximately (Delta_mu)^2/4

**Confidence level: HIGH** for the mixture model as primary explanation (well-established mathematics, directly testable). MEDIUM for the cusp catastrophe as mechanistic grounding (requires parameter estimation to validate). The mixture model alone is sufficient for the paper's quantitative predictions; the cusp catastrophe adds theoretical depth but is not strictly necessary.

## Sources

[1] [TIPMOC: Detecting and forecasting tipping points from sample variance alone](https://arxiv.org/abs/2602.10817) — Formalizes power-law variance divergence V(u)=a(uc-u)^{-gamma}+b near bifurcations; gamma=0.5 for saddle-node; assumes unidirectional approach to tipping point.

[2] [Law of total variance - Wikipedia](https://en.wikipedia.org/wiki/Law_of_total_variance) — States Var(X)=E[Var(X|Theta)]+Var[E(X|Theta)]; foundation for mixture variance decomposition into within- and between-component terms.

[3] [The variance of a mixture](https://statisticalmodeling.wordpress.com/2011/06/16/the-variance-of-a-mixture/) — Derives between-component variance p(1-p)*w^2 for two-component mixture; proves maximized at equal proportions p=0.5.

[4] [Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores](https://arxiv.org/abs/2601.13885) — Leverages Bernoulli variance p(1-p) peaking at p=0.5 for LLM evaluation design; confirms inverted-U variance shape.

[5] [Stochastic cusp catastrophe model and its Bayesian computations](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041743/) — Provides cusp SDE dX=(alpha+beta*X-X^3)dt+sigma*dW, stationary density, Cardan discriminant Delta=27alpha^2-4beta^3 for bimodality classification.

[6] [Cusp Catastrophe Regression and Its Application in Public Health](https://pmc.ncbi.nlm.nih.gov/articles/PMC5664721/) — Cusp potential V(Y;alpha,beta)=alpha*Y+beta*Y^2/2-Y^4/4; five catastrophe flags including anomalous variance; applied behavioral modeling.

[7] [Catastrophe Theory - ScienceDirect Topics](https://www.sciencedirect.com/topics/mathematics/catastrophe-theory) — Overview of catastrophe theory including cusp catastrophe structure, bifurcation set, and qualitative behavior.

[8] [Catastrophe Theory - Gilmore](https://onlinelibrary.wiley.com/doi/abs/10.1002/3527600434.eap052.pub2) — Defines catastrophe flags: bimodality, sudden jumps, inaccessibility, hysteresis, divergence, anomalous variance near bifurcation set.

[9] [Kramers' Theory - Chemistry LibreTexts](https://chem.libretexts.org/Bookshelves/Biological_Chemistry/Concepts_in_Biophysical_Chemistry_(Tokmakoff)/06:_Dynamics_and_Kinetics/23:_Barrier_Crossing_and_Activated_Processes/23.02:_Kramers_Theory) — Kramers escape rate with barrier height dependence; overdamped and underdamped regimes; Fokker-Planck foundation for double-well dynamics.

[10] [Dissipative Double-Well Potential: Kramers Rate and Stochastic Resonance](https://pubmed.ncbi.nlm.nih.gov/28009196/) — Experimental Kramers rate measurement in dissipative double-well; demonstrates stochastic resonance in switching dynamics.

[11] [Stochastic resonance - Wikipedia](https://en.wikipedia.org/wiki/Stochastic_resonance) — SNR peaks at moderate noise in bistable systems; periodic switching between wells produces variance peak at optimal noise-to-barrier ratio.

[12] [The Diffusion Decision Model: Theory and Data for Two-Choice Decision Tasks](https://pmc.ncbi.nlm.nih.gov/articles/PMC2474742/) — Comprehensive DDM review by Ratcliff: drift rate v determined by stimulus quality/difficulty; accumulation to boundary with noise.

[13] [A practical introduction to the drift diffusion model](https://pmc.ncbi.nlm.nih.gov/articles/PMC9784241/) — Practical DDM guide: drift rate relates to difficulty; larger boundary separation increases caution; speed-accuracy tradeoff.

[14] [The EZ Diffusion Model: An overview with derivation](https://www.tqmp.org/RegularArticles/vol16-2/p154/p154.pdf) — Three EZ equations: P(correct)=1/(1+exp(-2va/s^2)), MRT=a/v, VRT=a^2/v^3; derived via exponential martingale and optional stopping.

[15] [The leaky competing accumulator model](https://pubmed.ncbi.nlm.nih.gov/11488378/) — LCA by Usher & McClelland: mutual inhibition creates attractor dynamics; inhibition-dominant regime produces bimodal decision states.

[16] [Phase transitions and finite size scaling - Helsinki lecture notes](https://www.mv.helsinki.fi/home/rummukai/simu/fss.pdf) — FSS theory: susceptibility peaks rounded and shifted; pseudocritical shift T*(L)-Tc~L^{-1/nu}; peak height chi_max~L^{gamma/nu}.

[17] [Finite-size analysis in neural network classification of critical phenomena](https://arxiv.org/html/2305.03342) — NN output variance shows FSS with peak width sigma(L)~1/L^{1/nu}; connects neural network classification to universality classes.

[18] [Robustness of variance and autocorrelation as indicators of critical slowing down](https://esajournals.onlinelibrary.wiley.com/doi/10.1890/11-0889.1) — Dakos et al. 2012: variance can DECREASE near transitions; three conditions identified; autocorrelation always increases (more robust).

[19] [Dakos et al. 2012 - PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/22624308/) — Conditions for variance decrease: reduced environmental sensitivity, CSD high-frequency filtering, limited data underestimation.

[20] [Anticipating Critical Transitions - Scheffer et al. 2012 Science](https://www.science.org/doi/abs/10.1126/science.1225244) — Distinguishes CSD-based vs flickering-based early warnings; bimodality as direct diagnostic; flickering in highly stochastic systems.

[21] [Flickering as an early warning signal - Dakos et al. 2013](https://link.springer.com/article/10.1007/s12080-013-0186-4) — Flickering occurs in bistable region under strong noise; CSD not relevant in noisy conditions; produces rising variance and bimodality.

[22] [Flickering gives early warning signals of a critical transition to a eutrophic lake state](https://www.nature.com/articles/nature11655) — Empirical flickering detection in lake ecosystem; bimodality increases as early warning; works with sparse data.

[23] [Phase Transitions in Large Language Models and the O(N) Model](https://arxiv.org/abs/2501.16241) — Maps Transformer to O(N) model; two phase transitions at temperature Tc~1.2 and parameter count ~7B; specific heat divergence.

[24] [Rate-dependent transition delay in stochastic subcritical bifurcation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5882727/) — Finite ramping rates cause rate-dependent delays in tipping; dynamic hysteresis; relevant for non-quasi-static parameter sweeps.

[25] [Early warning signals for critical transitions in a thermoacoustic system](https://pmc.ncbi.nlm.nih.gov/articles/PMC5073343/) — Subcritical Hopf bifurcation: variance increases monotonically; autocorrelation behavior inconsistent across noise conditions.

[26] [Non-equilibrium early-warning signals for critical transitions](https://www.pnas.org/doi/10.1073/pnas.2218663120) — Non-equilibrium systems need multi-dimensional indicators; flux affects stability; 1D CSD approach may be insufficient.

[27] [Resilience indicators review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4247400/) — Comprehensive review: six mechanism types for regime shifts; CSD only detects gradually approached bifurcations; multiple indicators recommended.

## Follow-up Questions

- How well does the mixture variance formula Var(d) = p(d)*(1-p(d))*||Delta_mu||^2 fit the actual empirical variance data across different LLM models and benchmark tasks, and what is the residual pattern?
- Can the cusp catastrophe parameters (alpha, beta, sigma) be estimated from the empirical LLM data, and does the fitted Cardan discriminant correctly predict which difficulty levels show bimodal response distributions?
- Is the simpler mixture model sufficient for the paper, or does the cusp catastrophe formalism provide measurably better predictions that justify its inclusion as Layer 2 of the theoretical framework?

---
*Generated by AI Inventor Pipeline*
