# Revision queue — Introduction + framework section

*Status: pre-Methods. Framework setup only. Empirical contribution claim deferred to Par 13 and Results.*

*Last updated: 2026-05-14. Part I strategic decisions (D4 / E3 / D5) resolved and implemented in `main.tex`; see the resolution note in each section. The new `\subsection{The forward--inverse model pair}` added two intro paragraphs, so every Par numbered 7 or higher shifted up by 2. Par references throughout this file are updated to the new numbering.*

---

## Guiding principle

**The framework is presented as established background, not as the paper's contribution.** The paper's contribution is the robotic instantiation of all three regimes wired together (Par 13). This means there is zero cost to adding precursor citations, and substantial cost to under-citing — under-citing risks reading as if we claim the framework as novel.

Read every section of the introduction with this lens: *if a reader thought "they're claiming to have invented this," that's a problem to fix.*

---

# Part I — Strategic framing decisions

*These shape downstream citation and revision choices. Resolve before mechanical edits.*

## D. Unify motor control, model-based RL, and active inference under one framework

**This is not an issue; it is an opportunity.** The names "forward model" and "inverse model" overlap with motor control's usage on purpose: there is a deeper unification, and recognising it strengthens our paper rather than weakening it.

### D1. The unification

In motor control, "forward" and "inverse" mean:
- **Forward model**: motor command → predicted sensory consequence (= efference copy + predicted reafference)
- **Inverse model**: desired sensation → motor command (the controller)

In our framework, the labels map the same operations onto different inputs:
- **Forward model**: world feature ($\feats$) → predicted sensory data
- **Inverse model**: sensory data → world feature estimate ($\estfeats$)

Both are instances of the same general **forward–inverse model pair**: a mapping `state → observation` (forward / generative) and a mapping `observation → state` (inverse / recognition). This dual structure shows up under different names across communities — generative + recognition in Helmholtz machines and VAEs, observation equation + state estimation in classical state-space models, generative model + inference in predictive coding. The only thing that changes between motor control and our case is what counts as "state."

| Framework | "State" is... | Forward maps | Inverse maps |
|-----------|---------------|--------------|--------------|
| Motor control | Motor command / body configuration | command → expected sensation | desired sensation → command |
| Our framework | World feature ($\feats$) | feature → sensory data | sensory data → feature |
| **Unified view** | Anything the world (including the body) is in | state → observation | observation → state |

Under this unification, **the three learning regimes are not bespoke to cross-modal sensing** — they are the standard components of any model-based intelligent-agent architecture, specialised to the cross-modal sensory case:

| Our regime | Operation on the generative model | Where it's been called this |
|------------|------------------------------------|-----------------------------|
| Green | Cross-modal supervised inference (train one chain's inverse from another's labels) | Cross-modal distillation in ML; cross-modal calibration in development |
| Yellow | Fit the generative model from paired (state, observation) data | Generative model learning; system identification; predictive coding |
| Blue | Use the learned generative model to plan / train policies offline | Model-based RL (Dyna, World Models); active inference; mental simulation |

### D2. Major traditions to acknowledge

| Tradition | Canonical references | Vocabulary they use |
|-----------|---------------------|---------------------|
| Motor control | Jordan & Rumelhart 1992; Wolpert & Kawato; Miall | forward model, inverse model, distal teacher |
| Model-based RL | Sutton 1990 (Dyna); Ha & Schmidhuber 2018 (World Models); Hafner et al. (Dreamer) | world model, simulator, model-based planning |
| Predictive coding | Rao & Ballard 1999; Friston | generative model, prediction error, free energy |
| Active inference | Friston 2010 and later | inference, action selection, expected free energy |
| Cybernetics (deep root) | Conant & Ashby 1970 ("every good regulator is a model") | regulator, internal model |
| Cognitive science / embodied | Wilson 2002; Grush 2004; Bennett 2023 *(partially cited)* | emulation, off-line simulation, world model |

### D3. The framing implication

The honest contribution claim becomes:

> *Our paper instantiates a well-established model-based-agent architecture on a real robot with a non-standard modality pair (vision-supervised sonar). The architectural decomposition into inverse-learning, forward-learning, and offline-rehearsal regimes is the standard one in model-based intelligence; specialising it to cross-modal sensory learning, and demonstrating that the loop closes on a physical platform with a low-information sonar modality, is the novel contribution.*

This is a *stronger* claim than "we propose a new framework" because it doesn't require defending novelty of the components.

### D4. Open question — how to use this opportunity

Three options, in increasing weight:

1. **Light: one acknowledgement sentence at the end of Par 7.** Lists the three traditions and says "we specialise to the cross-modal sensory case." Minimal disruption to existing structure.
2. **Medium: a short standalone paragraph after Par 7 (or after Par 10).** Names the unification explicitly with grouped citations across motor control, model-based RL, and active inference. Pays clear tribute to precursor traditions without dominating the introduction.
3. **Heavy: reframe the introduction around the unified framework.** Open with "model-based agents have forward and inverse models; here we specialise to cross-modal sensory learning..." This is a bigger rewrite and might lose the bat-inspired motivation.

**Recommendation:** option 2. Strong enough to do the framing work; not so heavy that it changes the paper's voice.

**Resolved (2026-05-14): dedicated subsection.** Implemented as a new `\subsection{The forward--inverse model pair}` between "Modalities preserve different features" and "Cross-modal learning lives in the overlap" (new Pars 7-8). Heavier on *content* than option 2 but structure-preserving: the bat hook still opens Par 1 and the existing subsection flow is intact. Par 7 defines the forward-inverse pair and names the traditions (motor control, model-based RL, predictive coding, active inference, cybernetics) with grouped citations; Par 8 carries the D5 argument. Rationale (user's call): this text is also a proposal seed, so err on theoretical breadth now and trim for the paper later.

### D5. Why the forward–inverse pair is non-trivially powerful

If we adopt any version of the unified framing, a natural reader question is: *why is the bidirectional pair more than just "two directions of a mapping"?* Three substantive answers — but for our paper, only one is load-bearing:

**Primary (load-bearing for this paper):**

- **Counterfactual reasoning needs the forward; grounding needs the inverse.** Without the forward, the agent can interpret what *is* but cannot ask "what if?" Without the inverse, the agent can imagine what *would* happen under any hypothetical state but cannot ground its imagining in present sensory data. Together: ground in present data (inverse), imagine alternatives (forward), compare. This is mental simulation, planning, model-based control — the entire blue regime.

  This justification maps directly onto our three regimes:

  | Regime | Use of the pair |
  |--------|-----------------|
  | Green | Inverse provides the *grounding* the policy reads at deployment (sonar data → feature estimate). |
  | Yellow | Forward enables *counterfactual* simulation (what would sonar see at this hypothetical state?). |
  | Blue | The closed loop: train a policy on counterfactual rollouts (forward) that will be grounded by real sonar at deployment (inverse). |

  **This is the reason to cite in the prose itself.**

**Supporting (theoretical background; cite in passing, not as the operational reason):**

- **Bayesian inference requires both directions.** Principled inference is $p(\text{state}\mid\text{obs}) \propto p(\text{obs}\mid\text{state})\,p(\text{state})$ — the forward is the likelihood that justifies the inverse, and disambiguates many-to-one mappings (metamers, occlusion, sensor noise). True but not how our paper operates: we train point-estimate inverse and forward models, not posteriors. Worth a passing citation when noting the broader theoretical landscape; not load-bearing.

- **Conant–Ashby theorem (1970): "every good regulator of a system must be a model of that system."** Cybernetic foundation: any system maintaining a goal under disturbance must contain an internal model. Authority-argument level — it justifies having an internal model at all but does not specifically require the bidirectional pair. Worth one citation as the deepest historical anchor; not the reason our architecture has both halves.

**Possible textual handling (lead with Point 2):**

> "The forward–inverse pair is not merely two mappings but a *coupled* structure: the inverse grounds the agent in present sensory data, while the forward lets it ask what other states would have produced — counterfactual reasoning that the inverse alone cannot perform. This division of labour is what makes offline policy training (Section X) possible. (The principle that any closed-loop regulator must contain such an internal model is the formal content of Conant \& Ashby's theorem \citep{ConantAshby1970}; the broader probabilistic-inference reading is reviewed by \citealp{Friston2010}.)"

Citation needed if used: **Conant & Ashby (1970)** "Every good regulator of a system must be a model of that system." *International Journal of Systems Science.*

**Resolved (2026-05-14): deployed full.** The counterfactual-vs-grounding argument is now Par 8. The Conant-Ashby good-regulator theorem is cited in Par 7 as the cybernetic root of the forward-inverse pair, and Friston 2010 in Par 7 for active inference; both theoretical anchors land, so the separate Par-8 parenthetical drafted above was dropped as redundant prose rather than omitted. `ConantAshby1970` and `Friston2010` added to `references.bib`.

---

## E. Broader-impact framing — world models as the missing piece in AI

The contribution claim can be substantially broadened by leveraging the framing in **Bennett (2023)**, specifically chapters 11–13 ("Breakthrough #3: Simulating"), which itself draws on **LeCun's** widely-quoted claim that learned world models are the central missing piece of modern AI.

### E1. The Bennett / LeCun framing

Bennett positions simulation as the *third evolutionary breakthrough* in intelligence, distinguishing mammals (and birds, and cephalopods) from earlier reflex- and reinforcement-learning-only systems. The neocortex implements a generative model; this enables three new abilities:

1. **Vicarious trial and error** — mentally simulate paths before committing (Tolman; Redish & Johnson hippocampal recordings).
2. **Counterfactual learning** — simulate alternative pasts to support causal reasoning (Redish & Steiner restaurant-row).
3. **Episodic memory** — past events as simulations.

Bennett ties this directly to ML's model-based RL programme. He quotes LeCun:

> "Primates, dogs, cats, crows, parrots, octopi, and many other animals don't have human-like languages, yet exhibit intelligent behavior beyond that of our best AI systems. What they do have is an ability to learn powerful 'world models' that allow them to predict the consequences of their actions and to search for and plan actions to achieve a goal. **The ability to learn such world models is what's missing from AI systems today.**"

### E2. The impact lever for our paper

Our paper is a concrete instance of learned cross-modal world-model + policy training on a real robotic platform, with the additional difficulty that one of the two modalities (sonar) is a low-information channel where feature recovery is non-trivial. Under Bennett's / LeCun's framing, this is a direct contribution to the research programme they identify as central — not as "we solved world models" but as "here is what cross-modal world-model learning looks like in a real sensory system with structural information collapse."

Reframing the contribution this way turns the paper from "a bat-inspired robot demo" into "a concrete instance of the research programme everyone says we need to solve."

### E3. Where to use this

- **Par 10 (light):** Cite **Bennett 2023** alongside Wilson 2002 and Grush 2004 as an accessible recent synthesis bridging evolutionary, neuroscience, and AI perspectives on off-line simulation.
- **Par 11 (medium):** Add one sentence noting that learned world models have been identified as a key missing component of modern AI (Bennett 2023 citing LeCun). This connects the empirical roadmap to the broader programme.
- **Discussion (heavy, later):** Engage substantively with the world-models programme. Position our results as a concrete demonstration of cross-modal world-model learning, and discuss what specific lessons our experience teaches — e.g., the role of feature overlap, the value of low-information sonar as a stress test of forward-model fidelity, the asymmetry between training-time (cross-modal) and deployment-time (single-modal) operation.

### E4. Citation needed

**Bennett, Max (2023)** *A Brief History of Intelligence: Evolution, AI, and the Five Breakthroughs That Made Our Brains.* Mariner Books / HarperCollins. (Zotero: `2RPS3UXL`.)

**Resolved (2026-05-14): medium, heavy deferred.** Par 12 cites Bennett 2023 as an accessible recent synthesis on off-line simulation; Par 13 adds the world-models sentence (learned world models as a central missing component of contemporary AI, citing Bennett). `Bennett2023` added to `references.bib`. Heavy Discussion engagement (the E3 "heavy" option) is parked for Discussion drafting.

---

# Part II — Concrete revisions

*Mechanical edits, mostly local to specific paragraphs. Execute after Part I decisions resolve.*

*Par numbers below are updated to the post-2026-05-14 numbering (after the forward--inverse subsection added Pars 7-8).*

## A. Internal framework coherence

### A1. Define key terms at first mention

| Term | First use | Problem |
|------|-----------|---------|
| *readout* | Par 3 | Core concept, defined only implicitly via example |
| *inverse model* | Par 3 / Par 7 | Mostly resolved: new Par 7 formally defines it (`\emph{inverse model}`). Par 3's earlier parenthetical "(or inverse model)" could still be tightened. |
| *forward model* | Par 7 | Resolved: new Par 7 formally defines it (`\emph{forward model}`) before the regime labels use it. |

**Fix options:** largely overtaken by events. The forward--inverse subsection (new Pars 7-8) now serves as the terminology anchor that option (b) envisioned, and it cross-references motor-control terminology as D anticipated. What remains is the smaller A1 item: *readout* (Par 3) is still defined only by example, and Par 3's parenthetical "(or inverse model)" could be tightened now that Par 7 owns the formal definition.

### A2. Logic-chain gaps

| Transition | Missing link |
|------------|--------------|
| Par 1 → 2 | *Why* the channel/filter lens is the right abstraction for sensory chains |
| Par 2 → 3 | Explicit *therefore* from "channels discard info" to "estimates are readouts" |
| Par 6 → 7 | Partly addressed: the forward--inverse subsection (Pars 7-8) now sits between the overlap discussion and the three regimes. Remaining gap is the *because* link into Par 9's "three regimes follow"; Par 8's close could carry a forward-pointing sentence. |

### A3. Unjustified assumptions needing a citation, caveat, or example

- **Par 2:** "Top-down signals don't add information" — standard predictive-coding view, contested by some. Needs citation or hedged language.
- **Par 3:** "Channel preservation ≠ readout extraction" — key framework pillar. Currently bare. Needs example or short argument.
- **Par 5:** *Partially addressed.* "Overlap is generally non-empty" now anchored by concrete citations (dolphins, rodents). The *"generally"* claim itself may still want qualification (vision/sonar overlap varies with conditions; see the proposal-hook comment in the source).

### A4. Information-theoretic foundations

Par 2 announces the data-processing inequality and *defers* its consequences to the Information-theoretic Statement section. Decide: surface a one-sentence implication immediately in Par 2, or accept the punt as-is.

### A5. Terminology alignment

- Par 9 names regime (3) as **"cross-modal vicarious learning"**
- Par 12 calls it **"mental rehearsal"**
- Figure caption calls it **"in mental rehearsal"**
- Note: the new Par 8 introduces **"off-line policy training"** as a generic-ML description. That is fine as the general term; the regime-specific term still needs standardising per below.

**Recommended resolution:** standardise on **"vicarious rehearsal"** / **"vicarious learning"**, since the paper title is "Vicarious sonar learning." Treat "mental rehearsal" as a one-time gloss but not a primary term.

---

## B. Missing precursor citations (literature grounding)

*Note: Jordan & Rumelhart 1992 sits under section D (unified framework, motor-control specialisation), not under any individual pathway — their distal-teacher technique is not what our green pathway uses.*

### B1. Par 10 — Green pathway (cross-modal supervision of inverse models)

| Reference | Why it matters |
|-----------|----------------|
| **Gupta, Hoffman & Malik (2016)** "Cross-modal distillation for supervision transfer" *CVPR* | Identical mechanism in deep learning: RGB supervises depth network. Direct ML precedent for green pathway. |
| Hinton, Vinyals & Dean (2015) knowledge distillation | Same-modality precursor; cite if useful as the broader frame. |
| Subsequent cross-modal SSL: Aytar et al., Owens & Efros, Ngiam et al., audio-visual learning | Optional broader cross-modal SSL literature |

### B2. Par 11 — Yellow pathway (forward model learning)

| Reference | Why it matters |
|-----------|----------------|
| **Rao & Ballard (1999)** "Predictive coding in the visual cortex" *Nat. Neurosci.* | Broader cortical predictive framework. |
| Wolpert / Kawato cerebellar internal models | Sensorimotor forward-model learning; covered under section D if we adopt the unification. |
| ML model-based: forward-model fitting in Dreamer / PlaNet | Engineering instances of forward-model fitting from observation data. |

*Update (2026-05-14): Rao & Ballard 1999 is now cited in the forward--inverse subsection (Par 7), and the model-based-RL lineage with it. B2 reduces to an optional reinforcing cite inside the yellow-pathway paragraph (Par 11) itself.*

### B3. Par 12 — Blue pathway (vicarious rehearsal)

| Reference | Why it matters |
|-----------|----------------|
| **Sutton (1990)** Dyna architecture | Older precursor: model-based RL where a learned model is used for planning + training. |
| **Ha & Schmidhuber (2018)** "World Models" | Direct ML precedent: train forward model, then train policy entirely inside it. The blue pathway, in deep RL form. |
| **Bennett (2023)** *A Brief History of Intelligence*, chapters 11–13 | Accessible synthesis bridging evolutionary, neuroscience, and AI perspectives on simulation. See section E for the broader-impact use of this reference. |
| Hafner et al. — PlaNet, Dreamer (v1/v2/v3) | Optional: ongoing ML lineage. |
| Wilson 2002, Grush 2004 | Already cited. ✓ |

*Update (2026-05-14): Sutton 1990 and Ha & Schmidhuber 2018 are now cited in the forward--inverse subsection (Par 7); Bennett 2023 is cited in Pars 12-13. B3 reduces to an optional reinforcing cite inside the blue-pathway paragraph (Par 12) itself.*

### B4. Par 5 — Cross-modal feature overlap

| Reference | Why it matters |
|-----------|----------------|
| **Ernst & Banks (2002)** "Humans integrate visual and haptic information in a statistically optimal fashion" *Nature* | Foundational Bayesian multisensory integration. Adjacent strong precedent. |
| **Held & Hein (1963)** kitten carousel | Classic developmental cross-modal calibration. Foundational. |

### B5. General theoretical landscape

| Reference | Where it fits |
|-----------|---------------|
| **O'Regan & Noë (2001)** "A sensorimotor account of vision and visual consciousness" *BBS* | Philosophical adjacent to forward models. Could fit Par 3 or Par 12. |
| **Friston** active inference / free energy | Broader umbrella framework. Worth one citation somewhere (section D candidate). |
| Helmholtz, "perception as inference" | Foundational. Could be a single citation in Par 3. |

---

# Part III — Cross-cutting concerns

## C. Auditing once the substantive edits are done

### C1. Contribution framing audit

Read Pars 1-13 with the question "could this sentence be read as a novelty claim for the framework itself?" Adjust any such language. Par 13 is currently honest ("empirical demonstration of all three regimes wired together"). Earlier paragraphs may need light adjustment, especially the new Pars 7-8, which must read as established background.

If we adopt section D's unified-framework framing and/or section E's world-models framing, the contribution audit should align with those updated claims.

### C2. Citation density

Adding many references should be done without bloating prose. Use compact citation clusters (`\citep{A,B,C}`) where appropriate; let readers triangulate.

---

# Priority ordering

1. ~~**Strategic decisions** (D4 + E3 + D5 textual-handling pick)~~ — **DONE (2026-05-14).** Resolved: dedicated subsection (D4), medium with heavy deferred (E3), deployed full (D5). See the resolution note in each Part I section.
2. **A5** — terminology alignment (5-minute fix, prevents reader confusion).
3. **B1–B3** — missing precursor citations, informed by the strategic decisions above.
4. **A2** — logic-chain transitions (improves flow).
5. **A1** — early definitions (could combine with D2 terminology box if D is adopted).
6. **A3** — unjustified assumptions (defensive).
7. **B4–B5** — secondary literature grounding.
8. **A4** — DPI surfacing decision.
9. **C1** — contribution-framing audit (do *after* B1–B3, D, and E so the audit reflects the updated landscape).
10. **E2 heavy option (Discussion)** — defer until the Discussion is being written.
