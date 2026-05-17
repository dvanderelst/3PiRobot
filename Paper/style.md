# Style guide

Writing conventions for `main.tex`. Lives in the repo so it travels across machines.

## Symbols and terms

The prose should read naturally even if the reader skips every symbol. Lead with the natural-language term; put the symbol in parentheses as formal reinforcement. A symbol that comes first asks the reader to decode before they understand, and many readers will never bother.

- Yes: *"The features both modalities carry ($F_A \cap F_B$) is where the modalities can speak."*
- No: *"The intersection $F_A \cap F_B$, the features both modalities carry, is where the modalities can speak."*

Equally fine when the term and symbol form a tight phrase (term first, symbol immediately after, no parens needed): *"modality $M$"*, *"feature estimate $\hat{F}$"*, *"sensory data $\mathrm{Data}_M$"*.

Exception: dense formal passages (the Markov-chain / data-processing-inequality block in `\section{Information-theoretic statement}`) may use symbols without re-introducing terms each time. The narrative sections should not.

## Punctuation

No em-dashes (`---`) and no en-dashes (`--`) for parenthetical breaks. Use commas, colons, parentheses, or restructure the sentence.

Avoid colons used to introduce an elaboration or restatement of a prior clause. *"Vision's access to depth is comparable: distance must be recovered through specialised circuitry..."* reads as a journalistic flourish. Either join the clauses with a connective (*"Vision's access to depth is comparable, since distance must be recovered..."*) or split into two sentences. Colons remain fine before lists, definitions, and equations.

## Emphasis

`\emph{...}` sparingly. Reserve for:

- A term being formally defined on first mention.
- Short structural labels (e.g., *Preserved* / *Lossy* / *Destroyed* regime names in the technical section, or *Green* / *Yellow* / *Blue* in the theory figure caption).

Do not italicize ordinary words or short phrases for stylistic accentuation. Lean on word choice and sentence structure for emphasis instead.

## Tone

Formal academic register. Avoid:

- Hyperbole (e.g., *"the cleverest software ever written"*).
- Informal markers (*"quirk"*, *"a natural reading"*).
- Chatty constructions (*"tell you"*, *"put back"*).

Active voice where natural; passive where the actor is genuinely incidental.

## Spelling

UK English throughout. Common substitutions: *colour* (not color), *behaviour* / *behavioural* (not behavior / behavioral), *specialisation* (not specialization), *recognise* / *organise* / *analyse* (not -ize / -yze), *hypothesised* (not hypothesized), *parameterised* (not parameterized), *centre* (not center), *defence* (not defense).

Exception: the hyperref option `colorlinks=true` in the preamble is a package keyword and stays in US form.

## Paragraph numbering

Each paragraph is preceded by a `% Par N` comment on its own line, numbered globally from the top of `main.tex`. This lets us refer to paragraphs unambiguously in conversation ("Par 5", "Par 14").

```latex
% Par 1
All sensory systems are limited in what they perceive. ...

% Par 2
From an information-theoretic point of view ...
```

The numbers will drift as paragraphs are added, removed, or split. When that happens, renumber the affected range. Section/subsection headings and figures are not counted; only prose paragraphs (including `\paragraph{...}` bodies and `itemize` blocks that are part of their introducing paragraph).

## Voice and first person

"we" throughout, even in single-author drafting passes. No "I". Reserve passive for procedural steps where the actor is genuinely incidental.

- Yes: *"We start by collecting echoes ($N=1014$) in 21 natural bat habitats..."* (main01:197)
- Yes: *"We test this hypothesis by creating an agent-based simulation..."* (main02:87)
- Yes (passive, actor incidental): *"Echoes recorded by the microphone were sampled at 360 kHz."* (main01:204)
- No: *"The authors collected..."* or *"In this study, echoes were collected..."* when the actor is us.

## Hedging

Discussion and Introduction hedge interpretation; Methods and Results state procedure and findings directly. Standard hedges: *may*, *might*, *could*, *likely*, *probably*, *suggests*, *propose*, *seems*, *appears to*, *can be assumed to*.

- Yes (interpretation): *"This indicates that, at least in principle, the bat could encode the echoic information present at the cochlear nucleus level..."* (main01:396)
- Yes (interpretation): *"Our findings suggest that in dense swarms, bats can exploit the emergent acoustic environment to maintain safe distances passively."* (main02:66)
- Yes (procedure, flat): *"The networks had one output node, with a sigmoid activation function."* (main01:253)
- No (over-hedged procedure): *"The training may have been done using RMSProp..."*

Hedge interpretation, not method.

## Transitions

A small standard set, sentence-initial with comma: *However,* *Indeed,* *Hence,* *Therefore,* *Moreover,* *Furthermore,* *In particular,* *In contrast,* *First, / Next, / Finally,* *Note that*. Used liberally — most paragraphs contain one or two.

- Yes: *"Hence, while the previous experiment studied how the proposed compressive encoding preserves temporal information, this experiment tests how well spectral information is preserved."* (main01:293)
- Yes: *"Indeed, many samples contain hardly any energy (and, thus, information)."* (main01:383)
- Yes: *"In contrast, during the search for prey, \animal{tb}'s call varied in duration from about 7 to less than 1 ms..."* (main02:277)

One transition per sentence, not two. *"However, moreover, ..."* — no.

## Argument chaining

Make the logical structure of arguments explicit. When one sentence supplies a premise and the next states what follows from it, name the relation with a sentence-initial connective: *Therefore,* *Hence,* *It follows that,* *As a result,* *Consequently,* *This implies that,* *This indicates that,* *This suggests that,* *This shows that*. Do not leave the reader to infer the link.

Build chains: premise, premise, then connective + conclusion. The corpus does this routinely — in Methods justifications as well as Discussion arguments.

- main03:136 — three sentences listing limits of bat sonar (small field of view, low update rate, temporal resolution), closed with *"It follows that bats probably cannot extract much information about the 3D layout of the environment most of the time."*
- main01:374 — four-step chain in one paragraph: ideal low-pass assumed → only 2 kSamples/sec needed → *"Hence, at 360 kSamples/sec, the cochleograms are oversampled by a factor of 180."* → *"Note that the low-pass filter in the model is not ideal..."* → *"Therefore, in practice, sampling at a somewhat higher Nyquist rate is required."*
- main02:283 — *"... performance in the current simulations did not seem worse, using fewer assumptions. Consequently, we argue that the current simulations demonstrate that the assumptions ... can be further relaxed beyond those proposed by \citet{Mazar2024}."*

The connective is sentence-initial with a comma. Do not bury it mid-sentence.

- Yes: *"Therefore, we scaled all elements in our setup by a factor 2..."* (main03:162)
- Yes: *"Hence, this suggests that the encoded cochleograms do indeed retain sufficient information..."* (main01:349)
- No: *"We scaled all elements in our setup, therefore, by a factor 2..."* — connective buried.

Spell the chain even when the next step feels obvious. The reader is tracking the argument, not the prose.

## Equations

Lead in with a complete clause ending in a comma (often *"... as follows,"* or *"... can be written as,"*), then a display equation. Surround the equation environment with `%` comment lines immediately above `\begin{equation}` and below `\end{equation}`. This keeps source isolated without forcing a paragraph break.

```latex
this observed vector $\mathbf{x_j}$ can be written as a linear mixture of basic components,
%
\begin{equation}
\mathbf{x_j} = \sum _{i=1}^{N} c_{j,i} \cdot \mathbf{\Psi_i}= \mathbf{A} \cdot \mathbf{c_j}
\label{eq:filter}
\end{equation}
%
with the basic components $\mathbf{\Psi_i},\ i=1, \cdots, N$ making up the columns...
```
(main01:218–225; same pattern at main02:337–344)

In prose, refer back as *"equation \ref{eq:foo}"* or *"Equation \ref{eq:foo}"* (capital when sentence-initial). Not *Eq.*, not *Eqn.*

- Yes: *"In equation \ref{eq:calls}, $\emission_{\direction(i,j)}$ refers to the emission directionality..."* (main02:344)
- Yes: *"using equations \ref{eq:calls}, \ref{eq:body}, \ref{eq:walls}..."* (main02:422)

Punctuate the equation to match the surrounding sentence: comma if a clause continues after, period if the equation ends the sentence, nothing only if the equation stands alone as its own sentence-equivalent.

- Yes (clause continues): *"$x(t) = e(t) + a\,e(t-\tau),$"* followed by *"the spectrum of such an echo can be written as..."* (main01:274)
- Yes (terminal): equation ends with `.` when the prose stops there.
- No: equation with no punctuation when the next word is the continuation of its sentence.

## Citations

Group supporting references into one citation, not several.

- Yes: *"...for example, \cite{Aytekin2004,Chiu2007,Lawrence1982,Wotton2000}."* (main01:303)
- Yes: *"\citep{Goetze2016,Corcoran2017,Amichai2015}"* (main02:79)
- No: *"...for example, \cite{Aytekin2004}, \cite{Chiu2007}, \cite{Lawrence1982}, and \cite{Wotton2000}."*

Use natbib for new papers (load `\usepackage{natbib}`). `\citet{...}` when the author is part of the sentence; `\citep{...}` otherwise. Do not use plain `\cite{...}` in new work.

- *"\citet{Mazar2020} suggested that bats hunting near each other might experience less jamming..."* (main02:275)
- *"...generated by conspecifics \citep{Ulanovsky2004}."* (main02:79)

For illustrative or "see also" lists, use the prefix form: `\citep[e.g.,][]{...}`, `\citep[see][]{...}`, `\citep[See][for references]{...}`.

- *"\citep[e.g.,][]{Brooks1999,Beer1990,Pfeifer2006,DiPaolo2017}"* (main03:132)
- *"\citep[See][for references]{Mazar2020}"* (main02:275)

Citations sit at the end of the clause they support, immediately before the period or comma.

## Figure references

Use *Fig.* (with period) for singular and *Figs.* (with period) for plural. Always capitalised, including mid-sentence parentheticals. Not *Fig*, not *Figure*, not lowercase *fig.*

- Yes: *"Fig. 3 shows..."*, *"(see Fig. 3)"*, *"Figs. 3 and 4 together..."*
- No: *"Figure 3 shows..."*, *"(see fig. 3)"*, *"Figs 3 and 4"* (missing period).

Sub-panel letters attach directly to the ref: no space, no parentheses.

- Yes: *"Fig. \ref{fig:results}B shows that the neural network was also able to discriminate..."*
- Yes: *"Fig. \ref{fig:processing}A-D"*
- Yes: *"Figs. \ref{fig:a} and \ref{fig:b}a,b show that..."*
- No: *"Fig. \ref{fig:x} (a)"*, *"Fig. \ref{fig:x} a"*.

Two reference forms are in routine use, both fine:

- In-text subject: *"Fig. X shows..."*
- Parenthetical: *"(see Fig. X)"* / *"(Fig. X)"*

## Paragraph structure

Open each paragraph with an orientation sentence: either the claim the paragraph supports or the operation it describes. Do not bury the topic mid-paragraph.

- Yes (claim): *"Frequency selective cells are prevalent throughout the auditory pathway of bats, which is largely tonotopically organized \cite{Pollak1989}."* (main01:408) — paragraph then enumerates evidence.
- Yes (operation): *"To verify that this encoding retains the relevant spectrotemporal echo information useful to a bat, we simulate several previously published behavioral discrimination experiments with bats."* (main01:199) — paragraph then describes the simulation setup.

End paragraphs with a summary punchline or an explicit transition into the next paragraph. Trail-offs and "and so on" endings are not used.

- Yes (punchline): *"This indicates that the independent components we derived from the real echoes have correctly captured the physical constraints shaping the spectrotemporal cues present in the cochleograms."* (main01:328)
- Yes (transition setup): *"Below, we discuss both implications of the current results."* (main01:370)

## Vocabulary

Reach for these verbs when reporting our own work: *derive*, *propose*, *suggest*, *demonstrate*, *show*, *assess*, *investigate*, *test*, *model*, *simulate*, *implement*, *evaluate*, *report*.

- *"we derived an efficient encoding based on 25 independent components"* (main01:366)
- *"we propose an alternative hypothesis"* (main02:87)
- *"We tested the algorithm outlined above in two conditions."* (main03:234)

Avoid promotional verbs (*reveal*, *unveil*, *prove*) and product-pitch register (*leverage*, *empower*, *enable next-generation*, *cutting-edge*).

Stock connective phrases used routinely:

- *"Note that ..."* — main01:317, main02:245.
- *"It should be noted that ..."* — main01:336, main02:123.
- *"To the best of our knowledge, ..."* — main01:187, main02:107, main03:278.
- *"In particular, ..."* used to single out the most relevant case in a list — main01:259, main02:228.

## Latin abbreviations

Use *i.e.,* *e.g.,* *etc.* — lowercase, comma after. Also accepted but rarer: *i.c.,* (in this case).

- Yes: *"natural stimuli, i.c., echoes from different indoor and outdoor environments..."* (main01:189)
- Yes: *"\citep[e.g.,][]{Ulanovsky2004,Takahashi2014,Ibanez2004}"* (main02:85)

Do not use *cf.*, *viz.*, *q.v.* — none appear in the corpus.

## Acronyms

Always use the `glossaries` package. Load `\usepackage[acronym]{glossaries}` in the preamble, define every recurring acronym with `\newacronym{...}`, and use `\gls{...}` throughout the body. Do not hand-expand acronyms in parallel — `\gls{...}` handles first-use expansion automatically.

```latex
% Preamble
\usepackage[acronym]{glossaries}
\newacronym{ild}{ILD}{interaural level difference}
\newacronym{hrtf}{HRTF}{head-related transfer function}

% Body
\gls{ild}   % first use -> "interaural level difference (ILD)"
\gls{ild}   % later     -> "ILD"
```

Define standard recurring terms as acronyms (e.g., ILD, HRTF, JND, 2AFC). The terms we have already standardised on:

- *interaural level difference* (ILD) — not *interaural intensity difference*, not *IID*.
- *head-related transfer function* (HRTF).

