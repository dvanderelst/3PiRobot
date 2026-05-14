# Style guide

Writing conventions for `main.tex`. Lives in the repo so it travels across machines.

## Symbols and terms

The prose should read naturally even if the reader skips every symbol. Lead with the natural-language term; put the symbol in parentheses as formal reinforcement. A symbol that comes first asks the reader to decode before they understand, and many readers will never bother.

- Yes: *"The features both modalities carry ($F_A \cap F_B$) is where the modalities can speak."*
- No: *"The intersection $F_A \cap F_B$, the features both modalities carry, is where the modalities can speak."*

Equally fine when the term and symbol form a tight phrase (term first, symbol immediately after, no parens needed): *"modality $M$"*, *"feature estimate $\hat{F}$"*, *"sensory data $\mathrm{Data}_M$"*.

Exception: dense formal passages (the Markov-chain / data-processing-inequality block in `\section{Information-theoretic statement}`) may use symbols without re-introducing terms each time. The narrative sections should not.

## Punctuation

No em-dashes (`---`). Use commas, colons, parentheses, or restructure the sentence.

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
