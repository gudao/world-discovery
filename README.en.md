# World Discovery · MuJoCo

[中文版（默认）](README.md)

**Ultimate goal: train a world model that does not merely predict, but "grows" intelligence through exploration — discovering what its current understanding lacks, and actively seeking the evidence that fills the gap.**

This repository is the **groundwork** for that goal: on a MuJoCo block scenario (hidden mass/friction/fixed state), we use a strict protocol of explicit Bayesian system identification + information-gain probing + preregistered criteria to test, link by link, the mechanisms a world model needs. Each experimental version is not an endpoint; it answers which links in the mechanism chain toward "discovery-style" intelligence are real, and which are illusory.

**Current status (2026-10-10):** v0.1 mechanism baseline frozen; v0.2 protocol frozen, phase 1 (discrimination test) complete — the mechanism chain works as designed, but the environment is too cheap to discriminate, so the next step is the phase-4 environment upgrade. See the research log (in Chinese) in [README.md](README.md).

## Vision

The world model we want to train does not get its intelligence from a black box of stacked architecture. It stands on one explicit, testable design principle:

> **When the current representation can no longer stably explain the consequences of observations and actions, the system should seek probes that discriminate between competing explanations; if existing explanations keep failing, allow the representation to add new distinctions.**

Under this principle, all modes of exploration — looking from another angle, waiting, touching and nudging, changing conditions and re-observing — share one structure: **revealing hidden differences through different means.** The order of probing is not fixed in advance (vision first, touch second); it depends on what information is currently missing, which probe is most useful, and its cost and risk.

The crucial distinction is between two kinds of "unknown":

- **First kind: the question is known, the answer is not.** E.g., the system knows the object has weight, just not how much — hold a few candidate values, and one probe narrows them down. This is parameter identification, with ample prior research (active inference, Plan2Explore and related work already cover much of "active exploration to reduce uncertainty"; we do not claim it as our novelty).
- **Second kind: the system does not yet know which attributes to consider.** E.g., it has so far explained motion with shape and weight, yet finds two objects of similar shape and weight behaving differently on the same surface; after repeatedly ruling out existing explanations, it must form a new internal distinction — which it may only later come to interpret as friction, adhesion, or something else.

**The second kind is closer to intelligence that "grows" automatically: the system does not just fill gaps in existing knowledge, it expands its own way of describing the world.** This is the step this project genuinely pursues — can the system go from "my prediction fails here" to "I am missing a way of describing this world," and then form that description through a few probes?

The difficulty: a system cannot directly detect "things it has not yet thought of." It can only find clues — persistently unexplained outcomes, contradictions between senses, repeated prediction failures under specific conditions — and then probe to decide: is this observation noise, a mis-estimated known attribute, or a genuinely missing distinction? Therefore "adding a new distinction" need not mean adding a hand-named variable; it may also be a learned latent feature (u = g_φ(h)). Its value is judged by whether it improves subsequent prediction and transfers to other situations — not by merely memorizing the exception.

This does not guarantee that intelligence will emerge, but it turns "growing intelligence" into a **precise, testable question**:

> **After exploration, has the model acquired a discrimination ability it did not have before, and can it use it to explain and predict new situations?**

Hence the ultimate test is not prediction error (storing more data also lowers error) but three tests: the **discrimination test** (can world pairs that the old representation treats as identical be separated by autonomous probing), the **transfer test** (can a new distinction be carried to new worlds from the same distribution, ruling out the "just stored more data" explanation), and the **new-distinction test** (can the system autonomously discover, and accept under an MDL-style criterion, the correct new world variable).

Iteration roadmap (see `docs/ROADMAP.md`, in Chinese): v0.1 mechanism baseline (done) → v0.2 explicit hypothesis sets and the three tests (in progress) → v0.3 learnable representations (replacing fixed random features with a trainable latent-variable encoder) → v0.4 multimodal active perception (vision/touch/sensing cost, turning "look from another angle, touch" into controlled experimental variables) → v0.5 representation extension and knowledge integration (proposing, inferring, transferring and integrating new distinction modules). Every version accumulates rigorously tested mechanism conclusions for the final world model. **The output of this project is mechanism conclusions, not metric-chasing demos.**

## Experimental Objectives

Each version answers a set of preregistered research questions, all in service of the world-model goal. v0.2 tests the substance of the design principle above (full definitions in `docs/PROTOCOL-v002.md`, in Chinese):

- **R1' (discrimination):** Can information-gain probing on an explicit hypothesis posterior separate world pairs that the old representation treats as identical but differ in ground truth — and beat random and fixed-scan probing?
- **R2' (diagnosis and correction):** Once persistent misfit is detected, can the system accept the correct candidate among parallel explanations (update parameters / extend memory / inflate noise model / add variable) under a preregistered criterion?
- **R3' (transfer of new distinctions):** Can an accepted new variable be inferred from a few historical interactions and transferred to new worlds of the same distribution (ruling out the "just stored more data" explanation)?

The v0.1 A/B experiments (whether probing choice matters, whether on-demand capacity extension works) are kept as the mechanism baseline, with all outputs frozen read-only. v0.1's negative result (action collapse of the active scorer) is attributed to the hypothesis set H never being explicitly represented, and is tested head-on in v0.2 with the explicit mechanism replacing it.

> **For contributors: research-log convention.** After every experiment (including negative results and unfinished phases), the 「研究进展日志」 section in [README.md](README.md) must be updated — problem, idea, full procedure (commands and output directories), conclusions (with CIs / preregistered verdicts), and next step. Principles: freeze the protocol before running; absolute numbers + paired CIs over mean rankings; report negative results honestly. Protocol details: `docs/PROTOCOL.md` (v0.1) and `docs/PROTOCOL-v002.md` (v0.2, frozen).

The quick-start, environment description, and per-experiment details are in the [Chinese README](README.md).
