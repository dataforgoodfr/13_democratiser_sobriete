# Technical report — proposed addition

Drop-in section for `technical_report.md`, replacing the stub "Clustered policy sufficiency classification (to complete" and sitting between *Policy clustering* and *Policy impacts extraction*.

---

## Sufficiency classification and clustering: expert evaluation and recalibration

### Objective

The initial clustering described above was tuned by manual inspection only, and sufficiency was assessed at the cluster level, after clustering. We had no measure of how well the pipeline agreed with domain experts — neither on what counts as a sufficiency policy, nor on whether the clusters formed coherent, mechanism-consistent groups. This section describes the evaluation loop we built to answer both questions, and the recalibration it led to. The main structural change is a reordering of the pipeline: instead of clustering all 1.47M extracted policies and classifying clusters afterwards, we now classify each policy first, keep only sufficiency-related ones, and cluster within (sector, mechanism) strata.

### Methods

#### Expert ground truth

Domain experts produced two annotation campaigns that anchor all evaluations below.

First, at the policy level, 770 extracted policies were labelled with a 5-class category — sufficiency, efficiency, consistency, ambiguous, not a policy — and, when sufficiency, a sub-code from a 10-code mechanism taxonomy defined by the experts (caps/limits/bans, prices and taxes, reduce and right-size, passive design, modal shift, sharing schemes, public provisioning, etc.). This gold dataset serves as the benchmark for every classifier prompt and model.

Second, at the cluster level, the experts reviewed a stratified sample of 90 clusters from the recalibrated clustering (rating coherence, quality on a 1–5 scale, fit of the assigned mechanism label, and whether the cluster should be split), and individually adjudicated all 2,223 singleton clusters.

#### Calibrating the policy-level classifier

Our first classifier (a 3-class prompt: sufficiency / potential sufficiency / not sufficiency) reached only 63.4% binary accuracy against expert gold, with 34.1% strict recall on sufficiency: it systematically routed passive design and modal shift policies to efficiency ("same service with less input"). We rewrote the prompt around the expert taxonomy itself: 5-class output matching the expert categories, an optional 0–9 sub-code, and 10 few-shot examples drawn from the gold set. This v1 prompt reaches 79.3% binary accuracy, 81.7% strict sufficiency recall, and 83.5% sub-code agreement when both model and expert agree the policy is sufficiency.

A later attempt to enrich the prompt with a precise six-pillar definition of sufficiency (synthesised from Saheb 2021 and IPCC AR6 WGIII) produced an instructive negative result: the richer definition improved a large API model (DeepSeek, 79.3% → 81.1%) but *degraded* the small local model we actually deploy (Gemma 4 12B, 75.3% → 72.4%). What the small model needed was not a richer concept but a single calibration rule reining in its over-permissiveness ("a vague aspiration with no concrete mechanism is not a policy"). Prompt improvements validated on one model do not transfer to another size class; prompts must be validated on the model that will run at scale.

#### Full-corpus classification on a rented GPU

Running the 1.47M-policy classification through a commercial API was estimated at ~$280. Instead we ran Gemma 4 12B locally with vLLM on a single rented H100, using guided-JSON decoding to enforce the output schema. The full corpus took 10 h 45 min at a sustained ~38 policies/s, for ~$32 of rental — roughly 10× cheaper than the API path, though about 4× more carbon-intensive (2.46 kg CO₂eq, measured with codecarbon), since a dedicated GPU idles between batches where shared inference clusters run at high utilisation. The classified corpus is checkpointed as a HuggingFace dataset so that subsequent clustering iterations restart from it without re-running the classifier.

46.2% of policies survived the filter as sufficiency or ambiguous. The ambiguous class was subsequently dropped entirely: of 31,382 ambiguous rows, exactly one turned out to be sufficiency-related. The final input to clustering is 645,495 sufficiency policies.

#### Stratified re-clustering

Evaluation of the initial clustering showed that 58% of coherent clusters were topically coherent but *not* sufficiency-mechanism groups: a caps-and-limits policy and a public-provisioning policy about housing share vocabulary but represent entirely different sufficiency mechanisms. We therefore stratify clustering by (sector, sub-code) cell — 110 cells — making clusters mechanism-uniform by construction. Within each cell we keep the Leiden approach (on a FAISS-HNSW k-NN cosine graph, k=20, similarity threshold 0.55) with a size-adaptive resolution schedule. Re-clustering the full filtered corpus takes 9 minutes on CPU and yields 3,152 clusters with an average size of ~200 policies.

#### Automated evaluation: LLM-as-judge intrusion task

To evaluate cluster quality at scale between expert campaigns, we use an intrusion task: the judge model is shown a sample of cluster members plus one "intruder" policy from another cluster, and must identify the intruder — a standard proxy for cluster separability, calibrated against a 55-item hand-labelled reference set. On the initial clustering the judge scored 73.5%, degrading sharply on small clusters (58.6% for clusters ≤50). On the stratified clustering it scores **86.1%**, essentially flat across cluster sizes — the signature of a well-resolved clustering. Each evaluation round costs under a dollar of API calls.

#### Expert evaluation of the recalibrated clustering

The expert review of 90 clusters found 60% fully coherent, 28% partly coherent and 12% incoherent (mean quality 3.0/5). The dominant defect is not the grouping but the mechanism *label*: 32% of assigned sub-codes were judged wrong, half of them on clusters that are otherwise coherent. These are mostly lexical false matches — e.g. reducing *sedentary behaviour* mislabelled as consumption right-sizing, *knowledge*-sharing mislabelled as product-sharing schemes. One sector, INDUSTRY, was confirmed structurally problematic (mean 2.3/5, no fully coherent cluster).

The singleton adjudication produced an unexpected insight: 79.1% of the 2,223 singletons are not policies at all. Singletons are largely classifier false positives that the clustering graph strands by construction — stranding acts as a free noise filter. Only 9.4% (209 policies) are genuine sufficiency policies worth keeping.

Comparing judge and expert verdicts on overlapping clusters also revealed a limit of the intrusion task: its accuracy is essentially uncorrelated with expert-perceived quality. Intrusion measures separability *between* clusters, while the failures experts flag — coherent topic, wrong mechanism — are invisible to it. We consequently extended the judge with a second question rating whether the cluster's content fits its mechanism code, calibrated against the expert review.

#### Corrections: singleton rehoming and second-pass re-classification

The 209 expert-validated sufficiency singletons were rehomed into existing clusters by a similarity-weighted k-NN vote (k=20, cosine floor 0.45) restricted to the expert-assigned (sector, sub-code) cell, with a nearest-centroid cross-check: 203/209 were assigned, yielding an amended clustering of 2,949 clusters.

For the mislabelling problem, rather than hand-patching the flagged clusters, we designed a systematic second classification pass: the 645k clustered policies are replayed through the classifier with six additional calibration rules targeting the observed failure modes; policies that flip off sufficiency are dropped, and clusters with low retention or mechanism mismatch are flagged in an audit file. The expert flags serve as the validation set for this pass, which awaits GPU compute at the time of writing.

### Results

| metric | initial clustering | recalibrated |
|---|---:|---:|
| classifier binary accuracy vs expert gold | 63.4% | 79.3% |
| strict sufficiency recall | 34.1% | 81.7% |
| judge intrusion accuracy | 73.5% | 86.1% |
| judge accuracy, small clusters | 58.6% | 84.2% |
| clusters (after corrections) | 2,625 | 2,949 |
| expert coherence (yes / partly / no) | — | 60% / 28% / 12% |

The mechanism distribution itself is a substantive finding: public provisioning accounts for 43% of all sufficiency policies and caps/limits/bans for 27% — together over two-thirds of the corpus — while modal shift, despite its visibility in the transport-sufficiency literature, represents only 2.2%.

### Cost and environmental footprint

The entire evaluation-and-recalibration campaign cost about **$57** and **~2.5 kg CO₂eq**: ~$19 of API calls for the gold-standard evaluations and 100k-scale pilot, ~$33 of H100 rental for the full-corpus classification (2.46 kg CO₂eq, 99% of total emissions), ~$4 for prompt-variant validation, and $0 for the expert-review analyses and singleton rehoming (minutes of CPU). The pending second classification pass is estimated at ~$15.

### Future work

* Run the second classification pass and validate its audit flags against the 21 expert-flagged clusters.
* Re-judge with the mechanism-fit question and calibrate against the expert review.
* Size-adaptive Leiden resolution: uniform resolution over-fragments large cells into singletons, while naively lowering it over-merges them into blobs (one 34k-policy cluster in a trial run); the resolution should scale with cell size.
* An INDUSTRY-specific deep dive, and possible pruning of the four mechanism codes that each cover under 4% of policies.
* Packaging the recalibrated pipeline for the Jean-Zay SLURM environment for future full reruns.
