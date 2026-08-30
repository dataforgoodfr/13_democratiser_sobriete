# Technical report — bibliography pass (methods)

Verified references for the methods used in `technical_report.md` and `technical_report_additions.md`. All titles/venues/links checked against the sources on 2026-08-06. Domain (sufficiency) literature deliberately excluded. Part 1 maps each citation to the claim it should source; Part 2 is the full reference list; Part 3 lists caveats.

---

## Part 1 — Where each citation goes

### Existing section: Policy clustering

| claim in the report | cite |
|---|---|
| "We first tried K-means and HDBSCAN" | [MacQueen 1967] or [Lloyd 1982]; [Campello 2013] (+ [McInnes 2017] for the software) |
| "the silhouette score, commonly used for evaluating the quality of clustering" | [Rousseeuw 1987] |
| "using the Leiden algorithm" — **currently uncited** | [Traag 2019] |
| FAISS / HNSW index | already linked in the report ([Douze 2024], [Malkov 2020]) |
| "the NetworKit library … parallel implementation" | [Staudt 2016] |

### New section: expert evaluation and recalibration

| claim | cite |
|---|---|
| 10 few-shot examples drawn from the gold set | [Brown 2020] |
| LLM classification benchmarked against expert annotation (the whole gold-standard setup) | [Gilardi 2023]; optionally [Törnberg 2023], [Ziems 2024] |
| "Prompt improvements validated on one model do not transfer to another size class" | [Sclar 2024]; [Mizrahi 2024] (template choice changes model *rankings*) |
| Gemma 4 12B local classifier | [Gemma Team 2026] |
| DeepSeek judge/validation model | [DeepSeek-AI 2026] |
| vLLM serving | [Kwon 2023] |
| guided-JSON decoding | [Willard & Louf 2023]; optionally [Dong 2024] (XGrammar, current vLLM backend) |
| (optional footnote) structured decoding can affect task quality | [Tam 2024]; counter-argument [Geng 2025 / arXiv:2501.10868] |
| small local models competitive with API models for classification | [Bucher & Martini 2024]; [Alizadeh 2025] |
| carbon measured with codecarbon | [codecarbon (Zenodo)] + [Lacoste 2019] |
| intrusion task ("a standard proxy…") | [Chang 2009] — the origin of word/topic intrusion |
| automated coherence as proxy for human judgment (context for the judge) | [Lau 2014] |
| LLM-as-judge | [Zheng 2023]; survey [Li 2024] |
| precedent: LLMs performing Chang-style intrusion for topic-model evaluation | [Stammbach 2023] — nearest verified precedent; nobody does intrusion on *embedding clusters* as headline contribution, which supports a modest novelty claim |
| k-NN vote for singleton rehoming | [Cover & Hart 1967] (origin: [Fix & Hodges 1951]) |
| expert sub-code agreement figures (if you report agreement statistics) | [Cohen 1960] |

### Existing section: Policy impacts extraction

| claim | cite |
|---|---|
| vLLM + Automatic Prefix Caching ("computational cost of reading it is paid only once") | [Kwon 2023]; the academic reference for prefix/radix caching is [Zheng 2024] (SGLang/RadixAttention) — vLLM's APC has no dedicated paper |
| choice-based structured outputs (logits masked to valid tokens) | [Willard & Louf 2023] |

### Existing section: Ingestion

| claim | cite |
|---|---|
| Qwen3-Embedding-0.6B (currently only a model-card link) | [Zhang 2025] (Qwen3 Embedding report); base model [Yang 2025] |

---

## Part 2 — Reference list

- **[Alizadeh 2025]** Alizadeh, M., Kubli, M., Samei, Z., et al. "Open-Source LLMs for Text Annotation: A Practical Guide for Model Setting and Fine-Tuning." *Journal of Computational Social Science*, 2025 (preprint 2023). https://arxiv.org/abs/2307.02179
- **[Brown 2020]** Brown, T. B., et al. "Language Models are Few-Shot Learners." *NeurIPS 2020*. https://arxiv.org/abs/2005.14165
- **[Bucher & Martini 2024]** Bucher, M. J. J., & Martini, M. "Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification." arXiv preprint, 2024. https://arxiv.org/abs/2406.08660
- **[Campello 2013]** Campello, R. J. G. B., Moulavi, D., & Sander, J. "Density-Based Clustering Based on Hierarchical Density Estimates." *PAKDD 2013*, LNCS 7819. doi:10.1007/978-3-642-37456-2_14
- **[Chang 2009]** Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., & Blei, D. M. "Reading Tea Leaves: How Humans Interpret Topic Models." *NeurIPS 22*, 2009, pp. 288–296. https://www.umiacs.umd.edu/~jbg/docs/nips2009-rtl.pdf
- **[codecarbon (Zenodo)]** Courty, B., Schmidt, V., et al. "mlco2/codecarbon" (software), v3.1.2. doi:10.5281/zenodo.17772019
- **[Cohen 1960]** Cohen, J. "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement* 20(1), 1960, pp. 37–46. doi:10.1177/001316446002000104
- **[Cover & Hart 1967]** Cover, T. M., & Hart, P. E. "Nearest neighbor pattern classification." *IEEE Trans. Information Theory* 13(1), 1967, pp. 21–27. doi:10.1109/TIT.1967.1053964
- **[DeepSeek-AI 2026]** DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." 2026. https://arxiv.org/abs/2606.19348
- **[Dong 2024]** Dong, Y., et al. "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models." arXiv preprint, 2024. https://arxiv.org/abs/2411.15100
- **[Douze 2024]** Douze, M., et al. "The Faiss library." arXiv preprint, 2024. https://arxiv.org/abs/2401.08281 *(already cited in the report)*
- **[Fix & Hodges 1951]** Fix, E., & Hodges, J. L. "Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties." USAF School of Aviation Medicine, Report 4, 1951; reprinted *International Statistical Review* 57(3), 1989. doi:10.2307/1403797
- **[Gemma Team 2026]** Gemma Team, Google DeepMind. "Gemma 4 Technical Report." 2026. https://arxiv.org/abs/2607.02770
- **[Gilardi 2023]** Gilardi, F., Alizadeh, M., & Kubli, M. "ChatGPT outperforms crowd workers for text-annotation tasks." *PNAS* 120(30), 2023, e2305016120. doi:10.1073/pnas.2305016120
- **[Kwon 2023]** Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP '23*. https://arxiv.org/abs/2309.06180
- **[Lacoste 2019]** Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. "Quantifying the Carbon Emissions of Machine Learning." arXiv preprint, 2019. https://arxiv.org/abs/1910.09700
- **[Lau 2014]** Lau, J. H., Newman, D., & Baldwin, T. "Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality." *EACL 2014*, pp. 530–539. https://aclanthology.org/E14-1056/
- **[Li 2024]** Li, D., et al. "From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge." *EMNLP 2025* (preprint 2024). https://arxiv.org/abs/2411.16594
- **[Lloyd 1982]** Lloyd, S. P. "Least squares quantization in PCM." *IEEE Trans. Information Theory* 28(2), 1982, pp. 129–137. doi:10.1109/TIT.1982.1056489
- **[MacQueen 1967]** MacQueen, J. "Some methods for classification and analysis of multivariate observations." *Proc. 5th Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, 1967, pp. 281–297.
- **[Malkov 2020]** Malkov, Yu. A., & Yashunin, D. A. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI* 42(4), 2020, pp. 824–836. https://arxiv.org/abs/1603.09320 *(already cited in the report)*
- **[McInnes 2017]** McInnes, L., Healy, J., & Astels, S. "hdbscan: Hierarchical density based clustering." *JOSS* 2(11), 2017, 205. doi:10.21105/joss.00205
- **[Mizrahi 2024]** Mizrahi, M., et al. "State of What Art? A Call for Multi-Prompt LLM Evaluation." *TACL* 12, 2024, pp. 933–949. doi:10.1162/tacl_a_00681
- **[Rousseeuw 1987]** Rousseeuw, P. J. "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *J. Computational and Applied Mathematics* 20, 1987, pp. 53–65. doi:10.1016/0377-0427(87)90125-7
- **[Sclar 2024]** Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting." *ICLR 2024*. https://arxiv.org/abs/2310.11324
- **[Stammbach 2023]** Stammbach, D., Zouhar, V., Hoyle, A., Sachan, M., & Ash, E. "Revisiting Automated Topic Model Evaluation with Large Language Models." *EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.581/
- **[Staudt 2016]** Staudt, C. L., Sazonovs, A., & Meyerhenke, H. "NetworKit: A tool suite for large-scale complex network analysis." *Network Science* 4(4), 2016, pp. 508–530. doi:10.1017/nws.2016.20
- **[Tam 2024]** Tam, Z. R., et al. "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models." *EMNLP 2024 Industry Track*. https://arxiv.org/abs/2408.02442
- **[Törnberg 2023]** Törnberg, P. "ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning." arXiv preprint, 2023. https://arxiv.org/abs/2304.06588
- **[Traag 2019]** Traag, V. A., Waltman, L., & van Eck, N. J. "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports* 9:5233, 2019. doi:10.1038/s41598-019-41695-z
- **[Willard & Louf 2023]** Willard, B. T., & Louf, R. "Efficient Guided Generation for Large Language Models." arXiv preprint, 2023. https://arxiv.org/abs/2307.09702
- **[Yang 2025]** Yang, A., et al. (Qwen Team). "Qwen3 Technical Report." arXiv preprint, 2025. https://arxiv.org/abs/2505.09388
- **[Zhang 2025]** Zhang, Y., et al. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." arXiv preprint, 2025. https://arxiv.org/abs/2506.05176
- **[Zheng 2023]** Zheng, L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023* (Datasets and Benchmarks). https://arxiv.org/abs/2306.05685
- **[Zheng 2024]** Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." *NeurIPS 2024*. https://arxiv.org/abs/2312.07104
- **[Ziems 2024]** Ziems, C., et al. "Can Large Language Models Transform Computational Social Science?" *Computational Linguistics* 50(1), 2024, pp. 237–291. doi:10.1162/coli_a_00502

---

## Part 3 — Caveats and flags

- **Preprint-only references** (no peer-reviewed venue as of 2026-08): Törnberg 2023, Bucher & Martini 2024, Willard & Louf 2023, Dong 2024 (XGrammar — arXiv is the safe citation; the MLSys 2025 venue claim was not confirmed), Douze 2024, and the model technical reports. Fine for a technical report; avoid leaning on them for contested claims.
- **Chang 2009 has no DOI/arXiv** — cite via the NeurIPS proceedings.
- **Prompt non-transfer across model sizes**: no single canonical paper states exactly our finding. Sclar 2024 (formatting sensitivity) + Mizrahi 2024 (rankings change with template) are the safe pair; arXiv:2512.01420 ("PromptBridge", preprint, partially verified) makes the explicit cross-model claim if wanted.
- **Structured-decoding quality debate**: Tam 2024 found degradation; arXiv:2501.10868 (2025, verified by title/ID only) attributes it to prompt/schema setup rather than constrained decoding itself. Cite both or neither.
- **Small-model competitiveness caveat**: Bucher & Martini and Alizadeh et al. show *fine-tuned* small models beating *zero-shot* large ones — our setup (few-shot prompted, not fine-tuned) is adjacent, not identical; phrase accordingly.
- **codecarbon**: the Zenodo DOI above is version-specific (v3.1.2); check the record's "cite all versions" concept DOI before final submission.
- **Novelty note**: Stammbach 2023 is the nearest precedent (LLMs doing Chang-style intrusion on *topic models*); no verified paper does intrusion-style LLM evaluation of *embedding clusters* as its main contribution — the report can modestly claim that adaptation.
