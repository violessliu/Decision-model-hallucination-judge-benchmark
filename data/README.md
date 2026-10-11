# Evaluation data views

The eight JSONL files listed in the repository README are the exact binary views used in `share/hallucination_judge_results_20261010` after its 2026-10-11 FAVABench revision. One JSON object is one benchmark instance. `id` is the join key, `label=1` means the target hallucination type is present, and `label=0` means an eligible negative. `metadata` contains source annotations for analysis and is **not** sent to the judge.

The FAVABench FC and FF views use `binary_inclusive.jsonl`. The 153 negatives in each view have none of the six annotated FC/FF error types. The FC view has 224 positives; the FF view has 198 positives. HADTest positives use the corresponding human-validated injected errors, while its 1,124 `None`-labeled negatives appear in both factual views.

The runtime consumes these ready-made views directly. Upstream raw archives and preprocessing intermediates are not included. Source datasets are [FAVABench](https://huggingface.co/datasets/fava-uw/fava-data), [HADTest](https://github.com/pku0xff/HAD), [RAGTruth](https://github.com/ParticleMedia/RAGTruth), [ContraDoc](https://github.com/ddhruvkr/CONTRADOC), [ROSCOE](https://dl.fbaipublicfiles.com/parlai/projects/roscoe/annotations.zip), and [IF-RewardBench](https://github.com/thu-coai/IF-RewardBench).
