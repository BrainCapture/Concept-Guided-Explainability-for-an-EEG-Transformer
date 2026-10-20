# Florian, look at this: paper vs. code

As of 2026-10-20. Shared doc version: https://claude.ai/code/artifact/2293080a-d2b3-41c2-a8a9-6b040164fb25

Checking the manuscript against the code turned up gaps we cannot close without you. The headline findings hold:
gped, pled, high temporal delta and high occipital DFA are robust under every variant we tried. But several method
descriptions do not match the code, and Results IV-B/C do not match your own final TCAV tables.

We need answers to the questions below, any intact copies of the notebooks and model weights, and three short
re-runs if you still have the VM or bucket. Comment on this file in a PR, open an issue, or edit it directly.

## What survived

Most large files in both copies we have are cut off at exactly 512 KiB, 768 KiB or 1 MiB. In the folder export,
18,947 files are 0 bytes, including every HDF5 file. The cut points are identical in the GitHub repo and the export,
so the damage happened before either upload.

| Item | State |
| --- | --- |
| `python_files/*.py` | Intact |
| `TCAV_classifier_explorations`, `TCAV_concept_explorations`, `randomforest_pipeline`, `biomarker_pipeline` and 5 other notebooks | Truncated at 1 MiB; only the first code cells recovered |
| `transformer_explorations.ipynb` | Truncated at 768 KiB |
| `pretrained_models/bendr_tuh_02_2025.ckpt` | Truncated at 1.25 MiB |
| `model_weights/finetuned/*.ckpt` | Missing |
| `high_biomarker_region.csv`, `random_biomarker_region.csv` | Truncated at 512 KiB |
| `results/cav/figures/all_concepts_{01,13,14,17,18}[_FDR].pkl` | Intact |
| `results/feature_pruning/TUAB_clean.pkl` and log | Intact |
| `results/TCAV_concepts/tuh_eeg_events/events.csv` | Intact |
| `model_weights/finetuned/README.md` | Intact, but stops at model 15 |
| TUEV 60-s window extraction code | Not found anywhere |

This repo has the pipeline as scripts (format, biomarkers, RF, concepts, fine-tuning, TCAV, analysis). It runs end
to end on synthetic data. Preprocessing and fine-tuning still depend on the internal `eegprep` package.

## Already settled from your result files

These need no re-run. They need your confirmation and, where noted, a change to the manuscript.

1. **FDR.** The p-values in `all_concepts_<seed>_FDR.pkl` are Benjamini-Hochberg adjusted jointly over all 875 tests
   (5 seeds × 35 concepts × 5 layers), q = 0.05. `cgx/tcav/stats.py` reproduces your FDR tables exactly from the
   uncorrected ones. The paper could say the family is all 875 tests.
2. **Which RF is in the paper.** `TUAB_clean.pkl` holds 4 pruning runs of 15 iterations each. The second run is the
   paper's model: its 14-feature step has TUAB-eval F1 0.7783 (the reported 77.8%), CV F1 0.7777, and exactly the 14
   features used as TCAV top concepts. Its grid-search optimum was depth 11, min 2 samples per leaf and 280 trees,
   not the depth 20 / 5 per leaf / 100 trees in Sec. III-C.
3. **Encoder freezing.** Models 01, 13 and 14 were trained with `freeze_encoder: False` (seeds 0, 5 and 6 in your
   models README). Sec. III-B says the encoder was frozen and only the head was trained. Your README also shows a
   frozen encoder dropping F1 to about 70%, so the unfrozen setup was presumably intended; the text needs to change.
4. **Seeds.** The five TCAV models are 01, 13, 14, 17 and 18. Models 17 and 18 are not in the README, and there were
   at least 10 seeded TUAB runs.

## Open issues

| # | Issue | Evidence | Question for you | Needs a re-run? |
| --- | --- | --- | --- | --- |
| 1 | RF subset chosen on TUAB-eval | `iterative_feature_pruning` keeps the subset with the best `f1_test`, and `randomforest.py` passes TUAB-eval as the test set; SHAP pruning is also computed on it. The paper says eval was used only for the final report. | Was the 14-feature model picked as the best-by-eval step of the second run, and why 4 runs? | Yes: redo selection on CV F1 (TUAB train only), then check the top-14 set and eval accuracy |
| 2 | TCAV inputs not normalized | Training applies `StandardizeLabel` (clip to ±50 µV, divide by 1e-5). `make_input_tensor` and `get_dataset_from_examples` read raw `f['data']`, so TCAV inputs are about 1e5 times smaller than training inputs. | Was this intended, or were the HDF5 files already scaled? | Yes: model accuracy on the raw TCAV tensors, then TCAV with normalized inputs for at least one seed |
| 3 | `enc_augment` is random | `init_from_contextualizer` calls `load_state_dict` on the whole Lightning checkpoint with `strict=False`. No keys match, so nothing loads, and the layer is then frozen. One of the five probed layers therefore has random, seed-dependent weights. | Did you know? Should the paper describe it, or should we load `contextualizer.*` properly? | Only if we fix it (that needs retraining) |
| 4 | Quality screen threshold | `evaluate_quality` uses `min_nchans=10`; Sec. III-A says at least 15 valid channels. | Which one did you run? | No |
| 5 | Results IV-B/C wording | See the next section. | Which tables or settings produced Figs. 4 and 5? | No, if you have the figure notebook |
| 6 | TUEV windows | `events.csv` exists, but the code that built the 60-s event windows (merge, truncate, pad, 20-s overlap rule) is gone. | Do you still have it? | No |

## Results IV-B/C against your tables

The text of IV-B/C claims a sharper contrast than your FDR tables and your own `consistency_2L.png` show. The figure
and the tables agree with each other. The numbers below are the share of the 10 (layer × seed) conditions in the
summarizer and extended classifier that are both FDR-significant and accurate.

| Claim in the manuscript | Your FDR tables | With Bonferroni |
| --- | --- | --- |
| Median top-feature concept passes in "approximately half" of conditions | 0.85 | 0.70 |
| Bottom-feature concepts are "rarely robust" | median 0.80 | median 0.65 |
| `high alpha peak freq occipital` is "weaker" | 0.9, positive in 9 of 10 | not checked |
| Probe accuracy is below 0.60 in encoder and encoder-augment | Biomarker medians 0.600 and 0.589, the same as the summarizer (0.600). Events 0.714 / 0.687 vs 0.786 | same |
| gped and pled are robust; high delta abs temporal and high dfa occipital are most stable | gped 0.7, pled 0.9, delta 1.0, dfa 1.0 | 0.7, 0.9, 0.9, 1.0 |

The accuracy criterion decides most biomarker outcomes. Each probe is tested on 20 samples (10% of 100 concept plus
100 random examples), so accuracy moves in steps of 0.05. In 318 of the 725 biomarker (concept, layer, seed) cases
the median is exactly 0.60. These pass only because the stored float32 value 0.6000000238 is greater than 0.6. Under
the paper's wording, "must exceed 0.60", no median top or bottom feature would pass, while gped and pled still would.

We suggest:
- rewriting IV-B/C to match the figure
- stating the criterion as "at least 0.60"
- noting the 20-sample test split as a limitation; a larger probe test set would make the criterion meaningful

## What we need from you

Answers:

- [ ] Issue 1: how the 14-feature RF was chosen, and why there were 4 pruning runs
- [ ] Issue 2: whether TCAV inputs were meant to skip `StandardizeLabel`
- [ ] Issue 3: whether you knew `enc_augment` was random
- [ ] Issue 4: which channel threshold the quality screen used
- [ ] Issue 5: which notebook and settings produced Figs. 4 and 5
- [ ] Seeds and configs of models 17 and 18, and why these five models were chosen

Files, if a copy still exists on the VM, the GCS bucket or your laptop:

- [ ] Intact `TCAV_classifier_explorations.ipynb` and `TCAV_concept_explorations.ipynb`
- [ ] Intact `randomforest_pipeline.ipynb` and `biomarker_pipeline.ipynb`
- [ ] `bendr_tuh_02_2025.ckpt` and `tuh_eeg_tuab_60.0_clean_finetuned_{01,13,14,17,18}.ckpt`
- [ ] The TUEV 60-s window extraction code
- [ ] Full `high_biomarker_region.csv` and `random_biomarker_region.csv`

Re-runs, only if the data are still on the VM:

- [ ] RF selection on CV F1 only, reporting the top-14 set and eval accuracy
- [ ] Model accuracy on the raw TCAV input tensors for one seed
- [ ] TCAV for one seed with `StandardizeLabel` applied to inputs and concepts
