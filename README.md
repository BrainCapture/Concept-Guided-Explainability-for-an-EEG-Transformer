# Concept-Guided Explainability for a Transformer in EEG Abnormality Detection

Code for the paper

> **Concept-Guided Explainability for a Transformer in EEG Abnormality Detection**
> Florian David, William Lehn-Schiøler, Nicolai Spicher
> EPFL Neuro-X Institute · Technical University of Denmark · BrainCapture

We pair a transformer trained on raw 60-s EEG segments with a random forest (RF) trained on 150 qEEG biomarkers for the
same task (TUH Abnormal, ~82% accuracy each). RF feature importance (TreeSHAP) selects candidate biomarker concepts,
and Testing with Concept Activation Vectors (TCAV) measures whether the transformer's internal representations align
with those concepts and with clinically labelled EEG events. Robustness is assessed across five training seeds, five
probing layers and 100 random counter-concept sets.

## Pipeline

| Step | Paper | Script | Module |
|---|---|---|---|
| 1. Select the first clean 60-s window per recording | III-A | `scripts/format_tuh_60s_clean.py` | — |
| 2. Preprocess windows to HDF5 for the transformer | III-A | `eegprep` (see [below](#dependencies-on-eegprep)) | — |
| 3. Extract 25 channel-wise biomarkers | III-C, Table I | `scripts/extract_biomarkers.py` | `cgx/biomarkers/features.py` |
| 4. Train RF with SHAP-based feature pruning | III-C | `scripts/train_random_forest.py` | `cgx/biomarkers/random_forest.py` |
| 5. Build biomarker, age and random concepts | III-D | `scripts/build_concepts.py` | `cgx/concepts/concept_generation.py` |
| 6. Fine-tune Linear-Head BENDR (×5 seeds) | III-B | `scripts/finetune_transformer.py` | `cgx/transformer/` |
| 7. Run TCAV per concept, layer and seed | III-E | `scripts/run_tcav.py` | `cgx/tcav/tcav_utils.py` |
| 8. Significance (BH-FDR), separability and stability | III-E/F | `scripts/analyze_tcav.py` | `cgx/tcav/stats.py` |

TCAV probes the five sequential blocks of the model: `encoder`, `enc_augment`, `summarizer`, `extended_classifier`
and `clf`.

## Installation

```bash
git clone https://github.com/BrainCapture/Concept-Guided-Explainability-for-an-EEG-Transformer.git
cd Concept-Guided-Explainability-for-an-EEG-Transformer
python -m venv .venv && source .venv/bin/activate   # Python 3.10 or 3.11
pip install -r requirements.txt
pip install -e .
```

## Data

All data come from the Temple University Hospital EEG corpus. TUH data cannot be redistributed; request access at
<https://isip.piconepress.com/projects/nedc/html/tuh_eeg/>.

| Dataset | Version | Used for |
|---|---|---|
| TUH Abnormal (TUAB) | v3.0.1 | Training and evaluating both classifiers |
| TUAB Extended Balanced (TUABEXB) | Kiessner et al. 2023 | Biomarker-concept and random counter-concept pools |
| TUH Events (TUEV) | v2.0.1 | Event concepts (spsw, gped, pled, eyem, artf, bckg) |

## Usage

The commands below assume the raw data sit under `data/`. Every script takes `--help`.

```bash
# 1. Annotate the first clean 60-s window of every TUAB / TUABEXB recording
python scripts/format_tuh_60s_clean.py --input data/tuh_eeg_abnormal/v3.0.1/edf \
    --output data/tuh_eeg_abnormal/formatted_60.0_clean --temp data/temp/tuab
python scripts/format_tuh_60s_clean.py --input data/tuabexb/v2.0.1/edf --montage-subfolder "" \
    --output data/tuabexb/formatted_60.0_clean --temp data/temp/tuabexb

# 3. Biomarkers (writes <subset>/{normal,abnormal}.pkl)
python scripts/extract_biomarkers.py --formatted data/tuh_eeg_abnormal/formatted_60.0_clean --out-dir results/biomarkers/TUAB
python scripts/extract_biomarkers.py --formatted data/tuabexb/formatted_60.0_clean --out-dir results/biomarkers/TUABEXB

# 4. Random forest with iterative SHAP feature pruning
python scripts/train_random_forest.py --biomarker-dir results/biomarkers/TUAB --save-as results/feature_pruning/TUAB.pkl

# 5. Concepts (N=100 per concept, M=100 random counter-concepts)
python scripts/build_concepts.py --biomarker-dir results/biomarkers/TUABEXB --ages data/tuabexb/ages.csv \
    --out-dir results/concepts

# 6. Fine-tune the transformer, once per seed
python scripts/finetune_transformer.py --config configs/finetune_tuab_60s.yaml --seed 1 --name lhb_seed1

# 7. TCAV for biomarker/age concepts and for event concepts
python scripts/run_tcav.py --concept-set biomarkers \
    --model model_weights/finetuned/lhb_seed1.ckpt --encoder pretrained_models/bendr_tuh_02_2025.ckpt \
    --test-data data/tuh_eeg_abnormal/preprocessed_60.0_clean/eval \
    --random-concepts results/concepts/random_biomarker_region.csv --random-data data/tuabexb/preprocessed_60.0_clean/eval \
    --positive-concepts results/concepts/high_biomarker_region.csv --concept-data data/tuabexb/preprocessed_60.0_clean/train \
    --out-dir results/tcav/seed1
# ... and again with --concept-set events --positive-concepts results/concepts/events.csv \
#     --concept-data data/tuh_eeg_events/v2.0.1/preprocessed_60.0 --out-dir results/tcav/seed1

# 8. Mann-Whitney U vs. random concepts, BH-FDR (q = 0.05) jointly over seeds, concepts and layers,
#    probe accuracy > 0.60, and stability over the summarizer and extended classifier layers
python scripts/analyze_tcav.py --seed-dirs results/tcav/seed1 results/tcav/seed2 results/tcav/seed3 \
    results/tcav/seed4 results/tcav/seed5 --out-dir results/tcav/summary
```

`--ages` takes a CSV with columns `path_local` (path of the formatted EDF) and `age`. Without it, the age concept is
skipped.

For event concepts, run `run_tcav.py --concept-set events` with an `events.csv` holding 100 windows per event class:
the preprocessed TUEV recording (`files`), the window start in seconds (`t_starts`), the concept name (`concept`,
e.g. `gped_60`) and a numeric concept `id`.
Windows are 60 s long: contiguous same-label annotations are merged into blocks, longer blocks are truncated, shorter
ones are padded symmetrically, and windows overlapping another class by more than 20 s are discarded.

### Dependencies on eegprep

Step 2 (HDF5 preprocessing) and step 6 (fine-tuning) use BrainCapture's `eegprep` package, which is not yet public.
Biomarker extraction, the random forest, concept construction and TCAV run without it. `run_tcav.py` reads the
preprocessed HDF5 files directly. Each file holds `data` (windows × 19 channels × 15,360 samples at 256 Hz), `labels`,
`file_idxs`, `time_slices` and a `files` attribute listing the source EDF stems.

## Model weights

The BENDR encoder was pretrained with a contrastive self-supervised objective on the full TUH EEG corpus. Pretrained
and fine-tuned weights are available from the authors on request.

## Citation

```bibtex
@article{david2026conceptguided,
  title   = {Concept-Guided Explainability for a Transformer in EEG Abnormality Detection},
  author  = {David, Florian and Lehn-Schi{\o}ler, William and Spicher, Nicolai},
  year    = {2026}
}
```

## Acknowledgements

The BENDR encoder follows Kostas et al., *BENDR: Using transformers and a contrastive self-supervised learning task to
learn from massive amounts of EEG data* (Front. Hum. Neurosci., 2021). The Linear-Head BENDR follows Madsen et al.,
*Concept-based explainability for an EEG transformer model* (MLSP 2023). TCAV is computed with
[Captum](https://captum.ai/). Biomarkers use [FOOOF](https://fooof-tools.github.io/fooof/) and
[AntroPy](https://raphaelvallat.com/antropy/).

## License

MIT. See [LICENSE](LICENSE).
