"""Run TCAV on a fine-tuned LinearHeadBENDR for biomarker, age, or event concepts (Sec. III-D/E).

Each positive concept is tested against M random counter-concepts at every probing layer. Scores and linear-probe
statistics are written to `<out-dir>/<concept>.pkl`, and CAVs are cached by Captum in `<out-dir>/cav/`.
"""
import argparse
import os
import pickle

import torch
from captum.concept import TCAV

from cgx.transformer.linearheadbendr import LinearHeadBENDR
from cgx.tcav.tcav_utils import (TCAVClassifier, assemble_concept, get_experimental_set, get_experimental_stats,
                                 get_random_concepts, make_input_tensor)

LAYERS = ["encoder", "enc_augment", "summarizer", "extended_classifier", "clf"]

### BEST FEATURES FROM RF
TOP_BIOMARKERS = ['(alpha+beta)/(delta+theta)_occipital', '(alpha+beta)/(delta+theta)_parietal', 'offset_temporal',
                  '(alpha+beta)/(delta+theta)_temporal', 'delta_abs_temporal', 'alpha_rel_occipital',
                  'theta_rel_parietal', 'theta_abs_temporal', 'alpha_peak_freq_occipital', 'beta_rel_temporal',
                  'theta_abs_frontal', 'alpha_peak_bw_central', 'dfa_occipital', 'alpha_abs_occipital']
### WORST FEATURES FROM RF
BOTTOM_BIOMARKERS = ['delta/alpha_pre-frontal', 'delta/alpha_frontal', 'delta/alpha_temporal', 'delta/alpha_parietal',
                     'delta/alpha_occipital', 'delta/alpha_central', 'theta/beta_temporal', 'delta/beta_central',
                     'delta/beta_frontal', '(alpha+beta)/(delta+theta)_central', 'delta_abs_pre-frontal',
                     'theta/gamma_pre-frontal', 'beta_rel_pre-frontal', 'PermEn_5_pre-frontal']
EVENTS = ['artf_60', 'bckg_60', 'eyem_60', 'gped_60', 'pled_60', 'spsw_60']


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Fine-tuned LinearHeadBENDR checkpoint (.ckpt)')
    parser.add_argument('--encoder', required=True, help='Pretrained BENDR checkpoint (.ckpt)')
    parser.add_argument('--test-data', required=True, help='Preprocessed TUAB eval folder (.hdf5)')
    parser.add_argument('--random-concepts', required=True, help='CSV from build_concepts.py (random_*)')
    parser.add_argument('--random-data', required=True, help='Preprocessed TUABEXB eval folder (.hdf5)')
    parser.add_argument('--concept-set', choices=['biomarkers', 'events'], required=True)
    parser.add_argument('--positive-concepts', required=True,
                        help='Concept CSV: high_biomarker_region.csv (biomarkers) or events.csv (events)')
    parser.add_argument('--concept-data', required=True,
                        help='Preprocessed folder holding the concept examples (TUABEXB train or TUEV)')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--M', type=int, default=100, help='Number of random counter-concepts')
    parser.add_argument('--class-of-interest', type=int, default=1, help='1 = abnormal')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda available:", torch.cuda.is_available())

    # load model of interest

    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)

    model = LinearHeadBENDR(encoder=args.encoder,
                            encoder_h=512,
                            in_features=19,
                            out_features=2)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    tcav = TCAV(model=model, layers=LAYERS, save_path=os.path.join(args.out_dir, 'cav') + '/', classifier=TCAVClassifier())

    test_loader = make_input_tensor(args.test_data, args.class_of_interest).to(device)
    print(test_loader.shape)

    random_concepts = get_random_concepts(concepts_path=args.random_concepts, data_path=args.random_data, M=args.M, device=device)

    if args.concept_set == 'biomarkers':
        biomarker_names = TOP_BIOMARKERS + BOTTOM_BIOMARKERS + ['age']
        positive_concept_names = ['high_' + name for name in biomarker_names]
        events = False
    else:
        positive_concept_names = EVENTS
        events = True

    print("Running TCAV for:", positive_concept_names)

    for positive_concept_name in positive_concept_names:

        print(positive_concept_name)

        positive_concept = assemble_concept(name=positive_concept_name, concepts_path=args.positive_concepts,
                                            data_path=args.concept_data, device=device, events=events)

        experimental_set = get_experimental_set(positive_concept, random_concepts)

        scores_dict = tcav.interpret(inputs=test_loader, experimental_sets=experimental_set, target=args.class_of_interest, processes=1)
        stats_dict = get_experimental_stats(scores_dict, tcav.cavs)

        out_path = os.path.join(args.out_dir, positive_concept_name.replace('/', '-') + '.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump({'concept': positive_concept_name,
                         'random_concepts': [c.name for c in random_concepts],
                         'layers': LAYERS,
                         'scores': to_cpu(scores_dict),
                         'stats': stats_dict}, f)
        print('Saved', out_path)

        del positive_concept, experimental_set, scores_dict, stats_dict

    print("######## END ########")


if __name__ == '__main__':
    main()
