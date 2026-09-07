# Vision model card — PlantDoc tomato EfficientNet-B0

## Intended use

Preliminary classification of supported crop/disease classes from farmer-submitted field photographs, followed by calibrated abstention and human review. It is not intended for autonomous pesticide selection, dosage, legal compliance, poisoning response, or diagnosis outside the trained classes.

## Training data

- Dataset: official PlantDoc classification repository, downloaded 2026-09-02.
- License: Creative Commons Attribution 4.0 International; cite the PlantDoc authors and paper.
- Scope: eight tomato classes occurring in both official train and test splits.
- Split after cleaning: 552 train, 97 validation, and 68 official test images.
- Cleaning: two conflicting-label digest groups removed, four ambiguous images quarantined, and
  two exact train/test duplicates removed from training.
- Excluded label: `Tomato two spotted spider mites leaf` because it has no official test examples.
- Known gaps: small per-class samples, internet-source bias, unknown device/geography/season mix,
  and weak evidence for Indian farmer phone photographs.

The Windows CPU training profile uses a physical batch size of 4 and accumulates gradients over
four steps, retaining an effective batch size of 16 with lower peak activation memory. Training
writes both a best deployable checkpoint and a last-training checkpoint so interrupted runs can
resume.

## Split strategy

Record the pixel-hash grouping, train/validation/test counts, seed, and any manual duplicate review. State whether the same plant, capture burst, or source webpage might still cross splits.

## Evaluation

- Accuracy: 0.5294.
- Macro F1: 0.5094.
- Expected calibration error: 0.1081.
- Coverage at confidence 0.65: 0.2647.
- Selective accuracy at confidence 0.65: 0.8333 (18 covered examples).
- Best class F1: late blight 0.7619; weakest: early blight 0.2857.
- The frozen checkpoint SHA-256 is recorded by `scripts/prepare_deployment.ps1` in the deployment
  manifest because the trained binary is produced on the user's machine.

## Limitations

The model supports only its eight tomato classes and cannot reliably identify unknown diseases,
nutrient deficiencies, pests, multiple simultaneous conditions, or non-tomato crops. Look-alike
symptoms, early/late stages, occlusion, blur, lighting, background, and geographic shift can cause
errors. Training macro F1 approached 0.90 while validation macro F1 was about 0.59, indicating
overfitting. Official-test performance must not be represented as field performance.

## Human factors

Evaluate whether confidence wording is understood, whether farmers can provide a second image, reviewer response time, disagreement rates, and whether translated warnings retain their meaning.

## Reproducible walkthrough cases

`scripts/select_demo_images.py` scores the frozen official test split and exports held-out examples
for safe abstention, confidence-band review, dosage-triggered human review, and a healthy leaf when
available. Its manifest records the true label, measured prediction, confidence, and confirms that
the image was not used during training. These curated cases demonstrate control flow; they do not
replace aggregate evaluation.
