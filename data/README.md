# Data sources and licensing

## PlantDoc classification images

- Official repository: https://github.com/pratikkayal/PlantDoc-Dataset
- Paper: https://arxiv.org/abs/1911.10317
- License: Creative Commons Attribution 4.0 International
- Authors: Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra

The automated real-data workflow clones the official repository into
`data/raw/plantdoc-official`. It preserves the authors' official test set and derives only the
validation set from official training images. The first product scope includes labels matching
`tomato`; this is intentional because the current advisory and market layers are tomato-specific.
The official training split includes `Tomato two spotted spider mites leaf`, while the official
test split does not. The benchmark pipeline records and excludes train-only/test-only labels so
every deployed class has both training evidence and official holdout evaluation.

If Git reports `Already up to date` but `train/` or `test/` is absent, tracked files were
removed from the local checkout. `scripts/train_real_plantdoc.ps1` restores those official
folders from the current Git commit before preparation. The preparer also accepts one accidental
wrapper directory, such as `PlantDoc-Dataset-master/`.

The official dataset also contains some pixel-identical images assigned to different disease
labels. Those ambiguous groups are quarantined from every split instead of choosing a label. The
counts, affected labels, and image digests are recorded in `preparation_summary.json` for audit.

Do not commit downloaded images or derived model artifacts to ordinary Git. Preserve the official
license and citation in reports, model cards, demos, and publications.

## Market data

`scripts/bootstrap_demo.py` creates synthetic software-test data only. It must not be presented as
real market evidence. A verified government/provider source will replace it in the next data phase,
with source URL, retrieval date, field mapping, unit conversion, geographic coverage, and license
documented here.
