# RISS-AI

## Expanded dataset

The expanded profile set contains 50 brands and 200 creators. Profiles added
for this expansion are fictional synthetic examples. To regenerate the profile
files and pair inventories from the seed records, run `python
build_expanded_pairs.py` from the repository root.

- `data/brands.json` and `data/creators.json` contain all expanded profiles.
- `data/annotation_pairs.json` is the 2,000-pair Phase 1 annotation batch,
  with the specified 400/600/600/200/100/50/50 sampling categories.
- `data/all_annotation_pairs.json` contains all 10,000 brand–creator pairs.
- `data/labels.json` retains the original 400 manually supplied labels; newly
  generated pairs are unlabelled and ready for annotation.
