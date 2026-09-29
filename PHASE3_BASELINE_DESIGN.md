# Phase 3: Baseline ML Ranking System Design

## 1. Problem formulation

The production task is **learning to rank**: for one brand, order its candidate creators. The brand acts like a search query, and each creator is a candidate result.

For V0, treat the labels as a **three-class classification problem and turn the predicted probabilities into a ranking score**. With only 400 pairs and 10 brand groups, this is a lower-complexity starting point than a direct ranking model, while still using all three labels. For each creator, calculate:

```text
score = 2 × P(good) + 1 × P(maybe)
```

This encodes an explicit relevance scale: `poor = 0`, `maybe = 1`, `good = 2`. Sort creators within each brand by this score. It is a practical surrogate for the ranking task, not a claim that the label gaps are objectively equal.

A direct learning-to-rank model is a plausible later experiment: it learns comparisons within brand groups and can optimize ranking metrics such as NDCG. But there are only 10 groups to learn across. XGBoost's ranking interface explicitly uses query groups and graded relevance, and its documentation cautions that ranking is a sophisticated task requiring care in generalization and tuning. [XGBoost learning-to-rank documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)

## 2. Candidate models

| Candidate | What it learns | Strengths | Risks here |
|---|---|---|---|
| **L2 Logistic Regression** | Linear decision boundaries among `good`, `maybe`, and `poor` | Low complexity; works with 20 scaled numeric features; coefficients give a useful global explanation | Misses complex interactions; correlated features can make coefficients unstable; class imbalance can reduce minority-class recall |
| **Random Forest / Gradient Boosting** | Nonlinear feature splits and interactions | Can capture rules such as “niche match matters most when platform coverage is present” | Only 34 good examples; trees can fit annotation quirks, and feature importance may be unstable. Boosting adds more tuning choices |
| **XGBoost / LightGBM** | Boosted trees; both also offer ranking objectives | Can model nonlinearities; LambdaMART objectives directly optimize ranking surrogates | Added dependency and tuning complexity; 10 brand groups are a weak basis for judging generalization. Defer direct ranker comparison until there are more brands |
| **Learning-to-rank model** | Pairwise or listwise preferences within each brand's creator list | Closest match to production ranking; can directly use graded relevance and query groups | Only 10 query groups; model performance and tuning would be hard to assess reliably |

**Recommended V0 model: L2-regularized multinomial Logistic Regression**, with standardized features. Rank by expected relevance, `2P(good) + P(maybe)`. It is not the most powerful candidate; it is the simplest candidate that uses all three labels and gives a reviewable first comparison. Do not include brand IDs or creator IDs as predictors.

Evaluate unweighted and class-weighted fitting as a limited training-only comparison. Weighting can improve minority-class learning, but it changes the effective class priors, so its scores should not be treated as calibrated probabilities without checking calibration.

## 3. Data split and leakage

The files confirm **10 brands, 40 creators, and 400 labeled pairs**. Each brand is paired with all 40 creators. The observed labels are 34 good, 35 maybe, and 331 poor.

Randomly splitting pairs would put pairs from the same brand in both train and test. That tests ranking for a brand partly seen during training, while production asks about a **new brand**. Split by `brand_id` so every pair for a brand stays in one partition. Group splitters enforce non-overlapping groups; stratified group splitting tries to preserve class proportions, but cannot always do so when there are few groups. [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

A concrete **provisional** split:

| Partition | Brands | Pairs | good / maybe / poor |
|---|---|---:|---:|
| Train | b02–b09 | 320 | 25 / 16 / 279 |
| Validation | b10 | 40 | 4 / 12 / 24 |
| Test | b01 | 40 | 5 / 7 / 28 |

This keeps all three classes in each partition and holds out whole brands. Use brand IDs **only for splitting and grouping**, never as model features. Keep the test brand untouched until the final comparison.

One brand each in validation and test makes those metrics very noisy. Ten brands are not enough for a dependable estimate of performance on all future brands, and there is no split that removes that limitation. For development, also report leave-one-brand-out results across the eight training brands. Treat those as exploratory estimates, not substitutes for more labeled brands.

## 4. Class imbalance

Poor accounts for about **83%** of labels. A classifier that always predicts poor would have about 83% accuracy, while finding no good creators. This is why accuracy cannot be the main measure.

Do not automatically oversample. Duplicating rare rows can overemphasize particular brands, and synthetic rows could represent unrealistic brand–creator combinations. Instead:

- Keep the natural distribution for validation and test.
- Compare unweighted fitting with a modest, documented class-weighting option using training data only.
- Evaluate good-creator retrieval and ranking metrics.
- Avoid interpreting class-weighted probabilities as calibrated without validation.

For a direct ranker, the poor class is not simply “the majority class”; its labels contribute to within-brand comparisons. But only 10 brand groups remain a serious limitation.

## 5. Evaluation metrics

Define relevance consistently:

- For **Precision@K**, **Recall@K**, and **MRR**, count only `good` as relevant. A maybe is a possible compromise, not a known strong hire.
- For **NDCG@K**, use graded relevance: `poor = 0`, `maybe = 1`, `good = 2`. This gives partial credit for ranking maybes above poor matches. NDCG is designed to reward relevant items appearing near the top of a ranked list. [scikit-learn NDCG documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)

Report **Precision@3, Precision@5, Recall@5, MRR, and NDCG@5**, calculated separately per brand, then macro-averaged across brands. NDCG@5 should be the primary metric because the goal is a useful short list and it retains the three-level label information. Precision@5 and MRR make “good” creator retrieval easy to interpret. Also report each brand's scores; an average over one test brand is not evidence of broad generalization.

Accuracy, macro-F1, per-class precision/recall, and a confusion matrix are useful secondary diagnostics for the classification model. ROC-AUC can also be secondary, but a pair-level ROC-AUC does not assess within-brand top-of-list quality. For every metric, calculate within each brand's 40-creator list rather than treating all 400 pairs as one ranking.

## 6. Non-ML baseline

Build a fixed, transparent rule score before training. One workable version averages eight dimension scores:

1. **Niche:** combine primary match with capped secondary overlap.
2. **Audience:** average age overlap, gender match, and audience-location overlap.
3. **Geography:** use the target-country and mandatory-region indicators.
4. **Platforms:** platform coverage.
5. **Budget:** combine budget-fits-min with a capped, normalized log budget ratio.
6. **Size:** apply the annotation guide's in-range / near / outside follower rule directly to the profiles.
7. **Content:** content-type Jaccard similarity.
8. **Requirements:** mandatory-requirements-met.

Subtract a fixed penalty for `has_excluded_trait`. Freeze the weights and penalty before looking at validation or test results; do not tune the rule score on the test brand.

Compare this baseline with the logistic model on identical brand splits and metrics. It shows whether supervised learning adds value beyond explicit matching rules. It can also reveal label inconsistencies or feature problems if the model performs worse than the understandable baseline.

One feature caution: `followers_percentile_in_range` is clamped to `[0,1]`; creators above the brand's maximum can therefore receive the same value as creators at the top of the range. The rule baseline should use actual follower boundaries rather than relying on that feature alone.

## 7. Experiment structure

Keep the workflow explicit:

**profiles and labels → feature extraction → brand-group split → rule baseline → logistic model → per-brand ranking metrics → saved report**

For reproducibility, save:

- Dataset/version or hashes and label counts.
- Feature names, order, FX configuration, and feature-pipeline version.
- Exact brand IDs assigned to train, validation, and test.
- Label-to-relevance mapping and metric definitions, including K values.
- Model parameters, feature-scaling parameters, class-weighting choice, random seed, Python/package versions, and code revision.
- Per-brand predictions: brand ID for grouping, creator ID for joining results, true label, predicted probabilities, ranking score, and rank. IDs belong in evaluation outputs, not the model feature matrix.
- Aggregate and per-brand metrics, plus the fixed rule-baseline results.

## Final recommendation

- **Problem formulation:** Production is a grouped ranking task; V0 uses three-class classification probabilities to produce a graded ranking score.
- **V0 model:** Standardized, L2-regularized multinomial Logistic Regression; rank by `2P(good) + P(maybe)`.
- **Train/validation/test strategy:** Group by brand; provisionally train b02–b09, validate b10, test b01. Add leave-one-brand-out analysis on the training brands.
- **Primary metrics:** Macro per-brand NDCG@5; also Precision@3/5, Recall@5, and MRR with `good` as relevant.
- **Secondary metrics:** Macro-F1, per-class precision/recall, confusion matrix, and accuracy; ROC-AUC only as a supplementary diagnostic.
- **Baseline:** Frozen rule score built from fit dimensions and a fixed excluded-trait penalty.
- **Main risks:** Only 10 independent brand groups; severe class imbalance; incomplete mandatory-requirement parsing; possible annotation inconsistency; correlated features; and the follower-percentile boundary behavior.
- **What we should implement next:** After design review, implement the split manifest, rule baseline, logistic pipeline, grouped metric calculations, and reproducible experiment report. Add no model or training code before that review.
