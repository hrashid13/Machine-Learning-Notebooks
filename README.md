# Machine Learning Portfolio

Four assignments from **ISM 6251: Machine Learning** (University of South Florida, Fall 2025). Each folder holds the Jupyter notebook, the data it uses, and the assignment prompt. Everything is Python (pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, SciPy).

More context on the course and my write-up: [heshamrashid.org/Projects/Details/4](https://www.heshamrashid.org/Projects/Details/4)

## At a glance

| # | Assignment | Data | Methods | Headline result |
|---|---|---|---|---|
| 1 | [Data cleaning and decision trees](Assignment%201) | `hmeq.csv` (5,960 home-equity loans, target `BAD` = default) | Missing-value handling, outlier removal, decision tree (default vs. tuned), feature importance | Default tree: 86.9% accuracy, AUC 0.774. Tuned tree: AUC 0.864 and 85% precision on defaults, at 39% recall |
| 2 | [Predicting household income](Assignment%202) | `income_data.csv` (8,993 households, 9 income brackets, 13 demographic predictors) | Chi-square and Cramér's V, CART, pruning, spurious-variable test, gini vs. entropy, random forest | Tree overfit badly (91% train, 28% test); random forest and tuning gave only small gains |
| 3 | [Detecting earnings manipulation](Assignment%203) *(team)* | 220-company sample (39 manipulators) and 1,239-company full set (39 manipulators, 31:1 imbalance) | Logistic regression, CART, class weights, SMOTE, random forest, gradient boosting | Standard model caught 25% of manipulators; SMOTE raised recall to 75% at a precision cost |
| 4 | [Neural networks and Naive Bayes](Assignment%204) *(team)* | Worked problems, plus a breast-cancer recurrence dataset | Neural-network forward pass, building Boolean networks, Naive Bayes by hand | Written derivations (no notebook) |

## Assignment 1: Data cleaning and decision trees

- **Cleaning:** Profiled missing values across the loan data, wrote a function that removes outliers across the continuous variables together (206 rows dropped, 5,754 kept), and one-hot encoded categorical features. Also cleaned an ovarian-cancer clinical dataset (type fixes, recoding, coercing bad entries to missing).
- **Model:** 70/30 train/test split (4,027 / 1,727). A default tree (depth 22, 388 leaves) reached 86.9% accuracy but only 0.62 recall on defaults. A tuned tree reached AUC 0.864 and 0.85 precision on defaults, but recall fell to 0.39, so it flags fewer defaults and is right more often when it does.
- **Takeaway:** Accuracy hides the trade-off. The right tree depends on whether a missed default or a false alarm costs more. In the tuned tree, debt-to-income ratio (`DEBTINC`) accounts for about 89% of feature importance, followed by delinquencies (`DELINQ`).
- A by-hand decision-tree exercise (information gain on 10 students) is in the write-up PDFs.

## Assignment 2: Predicting household income

- **Setup:** 60/40 split; missing demographic values imputed; relationships explored without a model first (chi-square tests, Cramér's V, Spearman correlation).
- **What the exploration found:** Householder status, dual income, age and education had the strongest association with income.
- **Results:** The default CART tree grew to 3,163 leaves (depth 28) and hit 91% training accuracy but only about 28% on the test set (9 classes, so chance is about 11%). A tree pruned to 500/100 minimum node sizes fell to 34.8% training accuracy, trading fit for simplicity. Adding 10 random, irrelevant variables pushed training accuracy to 100% and test accuracy to 24.9%, showing how an unpruned tree memorizes noise.
- **Random forest:** 100 trees reached about 30.7% test accuracy, a 2.5-point gain. Going from 10 to 200 trees lowered test error from 70.6% to 68.7%, with returns flattening after about 100.
- **Takeaway:** A 9-way income target from demographics alone is hard. The useful lesson was how to diagnose overfitting and how much variance an ensemble removes.

## Assignment 3: Detecting earnings manipulation (team assignment)

Built with a teammate (Team Rodeo). Predicts whether a company manipulates earnings using eight financial ratios (DSRI, GMI, AQI, SGI, DEPI, SGAI, ACCR, LEVI), with only about 3% of companies being manipulators.

- **Balanced sample (220 cases):** Logistic regression reached 86.4% accuracy on the test set (precision 0.64, recall 0.58 for manipulators). A depth-limited CART tree did worse (76% accuracy, AUC 0.59).
- **Full data (1,239 cases, 31:1 imbalance):** A standard logistic regression scored 97.6% accuracy but found only 2 of 8 manipulators (25% recall). Accuracy was misleading because predicting "not a manipulator" for everyone scores about 97%.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression | 97.6% | 1.00 | 0.25 | 0.40 | 0.909 |
| Logistic regression, class weights | 84.7% | 0.13 | 0.63 | 0.21 | 0.915 |
| Logistic regression, SMOTE | 83.1% | 0.13 | 0.75 | 0.22 | 0.915 |
| Random forest (SMOTE) | 89.9% | 0.16 | 0.50 | 0.24 | 0.907 |
| Gradient boosting (SMOTE) | 92.7% | 0.22 | 0.50 | 0.31 | 0.901 |

- **Takeaway:** For fraud-style problems, recall and ROC-AUC matter more than accuracy. Rebalancing (class weights or SMOTE) caught more manipulators at the price of many false alarms, which is acceptable if a regulator reviews flagged cases. Test sets are tiny (8 manipulators), so these numbers are directional, not precise.
- Also covers deployment strategy and a suggested manipulation score (see the PDF write-up).

## Assignment 4: Neural networks and Naive Bayes (team assignment)

Worked by hand and written up as documents rather than a notebook: min-max normalization and a forward pass through a small neural network with sigmoid activations, constructing networks for Boolean functions (AND/OR combinations), and Naive Bayes classification, including a fruit example (about 67% "other") and a breast-cancer recurrence dataset.

## Running the notebooks

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy jupyter
jupyter notebook
```

Open the notebook in each assignment folder. Data files sit beside the notebooks, so paths are relative.

## What I took from the course

- Choose metrics that match the cost of errors; accuracy alone can mislead.
- Always compare training and test performance, and use pruning or ensembles when they diverge.
- Class imbalance changes the whole evaluation approach, not just the model.
- Simple, interpretable models (trees, logistic regression) are strong baselines worth beating before reaching for anything more complex.
