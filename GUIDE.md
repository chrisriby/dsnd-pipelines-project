# Guide: StyleSense Recommendation Pipeline

Step-by-step guidance for the Udacity "dsnd-pipelines-project". Each step maps to
one item on the task list and to the rubric. The code snippets are skeletons:
fill in the `...` parts yourself.

You work in `starter/starter.ipynb`. Start Jupyter from the project root:

```
venv\Scripts\activate
jupyter notebook
```

Then choose the kernel **Python (dsnd-pipelines)**.

---

## 0. What you know about the data

| Column | Type | Notes |
|---|---|---|
| Clothing ID | int, but really categorical | 531 distinct values, many rare |
| Age | numeric | |
| Positive Feedback Count | numeric | very skewed (mostly 0) |
| Division / Department / Class Name | categorical | 2 / 6 / 14 values |
| Title, Review Text | text | the strongest signal |
| **Recommended IND** | target (0/1) | **82 % are 1** → imbalanced |

- The CSV has no missing values. The rubric still asks you to "gracefully
  handle null values", so add imputers anyway (new reviews may have gaps).
- Because of the imbalance, accuracy alone is misleading: a model that always
  says "1" gets 82 %. Report precision, recall and F1, especially for class 0.
- Keep the provided `train_test_split` (test_size=0.1, random_state=27).
  **Never fit anything on `X_test`** until the final evaluation.

---

## 1. Data exploration (notebook section "Data Exploration")

Not a checklist item, but it earns the "visualizations" stand-out point and
justifies your later choices. Use `X_train` / `y_train` only.

Ideas:
- Bar chart of the target balance.
- Histogram of `Age` and `Positive Feedback Count` (note the skew).
- Recommendation rate per `Department Name` / `Class Name`.
- Distribution of review length (words) for recommended vs. not.

Write a short markdown cell after each plot: what you saw, and what it means
for the model.

---

## 2. Task 1 – Preprocessing for numerical and categorical data

Goal: separate column lists, one small pipeline per data type.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_FEATURES = ['Age', 'Positive Feedback Count']
CAT_FEATURES = ['Clothing ID', 'Division Name', 'Department Name', 'Class Name']
TEXT_FEATURES = ['Title', 'Review Text']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy=...)),   # median is robust to skew
    ('scaler', ...),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown=..., min_frequency=...)),
])
```

Hints:
- `Clothing ID` is a number but means "which product". Treat it as categorical.
- `handle_unknown='ignore'` (or `'infrequent_if_exist'`) prevents crashes when
  the test set has an ID never seen in training.
- `min_frequency=` groups rare IDs together; this is a good hyperparameter.
- Optional: `np.log1p` on `Positive Feedback Count` via `FunctionTransformer`.

---

## 3. Task 2 – Processing text data (NLP)

Goal: normalize, tokenize and lemmatize the text with spaCy. This covers the
"NLP techniques" rubric item.

Write a custom transformer (a class with `fit` and `transform`), so it can sit
inside the pipeline:

```python
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    """Lowercase, tokenize and lemmatize text columns with spaCy."""

    def __init__(self, text_columns, model='en_core_web_sm'):
        self.text_columns = text_columns
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nlp = spacy.load(self.model, disable=['parser', 'ner'])
        X = X.copy()
        for column in self.text_columns:
            texts = X[column].fillna('').astype(str)
            docs = nlp.pipe(texts, batch_size=500)
            X[column + '_clean'] = [self._clean(doc) for doc in docs]
        return X

    @staticmethod
    def _clean(doc):
        # keep lemmas of alphabetic tokens, drop stop words and punctuation
        ...
```

Hints:
- `token.lemma_`, `token.is_alpha`, `token.is_stop`, `token.lower_`.
- Careful with stop words: spaCy's list contains **"not", "no", "never"**.
  These matter a lot for sentiment ("not flattering"). Keep negations, e.g.
  `token.is_stop and token.lower_ not in {'not', 'no', 'never'}`.
- Load spaCy inside `transform`, not in `__init__` — otherwise saving the
  pipeline with pickle and cloning it in GridSearch causes trouble.
- Test it alone on `X_train.head()` before putting it in the pipeline.

**Speed:** spaCy needs about 2–3 minutes for the full training set. During
grid search it runs once per fold. Section 6 shows how to cache it.

---

## 4. Task 3 – Creating features from text

Goal: turn the cleaned text into numbers, plus a few hand-made features
("new features are created" in the rubric).

**a) TF-IDF** on the cleaned columns:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

TfidfVectorizer(ngram_range=(1, 2), min_df=..., max_features=...)
```
Bigrams catch phrases like "not fit" or "run small".

**b) Engineered features** — a second small transformer or a
`FunctionTransformer` that returns, e.g.:
- review length in words / characters,
- number of `!`,
- share of uppercase letters,
- counts of POS tags (adjectives, adverbs) → stand-out "advanced NLP" point.
  You can compute these in the same spaCy pass as the lemmas to save time
  (add extra `_n_adj` columns in `SpacyLemmatizer.transform`).

---

## 5. Task 4 – One pipeline from raw data to prediction

Goal: everything in one object. The rubric says preprocessing must not happen
outside the pipeline.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

features = ColumnTransformer([
    ('num', num_pipeline, NUM_FEATURES),
    ('cat', cat_pipeline, CAT_FEATURES),
    ('title', TfidfVectorizer(...), 'Title_clean'),
    ('review', TfidfVectorizer(...), 'Review Text_clean'),
    ('extra', ..., [...]),             # your engineered feature columns
])

model_pipeline = Pipeline([
    ('nlp', SpacyLemmatizer(TEXT_FEATURES)),
    ('features', features),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced')),
], memory='cache')

model_pipeline.fit(X_train, y_train)
```

Hints:
- A TF-IDF step gets a **single column name as a string** (`'Title_clean'`),
  not a list — it expects 1-D text.
- `class_weight='balanced'` helps the minority class (not recommended).
- Logistic Regression is fast and strong on TF-IDF. Try
  `RandomForestClassifier` or `LinearSVC` as a comparison if you like.
- Get a quick baseline first: `cross_val_score(model_pipeline, X_train,
  y_train, cv=3, scoring='f1_macro')`.

---

## 6. Task 5 – Fine-tuning

Goal: `GridSearchCV` or `RandomizedSearchCV` on the **whole pipeline** with
**training data only**.

Parameter names use `step__substep__param`:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'features__review__ngram_range': [(1, 1), (1, 2)],
    'features__review__min_df': [2, 5],
    'features__cat__encoder__min_frequency': [5, 20],
    'classifier__C': [0.1, 1, 10],
}

grid_search = GridSearchCV(
    model_pipeline, param_grid,
    cv=3, scoring='f1_macro', n_jobs=-1, verbose=1,
)
grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_
```

Why `memory='cache'` on the pipeline: the spaCy step's parameters never change
in the grid, so scikit-learn reuses its output per fold instead of
re-running spaCy for every parameter combination. Without it, the grid above
(24 combos × 3 folds) would take hours.

Hints:
- `scoring='f1_macro'` (or `'f1'` for class 0 via `make_scorer`) fits the
  imbalanced target better than accuracy.
- Start with a tiny grid to check it runs, then enlarge.
- Show `pd.DataFrame(grid_search.cv_results_)` sorted by rank — nice for the
  reviewer.

---

## 7. Task 6 – Final evaluation on the test set

`GridSearchCV` (with the default `refit=True`) already retrains the best
pipeline on **all** training data. Use it exactly once on the test set:

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

Then write a markdown conclusion:
- accuracy, precision, recall, F1 for **both** classes,
- what errors the model makes (look at a few misclassified reviews),
- comparison with the "always 1" baseline (82 % accuracy).

Stand-out: plot the 20 most positive and negative TF-IDF words from the
logistic regression coefficients (`get_feature_names_out()` on the
`features` step + `classifier.coef_`).

---

## 8. Finishing touches (code quality rubric)

- **README.md**: replace the template with a project summary, setup steps,
  file list, and results.
- **Notebook**: clear headings, a markdown cell explaining each step, helper
  functions with docstrings, PEP 8 names (`snake_case` functions/variables,
  `UPPER_CASE` constants, `CamelCase` classes).
- Run **Kernel → Restart & Run All** before submitting; it must run without
  errors.
- Optional dashboard: `joblib.dump(best_model, 'model.pkl')` and reuse the
  FastHTML setup from `dsnd-dashboard-project` to make live predictions.
- Submit as a public GitHub repo (create a new repo, then
  `git remote set-url origin <your repo url>` — right now `origin` still
  points to Udacity's repo).

---

## Checklist mapping

| Task list item | Guide section |
|---|---|
| Preprocessing for data types | 2 |
| Processing text data | 3 |
| Features from text | 4 |
| Combined model pipeline | 5 |
| Fine-tuning | 6 |
| Evaluate on test data | 7 |
