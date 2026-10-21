# StyleSense: Predicting Product Recommendations from Reviews

StyleSense, an online women's clothing retailer, has a backlog of product
reviews where customers did not say whether they recommend the product. This
project builds a machine learning pipeline that predicts the recommendation
from the rest of the review: the review text and title, the customer's age,
the product category, and how many other customers found the review helpful.

The whole process, from raw data to prediction, runs inside one scikit-learn
`Pipeline`: preprocessing of numerical and categorical data, NLP with spaCy,
feature engineering, and a logistic regression classifier. The pipeline is
tuned with a grid search on the training data and evaluated once on a held-out
test set.

## Results

Evaluated once on the held-out test set (10 %, 1,845 reviews):

| Metric | Final pipeline | Always "recommended" |
|---|---|---|
| Accuracy | 0.902 | 0.823 |
| Macro F1 | 0.845 | about 0.45 |
| Recall, not recommended | 0.832 | 0.000 |
| Recall, recommended | 0.917 | 1.000 |

The model finds 83 % of the reviews whose authors do not recommend the
product. The strongest signals come from the review text, including negated
word pairs such as *not flattering* and *not recommend*, which are kept
during text cleaning on purpose.

## Getting Started

### Dependencies

- Python 3.10 or higher (developed with 3.12)
- scikit-learn, pandas, numpy, spaCy (with the `en_core_web_sm` model),
  Jupyter, matplotlib, seaborn — exact versions in `requirements.txt`

### Installation

```
git clone https://github.com/chrisriby/dsnd-pipelines-project.git
cd dsnd-pipelines-project

python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate

python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On some machines spaCy needs a platform-specific install (for example
`pip install 'spacy[apple]'` on Apple Silicon); see
[spacy.io/usage](https://spacy.io/usage).

### Running the notebook

```
jupyter notebook starter/starter.ipynb
```

Run all cells from top to bottom. A full run takes about 20
minutes, mostly for spaCy and the grid search. The slow spaCy step is cached
in `starter/cache/`, so later runs are faster.

## Project Structure

```
├── README.md            this file
├── GUIDE.md             step-by-step development guide used for the project
├── requirements.txt     Python packages
├── LICENSE.txt
└── starter
    ├── README.md
    ├── starter.ipynb    the project notebook
    └── data
        └── reviews.csv  review dataset (18,442 reviews)
```

### Notebook outline

| Section | Content |
|---|---|
| Load data, split | 90 % training / 10 % test split (`random_state=27`) |
| Data exploration | target balance, numerical, categorical and text features (training data only) |
| Step 1 | preprocessing: median imputation + scaling for numbers; imputation + one-hot encoding (rare and unknown categories handled) for categories |
| Step 2 | NLP with spaCy: tokenization, lowercasing, lemmatization, stop-word removal while keeping negations (*not*, *n't*, *never*), repair of broken apostrophes, adjective counts from part-of-speech tags |
| Step 3 | text features: TF-IDF with word pairs for title and review; word, exclamation mark and negation counts |
| Step 4 | one pipeline combining all steps with a logistic regression classifier |
| Training | 3-fold cross-validation vs. a majority-class reference, training on all training data |
| Fine-tuning | grid search over 12 parameter combinations (regularization strength, TF-IDF vocabulary size and minimum term frequency) |
| Evaluation | test set metrics, confusion matrix, most important features, error analysis |

## Data

`starter/data/reviews.csv` contains 18,442 reviews with 8 features and the
target `Recommended IND` (1 = recommended, 0 = not recommended, 82 % / 18 %):

- **Numerical:** `Age`, `Positive Feedback Count`
- **Categorical:** `Clothing ID`, `Division Name`, `Department Name`, `Class Name`
- **Text:** `Title`, `Review Text`

## Built With

- [scikit-learn](https://scikit-learn.org/) - pipelines, preprocessing, model, grid search
- [spaCy](https://spacy.io/) - tokenization, lemmatization, part-of-speech tags
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) - data handling
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) - plots

## License

[License](LICENSE.txt)
