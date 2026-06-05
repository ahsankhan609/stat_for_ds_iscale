# Data Science & Machine Learning Study Plan

## How to use this plan

Each phase follows the pattern: **Theory first → Tools alongside**.
Learn the concept, then immediately implement it with the listed libraries.
This is exactly what you are already doing with statistics in [`notebooks/stats_iscale/desc_stats.ipynb`](notebooks/stats_iscale/desc_stats.ipynb).

---

## Phase 1 — Python Completion (1–2 weeks)

**What's left for you:** Iterators, generators, and decorators.
These are critical for understanding how ML libraries (NumPy, PyTorch) work under the hood.

- **Theory topics**
  - Iterators and the iterator protocol (`__iter__`, `__next__`)
  - Generator functions (`yield`) and generator expressions
  - Decorators and closures (`@functools.wraps`)
- **Tools to learn alongside**
  - `itertools` — chaining, grouping, product (used everywhere in data pipelines)
  - `functools` — `lru_cache`, `partial` (used in ML training loops)
  - No 3rd-party libraries needed here; pure Python is the tool

---

## Phase 2 — Statistics Implementation (3–5 weeks, currently here)

You have the theory for descriptive, probability, and inferential stats. Now solidify implementation.

### 2A — Descriptive Statistics (current)
- **Theory** (done): Mean, median, mode, variance, std, IQR, skewness, kurtosis
- **Tools to use now**
  - `statistics` (Python built-in) — mean, median, mode, stdev
  - `numpy` — array operations, `np.mean`, `np.std`, `np.percentile`
  - `pandas` — `df.describe()`, `df.value_counts()`, Series aggregations

### 2B — Probability & Distributions
- **Theory** (done): Normal, Binomial, Poisson, uniform distributions, PMF/PDF/CDF
- **Tools to learn alongside**
  - `numpy.random` — `np.random.normal`, `np.random.binomial` (sampling)
  - `scipy.stats` — `norm`, `binom`, `poisson` objects; `.pdf()`, `.cdf()`, `.ppf()`

### 2C — Inferential Statistics & Hypothesis Testing
- **Theory** (done): p-values, confidence intervals, t-test, chi-square, ANOVA
- **Tools to learn alongside**
  - `scipy.stats` — `ttest_ind`, `chi2_contingency`, `f_oneway`, `pearsonr`
  - `statsmodels` — `statsmodels.stats.proportion`, `statsmodels.stats.power` (effect size, power analysis)

### 2D — Regression Analysis (next theory stop)
- **Theory to study**: Simple linear regression, multiple regression, OLS, R², residuals, multicollinearity
- **Tools to learn alongside**
  - `numpy` — `np.polyfit`, `np.linalg.lstsq`
  - `statsmodels` — `OLS`, `summary()` output (full stats: coefficients, p-values, confidence intervals)
  - `sklearn.linear_model` — `LinearRegression` (preview of scikit-learn, used in Phase 5)

---

## Phase 3 — Data Visualization (2–3 weeks)

Learn visualization right after stats so you can plot every concept you've studied.

- **Theory topics**
  - Chart types and when to use each: histogram, boxplot, scatter, heatmap, bar
  - Principles of good visualization (scale, color, labels, distortion)
- **Tools to learn in order**
  1. `matplotlib` — foundational; learn `Figure`, `Axes`, subplots, styling. Everything else is built on this.
  2. `seaborn` — statistical plots: `histplot`, `boxplot`, `violinplot`, `heatmap`, `pairplot`. Learn after matplotlib basics.
  3. `plotly` — interactive charts (optional at this stage, revisit later for dashboards)

---

## Phase 4 — Data Manipulation & Exploratory Data Analysis (3–4 weeks)

- **Theory topics**
  - Tidy data principles; wide vs. long format
  - Handling missing data (MCAR, MAR, MNAR)
  - Outlier detection methods
  - Feature relationships: correlation, covariance
- **Tools to learn alongside**
  - `pandas` (deep dive) — `read_csv`, `merge`, `groupby`, `pivot_table`, `melt`, `apply`, `fillna`, `dropna`
  - `numpy` — advanced indexing, broadcasting, `np.where`, `np.corrcoef`
  - `missingno` — visualize missing data patterns
  - Practice: full EDA notebooks on public datasets (Titanic, House Prices on Kaggle)

---

## Phase 5 — Core Machine Learning (8–10 weeks)

Learn algorithm theory, then implement with scikit-learn.

### 5A — ML Fundamentals (1–2 weeks)
- **Theory**: Bias-variance tradeoff, overfitting/underfitting, train/val/test split, cross-validation, evaluation metrics
- **Tools**: `sklearn.model_selection` — `train_test_split`, `KFold`, `cross_val_score`; `sklearn.metrics`

### 5B — Supervised Learning — Classification (2–3 weeks)
- **Theory**: Logistic regression, KNN, Decision Trees, Random Forests, SVM, Naive Bayes; precision, recall, F1, ROC-AUC
- **Tools**: `sklearn.linear_model`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.svm`, `sklearn.naive_bayes`

### 5C — Supervised Learning — Regression (1–2 weeks)
- **Theory**: Ridge, Lasso, ElasticNet regularization; polynomial regression
- **Tools**: `sklearn.linear_model.Ridge/Lasso`, `sklearn.preprocessing.PolynomialFeatures`

### 5D — Unsupervised Learning (1–2 weeks)
- **Theory**: K-Means, DBSCAN, PCA, t-SNE, silhouette score
- **Tools**: `sklearn.cluster`, `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`

### 5E — Feature Engineering & Pipelines (1 week)
- **Theory**: Encoding, scaling, imputation, feature selection
- **Tools**: `sklearn.preprocessing`, `sklearn.pipeline.Pipeline`, `sklearn.feature_selection`

### 5F — Gradient Boosting (1 week)
- **Theory**: Boosting vs. bagging, gradient descent in tree ensembles
- **Tools**: `xgboost`, `lightgbm` — both have sklearn-compatible APIs

---

## Phase 6 — Deep Learning (6–8 weeks)

### 6A — Neural Network Foundations (2 weeks)
- **Theory**: Perceptrons, activation functions, backpropagation, gradient descent, loss functions, epochs/batches
- **Tools to learn**: `PyTorch` — tensors, `autograd`, building `nn.Module`, training loops
  - (Alternative: TensorFlow/Keras — choose one and stick to it; PyTorch is more popular in research and is recommended)

### 6B — Convolutional Neural Networks (2 weeks)
- **Theory**: Convolution, pooling, receptive field, transfer learning
- **Tools**: `torchvision` — pretrained models (`ResNet`, `VGG`); `torchvision.transforms`

### 6C — Recurrent Neural Networks / Sequence Models (1–2 weeks)
- **Theory**: RNN, LSTM, GRU, vanishing gradient problem, sequence-to-sequence
- **Tools**: `torch.nn.LSTM`, `torch.nn.GRU`

---

## Phase 7A — NLP Specialization (4–6 weeks)

- **Theory**: Tokenization, TF-IDF, word embeddings (Word2Vec, GloVe), attention, transformers (BERT, GPT)
- **Tools to learn in order**
  1. `nltk` — tokenization, stemming, lemmatization, stopwords (classical NLP)
  2. `spacy` — NER, POS tagging, dependency parsing (production NLP)
  3. `scikit-learn` — `TfidfVectorizer`, `CountVectorizer` for text features
  4. `gensim` — Word2Vec, topic modeling (LDA)
  5. `transformers` (HuggingFace) — pretrained BERT, GPT-2; pipelines for classification, NER, QA

---

## Phase 7B — Computer Vision Specialization (4–6 weeks)

- **Theory**: Image representation, convolutions, object detection (YOLO, R-CNN), segmentation
- **Tools to learn in order**
  1. `Pillow (PIL)` — image loading, resizing, basic transforms
  2. `OpenCV (cv2)` — image processing, edge detection, contours, video
  3. `torchvision` — datasets, pretrained CNNs, transforms
  4. `ultralytics (YOLOv8)` — object detection with minimal code

---

## Phase 8 — MLOps & Deployment (3–4 weeks, final phase)

- **Theory**: Model versioning, experiment tracking, serving models as APIs, containerization basics
- **Tools**
  - `MLflow` — experiment tracking, model registry
  - `FastAPI` — serve your model as a REST API
  - `Docker` (basics) — containerize your app
  - `Streamlit` — build interactive ML demos quickly

---

## Estimated Timeline (5–10 hrs/week)

| Phase | Topic | Duration |
|-------|-------|----------|
| 1 | Python completion (iterators, generators, decorators) | 1–2 weeks |
| 2 | Statistics implementation (descriptive → regression) | 3–5 weeks |
| 3 | Data visualization (matplotlib, seaborn) | 2–3 weeks |
| 4 | EDA & pandas deep dive | 3–4 weeks |
| 5 | Core Machine Learning (scikit-learn) | 8–10 weeks |
| 6 | Deep Learning (PyTorch) | 6–8 weeks |
| 7A | NLP specialization | 4–6 weeks |
| 7B | Computer Vision specialization | 4–6 weeks |
| 8 | MLOps & deployment | 3–4 weeks |
| | **Total** | **~10–12 months** |

---

## Library Learning Order (Quick Reference)

```
Python built-ins (itertools, statistics, functools)
    → numpy
    → pandas
    → matplotlib → seaborn
    → scipy.stats → statsmodels
    → scikit-learn
    → xgboost / lightgbm
    → PyTorch → torchvision
    → nltk → spacy → transformers (HuggingFace)
    → Pillow → OpenCV
    → MLflow → FastAPI → Streamlit
```

---

## Notebooks Structure (Recommended)

```
notebooks/
├── phase1_python/
│   ├── iterators_generators.ipynb
│   └── decorators.ipynb
├── phase2_stats/
│   ├── descriptive_stats.ipynb        ← currently here
│   ├── probability_distributions.ipynb
│   ├── inferential_stats.ipynb
│   └── regression_analysis.ipynb
├── phase3_visualization/
│   ├── matplotlib_basics.ipynb
│   └── seaborn_statistical_plots.ipynb
├── phase4_eda/
│   ├── pandas_deep_dive.ipynb
│   └── eda_titanic.ipynb
├── phase5_ml/
│   ├── ml_fundamentals.ipynb
│   ├── classification.ipynb
│   ├── regression.ipynb
│   ├── unsupervised.ipynb
│   ├── feature_engineering.ipynb
│   └── gradient_boosting.ipynb
├── phase6_deep_learning/
│   ├── pytorch_foundations.ipynb
│   ├── cnn.ipynb
│   └── rnn_lstm.ipynb
├── phase7a_nlp/
│   ├── nltk_spacy_basics.ipynb
│   ├── tfidf_vectorization.ipynb
│   └── transformers_huggingface.ipynb
└── phase7b_cv/
    ├── pillow_opencv_basics.ipynb
    └── yolov8_detection.ipynb
```
