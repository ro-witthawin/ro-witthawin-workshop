# Recommendation System 101

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ro-witthawin/ro-witthawin-workshop/blob/main/workshops/recommendation_system_101/RecommendationSystem.ipynb)

Hands-on notebook for building a simple recommendation system from user review data. The workshop starts with basic data exploration, builds product content profiles, and then compares content-based filtering with collaborative filtering using Non-negative Matrix Factorization (NMF).

The example dataset contains cigarette review data from smokers and is used only as an educational recommendation-system dataset.

## What You Will Learn

The notebook demonstrates how to:

- load and inspect a real user-item review dataset
- combine product fields into a unique item label
- create item content profiles from categorical attributes
- build a content-based product lookup using `Strength`, `Taste`, and `Price`
- create a user-item rating matrix with `pandas.crosstab`
- fill sparse ratings for matrix factorization
- train an NMF model with `scikit-learn`
- reconstruct predicted user-item ratings from latent factors
- generate top-k collaborative-filtering recommendations
- compare explicit content matching with latent-factor recommendations
- find a similar user from a target product preference using cosine similarity

## Files

- `RecommendationSystem.ipynb` - main workshop notebook
- `README.md` - this guide

## Main Stack

- Runtime: Python notebook environment
- Data tools: `pandas`, `numpy`
- Recommendation model: `sklearn.decomposition.NMF`
- Similarity metric: `scipy.spatial.distance.cosine`
- Recommended environment: Google Colab

## Dataset

The notebook downloads `smokerdata.csv` in Colab with `gdown`:

```python
!gdown --id 1ySNp84n4er1_rAYdLMcchMg7eMChoZcZ
```

Original dataset reference:

```text
https://www.kaggle.com/mikhailverghese/cigarette-reviews-by-smokers
```

Expected columns include:

- `User`
- `Brand`
- `Variety`
- `Rating`
- `Strength`
- `Taste`
- `Price`

## Quick Start in Google Colab

1. Open the notebook with the Colab badge above.
2. Run the download cell to fetch `smokerdata.csv`.
3. Run the import and data exploration cells.
4. Run the preprocessing cells to create `Brand_variety` and item content profiles.
5. Run the content-based filtering examples.
6. Run the collaborative filtering cells to train NMF and generate recommendations.

Google Colab usually includes the required data science libraries. If a dependency is missing, install it in a notebook cell:

```python
%pip install -U pandas numpy scikit-learn scipy gdown
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn scipy jupyter gdown
```

Download the dataset:

```bash
gdown --id 1ySNp84n4er1_rAYdLMcchMg7eMChoZcZ
```

The notebook currently reads the file from `/content/smokerdata.csv`, which is the default Colab path. For local runs, update the read cell to point to the downloaded file:

```python
df = pd.read_csv("smokerdata.csv")
```

Start Jupyter:

```bash
jupyter notebook workshops/recommendation_system_101/RecommendationSystem.ipynb
```

## Workshop Flow

Run the notebook sections in order:

1. Download `smokerdata.csv`.
2. Import `pandas` and `numpy`.
3. Load the review dataset into `df`.
4. Inspect users, brands, schema, and rating statistics.
5. Create `Brand_variety` by combining `Brand` and `Variety`.
6. Build `df_brand_clearn`, a cleaned item-profile table with common `Strength`, `Taste`, and `Price` values.
7. Create `dfnew`, the main recommendation table with user, item, rating, and product attributes.
8. Define `content_data()` for content-based filtering.
9. Create the user-item rating matrix `cmat`.
10. Fill missing values and replace user names with numeric IDs.
11. Fit an NMF model with 30 latent factors.
12. Create item-factor matrix `H` and user-factor matrix `W`.
13. Reconstruct predicted ratings with `np.dot(W, H)`.
14. Define recommendation helpers for reconstructed predictions and observed ratings.
15. Generate top-k recommendations for a selected user.
16. Build a synthetic preference vector and find the closest matching user with cosine similarity.
17. Recommend products based on the closest user's latent preference profile.

## Key Notebook Objects

- `df` - raw dataset loaded from CSV
- `Brand_variety` - combined product name from brand and variety
- `df_brand_clearn` - item content profile table
- `dfnew` - compact working table for recommendation
- `content_data(x, y, z)` - returns products matching strength, taste, and price
- `cmat` - user-item rating matrix
- `W` - user latent-factor matrix from NMF
- `H` - item latent-factor matrix from NMF
- `reconstructed` - predicted user-item rating matrix
- `recomendation(uid, topk=5)` - returns top-k recommendations from predicted ratings
- `recomendation_cmat(uid, topk=5)` - returns top-k items from observed ratings

## Troubleshooting

### `FileNotFoundError: /content/smokerdata.csv`

Run the `gdown` download cell first. If running locally, change the CSV path to the local file location:

```python
df = pd.read_csv("smokerdata.csv")
```

### `ModuleNotFoundError`

Install the notebook dependencies:

```python
%pip install -U pandas numpy scikit-learn scipy gdown
```

Restart the notebook kernel after installing packages.

### NMF warnings or slow training

The notebook uses `NMF(30)`. If training is slow or convergence warnings appear, try increasing `max_iter`:

```python
nmf = NMF(n_components=30, max_iter=1000, random_state=42)
```

### Empty content-based results

`content_data()` only returns exact matches for `Strength`, `Taste`, and `Price`. Check available category values in `df_brand_clearn` before querying.
