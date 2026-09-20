# Tabular ML: Embedding-Based MLP vs. XGBoost on House Prices

A tabular MLP pipeline that predicts housing prices, trained on the Ames
Housing / House Prices dataset. The goal of this project was to build a
neural network and compare its performance against XGBoost, the benchmark
for tabular data analysis.

## Results

| Model | Validation RMSE |
|---|---|
| Neural net (initial, untuned) | $28,387 |
| XGBoost (baseline) | $29,031 |
| **Neural net (tuned)** | **$25,084** |

The initial result was promising as the model only fell a couple of
thousand dollars behind the XGBoost in their respective RMSE (root mean
square error) calculations. I then pivoted from construction and towards a
tuning and debugging process which included finding typos in the
configuration and learning rate optimization through a LR range test and a
follow up sweep. The resulting optimized neural network outperformed the
XGBoost RMSE by approximately 14%. While the result did excite me, I do not
believe the result represents a definitive victory for the neural network.
The unoptimized XGBoost only served as a benchmark to beat, to test my
neural network further I would compare it against an optimized XGBoost
alongside further neural network development.

## Architecture

The neural network trains on categorical and numerical data. Each
categorical column (MS Zoning, Neighborhood, House Style, Sale Condition)
has its own embedding table as well as a cardinality-based heuristic
calculated as follows: `min(config.embedding_dim_cap, (cardinality + 1) // 2)`.
The heuristic allows high-cardinality category columns like Neighborhood to
have a greater impact than a low-cardinality columns like SaleCondition.
After the categorical columns are embedded the resulting vectors are then
concatenated with 8 numerical column features. The complete vector gets
processed in the forward pass: linear -> batch norm -> ReLU -> dropout. The
described pass repeats for every hidden layer; through the forward pass the
hidden dims `[128, 64]` get transformed into a singular price prediction.

The XGBoost, implemented through `XGBRegressor`, acts as a baseline to
compare the neural network against. It was trained on the same label
encoded features. To act as a baseline, XGBoost's hyperparameters,
`xgb_n_estimators` and `xgb_random_state`, were intentionally left as
untuned default values.

The decision to have a XGBoost came from a desire to judge performance
through comparison rather than solely interpreting the neural net's
results. The XGBoost fits its role well because with tabular data gradient
boosted trees are expected to outperform neural networks. My thought
process was that if my model can get close or even beat a XGBoost that
such results would validate the performance of my MLP.

## Project structure

```
src/tabular_ml/
├── config.py       # Central settings: paths, columns, hyperparameters
├── data.py         # Load, clean, split, and preprocess (leakage-free)
├── model.py        # Embedding + MLP architecture
├── baseline.py     # XGBoost baseline
├── train.py        # Training loop, evaluation, artifact saving/promotion
└── api.py          # FastAPI service serving the promoted model
tests/              # pytest suite covering data, model, baseline, and API
artifacts/production/   # The promoted, currently-served model
Dockerfile
.github/workflows/ci.yaml
```

## Running it

### Setup
```bash
pip install -r requirements.txt
```

### Train
Download `train.csv` from the
[Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
into `data/train.csv`, then:
```bash
python -m src.tabular_ml.train
```

### Serve
```bash
uvicorn src.tabular_ml.api:app --reload
```
Visit `http://127.0.0.1:8000/docs` for an interactive API explorer, or
`POST` to `/predict` directly:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LotArea": 8450, "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003,
    "GrLivArea": 1710, "FullBath": 2, "BedroomAbvGr": 3, "GarageCars": 2,
    "MSZoning": "RL", "Neighborhood": "CollgCr", "HouseStyle": "2Story",
    "SaleCondition": "Normal"
  }'
```

### Docker
```bash
docker build -t tabular-ml-api .
docker run -p 8000:8000 tabular-ml-api
```

### Tests
```bash
python -m pytest -v
```

## Tuning Process

I began by conducting an LR range test, over the course of 200 steps the
learning rate increased exponentially between 1e-7 to 1.0. The hope was to
find the steepest gradient descent possible while also maintaining some
semblance of stability. The range test pointed toward a higher learning
rate, but I later realized it had been run against a misconfigured
architecture (see the typo below) and once that was fixed, the range
test's suggestion no longer held.

I then pivoted my strategy and tried a learning rate sweep with the values
`[0.001, 0.01, 0.05, 0.1, 0.2]`. The sweep worked by training the model 5
separate times with the different learning rates. The RMSE decreased when
the LR decreased and resulted in the smallest value 0.001 producing the
lowest RMSE. To see whether this trend continued I trained the model on two
further LRs [0.0005, 0.0001] with 0.0005 returning the best results.

Two major bugs came up during this process.

The loss came back in the tens of billions on the very first run, barely
moving epoch to epoch. Tracing it back, the issue was scale: a freshly
initialized model outputs small numbers by default, not the desired figure
in the hundreds of thousands, and at a normal learning rate, it had no
realistic way to close a gap that large. The fix wasn't to make the model
output bigger numbers. It was to scale the target down to match the
model's natural output range, and only convert predictions back to real
dollars afterward, once training was done.

The second issue was a simple typo in the configuration. The
`config.hidden_dims` was set to `[12, 8, 64]` instead of `[128, 64]`. This
typo invalidated many of my first training runs and was only caught after
noticing the lack of discrepancies between two different runs.

## Known Limitations

A limitation of the method `apply_transf` in `data.py` revealed itself
during testing. The method will raise an error during validation, testing,
or from the API if a category is introduced not seen during training. For
the purpose of this project, this problem seems low risk as the project has set
categories. However, if this model was ever used in production a fix would
be necessary. A solution could be creating a fallback category that any
unseen category could fall back into.

`HouseFeatures` in `api.py` accepts NaN/Infinity as technically valid
floats via pydantic, which can produce a non-JSON-serializable prediction
and a 500 error. A stronger implementation would use pydantic's
`confloat(allow_inf_nan=False)` or a series of checks to ensure valid
floats were being used.
