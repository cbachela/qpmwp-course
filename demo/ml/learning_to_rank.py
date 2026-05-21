############################################################################
### QPMwP CODING EXAMPLES - LEARNING TO RANK (LTR)
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     01.05.2026
# First version:    01.05.2026
# --------------------------------------------------------------------------


# This script demonstrates the application of Learning to Rank to predict
# the cross-sectional ordering of stock returns for one point in time.


# Make sure to install the following packages before running the demo:

# uv pip install pyarrow fastparquet   # For reading and writing parquet files
# uv pip install xgboost               # For training the model with XGBoost
# uv pip install scikit-learn          # For pipeline, GridSearchCV, and custom scoring  
# uv pip install scipy                 # For evaluating the rankings (ndcg_score)





# Standard library imports
import os
import sys
from pathlib import Path

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from xgboost import XGBRanker

# Local imports
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.getcwd()))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

from helper_functions import (
    align_market_data_with_jkp_data,
)
from ml.utils.format import (
    check_if_multiindex,
    ensure_datetime_index,
)
from ml.utils.grid import (
    compound_returns_on_grid,
    shift_grid_returns,
)
from ml.utils.format import (
    check_if_multiindex,
    ensure_datetime_index,
)
from ml.transformers.pipeline import TransformPipeline
from ml.transformers.panel.cross_sectional import (
    CrossSectionalWinsorize,
    CrossSectionalRank,
    CrossSectionalPercentiles,
    CrossSectionalZScore,
)
from ml.metrics.scoring import ndcg_scorer
from ml.splitters.rolling_timeseries_split import PanelTimeSeriesSplit
from ml.splitters.rolling_timeseries_split import ObservationGridRollingSplit
from ml.model.xgb_ranker_wrapper import XGBRankerSklearnWrapper




# --------------------------------------------------------------------------
# Control parameters and paths
# --------------------------------------------------------------------------

# label_recompute = True
label_recompute = False
# feature_recompute = True
feature_recompute = False
start_date = "2000-01-01"
end_date = "2023-01-01"

data_path = Path('C:/Users/User/OneDrive/Documents/QPMwP/Data')  # Change this path if needed
return_series_path = data_path / "return_series.parquet"
feature_path = str(data_path / "features.parquet")
label_path = str(data_path / "labels.parquet")
prediction_path = str(data_path / "ml_signal.parquet")
shap_path = str(data_path / "shap_values.parquet")



# --------------------------------------------------------------------------
# Load data
# - market data (from parquet file)
# - jkp data (from parquet file)
# - swiss performance index, SPI (from csv file)
# --------------------------------------------------------------------------

# Load market and jkp data from parquet files
market_data = pd.read_parquet(path = data_path / "market_data.parquet")
jkp_data = pd.read_parquet(path = data_path / "jkp_data.parquet")

# Align market data with jkp data and forward fill missing values in market data
market_data_ffill, jkp_data = align_market_data_with_jkp_data(market_data=market_data, jkp_data=jkp_data)
market_data_ffill.index.names = ["DATE", "ID"]
jkp_data.index.names = ["DATE", "ID"]

# Calc daily returns and save as parquet file
X = market_data_ffill.pivot_table(index="DATE", columns="ID", values="price")
end_date = X.index.max().strftime("%Y-%m-%d")
width = X.shape[0] - 1
daily_ret = X[X.index <= end_date].tail(width + 1).pct_change(fill_method=None).iloc[1:]
daily_ret = daily_ret.stack()
daily_ret.index.names = ["DATE", "ID"]
daily_ret.name = "tot_return_gross"
daily_ret.dropna().to_frame().to_parquet(data_path / "return_series.parquet")




# --------------------------------------------------------------------------
# Create a features dataframe from the jkp_data
# --------------------------------------------------------------------------

if Path(feature_path).exists() and not feature_recompute:

    print(f"Loading features from {feature_path}")
    X = pd.read_parquet(feature_path)

else:

    print(f"Preparing features from raw data in {data_path}")

    feature_cols = [
        "ret_6_1",      # Momentum6
        "ret_12_1",     # Momentum12  
        "qmj",          # Quality minus Junk
        "qmj_growth",   # GrowthComp
        "qmj_safety",   # SafetyComp
        "gp_at",        # GrossProfit
        "op_at",        # OpProfit
        "be_me",        # BookValue
        "debt_me",      # Leverage
        "at_gr1",       # AssetGrowth
        "oaccruals_at", # Accruals
    ]

    # load signals
    X = pd.read_parquet(path=data_path / "jkp_data.parquet")
    X = X[feature_cols]
    X.index.names = ["DATE", "ID"]

    # keep only numeric columns (for now)
    X = X.select_dtypes(include="number")

    # forward fill, 1y max
    X = X.groupby(level="ID").ffill(limit=4)

    # only for the example, in practice more careful
    X = X.dropna()

    # find first DATE with more than 50 unique IDs and no NAs
    dates_id_counts = X.dropna().groupby(level="DATE").size()
    start_date_id = dates_id_counts[dates_id_counts > 50].index[0]
    print(f"Start date (first DATE with >50 IDs): {start_date_id}")

    X = X.loc[X.index.get_level_values("DATE") >= start_date_id]

    # remove duplicates
    X = X[~X.index.duplicated(keep="last")]

    # check if we have the proper panel format
    check_if_multiindex(X)

    # check if date level is datetime, if not convert
    X = ensure_datetime_index(X)

    # sort
    X = X.sort_index()

    # feature transformation pipeline
    feature_pipeline = TransformPipeline(
        [
            CrossSectionalZScore(),
            CrossSectionalWinsorize(lower=0.01, upper=0.99),
        ]
    )

    X = feature_pipeline.fit_transform(X)

    # assess the sample size over time
    # X.groupby(level="DATE").count().plot(title="Number of IDs per DATE")

print(f"Features prepared: X.shape={getattr(X, 'shape', None)}")



# --------------------------------------------------------------------------
# Inspect the features for one date
# --------------------------------------------------------------------------

X.loc['2024-01-31'].describe()
X.loc['2024-01-31'].plot(kind="density")




# --------------------------------------------------------------------------
# Prepare labels (i.e., ranks of period returns)
# --------------------------------------------------------------------------

# time grid for label creation
time_grid = X.index.get_level_values("DATE").unique().sort_values()

# if label file exists, load it otherwise compute and persist
if Path(label_path).exists() and not label_recompute:

    print(f"Loading labels from {label_path}")

    # load the label
    y_df = pd.read_parquet(label_path)

    # downstream needs a series
    y = y_df.squeeze()

else:

    print(f"Preparing labels from raw data in {data_path}")
    # on s3 we still have multi columns

    return_series = pd.read_parquet(return_series_path, columns=["tot_return_gross"])
    # return_series = pd.read_parquet(return_series_path)

    # check if we have the proper panel format
    check_if_multiindex(return_series)

    # check if date level is datetime, if not convert
    return_series = ensure_datetime_index(return_series)

    # daily returns
    daily_ret = return_series.sort_index()

    # grid returns
    grid_ret = compound_returns_on_grid(
        return_series=daily_ret.squeeze(),
        time_grid=time_grid,
    )

    # shift back one unit on the grid
    y = shift_grid_returns(grid_ret, shift=-1)

    # remove duplicates
    y = y[~y.index.duplicated(keep="last")]

    # label transformation pipelines
    label_pipeline1 = TransformPipeline([
        CrossSectionalWinsorize(lower=0.01, upper=0.99),
        CrossSectionalRank(as_percentile=True),
    ])
    y_ranks = label_pipeline1.fit_transform(y)

    label_pipeline2 = TransformPipeline([
        CrossSectionalWinsorize(lower=0.01, upper=0.99),
        CrossSectionalPercentiles(n_bins=10),
    ])
    y_deciles = label_pipeline2.fit_transform(y)

    # Concatenate the transformed labels into a single DataFrame
    y = pd.concat([
        y,
        y_ranks.rename("return_ranks"),
        y_deciles.rename("return_deciles")
    ], axis=1)
    y = y.dropna()

print(f"Labels prepared: y.shape={getattr(y, 'shape', None)}")


# Inspect the labels for one date
y.loc['2024-01-31']
y.loc['2024-01-31'].describe()
y.loc['2024-01-31'].plot(kind="density")



# --------------------------------------------------------------------------
# Alignment of X and y
# --------------------------------------------------------------------------

common_index = X.index.intersection(y.index).sort_values()
X = X.loc[common_index]
y = y.loc[common_index]

print(f"Aligned X and y: X.shape={X.shape}, y.shape={y.shape}")
assert X.index.equals(y.index)


# persist the aligned features for future use
if not Path(feature_path).exists() or feature_recompute:
    print("Persisting Features")
    X.to_parquet(feature_path)

# persist the label for future use
if not Path(label_path).exists() or label_recompute:
    print("Persisting Labels")
    y.to_parquet(label_path)


print(f"Aligned X and y: X.shape={X.shape}, y.shape={y.shape}")




# --------------------------------------------------------------------------
# Rolling split configuration
# --------------------------------------------------------------------------

rolling_splitter = ObservationGridRollingSplit(
    observation_dates=time_grid,
    train_window_obs=36,
    skip_obs_between_train_test=0,
    retrain_stride=1,
)

rolling_splitter.print_splits(X=X)




# --------------------------------------------------------------------------
# Model / Pipeline / Hyperparameter tuning configuration and GridSearchCV setup
# --------------------------------------------------------------------------

# Pipeline with XGBRanker
pipeline = Pipeline([
    ("ranker", XGBRankerSklearnWrapper(
        objective="rank:ndcg",
        # objective='rank:pairwise',
        ndcg_exp_gain=False,
        # ndcg_exp_gain=True,
    ))
])

# Hyperparameter grid — analog to Ridge param_grid
param_grid = {
    "ranker__max_depth": [4, 6, 8],
    "ranker__eta": [0.05, 0.1],
    "ranker__n_estimators": [100, 200],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=make_scorer( # Make custom scorer (wrapper around an arbitrary metric or loss function)
        score_func=ndcg_scorer,
        greater_is_better=True,
        # k=10,
        k=None,
    ),
    # cv=KFold(n_splits=5),
    cv=PanelTimeSeriesSplit(
        n_splits=4,
        date_level="DATE"
    ),
    n_jobs=-1,
    refit=True,
)






# --------------------------------------------------------------------------
# Prepare the training data for one split
# --------------------------------------------------------------------------

# Choose the label to train on
label_name = "return_ranks"
# label_name = "return_deciles"

# Get the splits (generator of train/test indices)
splits = rolling_splitter.split(X=X)

# # get training info of first split
# train_idx, test_idx = next(splits)

# get training info of t'th split
train_idx, test_idx = rolling_splitter.get_split_at(X=X, t=200)
train_idx, test_idx

# Subset the data for the first split
X_train = X.loc[train_idx]
y_train = y.loc[train_idx][label_name]



# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

# clone model and fit
model = clone(grid_search)
# model = clone(pipeline)
model.fit(X=X_train, y=y_train)

# Extract the best model and parameters
best_model = model.best_estimator_
best_params = model.best_params_
best_score = model.best_score_

print("Best parameters:", best_params)
print(f"Best CV score: {best_score:.4f}")
print("Best model:", best_model)

# You can also get all parameters from the best model
print("\nAll parameters of best model:")
print(best_model.get_params())



# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------

# # Just as a sanity check, look at in-sample fit
# test_idx = train_idx
# y_test = y.loc[test_idx][label_name]

# Look at out-of-sample predictions
y_test = y.loc[test_idx][label_name]

# Extract prediction
# preds = model.predict(X.loc[test_idx])
preds = best_model.predict(X.loc[test_idx])
y_pred_score = pd.Series(preds, index=test_idx)
if label_name == "return_ranks":
    y_pred = CrossSectionalRank(as_percentile=True).transform(y_pred_score)
elif label_name == "return_deciles":
    y_pred = CrossSectionalPercentiles(n_bins=10).transform(y_pred_score)
else:
    print("Incorrect label_name")

# Predictions vs. true (future) labels
out = pd.concat([
    y.loc[test_idx]['return'],
    y_test,
    y_pred_score,
    y_pred,
], axis=1)
out = out.droplevel('DATE')
out.columns = ['ret', 'y_true', 'y_pred_score', 'y_pred']

# Inspect predictions vs. true labels
out.sort_values('ret', ascending=False).head(20)
out.sort_values('y_true', ascending=False).tail(20)

out.plot(kind='scatter', x='y_true', y='y_pred')

# Compute the mean per decile
if label_name == "return_ranks":
    out['decile'] = pd.qcut(out['y_true'], q=10, labels=False) + 1
    decile_means = out.groupby('decile')['y_pred'].mean()
    print(decile_means)

# Calculate the NDCG score
ndcg_scorer(y_true=y_test, y_pred=y_pred, k=5)
ndcg_scorer(y_true=y_test, y_pred=y_pred, k=10)
ndcg_scorer(y_true=y_test, y_pred=y_pred, k=100)




# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------

# Extract the actual XGBoost model from the sklearn pipeline
xgb_model = best_model.named_steps['ranker'].ranker_

# Use feature importance plot from XGBoost
import xgboost as xgb

xgb.plot_importance(
    xgb_model,
    importance_type='weight',
    max_num_features=20,
    title='Feature Importance (weight)',
)
xgb.plot_importance(
    xgb_model,
    importance_type='gain',
    max_num_features=20,
    title='Feature Importance (gain)',
    show_values=False,
)



