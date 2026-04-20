# import pandas as pd
# import numpy as np


# def generate_features(prices, horizons=[10, 21, 63, 252]) -> pd.DataFrame:
#     """
#     Generates financial features such as returns and volatility over specified horizons.

#     Parameters:
#     ----------
#     prices : pd.DataFrame
#         A DataFrame containing price data for one or more assets, indexed by date.
#     horizons : List[int], optional
#         A list of time horizons (in periods) over which features will be calculated.
#         Default is [10, 21, 63, 252].

#     Returns:
#     -------
#     result : pd.DataFrame
#         A DataFrame containing the generated features, with multi-level column names
#         where the first level is the feature name (e.g., 'ret_10', 'volatility_21')
#         and the second level is the asset.

#     """

#     feature_map = {}

#     # rets
#     for h in horizons:
#         rets = prices.pct_change(h)
#         feature_map[f"ret_{h}"] = rets
#         feature_map[f"ret_{h}_zscore"] = rets.apply(lambda x: (x - x.mean()) / x.std())

#     # volatility
#     for h in horizons:
#         volatility = prices.pct_change(1).rolling(h).std()
#         feature_map[f"volatility_{h}"] = volatility
#         feature_map[f"volatility_{h}_zscore"] = volatility.apply(lambda x: (x - x.mean()) / x.std())

#     # add more technical indicators (the above is more of a poc)
#     result = pd.concat(feature_map, axis=1, keys=feature_map.keys())

#     return result
