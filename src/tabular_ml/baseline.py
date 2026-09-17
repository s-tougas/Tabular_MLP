from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def train_xgboost(train_df,val_df,config):
    xtr = train_df[config.numeric_cols+ config.categorical_cols]
    ytr = train_df[config.target_col]

    xval = val_df[config.numeric_cols+ config.categorical_cols]
    yval = val_df[config.target_col]

    xgb = XGBRegressor(n_estimators= config.xgb_n_estimators, random_state = config.xgb_random_state)
    xgb.fit(xtr,ytr)
    prediction = xgb.predict(xval)

    mse = mean_squared_error(yval,prediction)
    rmse = mse ** 0.5

    return xgb, rmse