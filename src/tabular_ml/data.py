import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean(df, config):
    cols = config.numeric_cols+ config.categorical_cols + [config.target_col]
    df = df[cols].copy()
    
    # fill missing val with median val
    
    for col in config.numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in config.categorical_cols:
        df[col] = df[col].fillna("missing")
    return df


def split_data(df, config):
    #Train 80% of dataset
    #Validate 10% of dataset
    #Test 10% of dataset
    train_val, test = train_test_split(df,test_size= config.test_size,random_state=config.random_seed)
    train, val = train_test_split( train_val, test_size= (config.val_size)/(1-config.test_size), random_state= config.random_seed)

    return train ,val, test


def fit_transform_data(df, config):
    n_df = df.copy()
    scaler = StandardScaler() # Subtract the mean and divide by standard deviation
    n_df[config.numeric_cols]= scaler.fit_transform(df[config.numeric_cols])
    encoders = {}
    for col_ in config.categorical_cols:
        lab_enc = LabelEncoder() 
        n_df[col_] = lab_enc.fit_transform(df[col_])
        encoders[col_] = lab_enc

    return n_df, scaler, encoders


#TODO Currently apply_transf will raise a ValueError if a category not seen
#during training appears in val or test sets. For the moment with the set numerical cols 
# and categorical cols this should be fine but this can and will be sorted out.
# I'm thinking about implementing a fallback category to mitigate.
def apply_transf(df, config, scaler, encoders):
    #
    n_df = df.copy()
    n_df[config.numeric_cols]= scaler.transform(df[config.numeric_cols])

    for col in config.categorical_cols:
        n_df[col]= encoders[col].transform(df[col])

    return n_df







