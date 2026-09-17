import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.tabular_ml.config import config
from src.tabular_ml.baseline import train_xgboost
from src.tabular_ml.model import tabular_mlp
from src.tabular_ml.data import load_data, clean, split_data, fit_transform_data, apply_transf


import json
import joblib
from datetime import datetime
from pathlib import Path
import shutil


def evaluate_results(model, numeric, categorical, target, loss_fn):
    model.eval()
    with torch.no_grad():
        predictions= model(numeric, categorical)
        val_loss = loss_fn(predictions, target.view(-1,1))
    model.train()
    return val_loss.item()


def train_model(model, train_df, val_df, config):

    numeric_tensor = torch.tensor(train_df[config.numeric_cols].values, dtype= torch.float32)
    categorical_tensor = torch.tensor(train_df[config.categorical_cols].values, dtype= torch.long)
    
    #Scaling Fix
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(train_df[[config.target_col]].values) 
    target_tensor = torch.tensor(target_scaled, dtype= torch.float32)

    val_num_ten = torch.tensor(val_df[config.numeric_cols].values, dtype = torch.float32)
    val_cat_ten = torch.tensor(val_df[config.categorical_cols].values, dtype= torch.long)

    val_target_scaled = target_scaler.transform(val_df[[config.target_col]].values)
    val_target_tensor = torch.tensor(val_target_scaled,dtype=torch.float32)


    tensor_dataset = TensorDataset(numeric_tensor, categorical_tensor, target_tensor)
    loader = DataLoader(dataset= tensor_dataset, batch_size= config.batch_size, shuffle= True)

    loss =nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr= config.learning_rate)

    
    for ep in range(config.epochs):
        val_loss= epoch_loss = 0.0
        
        for num_b, cat_b, tar_b in loader:
            optimizer.zero_grad()
            predictions = model(num_b,cat_b)
            batch_loss = loss(predictions,tar_b.view(-1,1))
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        val_loss =evaluate_results(model,val_num_ten,val_cat_ten,val_target_tensor,loss)

        print(f"Epoch {ep+1}/{config.epochs} - Train loss: {epoch_loss / len(loader):.2f}")
        print(f"Epoch {ep+1}/{config.epochs} - Val loss: {val_loss:.2f}")

    model.eval()
    with torch.no_grad():
        scaled_predictions = model(val_num_ten,val_cat_ten)
    model.train()

   
    return target_scaler, scaled_predictions, val_target_tensor

def run_pipeline(config = config, save =True ):
    df = load_data(config.data_path)
    clean_df = clean(df, config)
    train, val, test = split_data(clean_df,config)
    train_t, scaler, encoders = fit_transform_data(train,config)
    val_t = apply_transf(val,config,scaler,encoders)
    c_categories = {col: len(encoders[col].classes_)for col in  config.categorical_cols}

    model = tabular_mlp(config,c_categories)
    tg_scaler, scaled_pred, val_target_ten = train_model(model,train_t, val_t,config)

    pred_figure = tg_scaler.inverse_transform(scaled_pred.numpy())
    act_figure = tg_scaler.inverse_transform(val_target_ten.numpy())

    nn_rmse = mean_squared_error(pred_figure,act_figure)** 0.5

    xgb, rmse = train_xgboost(train_t,val_t, config)
    print(f"Neural net val RMSE: ${nn_rmse:,.2f}")
    print(f'XGBoost val RMSE: ${rmse:,.2f}')

    if save:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config.artifacts_dir) / run_id
        run_dir.mkdir(parents=True,exist_ok = True)
        torch.save(model.state_dict(),run_dir/ "model.pt")

        joblib.dump(scaler,run_dir/ "scaler.pkl")
        joblib.dump(encoders,run_dir / "encoders.pkl")
        joblib.dump(tg_scaler,run_dir / "target_scaler.pkl")
        joblib.dump(xgb,run_dir/ "xgboost_model.pkl")

        metadata = {
            "run_id": run_id,
            "nn_rmse": nn_rmse,
            "xgb_rmse": rmse,
            "winner": "neural_net" if nn_rmse < rmse else "xgboost",
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "hidden_dims": config.hidden_dims 
        }
        with open(run_dir/ "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

def promote_run(run_id, config):
    source = Path(config.artifacts_dir)/ run_id
    dest = Path(config.artifacts_dir) / "production"

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source,dest)

if __name__ == "__main__":
    run_pipeline(config, save= True)