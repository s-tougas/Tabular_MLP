import torch
import torch.nn as nn

from src.tabular_ml import config

class tabular_mlp(nn.Module):
    def __init__(self, config, categories):
        super().__init__()
        category_dict = { }
        for col in config.categorical_cols:
            category_dict[col]=  min(config.embedding_dim_cap, (categories[col] + 1) //2)



        self.embeddings = nn.ModuleList([ nn.Embedding(categories[col],category_dict[col]) for col in config.categorical_cols])
        self.mlp_input_dim = sum(category_dict.values()) + len(config.numeric_cols)

        curr_dim = self.mlp_input_dim
        layers = []

        for hd in config.hidden_dims:
            layers.append(nn.Linear(curr_dim,hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            curr_dim = hd

        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, numeric_input, categorical_input):

        tensor_list = [self.embeddings[col](categorical_input[:, col])for col in range(len(self.embeddings))]

        output = torch.cat((tensor_list+ [numeric_input]), dim = 1 )
            
        prediction = self.mlp(output)
        return prediction
