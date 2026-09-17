from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    data_path: Path = PROJECT_ROOT / "data" / "train.csv"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    #Target

    target_col: str = "SalePrice"

    # Features
    #   My initial data will be a subset of the columns in the data.
    #   This handful of categories are of enough significance and impact 
    #   such that I am confident that they will build a solid foundation
    #   to build on.

    numeric_cols: list[str] = field(default_factory= lambda:[
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "GrLivArea",
        "FullBath",
        "BedroomAbvGr",
        "GarageCars"
    ])

    categorical_cols: list[str] = field(default_factory=lambda:[
        "MSZoning",
        "Neighborhood",
        "HouseStyle",
        "SaleCondition"
    ])

    # Split
    test_size: float = 0.1
    val_size:float = 0.1
    random_seed = 42

    # Model hyperparameters (MLP with embeddings)
    embedding_dim_cap: int = 50

    hidden_dims: list[int] = field(default_factory= lambda: [128, 64])

    dropout: float = 0.2
    learning_rate: float = 5e-3
    batch_size: int = 64
    epochs: int = 30

    xgb_n_estimators: int = 200
    xgb_random_state: int = 42

config = Config()