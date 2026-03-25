from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

@dataclass
class PreprocessorConfig:
    
    # general
    target_col: str = "SalePrice"
    n_splits: int = 5
    random_state: int = 1234

    # missing Values
    missing_threshold: float = 0.5
    categorical_missing_method: str = 'mode'
    numerical_missing_method: str = 'median'

    # outliers
    outlier_method: str = 'cap'
    outlier_threshold: float = 1.5

    # encoding
    binary_encoding: str = 'label'
    categorical_encoding: str = 'onehot'
    max_onehot_unique_count: int = 20
    gradations: Dict[str, List[Any]] = field(default_factory=dict)

    # scaling
    scaling: str = 'standard'

    # model config    
    model_type: str = "linear"  # Options: "linear", "ridge", "lasso"
    model_params: Dict[str, Any] = field(default_factory=dict)

    feature_engineering: Dict[str, Any] = field(default_factory=dict)

    def get_scaler(self):
        mapping = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': None
        }
        return mapping.get(self.scaling)

    # "imputer" = missing value handler

    def get_numerical_imputer(self) -> Callable:
        mapping = {
            'median': lambda col: col.median(),
            'mean': lambda col: col.mean(),
        }
        return mapping.get(self.numerical_missing_method)

    def get_categorical_imputer(self) -> Callable:
        mapping = {
            'mode': lambda col: col.mode()[0] if not col.mode().empty else None,
        }
        return mapping.get(self.categorical_missing_method)