import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder

from .preprocessor_config import PreprocessorConfig

class Preprocessor:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame, cfg: PreprocessorConfig):
        self.df = df.copy()
        self.cfg = cfg
        
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

    def prepare_base_data(self):
        """Universal cleaning that doesn't depend on the split."""
        self._drop_ids()
        self._feature_engineering()
        self._handle_gradations()
        self._handle_missing_values()
        self._handle_outliers()
        self._encode_low_cardinality()

    def get_folds(self):
        """A generator that yields split, and scaled data."""
        
        kf = RepeatedKFold(
            n_repeats=self.cfg.n_repeats,
            n_splits=self.cfg.n_splits,
            random_state=self.cfg.random_state)
        
        y = self.df[self.cfg.target_col]
        X = self.df.drop(columns=[self.cfg.target_col])

        for train_idx, test_idx in kf.split(X):
            self.X_train, self.X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            self.y_train, self.y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # post-split processing
            self._encode_high_cardinality()
            self._scale_data()
            
            yield self.X_train, self.X_test, self.y_train, self.y_test

    def _drop_ids(self):
        self.df = self.df.drop(columns=['Id'])

    def _feature_engineering(self):
        settings = self.cfg.feature_engineering

        self.df['LotFrontage'] = self.df['LotFrontage'].replace('?', np.nan)
        # self.df['LotFrontage'] = self.df['LotFrontage'].replace('?', 0.0) # gorsze
        self.df['LotFrontage'] = pd.to_numeric(self.df['LotFrontage'], errors="coerce")

        if settings['calc_age'] != 'no':
            self.df['Age'] = self.df['YrSold'] - self.df['YearBuilt']
            if settings['calc_age'] == 'replace':
                self.df = self.df.drop(columns=['YrSold', 'YearBuilt'])
                
        if settings['calc_total_sf'] != 'no':
            self.df['TotalSF'] = (
                self.df['1stFlrSF'] + 
                self.df['2ndFlrSF'] + 
                self.df['TotalBsmtSF']
            )
            if settings['calc_total_sf'] == 'replace':
                self.df = self.df.drop(columns=['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'])

    def _handle_gradations(self):
        for column, order in self.cfg.gradations.items():
            if column in self.df.columns:
                # Create a mapping dictionary, e.g. {"Po": 0, "Fa": 1, ...}.
                mapping = {val: i for i, val in enumerate(order)}
                self.df[column] = self.df[column].map(mapping).astype(float)

    # TODO: some Nones are actual category values
    def _handle_missing_values(self):
        num_imputer = self.cfg.get_numerical_imputer()
        cat_imputer = self.cfg.get_categorical_imputer()

        # for junk in ['None', '?']:
        #     self.df.replace(junk, np.nan, inplace=True)

        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue

            if self.df[col].dtype == 'object':
                # Try to force it to numeric if it looks like it should be.
                converted = pd.to_numeric(self.df[col], errors='coerce')
                if converted.dtype != 'object':
                    self.df[col] = converted

            if self.df[col].dtype in ['object', 'category', 'string']:
                val = cat_imputer(self.df[col])
            else:
                val = num_imputer(self.df[col])
            self.df[col] = self.df[col].fillna(val)

    def _handle_outliers(self):
        columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.cfg.outlier_threshold * IQR
            upper_bound = Q3 + self.cfg.outlier_threshold * IQR

            # todo na co najmniej 3 cechach, prog ilosci cech ktory musi byc outlierem, zeby wywalic record

            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

    def _encode_low_cardinality(self):
        """Handles binary and one-hot encoding (safe to do before split)."""

        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            
            # binary data => label encoding
            if self.df[col].nunique() == 2:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
            
            # several categories => one-hot encoding
            # TODO "reszte do jednego worka, pewien prog, gdy mamy malo wartosci dla pewnej kolumny"
            elif self.df[col].nunique() < self.cfg.max_onehot_unique_count:
                self.df = pd.get_dummies(
                    self.df,
                    columns=[col],
                    drop_first=True,
                    prefix=col
                )

    def _encode_high_cardinality(self):

        # fallback
        global_mean = self.y_train.mean()

        for col in self.X_train.select_dtypes(include=['object', 'category']).columns:
            if self.df[col].nunique() >= self.cfg.max_onehot_unique_count:
                target_means = self.y_train.groupby(self.X_train[col]).mean()
                self.X_train[col] = self.X_train[col].map(target_means)
                
                # `.fillna(global_mean)` handles categories in Test that weren't in Train
                self.X_test[col] = self.X_test[col].map(target_means).fillna(global_mean)
                
                self.X_train[col] = self.X_train[col].astype(float)
                self.X_test[col] = self.X_test[col].astype(float)
                target_means = self.df.groupby(col)[self.cfg.target_col].mean()
                self.df[col] = self.df[col].map(target_means)
        
        # cols_to_encode = [
        #     col for col in X_train.columns 
        #     if X_train[col].dtype == 'object' 
        #     and X_train[col].nunique() >= max_onehot_unique_count
        # ]
        #
        # if not cols_to_encode:
        #     return
        #
        # # Initialize the encoder
        # # 'smooth' automatically handles the weight between category and global mean
        # encoder = TargetEncoder(target_type='continuous', random_state=self.random_state)
        #
        # # Fit only on training data
        # encoder.fit(self.X_train[cols_to_encode], self.y_train)
        #
        # # Transform both
        # self.X_train[cols_to_encode] = encoder.transform(self.X_train[cols_to_encode])
        # self.X_test[cols_to_encode] = encoder.transform(self.X_test[cols_to_encode])

    def _scale_data(self):
        scaler = self.cfg.get_scaler()
        scaler.fit(self.X_train)
        
        # save column names, because NumPy doesn't save them
        columns = self.X_train.columns
        
        self.X_train = pd.DataFrame(
            scaler.transform(self.X_train), 
            columns=columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test), 
            columns=columns,
            index=self.X_test.index
        )