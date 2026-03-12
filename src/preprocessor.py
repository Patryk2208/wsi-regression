import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler


class DataPreprocessor:
    df: pd.DataFrame

    def __init__(self, df, config = None):
        self.df = df
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        self.config = config or {}

        #config params and defaults
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 1234)

        self.missing_threshold = self.config.get('missing_threshold', 0.5)
        self.categorical_missing_method = self.config.get('categorical_missing_method', 'mode')
        self.numerical_missing_method = self.config.get('numerical_missing_method', 'median')

        self.outlier_method = self.config.get('outlier_method', 'cap')
        self.outlier_threshold = self.config.get('outlier_threshold', 1.5)

        self.binary_encoding_method = self.config.get('binary_encoding', 'label')
        self.categorical_encoding_method = self.config.get('categorical_encoding', 'onehot')
        self.max_onehot_unique_count = self.config.get('max_onehot_unique_count', 20)

        self.scaling_method = self.config.get('scaling', 'standard')

        self.target_col = self.config.get('target_col', "SalePrice")

        # mappings for config
        self.scaler_mapping = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            "none": None
        }
        self.categorical_missing_method_mapping = {
            'mode': lambda col: col.mode()[0] if not col.mode().empty else None,
        }
        self.numerical_missing_method_mapping = {
            'median': lambda col: col.median(),
            'mean': lambda col: col.mean(),
        }

        self.scaler = self.scaler_mapping[self.scaling_method]
        self.categorical_missing_method_pointer = self.categorical_missing_method_mapping[self.categorical_missing_method]
        self.numerical_missing_method_pointer = self.numerical_missing_method_mapping[self.numerical_missing_method]

    def preprocess(self):
        self.drop_ids()
        self.handle_missing_values()
        self.handle_outliers()
        self.encode_non_numeric_data()
        self.split_data()
        self.scale_data()

    def drop_ids(self):
        self.df = self.df.drop(columns=['Id'])

    def handle_missing_values(self):
        for junk in ['None', '?']:
            self.df.replace(junk, np.nan, inplace=True)

        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue

            if self.df[col].dtype == 'object':
                # Try to force it to numeric if it looks like it should be.
                converted = pd.to_numeric(self.df[col], errors='coerce')
                if converted.dtype != 'object':
                    self.df[col] = converted

            if self.df[col].dtype in ['object', 'category', 'string']:
                val = self.categorical_missing_method_pointer(self.df[col])
            else:
                val = self.numerical_missing_method_pointer(self.df[col])
            self.df[col] = self.df[col].fillna(val)

    def handle_outliers(self):
        columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR

            # todo na co najmniej 3 cechach, prog ilosci cech ktory musi byc outlierem, zeby wywalic record

            self.df[col] = self.df[col].clip(lower_bound, upper_bound)

    def encode_non_numeric_data(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if self.df[col].nunique() == 2:
                # binary data => label encoding
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
            elif self.df[col].nunique() < self.max_onehot_unique_count:
                # todo reszte do jdnego worka, pewien prog, gdy mamy malo wartosci dla pewnej kolumny

                # several categories => one-hot encoding
                self.df = pd.get_dummies(
                    self.df,
                    columns=[col],
                    drop_first=True,
                    prefix=col
                )
            else:
                # lots of categories => target encoding
                target_means = self.df.groupby(col)[self.target_col].mean()
                self.df[col] = self.df[col].map(target_means)

    def split_data(self):
        y = self.df[self.target_col]
        X = self.df.drop(self.target_col, axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def scale_data(self):
        self.scaler.fit(self.X_train)
        
        # save column names, because NumPy doesn't save them
        columns = self.X_train.columns
        
        self.X_train = pd.DataFrame(
            self.scaler.transform(self.X_train), 
            columns=columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test), 
            columns=columns,
            index=self.X_test.index
        )
    