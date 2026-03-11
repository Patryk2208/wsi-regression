import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler


class DataPreprocessor:
    raw_data: pd.DataFrame

    def __init__(self, df, config = None):
        self.raw_data = df
        self.preprocessed_data = self.raw_data.copy()
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
        self.preprocessed_data = self.preprocessed_data.drop(columns=['Id'])

    def handle_missing_values(self):
        for junk in ['None', '?']:
            self.preprocessed_data.replace(junk, np.nan, inplace=True)

        for col in self.raw_data.columns:
            if self.raw_data[col].isnull().sum() > 0:
                if self.raw_data[col].dtype in ['object', 'category', 'string']:
                    val = self.categorical_missing_method_pointer(self.raw_data[col])
                else:
                    val = self.numerical_missing_method_pointer(self.raw_data[col])
                self.preprocessed_data[col] = self.preprocessed_data[col].fillna(val)

    def handle_outliers(self):
        columns = self.raw_data.select_dtypes(include=[np.number]).columns

        for col in columns:
            Q1 = self.preprocessed_data[col].quantile(0.25)
            Q3 = self.preprocessed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR

            self.preprocessed_data[col] = self.preprocessed_data[col].clip(lower_bound, upper_bound)

    def encode_non_numeric_data(self):
        categorical_cols = self.preprocessed_data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if self.preprocessed_data[col].nunique() == 2:
                # binary data => label encoding
                le = LabelEncoder()
                self.preprocessed_data[col] = le.fit_transform(self.preprocessed_data[col].astype(str))
            elif self.preprocessed_data[col].nunique() < self.max_onehot_unique_count:
                # several categories => one-hot encoding
                self.preprocessed_data = pd.get_dummies(
                    self.preprocessed_data,
                    columns=[col],
                    drop_first=True,
                    prefix=col
                )
            else:
                # lots of categories => target encoding
                target_means = self.preprocessed_data.groupby(col)[self.target_col].mean()
                self.preprocessed_data[col] = self.preprocessed_data[col].map(target_means)

    def split_data(self):
        y = self.preprocessed_data[self.target_col]
        X = self.preprocessed_data.drop(self.target_col, axis=1)

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
    