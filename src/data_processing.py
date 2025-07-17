import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, data):
        self.data = data

    def handle_missing_values(self, method='drop', fill_value=None):
        if method == 'drop':
            self.data = self.data.dropna()
        elif method == 'fill':
            self.data = self.data.fillna(fill_value)
        return self.data

    def remove_duplicates(self):
        self.data = self.data.drop_duplicates()
        return self.data

    def correct_data_types(self):
        if 'signup_time' in self.data.columns:
            self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
        if 'purchase_time' in self.data.columns:
            self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])
        return self.data

    def normalize_and_scale(self, columns):
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data

    def encode_categorical(self, columns):
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)
        return self.data