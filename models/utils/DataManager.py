import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class DataManager:
    
    def __init__(self,path):
        self.path = path
        self.df_original = pd.read_excel(self.path, index_col=0)
        self.X = self.df_original.iloc[:,:21]
        self.y = self.df_original.iloc[:, 21]
        
    def get_original_df(self):
        
        print(self.get_class_propotion(self.y))
        
        return self.df_original

    def train_test_splits(self, train_pct: float):
        train_test = train_test_split(self.X, self.y, train_size=train_pct, random_state=42, stratify= self.y)
        return train_test
    
    @staticmethod
    def get_class_propotion(df:pd.Series):
        
        return df.value_counts(normalize=True) * 100
    