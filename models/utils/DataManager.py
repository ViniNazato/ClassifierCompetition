import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

class DataManager:
    
    def __init__(self,path):
        self.path = path
        self.df_original = pd.read_excel(self.path)
        self.X = self.df_original.iloc[:,:22]
        self.y = self.df_original.iloc[:, 22]
        
    def get_original_df(self):
        return self.df_original

    def train_test_splits(self, train_pct: float):
        train_test = train_test_split(self.X, self.y, train_size=train_pct, random_state=42, stratify= self.y)
        return train_test
    
    def StratifiedFoldCV(self, n_splits):
        skf = StratifiedKFold(n_splits=n_splits)
        
    