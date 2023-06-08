from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline


def sampling_SMOTEK(X, y) -> tuple:
    
    smotek =SMOTETomek(sampling_strategy='auto')
    steps = [('1', smotek)]
    pipeline = Pipeline(steps=steps)
    
    X_sampled, y_sampled = pipeline.fit_resample(X, y)
    
    return X_sampled, y_sampled

def sampling_Random(X,y) -> tuple:
    over = RandomOverSampler(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)
    
    X_sampled, y_sampled = pipeline.fit_resample(X, y)
    
    return X_sampled, y_sampled