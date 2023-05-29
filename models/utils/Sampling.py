from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def sampling_SMOTE(X, y) -> tuple:
    
    # Create Pipeline
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    
    X_sampled, y_sampled = pipeline.fit_resample(X, y)
    
    return X_sampled, y_sampled