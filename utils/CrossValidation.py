from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import sklearn

@dataclass
class CrossValidation:
    
    X: pd.DataFrame
    y: pd.DataFrame
    model: sklearn.base.BaseEstimator
    n_splits: int
    
    def __post_init__(self):
        self.SKFcv = StratifiedKFold(n_splits=self.n_splits) 
          
    def grid_search(self, params: dict,scoring:str ='roc-auc',verbose:bool=True):
       
        verbose = 1 if verbose == True else 0
        
        grid_model = GridSearchCV(estimator=self.model, param_grid=params, scoring=scoring,verbose=verbose, cv= self.SKFcv, n_jobs=-1)
        grid_model.fit(self.X, self.y)
        
        print(grid_model.best_estimator_)
        
        return grid_model, grid_model.best_params_
    
    def roc_cv_visualization(self, model):

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(6, 6))

        for fold, (train, test) in enumerate(self.SKFcv.split(self.X,self.y)):    
            
            model.fit(self.X.iloc[train], self.y.iloc[train])
            
            viz = RocCurveDisplay.from_estimator(
                model,
                self.X.iloc[test],
                self.y.iloc[test],
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=f"Mean ROC (AUC = {round(mean_auc,2)} +- {round(std_auc,2)}) ",
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability",
        )
        ax.axis("square")
        ax.legend(loc="lower right")
        plt.show()