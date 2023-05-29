from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, RocCurveDisplay, matthews_corrcoef, balanced_accuracy_score 
from imblearn.metrics import classification_report_imbalanced


def calculate_metrics(X_true, y_true, y_pred, model):
    
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    ConfusionMatrixDisplay(confusion_matrix = conf_matrix).plot()
    plt.show()

    print('Classification Report for Unbalanced Series: \n', classification_report_imbalanced(y_true, y_pred))
    print('Balanced Accuracy Score:', round(balanced_accuracy_score(y_true, y_pred), 2))
    print('Matthews Correlation Coefficient:', matthews_corrcoef(y_true, y_pred))
    print('ROC AUC Score:', roc_auc_score(y_true, model.predict_proba(X_true)[:,1]))


    RocCurveDisplay.from_estimator(model, X_true, y_true)
    plt.plot([0,1],[0,1],'k--',label='Benchmark')
    plt.title('ROC Curve Prediction')
    plt.legend()
    plt.show()
