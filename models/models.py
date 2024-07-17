


import xgboost as xgb
from xgboost import XGBClassifier

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score

import matplotlib.pyplot as plt 


class XGBClassifierWrapper:
    def __init__(self, verbose=False):
        """
        Initialize the XGBClassifierWrapper with model parameters.
        """
        self.params = {
                    'n_estimators': 3000,
                    'objective': 'binary:logistic',
                    'learning_rate': 0.005,
                    'subsample': 0.555,
                    'colsample_bytree': 0.7,
                    'min_child_weight': 3,
                    'max_depth': 8,
                    'n_jobs': -1
                    }
        
        self.model = XGBClassifier(**self.params)
        self.verbose = verbose

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Fit the XGBoost model with the given training and test data.
        """
        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.model.fit(
            X_train, y_train, eval_set=eval_set,
            eval_metric='auc', early_stopping_rounds=100, verbose=self.verbose
        )

    def predict(self, X_test):
        """
        Make predictions using the trained XGBoost model.
        """
        if hasattr(self.model, 'best_iteration'):
            iteration_range = (0, self.model.best_iteration + 1)
            return self.model.predict(X_test, iteration_range=iteration_range)
        else:
            return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data and print various metrics.
        """
        y_pred = self.predict(X_test)
        if self.verbose:
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            print("Model Final Generalization Accuracy: %.6f" % accuracy_score(y_test, y_pred))
        
        # Assuming get_roc is defined to calculate and display the ROC curve
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        self.get_roc(y_test, y_pred_proba)  # This function needs to be defined elsewhere

    def get_roc (self, y_test, y_pred):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        #Plot of a ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="upper left")
        plt.show()
        return

    def get_model(self):
        """
        Return the internal XGBoost model.
        """
        return self.model
    
    def get_params(self):

        return self.model.get_params()


