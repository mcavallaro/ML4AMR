import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from scipy.stats import entropy    

def gini(x):
    # Gini impurity
    return 1 - np.sum(x * x)

from jenkspy import jenks_breaks


class Scaler():
    """
    Scaler = xw.Scaler(X_train)
    X_train = Scaler.transform(X_train)
    X_test = Scaler.transform(X_test)
    """
    def __init__(self, X):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)

        
    def transform(self, X):
        if type(X) is pd.DataFrame:
            return pd.DataFrame(data=self.scaler.transform(X), index=X.index, columns=X.columns)
        else:
            return self.scaler.transform(X)
        
        

class xgbClassifier():
    """
    This class is a wrapper to the xgboost classifier that deals with pandas 
    """
    def __init__(self, num_boost_round=10, **kwargs):
        self.bst = None
        self.num_boost_round = num_boost_round
        self.params = kwargs['params']
        self.params.update({'objective': 'multi:softprob'}) # error evaluation for multiclass training
        if 'max_dept' in self.params.keys():
            self.params.update({'max_depth': int(self.params['max_depth'])})

 
    def fit(self, X, y, num_boost_round=None):
        
        num_boost_round = num_boost_round or self.num_boost_round
        
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        self.num2label = {i: label for label, i in self.label2num.items()}

        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        
        # this saves object of class booster in `self.bst`
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
        
    def predict_proba(self, X, output_margin=False):
        dtest = xgb.DMatrix(X.values)
        prediction = self.bst.predict(dtest, output_margin=output_margin)            

        if type(X) is pd.DataFrame:
            return pd.DataFrame(prediction, index=X.index, columns=[self.num2label[i] for i in sorted(list(self.num2label.keys()))])
        else:
            return prediction
        
    def predict(self, X, output_margin=True):
        return self.predict_proba(X, output_margin=output_margin)

    def score(self, X, y_test):
        Y = self.predict_proba(X)
        return 1 / self.logloss(Y, y_test)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    def logloss(self, Y, y_test):
        import math
        if type(Y) is pd.DataFrame:
            Y = Y.values
            
        if type(y_test) is pd.DataFrame:
            y_test = y_test.values
            
        return - sum(math.log(y[self.label2num[label]]) if y[self.label2num[label]] > 0 else -np.inf for y, label in zip(Y, y_test)) / len(y_test)
    
    
    def get_score(self, importance_type='gain'):
        """
        returns a dictionary {feature: score}
        """
        return self.bst.get_score(importance_type=importance_type)
    
    
    def plt_importance(self, ax=None, key_list=None, key_list_color=None, importance_type='gain', dictionary_code=None, height=0.2, show_values=True, grid=True, max_num_features=None):
        """
        """
        importance = self.bst.get_score(importance_type=importance_type)
        
        if key_list is not None:
            importance = {key:importance.get(key) for key in key_list if key in importance}

        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(1, 1)

        if dictionary_code is None:
            tuples = [(k, importance[k]) for k in importance]
        else:
            tuples = [(k, importance[k], dictionary_code(k)) for k in importance]
        
        tuples = sorted(tuples, key=lambda x: x[1])
        
        if max_num_features is not None:
            tuples = tuples[-max_num_features:]

        if dictionary_code is None:
            labels, values = zip(*tuples)
        else:
            _, values, labels = zip(*tuples)
    
        
        if key_list_color is not None:
            color = ['tab:red' if _ in key_list_color else 'tab:blue']
        else:
            color = 'tab:blue'
            
        ylocs = np.arange(len(values))
        ax.barh(ylocs, values, align='center', height=height, color=color)
            
        if show_values is True:
            for x, y in zip(values, ylocs):
                ax.text(x + 0.1 * max(values), y, np.round(x, 3), va='center')

        ax.set_yticks(ylocs)
        ax.set_yticklabels(labels)
        ax.grid(grid)
        ax.set_title('Feature importance (' + importance_type +')')
        return ax
        


def buildROC(y_test, prediction_prob, ax, color=None, label=None, CI=False, plot_CI=False):
    """
    Works only with binary predictions!
    Usage:
    fig, ax = plt.subplots(1)
    prediction_p_test = clf.predict_proba(X_test).iloc[:,0]
    prediction_p_train = clf.predict_proba(X_train).iloc[:,0]
    #
    buildROC(y_test, prediction_p_test, ax, label='test')
    buildROC(y_train, prediction_p_train, ax, label='train')
    """
        
    if type(y_test) is pd.Series:
        y_test = y_test.values
    if type(prediction_prob) is pd.Series:
        prediction_prob = prediction_prob.values
    fpr, tpr, threshold = roc_curve(y_test, prediction_prob)
    roc_auc = auc(fpr, tpr)
    ax.set_title('Receiver Operating Characteristic')
    if color is None:
        color='black'
    if CI:
        import auc_boot
        import imp
        imp.reload(auc_boot)
        roc_auc_ci = auc_boot.bootstrapped_auc(y_test, prediction_prob, ax=ax, color='tab:blue', plot_CI=plot_CI)
        if plot_CI:
            value = 'AUC=%0.2f (95%% CI %0.2f - %0.2f)' % (roc_auc_ci[1], roc_auc_ci[0], roc_auc_ci[2])
        else:
            value = 'AUC=%0.2f' % (roc_auc_ci[1])
    else:
        value = 'AUC=%0.2f' % roc_auc
        roc_auc_ci = None
        
    if (label is None):
        label = value
    else:
        label = label + ' (' + value + ')'
        
    ax.plot(fpr, tpr, 'b', label=label, color=color)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1], '--', alpha=0.1, color='gray')
    ax.set_ylabel('true-positive rate (sensitivity)')
    ax.set_xlabel('false-positive rate (1-specificity)')
    return (roc_auc, roc_auc_ci)


    
if __name__ == '__main__':
    from sklearn.metrics import precision_score
    from sklearn.model_selection import train_test_split

    import pandas as pd
    import matplotlib.pyplot as plt

    Iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        Iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]], Iris['species'], test_size=0.2, random_state=42)

    clf = xgbClassifier(
            max_depth = 3, # the maximum depth of each tree
            num_class = 3, # corresponds to the species I want to detect
            silent = 1, # logging mode - quiet
            eta=0.3,  # the training step for each iteration
            objective='multi:softprob',  # error evaluation for multiclass training
            num_round = 20  # the number of training iterations
    )

    clf.fit(X_train, y_train)

    clf.predict(X_test)
    
    print('Importance scores are:', clf.bst.get_score())
    fig, ax = plt.subplots(1) # figsize=(100,100))
    xgb.plot_importance(clf.bst, ax=ax)
    fig.tight_layout()
#     xgb.plot_tree(bst, num_trees=0, ax=ax)
    fig.savefig('importance.pdf')
    print('Precision score is: ', precision_score(y_test, clf.predict(X_test), average='macro'))
    
    
    print("the parameters used where: ", clf.get_params())
 