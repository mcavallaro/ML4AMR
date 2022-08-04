
from hyperopt import hp, tpe
from hyperopt.fmin import fmin


space = {
    'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'adaptive']),
    'alpha': hp.quniform('alpha', 0.1, 1, 0.00001),
    'epsilon': hp.quniform('epsilon', 0.01, 1, 0.001),
    'eta0': hp.quniform('eta0', 0.1, 2, 0.001),
#     'max_iter': hp.choice('max_iter', [500, 1000, 2000, 2500]) 
}
    
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

Iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

Iris['species'] = Iris['species'].apply(lambda x: 1 if x == 'setosa' else 0)

X_train, X_test, y_train, y_test = train_test_split(
    Iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]], Iris['species'], test_size=0.2, random_state=42)


def objective(params, X=X_train, y=y_train):
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.linear_model import SGDClassifier
    
    params = {
#         'max_iter': params['max_iter'],
        'learning_rate': params['learning_rate'],
        'alpha': params['alpha'],
        'epsilon': params['epsilon'],
        'eta0': params['eta0']
    }

    clf = SGDClassifier(
        tol=None,
        **params 
    ) 
    
    score = cross_val_score(clf, X.values, y.values, scoring='roc_auc', cv=StratifiedKFold(n_splits=5))
    print(score)
#     print("AUC %.3f"%(-score,))

    return -score.mean()



# best_params = fmin(fn=hoc.objective,
#     space=hoc.space,
#     algo=hoc.tpe.suggest,
#     max_evals=100)
    
# print("The best parameters where: ", best_params)
 

