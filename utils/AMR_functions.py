import sys
import imp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
import xgboost as xgb
import stratify as st
import xgboost_wrapper as xw
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, accuracy_score
from datetime import timedelta
import warnings
import utils as ut
import shap
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from hyperopt import STATUS_OK
from sklearn.metrics import roc_auc_score

def my_train_test_split(df, response_name, all_response_names, test_size=0.3, random_state=42):
    """
    Usage:
    all_response_names = ['AMP', 'AUG', 'GT', 'CIP', 'CPD', 'TAZ', 'MEM', 'ETP', 'CAZ', 'CTX', 'TEM', 'AK', 'ATM', 'COL']
    X_train, X_test, y_train, y_test = my_train_test_split(pharma_lab_demo_df,
                                                        'AUG',
                                                        all_response_names)
    """
    locs = df[response_name].notnull()
    response = df.loc[locs, response_name] 
    Data = df.loc[locs].drop(all_response_names, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(
        Data, response, test_size=test_size, random_state=random_state)
    y_train = y_train.apply(lambda x: 1 if x=='R' else 0) 
    y_test = y_test.apply(lambda x: 1 if x=='R' else 0)
    return X_train, X_test, y_train, y_test



class binary_Classifier(xw.xgbClassifier):
    def __init__(self, num_boost_round=10, **kwargs):
        self.bst = None
        self.num_boost_round = num_boost_round
        self.params = kwargs['params']
        self.params.update({'objective': 'binary:logistic'})
        print(self.params) #xgb.XGBClassifier
        if 'max_depth' in self.params.keys():
            self.params.update({'max_depth': int(self.params['max_depth'])})
#         if 'scale_pos_weight' in self.params.keys():
#             self.params.update({'scale_pos_weight': int(self.params['scale_pos_weight'])})            

# There are in general two ways that you can control overfitting in XGBoost:
# The first way is to directly control model complexity.
# This includes `max_depth`, `min_child_weight` and `gamma`.
# The second way is to add randomness to make training robust to noise.
# This includes `subsample` and `colsample_bytree`.
# You can also reduce stepsize `eta`. Remember to increase `num_round` when you do so.

# params = {"max_depth":6, #default=6, Maximum depth of a tree. Increasing this value will make the model more complex.
#           "n_estimators":1000, 
#           "min_child_weight":4, #default=1 control model complexity
#           "gamma":1,  # default=0, [\in\0,infty], minimum loss reduction, increasing this value will make model more conservative, control model complexity
#           "silent":1, # logging mode - quiet
# #          "eta":0.3,  # default=0.3, [\in\0,1], learning rate, Increasing this value will make model more conservative.
#           "colsample_bytree":0.6, # default=1, [\in[\0,1] (add randomness to make training robust to noise)
#           "subsample":0.85, #default=1 (add randomness to make training robust to noise)
#           "reg_alpha":0.25, #default=0, L1 regularization term on weights, Increasing this value will make model more conservative.
#           "reg_lambda":0.5, #default=1, L2 regularization term on weights, Increasing this value will make model more conservative.
# #           "num_round":100 # the number of training iterations
#          }



# class binary_Classifier2(binary_Classifier):
#     def __init__(self, num_boost_round=10, verbose_eval=True, **kwargs):
#         self.bst = None
#         self.num_boost_round = num_boost_round
#         self.params = kwargs['params']
#         self.params.update({'objective': 'binary:logistic'})
#         self.verbose_eval = verbose_eval
#         print(self.params) #xgb.XGBClassifier
#         if 'max_depth' in self.params.keys():
#             self.params.update({'max_depth': int(self.params['max_depth'])})

 
#     def fit(self, X, y, num_boost_round=None):
        
#         num_boost_round = num_boost_round or self.num_boost_round
        
#         self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
#         self.num2label = {i: label for label, i in self.label2num.items()}

#         dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        
#         # this saves object of class booster in `self.bst`
#         self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round, verbose_eval=self.verbose_eval)


        
def get_prediction(clf, X, y):
    import xgboost as xgb
    import xgboost_wrapper as xw
    prediction = clf.bst.predict(xgb.DMatrix(X), output_margin=False)
    prediction = pd.DataFrame({'p': prediction})
    prediction['predicted response'] = prediction['p'].apply(lambda p: 1 if p > 0.5 else 0)
#     prediction_from_train.head()
    prediction['1-p'] = prediction['p'].apply(lambda p: 1 - p)
    prediction['entropy'] = prediction[['p','1-p']].apply(lambda x: xw.entropy(x), axis=1)
    prediction['Gini coeff.'] = prediction[['p','1-p']].apply(lambda x: xw.gini(x), axis=1)
    prediction['true response'] = y.values    
    return prediction

import bisect as bs

def plot_breakpoint(prediction, ax=None, impurity='Gini coeff.', colors=None, labelsize=12, bins=10, accuracy=False, legend=True):
    if ax is None:
        fig, ax = plt.subplots(1,1,  figsize=(4 * 1.2, 3 * 1.2))
        
    #.hist(ax=ax)
    h, breaks = np.histogram(prediction[impurity].values, density=True, bins=bins)
    d = breaks[1] - breaks[0]
    breakpoint = xw.jenks_breaks(prediction[impurity].values, 2)[1]
    ymax = h[bs.bisect_left(breaks, breakpoint)] / np.max(h)
#     ax.vlines(breakpoint, ymin=0, ymax=ymax, color='black', lw=0.5);
    ax.set_xlabel('Impurity', fontsize=labelsize+1);
    
    centers = breaks[1:] - d/2
    H = h /np.max(h)
    if colors is None:
        colors = cm.get_cmap('coolwarm', len(H))(range(len(H)))
    else:
        colors = np.repeat(colors, len(H))
    for i in range(len(H)):
        if i == 0:
            ax.bar(centers[i], H[i], color=colors[i],
                   width=d, alpha=0.2, label='Imp. freq.')
        else:
            ax.bar(centers[i], H[i], color=colors[i],
                   width=d, alpha=0.2)                
    
    break_point_entropy = xw.jenks_breaks(prediction['entropy'].values, 2)[1]
    break_point_gini = xw.jenks_breaks(prediction['Gini coeff.'].values, 2)[1]
#     print("Entropy break point: %.3f, Gini break point %.3f"%(break_point_entropy, break_point_gini))
    
    auc = []
    mse = []

    for i in range(len(breaks)-1):
        idx = (prediction[impurity].values < breaks[i+1]) & (prediction[impurity].values > breaks[i])
        prediction_df = prediction[idx]
#         ac.append(accuracy_score(prediction_df['true response'],
#                             prediction_df['predicted response']))
        if prediction_df.shape[0] > 0:
            auc.append(roc_auc_score(prediction_df['true response'],
                                     prediction_df['predicted response']))
            mse.append(mean_squared_error(prediction_df['true response'],
                                     prediction_df['predicted response']))
            if accuracy:
                bac.append(balanced_accuracy_score(prediction_df['true response'],
                                                   prediction_df['predicted response']))

        else:
            auc.append(np.nan)
            mse.append(np.nan)
            if accuracy:
                bac.append(np.nan)
            
#     ax.plot(centers, ac, label='accuracy')
    ax.plot(centers, auc, '-x', label='AUC', markersize=11, color='tab:blue')
    ax.plot(centers, mse, '-+', label='MSE', markersize=11, color='tab:orange')
    if accuracy:
        ax.plot(centers, bac, '-1', label='Balanced accuracy', markersize=11, color='tab:green')
                                                   
    
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)

    ax.set_ylim([-0.05, 1.3])
    if legend:
        ax.legend(loc='upper right', fontsize=labelsize-1)
    
    return (break_point_entropy, break_point_gini)

def plot_ROC_entropy(prediction_df, clf, break_point, impurity='entropy', ax=None, CI=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    blue = [0.2298057 , 0.29871797, 0.75368315, 1.        ]#'tab:blue'
    red = [0.70567316, 0.01555616, 0.15023281, 1.        ]#'tab:red'
    _all = xw.buildROC(prediction_df['true response'],
                    prediction_df['p'],#.apply(lambda x: clf.label2num.get(x)),
                    ax, CI=CI, label='all', color='tab:grey')
    loc = prediction_df[impurity] < break_point
    _low = xw.buildROC(prediction_df.loc[loc, 'true response'], #y[loc.values].apply(lambda x: clf.label2num.get(x)),
                    prediction_df.loc[loc,  'p'],#''.apply(lambda x: clf.label2num.get(x)),
                    ax, CI=CI, label='low entropy group', color=blue)
    loc = prediction_df[impurity] > break_point
    _high = xw.buildROC(prediction_df.loc[loc, 'true response'], #y[loc.values].apply(lambda x: clf.label2num.get(x)),
                    prediction_df.loc[loc,  'p'],#.apply(lambda x: clf.label2num.get(x)),
                    ax, CI=CI, label='high entropy group', color=red)
    if _all[1] is None:
        print('all %0.2f'%(_all[0])) 
        print('low entropy group %0.2f'%(_low[0])) 
        print('high entropy group %0.2f'%(_high[0])) 
    else:
        print('all %0.2f %0.2f %0.2f'%(_all[1][1], _all[1][0], _all[1][2]))
        print('low entropy group %0.2f %0.2f %0.2f'%(_low[1][1], _low[1][0], _low[1][2]))
        print('high entropy group %0.2f %0.2f %0.2f'%(_high[1][1], _high[1][0], _high[1][2]))
        
    ax.set_title('Receiver Operating Characteristics')
    plt.show()


def Goodness_of_prediction(prediction_df, clf, break_point=None, impurity='entropy'):
    ac = accuracy_score(prediction_df['true response'],#.apply(lambda x: clf.label2num.get(x)),
                        prediction_df['predicted response']) #.apply(lambda x: clf.label2num.get(x)))
    bac = balanced_accuracy_score(prediction_df['true response'], #y.apply(lambda x: clf.label2num.get(x)),
                                  prediction_df['predicted response']) #.apply(lambda x: clf.label2num.get(x)))
    mse = mean_squared_error(prediction_df['true response'],
                             prediction_df['predicted response'])#.apply(lambda x: clf.label2num.get(x)))
    print("accuracy score\t\t%.3f"%ac)
    print("balanced accur. score\t%.3f"%bac)
    print("mean squared error\t%.3f"%mse)

    ret = pd.DataFrame({'all':[ac, bac, mse]})
    if break_point != None:
        loc = prediction_df[impurity] < break_point
        for l,g in zip([loc, ~loc], ['low', 'high']):
            ac = accuracy_score(prediction_df.loc[l, 'true response'],
                                prediction_df.loc[l, 'predicted response']) #.apply(lambda x: clf.label2num.get(x)))
            bac = balanced_accuracy_score(prediction_df.loc[l, 'true response'], #y[l].apply(lambda x: clf.label2num.get(x)),
                                          prediction_df.loc[l, 'predicted response'])#.apply(lambda x: clf.label2num.get(x)))
            mse = mean_squared_error(prediction_df.loc[l, 'true response'], #y[l].apply(lambda x: clf.label2num.get(x)),
                                     prediction_df.loc[l, 'predicted response'])#.apply(lambda x: clf.label2num.get(x)))
            print("\nThe goodness of prediction for the %s-impurity group is:"%g)
            print(" accuracy score\t\t%.3f"%ac)
            print(" balanced accur. score\t%.3f"%bac)
            print(" mean squared error\t%.3f"%mse)
            ret[g] = [ac,bac,mse]
    return ret

space = {
    'learning_rate':hp.uniform('learning_rate', 0, 1),
#     'n_estimators': hp.quniform('n_estimators', 1000, 1500, 100),
    'max_depth': hp.quniform('max_depth', 6, 14, 1),
    'min_child_weight': hp.uniform('min_child_weight', 0.5, 10),
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
#     'num_boost_round': hp.quniform('num_boost_round', 2000, 3500, 10)}
    'subsample': hp.uniform('subsample', 0.5, 1),
    'gamma': hp.uniform('gamma', 0.0, 100.0),
    'max_delta_step': hp.uniform('max_delta_step', 1.0, 10.0)
#     'alpha': hp.uniform('alpha', 0, 10)
}

# def fspace(y):
#     max_w = (len(y) - y.sum()) / y.sum() + ?
#     min_w = (len(y) - y.sum()) / y.sum() - ?
#     d = {
#     'learning_rate':hp.uniform('learning_rate', 0, 1),
#     'max_depth': hp.quniform('max_depth', 6, 14, 1),
#     'min_child_weight': hp.uniform('min_child_weight', 0.5, 10),
#     'subsample': hp.uniform('subsample', 0.5, 1),
#     'gamma': hp.uniform('gamma', 0.0, 100.0),
#     'scale_pos_weight': hp.uniform('scale_pos_weight', min_w, max_w),        
#     }
#     return d

# def objective(params, X, y, sign=-1):
#     params = {
#         'learning_rate': "{:.3f}".format(params['learning_rate']),
# #         'n_estimators': int(params['n_estimators']),
#         'max_depth': int(params['max_depth']),
#         'min_child_weight': params['min_child_weight'],
#         'gamma': "{:.3f}".format(params['gamma']),
# #         'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
#         'subsample': '{:.3f}'.format(params['subsample']),
# #         'max_delta_step': '{:.3f}'.format(params['max_delta_step'])
# #         'alpha': "{:.3f}".format(params['alpha'])
# #         'lambda':  "{:.3f}".format(params['lambda'])
#     }

#     clf = xgb.XGBClassifier(
#         objective='binary:logistic',
#         nthread=4, seed=2,
#         num_boost_round=200,
#         scale_pos_weight = np.sqrt((len(y) - y.sum()) / y.sum()),
#         alpha=0.5,
#         reg_lambda=0,
#         **params
#     )
#     score = cross_val_score(clf, X.values, y.values, scoring='roc_auc', cv=StratifiedKFold(n_splits=5))
#     return{'loss': sign * np.mean(score), 'status': STATUS_OK}


def objective(params, X, y, sign=-1, scale_pos_weight = True):
    params = {
        'learning_rate': "{:.3f}".format(params['learning_rate']),
#         'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'min_child_weight': params['min_child_weight'],
        'gamma': "{:.3f}".format(params['gamma']),
#         'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'subsample': '{:.3f}'.format(params['subsample']),
#         'max_delta_step': '{:.3f}'.format(params['max_delta_step'])
#         'alpha': "{:.3f}".format(params['alpha'])
#         'lambda':  "{:.3f}".format(params['lambda'])
    }
    
    if scale_pos_weight:
        s = np.sqrt((len(y) - y.sum()) / y.sum())
    else:
        s = 1

    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        nthread=4, seed=2,
        num_boost_round=200,
        scale_pos_weight = s,
        alpha=0.5,
        reg_lambda=0,
        **params
    )
    score = cross_val_score(clf, X.values, y.values, scoring='roc_auc', cv=StratifiedKFold(n_splits=5))
    return{'loss': sign * np.mean(score), 'status': STATUS_OK}

def volcano_plot(shap_values, idx_match, prediction):
    fig = plt.figure(figsize=(14,7))
    
    from matplotlib import cm
    color = cm.get_cmap('plasma')(prediction.entropy)

    widths = [0.5, 4, 4, 0.5]
    heights = [0.5, 4]
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths, height_ratios=heights)

    ax = {}
    for i in range(len(heights)*len(widths)):
        ax[i] = fig.add_subplot(spec[i//len(widths), i%len(widths)])
    
    mean_idx_match = [np.mean(i) for i in shap_values[idx_match,]]
    mean_idx_not_match = [np.mean(i) for i in shap_values[~idx_match,]]
    std_idx_match = [np.std(i) for i in shap_values[idx_match,]]
    std_idx_not_match = [np.std(i) for i in shap_values[~idx_match,]]
    
    mm = [np.mean(i) for i in shap_values[:,]]
    xrange = np.array([np.min(mm), np.max(mm)]) * np.array([0.9,1.1]) 
    mm = [np.std(i) for i in shap_values[:,]]
    yrange = np.array([np.min(mm), np.max(mm)]) * np.array([0.9,1.1])
    sns.kdeplot(mean_idx_match, bw=0.005, ax=ax[1])
    sns.kdeplot(mean_idx_not_match, bw=0.005, ax=ax[2])
    sns.kdeplot(std_idx_match, bw=0.005, vertical=True, ax=ax[4])
    sns.kdeplot(std_idx_not_match, vertical=True, bw=0.005, ax=ax[7])
    
#     sns.kdeplot(mean_idx_match, label='non-matching', bw=0.005, ax=ax[2],
#                 fill=True, alpha=.5, linewidth=0)
#     sns.kdeplot(mean_idx_not_match, label='non-matching', bw=0.005, ax=ax[2])    
    
#     sns.kdeplot(std_idx_match, label='non-matching', bw=0.005, ax=ax[4],
#                 fill=True, alpha=.5, linewidth=0)
#     sns.kdeplot(std_idx_not_match, label='non-matching', bw=0.005, ax=ax[4])
#     sns.kdeplot(std_idx_match, label='non-matching', bw=0.005, ax=ax[7],
#                 fill=True, alpha=.5, linewidth=0)
#     sns.kdeplot(std_idx_not_match, label='non-matching', bw=0.005, ax=ax[7])
    ax[5].scatter(
        mean_idx_match,
        std_idx_match, label='matching', s=10, color=color[idx_match])
    ax[6].scatter(
        mean_idx_not_match,
        std_idx_not_match , label='non-matching', s=10, color=color[~idx_match])
    ax_ = ax[6].twinx()
    ax[5].set_xlabel('mean of SHAP values')
    ax[5].set_ylabel('std of SHAP values')
    ax[6].set_xlabel('mean of SHAP values')
    ax_.set_ylabel('std of SHAP values')
    
    ax[5].set_title('Matching cases')
    ax[6].set_title('Non-matching cases')

    ax[1].set_xlim(xrange)
    ax[2].set_xlim(xrange)
    ax[5].set_xlim(xrange)
    ax[6].set_xlim(xrange)    
    ax[4].set_ylim(yrange)
    ax[5].set_ylim(yrange)
    ax[6].set_ylim(yrange)
    ax_.set_ylim(yrange)    
    ax[7].set_ylim(yrange)
    
    for a in [ax[0], ax[1], ax[2], ax[3]]:
        a.get_yaxis().set_visible(False)
        a.spines['top'].set_visible(False)
        a.get_xaxis().set_ticklabels([])
    ax[6].get_yaxis().set_visible(False)

    for a in [ax[4], ax[7]]:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_ticklabels([])
        
    ax[4].spines['left'].set_visible(False)
    ax[7].spines['right'].set_visible(False)

    for a in [ax[0], ax[3]]:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.get_yaxis().set_visible(False)
        a.get_xaxis().set_visible(False)

    plt.tight_layout()
#     return fig



def volcano_plot2(shap_values, idx_match, fig=None, label1='Matching', label2='Non-matching'):
    if fig is None:
        fig = plt.figure(figsize=(7,7))
    color_warm = 'tab:orange'
    color_cold = 'tab:grey'
    widths = [4, 0.5]
    heights = [0.5, 4]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)

    ax = {}
    for i in range(len(heights)*len(widths)):
        ax[i] = fig.add_subplot(spec[i//len(widths), i%len(widths)])
    
    mean_idx_match = [np.mean(i) for i in shap_values[idx_match,]]
    mean_idx_not_match = [np.mean(i) for i in shap_values[~idx_match,]]
    std_idx_match = [np.std(i) for i in shap_values[idx_match,]]
    std_idx_not_match = [np.std(i) for i in shap_values[~idx_match,]]
    
    mm = [np.mean(i) for i in shap_values[:,]]
    xrange = np.array([np.min(mm), np.max(mm)]) * 1.1 #np.array([0.9,1.1]) 
    mm = [np.std(i) for i in shap_values[:,]]
    yrange = np.array([np.min(mm), np.max(mm)]) * np.array([0.9,1.1])
    
    sns.kdeplot(mean_idx_match, bw=0.005, ax=ax[0], color=color_cold, label=label1)
    sns.kdeplot(mean_idx_not_match, bw=0.005, ax=ax[0], color=color_warm, label=label2)
    
    sns.kdeplot(std_idx_match, bw=0.005, vertical=True, ax=ax[3], color=color_cold)
    sns.kdeplot(std_idx_not_match, vertical=True, bw=0.005, ax=ax[3], color=color_warm)

    ax[2].scatter(
        mean_idx_match,
        std_idx_match, label=label1, s=10, alpha=0.5, color=color_cold)
    ax[2].scatter(
        mean_idx_not_match,
        std_idx_not_match , label=label2, s=10, alpha=0.5, color=color_warm)
    ax[2].set_ylabel('Std of SHAP values')
    ax[2].set_xlabel('Mean of SHAP values')
    
    ax[2].legend(loc='lower right')
    ax[2].set_xlim(xrange)
    ax[2].set_ylim(yrange)
    ax[0].set_xlim(xrange)
    ax[3].set_ylim(yrange)

    for a in [ax[0], ax[1]]:
        a.get_yaxis().set_visible(False)
        a.spines['top'].set_visible(False)
        a.get_xaxis().set_ticklabels([])
        
    for a in [ax[1], ax[3]]:
        a.get_xaxis().set_visible(False)
        a.spines['right'].set_visible(False)        
        a.get_yaxis().set_ticklabels([])
    
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)

    plt.tight_layout()


def impurity_vs_mean(shap_values, prediction,  ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([np.sum(i) for i in shap_values[:,]],prediction['entropy'],'.', label='entropy')
    ax.plot([np.sum(i) for i in shap_values[:,]],prediction['Gini coeff.'],'.', label='Gini coeff.')
    ax.legend()
    ax.set_ylabel('impurity')
    ax.set_xlabel('sum of SHAP values')


def is_mismatch(x, abx):
    """
    return 1 if laboratory test reports resistance to `abx` (lab dataset)
    and `abx` has been prescribed (pharma dataset).
    """
    if (x[abx] == "R") and (x['ABX_' + abx] == True):
        return 1
    elif (x[abx] is None) or (x['ABX_' + abx] is None):
        return np.nan
    else:
        return 0

    
def is_mismatch_b72(x, abx):
    """
    return 1 if laboratory test reports resistance to `abx` (lab dataset)
    and `abx` has been prescribed during the first 72 hrs from admissio (pharma dataset).
    """
    if (x[abx] == "R") and (x['ABX_b72_' + abx] == True):
        return 1
    elif (x[abx] is None) or (x['ABX_b72_' + abx] is None):
        return np.nan
    else:
        return 0


def is_mismatch_2(x):
    """
    return 1 if laboratory test reports resistance to `abx` (lab dataset)
    and `abx` has been prescribed (pharma dataset).
    """
    if (x[0] == 1) and (x[1] == 1):
        return 1
    elif (x[0] is None) or (x[1] is None):
        return np.nan
    else:
        return 0

    
def bool_to_int(df):
    tmp  = df.copy()
    for column in tmp.columns.values:
        if tmp[column].dtype == np.dtype('bool'):
            tmp[column] = tmp[column].astype(int)
    return tmp


def partial_dependence(model, X, selected_feature):
    # The model could be an XGBoost sklearn fitted instance (or anything else with a 
    # predict method)
    X_temp = X.copy()
    # Works only for numerical features. Find something else for categorical features.
    grid = np.linspace(np.percentile(X_temp.loc[:, selected_feature], 0.05), 
                       np.percentile(X_temp.loc[:, selected_feature], 99.95), 
                       200)
    y_pred = np.zeros(len(grid))
    
    for i, val in enumerate(grid):
        X_temp.loc[:, selected_feature] = val
        a = model.predict(xgb.DMatrix(X_temp))
        y_pred[i] = a.mean()
    
    return grid, y_pred

def plot_mismatch(prediction, X, response_name, ax=None, impurity='entropy', colors=None, labelsize=12, accuracy=False, legend=True, bins=10):
    if ax is None:
        fig, ax = plt.subplots(1,1,  figsize=(4 * 1.2, 3 * 1.2))
        
    #.hist(ax=ax)
    h, breaks = np.histogram(prediction[impurity].values, density=True, bins=bins)
    d = breaks[1] - breaks[0]
#     breakpoint = xw.jenks_breaks(prediction[impurity].values, 2)[1]
#     ymax = h[bs.bisect_left(breaks, breakpoint)] / np.max(h)
#     ax.vlines(breakpoint, ymin=0, ymax=ymax, color='black', lw=0.5);
    ax.set_xlabel('Impurity', fontsize=labelsize+1);
    
    centers = breaks[1:] - d/2
    H = h /np.max(h)
    if colors is None:
        colors = cm.get_cmap('coolwarm', len(H))(range(len(H)))
    else:
        colors = np.repeat(colors, len(H))
    for i in range(len(H)):
        if i == 0:
            ax.bar(centers[i], H[i], color=colors[i],
                   width=d, alpha=0.1)
        else:
            ax.bar(centers[i], H[i], color=colors[i],
                   width=d, alpha=0.1)                
        
    mismatch = []
    mismatch_a72 = []
    mismatch_b72 = []
    mismatch_true = []
    mismatch_a72_true = []
    mismatch_b72_true = []
    
    for i in range(len(breaks)-1):
        idx = (prediction[impurity].values < breaks[i+1]) & (prediction[impurity].values > breaks[i])
        prediction_df = prediction[idx]
        X_df = X[idx]
        

        if prediction_df.shape[0] > 0:
            mism = Mism(prediction_df, X_df, response_name)
            mismatch.append(np.mean(mism.values) )
            mism = Mism_a72(prediction_df, X_df, response_name)
            mismatch_a72.append(np.mean(mism.values) )
            mism = Mism_b72(prediction_df, X_df, response_name)
            mismatch_b72.append(np.mean(mism.values))
            #
            mism = Mism(prediction_df, X_df, response_name, response='true response')
            mismatch_true.append(np.mean(mism.values) )
            mism = Mism_a72(prediction_df, X_df, response_name, response='true response')
            mismatch_a72_true.append(np.mean(mism.values) )
            mism = Mism_b72(prediction_df, X_df, response_name, response='true response')   
            mismatch_b72_true.append(np.mean(mism.values) )

        else:
            mismatch.append(np.nan)
            mismatch_a72.append(np.nan)            
            mismatch_b72.append(np.nan)
            mismatch_true.append(np.nan)
            mismatch_a72_true.append(np.nan)            
            mismatch_b72_true.append(np.nan)
            
   
    ax.plot(centers, mismatch, marker='.', linestyle='-', label='Pred. respon.', linewidth=2, markersize=7, color='tab:blue')
#     ax.plot(centers, mismatch_a72, marker='.', linestyle=':', label='Pred. a72', linewidth=2, markersize=7, color='tab:orange')
    ax.plot(centers, mismatch_b72, marker='.', linestyle='-', label='Pred. respon. (b72)', linewidth=2, markersize=7, color='tab:green')                                                   
    ax.plot(centers, mismatch_true, marker='.', linestyle=':', label='True respon.', linewidth=2, markersize=7, color='tab:blue')
#     ax.plot(centers, mismatch_a72_true, marker='.', linestyle='-', label='True respon. a72', linewidth=2, markersize=7, color='tab:orange')
    ax.plot(centers, mismatch_b72_true, marker='.', linestyle=':', label='True respon. (b72)', linewidth=2, markersize=7, color='tab:green')
    

    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)

    ax.set_ylim([-0.05, 1.05])
#     ax.set_xlim([breaks[0] - breaks[0] * 0.4, breaks[-1] + breaks[-1] * 0.1])
    if legend:
        ax.legend(loc='upper left', fontsize=labelsize * 0.7)
        
    return (mismatch, mismatch_true)
    
def Mism(prediction, X, response_name, response='predicted response'):
    mism = pd.concat([X['ABX_' + response_name].reset_index(drop=True),
                      prediction[response].reset_index(drop=True)], axis=1).apply(is_mismatch_2, axis=1)
    return mism


def Mism_b72(prediction, X, response_name, response='predicted response'):
    mism = pd.concat([X['ABX_' + response_name].reset_index(drop=True),
                      prediction[response].reset_index(drop=True)], axis=1).loc[X['ABX_b72_' + response_name].reset_index(drop=True).astype(bool),:].apply(is_mismatch_2, axis=1)
    return mism

def Mism_a72(prediction, X, response_name, response='predicted response'):
    df1 = X['ABX_' + response_name].reset_index(drop=True)
    df2 = prediction[response].reset_index(drop=True)
    pos = ~X['ABX_b72_' + response_name].reset_index(drop=True).astype(bool)

    mism = pd.concat([df1, df2], axis=1).loc[pos,:]
    mism = mism.apply(is_mismatch_2, axis=1)
    return mism



def wald_test(x, n):
    p = (x+2) / (n+4)
    p1 = 1.96 * np.sqrt(p*(1-p) / (n+4))
    return (p-p1,p+p1)

def binom_test(x,n):
    return binomtest(x,n).proportion_ci(confidence_level=0.95)

from math import sqrt
import numpy as np
# from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from scipy.stats import binom
# from ._common import ConfidenceInterval


class BinomTestResult:
    """
    Result of `scipy.stats.binomtest`.

    Attributes
    ----------
    k : int
        The number of successes (copied from `binomtest` input).
    n : int
        The number of trials (copied from `binomtest` input).
    alternative : str
        Indicates the alternative hypothesis specified in the input
        to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
        or ``'less'``.
    pvalue : float
        The p-value of the hypothesis test.
    proportion_estimate : float
        The estimate of the proportion of successes.

    Methods
    -------
    proportion_ci :
        Compute the confidence interval for the estimate of the proportion.

    """
    def __init__(self, k, n, alternative, pvalue, proportion_estimate):
        self.k = k
        self.n = n
        self.alternative = alternative
        self.proportion_estimate = proportion_estimate
        self.pvalue = pvalue

    def __repr__(self):
        s = ("BinomTestResult("
             f"k={self.k}, "
             f"n={self.n}, "
             f"alternative={self.alternative!r}, "
             f"proportion_estimate={self.proportion_estimate}, "
             f"pvalue={self.pvalue})")
        return s

    def proportion_ci(self, confidence_level=0.95, method='exact'):
        """
        Compute the confidence interval for the estimated proportion.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is 0.95.
        method : {'exact', 'wilson', 'wilsoncc'}, optional
            Selects the method used to compute the confidence interval
            for the estimate of the proportion:

            'exact' :
                Use the Clopper-Pearson exact method [1]_.
            'wilson' :
                Wilson's method, without continuity correction ([2]_, [3]_).
            'wilsoncc' :
                Wilson's method, with continuity correction ([2]_, [3]_).

            Default is ``'exact'``.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence interval.

        References
        ----------
        .. [1] C. J. Clopper and E. S. Pearson, The use of confidence or
               fiducial limits illustrated in the case of the binomial,
               Biometrika, Vol. 26, No. 4, pp 404-413 (Dec. 1934).
        .. [2] E. B. Wilson, Probable inference, the law of succession, and
               statistical inference, J. Amer. Stat. Assoc., 22, pp 209-212
               (1927).
        .. [3] Robert G. Newcombe, Two-sided confidence intervals for the
               single proportion: comparison of seven methods, Statistics
               in Medicine, 17, pp 857-872 (1998).

        Examples
        --------
        >>> from scipy.stats import binomtest
        >>> result = binomtest(k=7, n=50, p=0.1)
        >>> result.proportion_estimate
        0.14
        >>> result.proportion_ci()
        ConfidenceInterval(low=0.05819170033997342, high=0.26739600249700846)
        """
        if method not in ('exact', 'wilson', 'wilsoncc'):
            raise ValueError("method must be one of 'exact', 'wilson' or "
                             "'wilsoncc'.")
        if not (0 <= confidence_level <= 1):
            raise ValueError('confidence_level must be in the interval '
                             '[0, 1].')
        if method == 'exact':
            low, high = _binom_exact_conf_int(self.k, self.n,
                                              confidence_level,
                                              self.alternative)
        else:
            # method is 'wilson' or 'wilsoncc'
            low, high = _binom_wilson_conf_int(self.k, self.n,
                                               confidence_level,
                                               self.alternative,
                                               correction=method == 'wilsoncc')
        return (low, high)


def _findp(func):
    try:
        p = brentq(func, 0, 1)
    except RuntimeError:
        raise RuntimeError('numerical solver failed to converge when '
                           'computing the confidence limits') from None
    except ValueError as exc:
        raise ValueError('brentq raised a ValueError; report this to the '
                         'SciPy developers') from exc
    return p


def _binom_exact_conf_int(k, n, confidence_level, alternative):
    """
    Compute the estimate and confidence interval for the binomial test.

    Returns proportion, prop_low, prop_high
    """
    if alternative == 'two-sided':
        alpha = (1 - confidence_level) / 2
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'less':
        alpha = 1 - confidence_level
        plow = 0.0
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'greater':
        alpha = 1 - confidence_level
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        phigh = 1.0
    return plow, phigh


def _binom_wilson_conf_int(k, n, confidence_level, alternative, correction):
    # This function assumes that the arguments have already been validated.
    # In particular, `alternative` must be one of 'two-sided', 'less' or
    # 'greater'.
    p = k / n
    if alternative == 'two-sided':
        z = ndtri(0.5 + 0.5*confidence_level)
    else:
        z = ndtri(confidence_level)

    # For reference, the formulas implemented here are from
    # Newcombe (1998) (ref. [3] in the proportion_ci docstring).
    denom = 2*(n + z**2)
    center = (2*n*p + z**2)/denom
    q = 1 - p
    if correction:
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            dlo = (1 + z*sqrt(z**2 - 2 - 1/n + 4*p*(n*q + 1))) / denom
            lo = center - dlo
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            dhi = (1 + z*sqrt(z**2 + 2 - 1/n + 4*p*(n*q - 1))) / denom
            hi = center + dhi
    else:
        delta = z/denom * sqrt(4*n*p*q + z**2)
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            lo = center - delta
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            hi = center + delta

    return lo, hi


def binomtest(k, n, p=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    The binomial test [1]_ is a test of the null hypothesis that the
    probability of success in a Bernoulli experiment is `p`.

    Details of the test can be found in many texts on statistics, such
    as section 24.5 of [2]_.

    Parameters
    ----------
    k : int
        The number of successes.
    n : int
        The number of trials.
    p : float, optional
        The hypothesized probability of success, i.e. the expected
        proportion of successes.  The value must be in the interval
        ``0 <= p <= 1``. The default value is ``p = 0.5``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Indicates the alternative hypothesis. The default value is
        'two-sided'.

    Returns
    -------
    result : `~scipy.stats._result_classes.BinomTestResult` instance
        The return value is an object with the following attributes:

        k : int
            The number of successes (copied from `binomtest` input).
        n : int
            The number of trials (copied from `binomtest` input).
        alternative : str
            Indicates the alternative hypothesis specified in the input
            to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
            or ``'less'``.
        pvalue : float
            The p-value of the hypothesis test.
        proportion_estimate : float
            The estimate of the proportion of successes.

        The object has the following methods:

        proportion_ci(confidence_level=0.95, method='exact') :
            Compute the confidence interval for ``proportion_estimate``.

    Notes
    -----
    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Binomial test, https://en.wikipedia.org/wiki/Binomial_test
    .. [2] Jerrold H. Zar, Biostatistical Analysis (fifth edition),
           Prentice Hall, Upper Saddle River, New Jersey USA (2010)

    Examples
    --------
    >>> from scipy.stats import binomtest

    A car manufacturer claims that no more than 10% of their cars are unsafe.
    15 cars are inspected for safety, 3 were found to be unsafe. Test the
    manufacturer's claim:

    >>> result = binomtest(3, n=15, p=0.1, alternative='greater')
    >>> result.pvalue
    0.18406106910639114

    The null hypothesis cannot be rejected at the 5% level of significance
    because the returned p-value is greater than the critical value of 5%.

    The estimated proportion is simply ``3/15``:

    >>> result.proportion_estimate
    0.2

    We can use the `proportion_ci()` method of the result to compute the
    confidence interval of the estimate:

    >>> result.proportion_ci(confidence_level=0.95)
    ConfidenceInterval(low=0.05684686759024681, high=1.0)

    """
    if k > n:
        raise ValueError('k must not be greater than n.')

    if not (0 <= p <= 1):
        raise ValueError("p must be in range [0,1]")

    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized; \n"
                         "must be 'two-sided', 'less' or 'greater'")
    if alternative == 'less':
        pval = binom.cdf(k, n, p)
    elif alternative == 'greater':
        pval = binom.sf(k-1, n, p)
    else:
        # alternative is 'two-sided'
        d = binom.pmf(k, n, p)
        rerr = 1 + 1e-7
        if k == p * n:
            # special case as shortcut, would also be handled by `else` below
            pval = 1.
        elif k < p * n:
            ix = _binary_search_for_binom_tst(lambda x1: -binom.pmf(x1, n, p),
                                              -d*rerr, np.ceil(p * n), n)
            # y is the number of terms between mode and n that are <= d*rerr.
            # ix gave us the first term where a(ix) <= d*rerr < a(ix-1)
            # if the first equality doesn't hold, y=n-ix. Otherwise, we
            # need to include ix as well as the equality holds. Note that
            # the equality will hold in very very rare situations due to rerr.
            y = n - ix + int(d*rerr == binom.pmf(ix, n, p))
            pval = binom.cdf(k, n, p) + binom.sf(n - y, n, p)
        else:
            ix = _binary_search_for_binom_tst(lambda x1: binom.pmf(x1, n, p),
                                              d*rerr, 0, np.floor(p * n))
            # y is the number of terms between 0 and mode that are <= d*rerr.
            # we need to add a 1 to account for the 0 index.
            # For comparing this with old behavior, see
            # tst_binary_srch_for_binom_tst method in test_morestats.
            y = ix + 1
            pval = binom.cdf(y-1, n, p) + binom.sf(k-1, n, p)

        pval = min(1.0, pval)

    result = BinomTestResult(k=k, n=n, alternative=alternative,
                             proportion_estimate=k/n, pvalue=pval)
    return result


def _binary_search_for_binom_tst(a, d, lo, hi):
    """
    Conducts an implicit binary search on a function specified by `a`.

    Meant to be used on the binomial PMF for the case of two-sided tests
    to obtain the value on the other side of the mode where the tail
    probability should be computed. The values on either side of
    the mode are always in order, meaning binary search is applicable.

    Parameters
    ----------
    a : callable
      The function over which to perform binary search. Its values
      for inputs lo and hi should be in ascending order.
    d : float
      The value to search.
    lo : int
      The lower end of range to search.
    hi : int
      The higher end of the range to search.

    Returns
    ----------
    int
      The index, i between lo and hi
      such that a(i)<=d<a(i+1)
    """
    while lo < hi:
        mid = lo + (hi-lo)//2
        midval = a(mid)
        if midval < d:
            lo = mid+1
        elif midval > d:
            hi = mid-1
        else:
            return mid
    if a(lo) <= d:
        return lo
    else:
        return lo-1


def OR_function(x,y):
    '''
    x and y must be binary vectors
    '''
    Table = np.zeros((2,2))
    Table[0,0] = np.sum( (x==0) & (y==0))
    Table[1,1] = np.sum( (x==1) & (y==1))
    Table[0,1] = np.sum( (x==0) & (y==1))
    Table[1,0] = np.sum( (x==1) & (y==0))
    Table = Table+0.5
    logOR = np.log(Table[1,1]) + np.log(Table[0,0]) - np.log(Table[0,1]) - np.log(Table[1,0])
    SE = np.sqrt(1/Table[1,1] + 1/Table[0,1] + 1/Table[1,0] + 1/Table[0,0])
    low = logOR - 1.96 * SE
    upp = logOR + 1.96 * SE
    return logOR, low, upp
    
    
def ORs(X, y):
    OR = []
    lowOR = []
    uppOR = []
    features = X.columns.values
    r = pd.DataFrame({'key': features})
    for f in features:
        if (X[f].dtype == np.int64) & np.all(X[f] < 2):
            val, low, upp = OR_function(X[f],y)
            OR.append(val)
            lowOR.append(low)
            uppOR.append(upp)            
        else:
            OR.append(np.nan)
            lowOR.append(np.nan)
            uppOR.append(np.nan)            
    r['logOR'] = OR
    r['lowOR'] = lowOR
    r['uppOR'] = uppOR    
    return r
    
    
    
    
    
    
    
    