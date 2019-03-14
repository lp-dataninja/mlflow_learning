#Load packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

#eval_metrics function
def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return acc, precision, recall, f1

#starting point
if __name__ == "__main__":
    #Download dataset from kaggle : https://www.kaggle.com/c/santander-customer-transaction-prediction
    #reading dataset
    train_df = pd.read_csv("train.csv.zip")
    train_df_sample = train_df.sample(100000)
    train_df_sample['target'].value_counts()

    #feature engineering
    idx = features = train_df_sample.columns.values[2:202]
    train_df_sample['sum'] = train_df_sample[idx].sum(axis=1)
    train_df_sample['min'] = train_df_sample[idx].min(axis=1)
    train_df_sample['max'] = train_df_sample[idx].max(axis=1)
    train_df_sample['mean'] = train_df_sample[idx].mean(axis=1)
    train_df_sample['std'] = train_df_sample[idx].std(axis=1)
    train_df_sample['skew'] = train_df_sample[idx].skew(axis=1)
    train_df_sample['kurt'] = train_df_sample[idx].kurtosis(axis=1)
    train_df_sample['med'] = train_df_sample[idx].median(axis=1)

    #Train and Test split
    features = [c for c in train_df_sample.columns if c not in ['ID_code', 'target']]
    X = train_df_sample[features]
    y = train_df_sample['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #Hyperparameters for the model
    param = {
        'num_leaves': 6,
        'max_bin': 63,
        'min_data_in_leaf': 45,
        'learning_rate': 0.01,
        'min_sum_hessian_in_leaf': 0.000446,
        'bagging_fraction': 0.55,
        'bagging_freq': 5,
        'max_depth': 14,
        'save_binary': True,
        'seed': 31452,
        'feature_fraction_seed': 31415,
        'feature_fraction': 0.51,
        'bagging_seed': 31415,
        'drop_seed': 31415,
        'data_random_seed': 31415,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    #Mlflow run
    with mlflow.start_run():
        #train model and evaluate model
        clf = lgb.train(param, train_data)
        y_pred = clf.predict(X_test)

        for i in range(len(y_pred)):
            if y_pred[i] >= .5:  # setting threshold to .5
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        (acc, precision, recall, f1) = eval_metrics(y_test, y_pred)
        #print metrics
        print(" Acc: " , acc)
        print(" precision: " , precision)
        print(" recall: " , recall)
        print(" f1: %s" , f1)

        #log hyperparamter and metrics
        mlflow.log_param("num_leaves", 6)
        mlflow.log_param("metric", "auc")
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("Acc", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric(" f1", f1)
        
        #features_importance
        feature_importance_df = pd.DataFrame()
        feature_importance_df["Feature"] = features
        feature_importance_df["importance"] = clf.feature_importance()
        cols = (feature_importance_df[["Feature", "importance"]]
                .groupby("Feature").mean()
                .sort_values(by="importance", ascending=False)[:150].index)
        best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

        plt.figure(figsize=(14,28))
        sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
        plt.title('Features importance')
        plt.tight_layout()
        plt.savefig('features_importance.png')
        #save features_importance
        mlflow.log_artifact("features_importance.png")
       
        #save model
        mlflow.sklearn.log_model(clf, "model")
        
        #save input_data
        mlflow.log_artifact("train.csv.zip")
