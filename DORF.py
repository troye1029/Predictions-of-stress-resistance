import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from skopt import BayesSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import RFE
import joblib

# 随机数种子
RANDOM_STATE = 42

# 读取数据
df = pd.read_csv(r'data.csv', encoding='gb2312', engine='python')

X = df[['取样深度','岩性','视电阻率x', '自然伽玛x', '双收时差x']]
y = df['抗压强度']

# 数据分割
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

# 使用RFE进行特征选择
from sklearn.linear_model import LinearRegression

selector = RFE(LinearRegression(), n_features_to_select=5)
selector.fit(xtrain, ytrain)
selected_features = X.columns[selector.support_]

xtrain_selected = selector.transform(xtrain)
xtest_selected = selector.transform(xtest)

print("Selected features:", selected_features)

# 模型评估函数
def evaluate_model(train_pred, test_pred, ytrain, ytest, model_name):
    results = {
        '模型': model_name,
        '训练 R²': r2_score(ytrain, train_pred),
        '测试 R²': r2_score(ytest, test_pred),
        '训练 MSE': mean_squared_error(ytrain, train_pred),
        '测试 MSE': mean_squared_error(ytest, test_pred),
        '训练 MAE': mean_absolute_error(ytrain, train_pred),
        '测试 MAE': mean_absolute_error(ytest, test_pred),
        '训练 EVS': explained_variance_score(ytrain, train_pred),
        '测试 EVS': explained_variance_score(ytest, test_pred)
    }
    return results

# 基线随机森林模型
baseline_rf = RandomForestRegressor(random_state=RANDOM_STATE)
baseline_rf.fit(xtrain, ytrain)
baseline_rf_train_pred = baseline_rf.predict(xtrain)
baseline_rf_test_pred = baseline_rf.predict(xtest)
baseline_results = evaluate_model(baseline_rf_train_pred, baseline_rf_test_pred, ytrain, ytest, "基线随机森林")

# 网格搜索优化随机森林
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(xtrain_selected, ytrain)
grid_rf = grid_search.best_estimator_
grid_rf_train_pred = grid_rf.predict(xtrain_selected)
grid_rf_test_pred = grid_rf.predict(xtest_selected)
grid_rf_results = evaluate_model(grid_rf_train_pred, grid_rf_test_pred, ytrain, ytest, "网格搜索优化随机森林")
grid_best_params = grid_search.best_params_

# 贝叶斯优化随机森林
bayes_search = BayesSearchCV(RandomForestRegressor(random_state=RANDOM_STATE), param_grid, cv=5, scoring='r2', n_jobs=-1, n_iter=50, random_state=RANDOM_STATE)
bayes_search.fit(xtrain_selected, ytrain)
bayes_rf = bayes_search.best_estimator_
bayes_rf_train_pred = bayes_rf.predict(xtrain_selected)
bayes_rf_test_pred = bayes_rf.predict(xtest_selected)
bayes_rf_results = evaluate_model(bayes_rf_train_pred, bayes_rf_test_pred, ytrain, ytest, "贝叶斯优化随机森林")
bayes_best_params = bayes_search.best_params_

# 优化后的Stacking模型
stacking_regressor = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        ('lasso', Lasso(random_state=RANDOM_STATE))
    ],
    final_estimator=Ridge(alpha=1.0)
)
stacking_regressor.fit(xtrain_selected, ytrain)
stacking_train_pred = stacking_regressor.predict(xtrain_selected)
stacking_test_pred = stacking_regressor.predict(xtest_selected)
stacking_results = evaluate_model(stacking_train_pred, stacking_test_pred, ytrain, ytest, "优化后的Stacking模型")

# 打印结果
print("\n基线随机森林模型评估结果：")
for metric, value in baseline_results.items():
    print(f"{metric}: {value}")

print("\n网格搜索优化随机森林最佳超参数：")
print(grid_best_params)
print("\n网格搜索优化随机森林评估结果：")
for metric, value in grid_rf_results.items():
    print(f"{metric}: {value}")

print("\n贝叶斯优化随机森林最佳超参数：")
print(bayes_best_params)
print("\n贝叶斯优化随机森林评估结果：")
for metric, value in bayes_rf_results.items():
    print(f"{metric}: {value}")

print("\n优化后的Stacking模型评估结果：")
for metric, value in stacking_results.items():
    print(f"{metric}: {value}")

# 保存最佳模型
joblib.dump(grid_rf, 'best_grid_search_random_forest_model.pkl')
joblib.dump(bayes_rf, 'best_bayes_search_random_forest_model.pkl')
joblib.dump(stacking_regressor, 'best_stacking_model.pkl')
