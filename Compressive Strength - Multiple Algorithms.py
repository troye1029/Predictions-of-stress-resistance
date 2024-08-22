import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt

# 固定随机种子以确保结果可复现
RANDOM_STATE = 42

# 读取数据
df = pd.read_csv(r'data.csv', encoding='gb2312', engine='python')

# 选择粉砂岩数据
X = df[['取样深度','岩性','视电阻率x', '自然伽玛x', '双收时差x']]
Y = df['抗压强度']

# 划分训练集和测试集，同时设置随机状态
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_STATE)

# 定义模型列表，同时为接受随机状态的模型设置随机状态
models = {
    "SVR": SVR(),
    "BPnet": MLPRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "LR": LinearRegression(),
    "KNN": KNeighborsRegressor()
}

# 训练并评估每个模型
results = {}
for name, model in models.items():
    model.fit(xtrain, ytrain)
    train_predictions = model.predict(xtrain)
    test_predictions = model.predict(xtest)
    results[name] = {
        "Training R²": r2_score(ytrain, train_predictions),
        "Testing R²": r2_score(ytest, test_predictions),
        "Training MSE": mean_squared_error(ytrain, train_predictions),
        "Testing MSE": mean_squared_error(ytest, test_predictions),
        "Training MAE": mean_absolute_error(ytrain, train_predictions),
        "Testing MAE": mean_absolute_error(ytest, test_predictions),
        "Training Explained Variance Score": explained_variance_score(ytrain, train_predictions),
        "Testing Explained Variance Score": explained_variance_score(ytest, test_predictions)
    }

# 打印结果
for name, result in results.items():
    print(f"\n{name}:")
    for metric, value in result.items():
        print(f"{metric}: {value}")

# 绘制并排柱状图
model_names = list(results.keys())
train_r2_values = [result["Training R²"] for result in results.values()]
test_r2_values = [result["Testing R²"] for result in results.values()]

# 设置字体和图形参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建一个图形对象和一个子图
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱状图的位置和宽度
bar_width = 0.35
index = range(len(model_names))

# 绘制训练集R²分数的柱状图
bars_train = ax.bar(index, train_r2_values, bar_width, color='skyblue', label='Training R$^2$')

# 绘制测试集R²分数的柱状图
bars_test = ax.bar([i + bar_width for i in index], test_r2_values, bar_width, color='orange', label='Testing R$^2$')

# 设置x轴刻度和标签
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(model_names)
ax.tick_params(axis='both', which='major', labelsize=12)

# 自动调整y轴范围，确保所有数据可见
ax.relim()  # 重新计算数据范围
ax.autoscale_view()  # 自动调整视图

# 添加标题和图例
ax.set_title('Performance of different model R$^2$ scores')
ax.set_xlabel('Model')
ax.set_ylabel('The score of R$^2$')
ax.legend()

# 调整布局以适应较宽的图像
plt.tight_layout()
plt.show()
