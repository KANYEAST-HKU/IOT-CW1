import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns  # 导入 seaborn 库，用于数据可视化

# 数据路径
train_path = "./TrainingData.txt"
test_path = "./TestingData.txt"
save_path = "TestingResults.txt"


# 加载训练数据
def load_train_data():
    """
    从训练数据文件中加载数据，分离特征和标签，并返回训练集和测试集。
    :return: X_train, X_test, y_train, y_test
    """
    with open(train_path,'r') as f:
        data = f.readlines()  # 读取文件中的每一行
    data = [line.strip().split(',') for line in data]  # 去除每行的换行符并分割逗号
    data = [[float(element) for element in row] for row in data]  # 转换为浮点数

    np_data = np.array(data)  # 转换为 NumPy 数组

    X = np_data[:,:24]  # 选择前24列作为特征
    y = np_data[:,24]   # 最后一列作为标签

    # 数据分割，将数据分为训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# 加载测试数据
def load_test_data():
    """
    从测试数据文件中加载数据，仅提取特征部分。
    :return: X
    """
    with open(test_path,'r') as f:
        data = f.readlines()  # 读取文件中的每一行
    data = [line.strip().split(',') for line in data]  # 去除每行的换行符并分割逗号
    data = [[float(element) for element in row] for row in data]  # 转换为浮点数

    np_data = np.array(data)  # 转换为 NumPy 数组

    X = np_data[:,:24]  # 选择前24列作为特征

    return X

# 使用 XGBoost 分类器进行模型训练与预测
def XGBoost_solver(X_train, X_test, y_train, y_test):
    """
    使用 XGBoost 分类器进行训练并评估模型准确性。
    :param X_train: 训练特征
    :param X_test: 测试特征
    :param y_train: 训练标签
    :param y_test: 测试标签
    :return: 训练好的模型
    """
    # 初始化 XGBoost 分类器
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # 在训练数据上训练模型
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算并打印模型的准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 输出详细的分类报告
    print(classification_report(y_test, y_pred))

    return model

# 使用 SVM 分类器进行模型训练与预测
def SVM_solver(X_train, X_test, y_train, y_test):
    """
    使用支持向量机（SVM）进行训练并评估模型准确性。
    :param X_train: 训练特征
    :param X_test: 测试特征
    :param y_train: 训练标签
    :param y_test: 测试标签
    :return: 训练好的模型
    """
    # 初始化 SVM 分类器（使用线性核）
    svm_model = SVC(kernel='linear', random_state=30)

    # 在训练数据上训练模型
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svm_model.predict(X_test)

    # 计算并打印模型的准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # 输出详细的分类报告
    print(classification_report(y_test, y_pred))

    return svm_model

# 计算测试集的结果并保存到文件中
def write2txt(model):
    """
    使用训练好的模型对测试集进行预测并将结果保存到文件中。
    :param model: 训练好的分类模型
    """
    # 加载测试数据
    X = load_test_data()

    # 使用模型进行预测，结果是一个二维数组，需要reshape成列向量
    res = model.predict(X).reshape((X.shape[0],1))

    # 将预测结果与测试特征合并，形成最终输出格式
    to_write = np.hstack((X,res))

    # 保存结果到指定的文本文件
    np.savetxt(save_path, to_write, delimiter=',', fmt=['%.14f'] * 24 + ['%d'])

    print(f"Results saved in {save_path}")

# 主程序
if __name__ == "__main__":
    # 加载训练数据
    X_train, X_test, y_train, y_test = load_train_data()

    # 使用 XGBoost 模型进行训练和评估
    xgb_solver = XGBoost_solver(X_train, X_test, y_train, y_test)

    # 使用 SVM 模型进行训练和评估
    svm_solver = SVM_solver(X_train, X_test, y_train, y_test)

    # 将 SVM 模型的预测结果写入文件
    write2txt(svm_solver)


# 线性规划作图部分
# 读取 Excel 文件中的数据
excel_file = './IMSE7143CW1Input.xlsx'  # 替换为你的 Excel 文件路径
df = pd.read_excel(excel_file, sheet_name='User & Task ID')  # 读取特定 sheet 的数据

# 将 DataFrame 转换为 NumPy 数组，便于后续处理
excel_data = df.to_numpy()

# 读取保存的预测结果
with open("./TestingResults.txt",'r') as f:
    data = f.readlines()  # 读取文件中的每一行
    data = [line.strip().split(',') for line in data]  # 去除每行的换行符并分割逗号
    data = [[float(element) for element in row] for row in data]  # 转换为浮点数

    np_data = np.array(data)  # 转换为 NumPy 数组

# 辅助函数：线性规划求解
def helper(coefficient):
    """
    线性规划求解过程，优化用户任务分配问题。
    :param coefficient: 当前行的系数（任务的特征）
    :return: 线性规划的解
    """
    A = np.zeros((50, 24*50))  # 初始化约束矩阵 A
    b = np.zeros((50, 1))       # 初始化约束向量 b
    for i,j in enumerate(excel_data):
        # 设置任务分配的约束
        A[i][j[1] + i * 24 :j[2]+1 + i * 24] = 1
        b[i] = j[4]  # 设置任务的最大需求

    first_24 = coefficient  # 当前任务的系数
    c = np.tile(first_24, (50, 1)).reshape(1200,1)  # 目标函数的系数

    # 使用 scipy 的 linprog 求解线性规划
    result = linprog(c.flatten(), A_ub=-A, b_ub=-b.flatten(), bounds=[(0, 1)] * 1200, method='highs')

    return result.x

# 结果累加
res = np.zeros((1200))  # 初始化结果数组
for line in np_data:
    if line[-1] == 0:
        continue  # 跳过标签为 0 的行
    else:
        # 使用线性规划辅助函数进行计算
        res += helper(line[:24])

# 设置 seaborn 的样式
sns.set(style="darkgrid")

# 绘制堆叠条形图
def bar_plot(res):
    """
    绘制每个用户在24小时内的能耗变化图。
    :param res: 预测的能耗结果
    """
    res = res.reshape((50,24))  # 将结果重塑为 50 用户，24 小时
    res = res.reshape(5, 10, 24).sum(axis=1)  # 将 5 用户分成 5 行，显示不同用户的能耗

    power_usage = res  # 将重塑后的结果作为能耗数据

    num_users, num_hours = power_usage.shape  # 用户数和小时数
    hours = np.arange(num_hours)  # 小时的索引
    user_labels = [f'User {i+1}' for i in range(num_users)]  # 用户标签

    # 定义每个用户的颜色
    colors = ['#440154', '#31688E', '#21918C','#35B779','#FDE725']

    # 创建一个图形和坐标轴对象
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制堆叠条形图
    bottom = np.zeros(num_hours)
    for i in range(num_users):
        ax.bar(hours, power_usage[i], bottom=bottom, color=colors[i], label=user_labels[i])
        bottom += power_usage[i]  # 更新每一层的底部位置

    # 设置图形标签和标题
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Power Usage")
    ax.set_title("Energy Usage by Hour for Each User")
    ax.legend(title="Users")
    ax.set_xticks(hours)

    plt.tight_layout()
    plt.show()  # 显示图形

# 绘制图表
bar_plot(res)

