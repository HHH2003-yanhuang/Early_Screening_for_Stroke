#%%
#导入库并定义一些函数

#导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import joblib
#%%
#抑制警告信息，并设置 matplotlib 的字体和格式，以便在绘制图表时支持中文字体
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
#%%
#定义模型性能评估函数
def performance(labelArr, predictArr):
    TP = 0  # True Positive
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative
    
    # 计算 TP, TN, FP, FN
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1
        elif labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1
        elif labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1
        elif labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1
            
    #计算归一化代价NC
    Normalized_Cost = (FP * 1 + FN * 400) / ((TN + FP) * 1 + (TP + FN) * 400)
        
    # 计算敏感性
    Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    
    # 计算特异性
    Specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    
    # 计算假阴率
    false_negative_rate = FN / (TP + FN) if (TP + FN) != 0 else 0

    # 计算假阳率
    false_positive_rate = FP / (TN + FP) if (TN + FP) != 0 else 0

    # 计算准确率
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    # 计算 Youden 指数
    Youden = Specificity + Sensitivity - 1

    # 计算 MCC
    MCC = ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))**0.5 if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0 else 0
    
    
    return {
        'Normalized_Cost':Normalized_Cost,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'False Negative Rate': false_negative_rate,
        'False Positive Rate': false_positive_rate,
        'Accuracy': Accuracy,
        'Youden': Youden,
        'MCC': MCC
    }


#%%
#定义特征重要性绘制函数
def plot_feature_importances(feature_importances, title, feature_names):
    # 标准化特征重要性
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    
    # 获取排序后的索引，确保是整数
    index_sorted = np.argsort(feature_importances)
    
    # 检查 feature_names 是否为正确的类型，并转换为 NumPy 数组
    if not isinstance(feature_names, np.ndarray):
        feature_names = np.array(feature_names)
    
    # 确保特征名称与特征重要性的数量匹配
    if len(feature_names) != len(feature_importances):
        raise ValueError("特征名称的数量和特征重要性的数量不匹配！")
    
    pos = np.arange(index_sorted.shape[0]) + 0.5  # 用于y轴位置的标记
    
    # 绘制特征重要性图
    plt.figure(figsize=(15, 10))
    plt.barh(pos, feature_importances[index_sorted], align='center')
    plt.yticks(pos, feature_names[index_sorted])  # 使用按排序后的特征名称
    plt.xlabel('Relative Importance')
    plt.title(title)
    
    
    # 保存为 SVG 格式
    plt.savefig(f"feature_importance_{title}.svg", format="svg")

    plt.show()




#%%
#数据预处理

#导入数据
df = pd.read_csv("dataset.csv")
dfhz= pd.read_excel("患者数据.xlsx")
dfjk = pd.read_excel("健康人数据.xlsx")

df.head()

#%%id列删除

# 删除id列
df = df.drop(columns=['id'])
dfhz = dfhz.drop(columns=['id'])
dfjk = dfjk.drop(columns=['id'])

#%%
#让我们看看数据是平衡的还是不平衡的

print(df['stroke'].value_counts())
print('***'*30)
print(df.shape)
print('***'*30)
print(df.isna().sum())

#stroke
#0    42617
#1      783
# so，数据不平衡

#绘制数据处理前stroke的分布

import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据准备
categories = ['健康样本', '卒中样本']
counts = [42617, 783]  # 0: 健康, 1: 卒中
total_samples = sum(counts)
percentages = [f'({count/total_samples*100:.1f}%)' for count in counts]

# 创建图形
plt.figure(figsize=(10, 6), dpi=100)

# 绘制柱状图
bars = plt.bar(categories, counts, color=['#4C72B0', '#C44E52'], width=0.7)

# 添加数据标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 500,
             f'{counts[i]:,}\n{percentages[i]}',
             ha='center', va='bottom', fontsize=12)

# 设置标题和标签
plt.title('脑卒中数据集分布（总样本：43,400）', fontsize=14, pad=20)
plt.ylabel('样本数量', fontsize=12)
plt.ylim(0, 45000)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加不平衡比例标注
plt.annotate('类别不平衡比例：54.4:1', 
             xy=(0.5, 0.85), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1),
             fontsize=12, ha='center')

# 显示图形
plt.tight_layout()
plt.savefig('stroke_data_distribution.png', bbox_inches='tight')
plt.show()




#%%缺失值处理

#缺失值比例
#bmi
print("percentage of missing values in bmi :: ",round(df['bmi'].isna().sum()/len(df['bmi'])*100,2))
print("percentage of missing values in bmi of jiank :: ",round(dfjk['bmi'].isna().sum()/len(dfjk['bmi'])*100,2))
print("percentage of missing values in bmi of huanz :: ",round(dfhz['bmi'].isna().sum()/len(dfhz['bmi'])*100,2))
#smoking
print("percentage of missing values in Smoking Status :: ",round(df['smoking_status'].isna().sum()/len(df['smoking_status'])*100,2))
print("percentage of missing values in Smoking Status of jk :: ",round(dfjk['smoking_status'].isna().sum()/len(dfjk['smoking_status'])*100,2))
print("percentage of missing values in Smoking Status of hz :: ",round(dfhz['smoking_status'].isna().sum()/len(dfhz['smoking_status'])*100,2))

#缺失比例 bmi ::  3.37%  Smoking Status ::  30.63%

# so，bmi and smoking status have 缺失值
#bmi          缺         1462
#smoking_status 缺      13292

#%%smoking_status

#吸烟史缺失比例较大，是否删?

import matplotlib.pyplot as plt

sns.countplot(data=df, x='stroke', hue='smoking_status')
plt.title('Distribution of Target Variable by Smoking Status')

# 保存为高清矢量图（PDF）
plt.savefig('stroke_smoking_status.pdf', format='pdf', dpi=300)  # 保存为 PDF

plt.show()  # 显示图表

#发现，吸不吸烟中风比例是不同的，所以认为有关，所以不删

#%%为smoking_status补缺失值

#用众数补

#患者和健康人的数据有很大差异的，故想将其分为两组数据分别分析补缺失值（后文有差异性分析）

# 对患者数据补缺失值
dfhz['smoking_status'].fillna(dfhz['smoking_status'].mode()[0], inplace=True)

# 对健康人数据补缺失值
dfjk['smoking_status'].fillna(dfjk['smoking_status'].mode()[0], inplace=True)

#%%bmi

#由于 bmi 只有 3.37% % 的缺失值，我们将用新的适当值填充它
#可以是平均值、中位数

#补平均值 or 中位数?
#看bmi的分布，偏度
sns.distplot(dfhz['bmi'])
plt.title('患者的BMI的分布')

# 保存为高清矢量图（PDF）
plt.savefig('患者的BMI的分布.pdf', format='pdf', dpi=300)  # 保存为 PDF

plt.show()  # 显示图表

sns.distplot(dfjk['bmi'])
plt.title('健康人的BMI的分布')

# 保存为高清矢量图（PDF）
plt.savefig('健康人的BMI的分布.pdf', format='pdf', dpi=300)  # 保存为 PDF

plt.show()
print(dfhz['bmi'].skew())
print(dfjk['bmi'].skew())

#dfhz：0.5413016662177812
#dfjk：0.6411279823579266
#偏度值大于 1 或小于 -1 表示分布高度偏斜。 0.5 到 1 或 -0.5 到 -1 之间的值是适度倾斜的。 -0.5 和 0.5 之间的值表示分布相当对称。
#所以 df['bmi'] 是适度倾斜的。因此 w 不能使用平均值来填充 df['bmi'] 中的空值，我们可以使用中位数（仅用于数字数据）或众数（可用于数字和分类）

#连续数值变量，故选择中位数补

#bmi补中位数
dfhz['bmi'].fillna(dfhz['bmi'].median(),inplace=True)
dfjk['bmi'].fillna(dfjk['bmi'].median(),inplace=True)


#%%
#检查是否还有缺失值

print("填补后bmi缺失值数量（患者）:", dfhz['bmi'].isna().sum())
print("填补后bmi缺失值数量（健康人）:", dfjk['bmi'].isna().sum())

print("填补后smoking缺失值数量（患者）:", dfhz['smoking_status'].isna().sum())
print("填补后smoking缺失值数量（健康人）:", dfjk['smoking_status'].isna().sum())

#%%异常值检测和处理

#分别对患者、健康人异常值检测和处理
#血糖，bmi
#绘制箱线图

import matplotlib.pyplot as plt
import seaborn as sns

# 绘制带异常值标记的箱线图
def plot_boxplot_with_outliers(data, column_name, title, lower_bound, upper_bound, save_as_svg=False):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column_name])
    
    # 绘制上下界的竖线
    plt.axvline(x=lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axvline(x=upper_bound, color='blue', linestyle='--', label='Upper Bound')
    
    plt.title(f'Box Plot of {title}', fontsize=16)
    plt.xlabel(column_name, fontsize=14)
    plt.legend()

    # 保存为SVG矢量图
    if save_as_svg:
        plt.savefig(f'{title}_boxplot.svg', format='svg')
    else:
        plt.show()

# 定义异常值检测函数
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 检测出异常值（小于lower_bound 或大于 upper_bound）
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    # 返回异常值及上下界
    return lower_bound, upper_bound, outliers

# 计算异常值比例并打印
def print_outlier_info(data, outliers, feature_name, group_name, lower_bound, upper_bound):
    total = len(data)
    outliers_count = len(outliers)
    outlier_percentage = (outliers_count / total) * 100
    print(f"{group_name} {feature_name} 异常值范围: 小于 {lower_bound:.2f} 或大于 {upper_bound:.2f}")
    print(f"检测到 {group_name} {feature_name} 异常值数量: {outliers_count} ({outlier_percentage:.2f}%)")

# 患者数据箱线图（保存为矢量图）
lower_bound_glucose_hz, upper_bound_glucose_hz, outliers_glucose_hz = detect_outliers_iqr(dfhz['avg_glucose_level'])
plot_boxplot_with_outliers(dfhz, 'avg_glucose_level', 'Average Glucose Level of Patients', lower_bound_glucose_hz, upper_bound_glucose_hz, save_as_svg=True)
print_outlier_info(dfhz['avg_glucose_level'], outliers_glucose_hz, 'avg_glucose_level', '患者', lower_bound_glucose_hz, upper_bound_glucose_hz)

lower_bound_bmi_hz, upper_bound_bmi_hz, outliers_bmi_hz = detect_outliers_iqr(dfhz['bmi'])
plot_boxplot_with_outliers(dfhz, 'bmi', 'BMI of Patients', lower_bound_bmi_hz, upper_bound_bmi_hz, save_as_svg=True)
print_outlier_info(dfhz['bmi'], outliers_bmi_hz, 'BMI', '患者', lower_bound_bmi_hz, upper_bound_bmi_hz)

# 健康人数据箱线图（保存为矢量图）
lower_bound_glucose_jk, upper_bound_glucose_jk, outliers_glucose_jk = detect_outliers_iqr(dfjk['avg_glucose_level'])
plot_boxplot_with_outliers(dfjk, 'avg_glucose_level', 'Average Glucose Level of Healthy Individuals', lower_bound_glucose_jk, upper_bound_glucose_jk, save_as_svg=True)
print_outlier_info(dfjk['avg_glucose_level'], outliers_glucose_jk, 'avg_glucose_level', '健康人', lower_bound_glucose_jk, upper_bound_glucose_jk)

lower_bound_bmi_jk, upper_bound_bmi_jk, outliers_bmi_jk = detect_outliers_iqr(dfjk['bmi'])
plot_boxplot_with_outliers(dfjk, 'bmi', 'BMI of Healthy Individuals', lower_bound_bmi_jk, upper_bound_bmi_jk, save_as_svg=True)
print_outlier_info(dfjk['bmi'], outliers_bmi_jk, 'BMI', '健康人', lower_bound_bmi_jk, upper_bound_bmi_jk)

#%%使用上下界替换异常值
# 
def replace_outliers(data, column_name, lower_bound, upper_bound):
    for index, value in data[column_name].items():
        if value < lower_bound:
            data.at[index, column_name] = lower_bound
        elif value > upper_bound:
            data.at[index, column_name] = upper_bound
    return data

# 替换患者的异常值
dfhz = replace_outliers(dfhz.copy(), 'avg_glucose_level', lower_bound_glucose_hz, upper_bound_glucose_hz)
dfhz = replace_outliers(dfhz, 'bmi', lower_bound_bmi_hz, upper_bound_bmi_hz)

# 替换健康人的异常值
dfjk = replace_outliers(dfjk.copy(), 'avg_glucose_level', lower_bound_glucose_jk, upper_bound_glucose_jk)
dfjk = replace_outliers(dfjk, 'bmi', lower_bound_bmi_jk, upper_bound_bmi_jk)

# 检查替换后的结果
print("患者数据替换后的 avg_glucose_level:\n", dfhz['avg_glucose_level'].describe())
print("患者数据替换后的 BMI:\n", dfhz['bmi'].describe())
print("健康人数据替换后的 avg_glucose_level:\n", dfjk['avg_glucose_level'].describe())
print("健康人数据替换后的 BMI:\n", dfjk['bmi'].describe())



#%%

#将患者数据和健康人数据合成一个combined_df

import pandas as pd

# 合并数据
combined_df = pd.concat([dfhz, dfjk], ignore_index=True)

# 查看合并后的数据
print(combined_df.head())
print("合并后的数据形状:", combined_df.shape)

#%%
#采用独热编码将类别变量转换

import pandas as pd

# 确认数据中哪些列是类别变量（类型为'object'或者其他类别变量）
print("数据中的类别变量有:")
print(combined_df.select_dtypes(include='object').columns)

# 列出所有要进行独热编码的类别变量（包括多元和二元类别变量）
categorical_columns = ['gender', 'ever_married', 'work_type','Residence_type','smoking_status']  

# 使用 pd.get_dummies() 进行独热编码转换
df_encoded = pd.get_dummies(combined_df, columns=categorical_columns, drop_first=True)

# 查看转换后的数据
print("转换后的数据：")
print(df_encoded.head())

# 输出数据集的维度
print("数据集的形状:", df_encoded.shape)

#%%
#单因素分析 →
#相关性分析 →
#组合采样（解决不平衡问题） →
#数据标准化（模型训练前的最后一步
#%%
#单因素分析
#判断每个变量在患者和健康人之间是否有显著差异

#连续变量
#连续变量：对于如年龄、BMI、血糖等连续变量，常用方法是：
#t 检验：比较两个组（如健康人 vs 患者）之间的均值差异。
#Mann-Whitney U 检验：当连续变量不符合正态分布时，可以使用此非参数检验

#%%
#故下一步，检验age、bmi、血糖是否在患者和健康人两个总体分布都正态
from scipy.stats import shapiro

# 检验正态分布的函数
def check_normality(df_encoded, column, group):
    data = df_encoded[df_encoded['stroke'] == group][column].dropna()  # 获取相应组的数据
    stat, p_value = shapiro(data)
    return p_value

# 变量列表
continuous_vars = ['age', 'bmi', 'avg_glucose_level']

# 分别检验患者和健康人
for var in continuous_vars:
    # 对患者进行正态分布检验
    p_value_hz = check_normality(df_encoded, var, group=1)  # group=1 表示患者
    print(f"患者组 {var} 的正态性检验 P 值: {p_value_hz:.4f}")
    
    # 对健康人进行正态分布检验
    p_value_jk = check_normality(df_encoded, var, group=0)  # group=0 表示健康人
    print(f"健康人组 {var} 的正态性检验 P 值: {p_value_jk:.4f}")
    print('---' * 10)

#所有连续变量（age、BMI、avg_glucose_level）在患者和健康人组中都显著偏离了正态分布，因为它们的 Shapiro-Wilk 正态性检验的 p 值都是 0.0000（也就是非常小，远小于 0.05）。这说明这些变量在两个群体中都不符合正态分布。
#故不能用t分布检验，应该非参数检验差异性

#%%
#使用 Mann-Whitney U 检验来比较患者和健康人群之间这些变量的差异
from scipy.stats import mannwhitneyu

# 对age进行Mann-Whitney U检验
stat, p = mannwhitneyu(df_encoded[df_encoded['stroke'] == 1]['age'],
                       df_encoded[df_encoded['stroke'] == 0]['age'])
print(f'年龄的 Mann-Whitney U 检验 P 值: {p:.4f}')

# 对bmi进行Mann-Whitney U检验
stat, p = mannwhitneyu(df_encoded[df_encoded['stroke'] == 1]['bmi'],
                       df_encoded[df_encoded['stroke'] == 0]['bmi'])
print(f'BMI的 Mann-Whitney U 检验 P 值: {p:.4f}')

# 对avg_glucose_level进行Mann-Whitney U检验
stat, p = mannwhitneyu(df_encoded[df_encoded['stroke'] == 1]['avg_glucose_level'],
                       df_encoded[df_encoded['stroke'] == 0]['avg_glucose_level'])
print(f'血糖的 Mann-Whitney U 检验 P 值: {p:.4f}')

#设置x显著性水平0.05，故显著。

#%%

#分类变量单因素分析

#卡方检验
import pandas as pd
from scipy.stats import chi2_contingency

# 目标变量为 'stroke'，且取值为 0（未中风）和 1（患中风）

# 对于每个独热编码的分类变量，执行卡方检验
for col in ['hypertension', 'heart_disease', 'gender_Male', 'gender_Other', 'ever_married_Yes', 
            'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban', 'smoking_status_never smoked', 'smoking_status_smokes']:
    contingency_table = pd.crosstab(df_encoded[col], df_encoded['stroke'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f'{col}: p-value = {p}')


#So，gender_Other (其他性别): p 值为 1.0，表示没有显著关联。删
#work_type_Never_worked (从未工作): p 值为 0.127，表示没有显著关联。删
#work_type_Private: p-value = 0.633463235370685 删 
#smoking_status_smokes: p-value = 0.15542539368489733  删

#对于 Residence_type_Urban，p 值大于 0.05。但，由于居住类型只剩他一个变量，且基于现有文献或领域知识支持，脑中风与居住在城市或农村有关。故保留
#%%
#那删除
df_encoded.drop(columns=['gender_Other', 'work_type_Never_worked','work_type_Private','smoking_status_smokes'], inplace=True)


#%%
#相关性分析
#对于连续变量（如BMI、年龄、血糖），可以使用皮尔逊/spearman相关系数；
#对于分类变量，独热编码后可以使用点双列相关系数或卡方检验。
#这一步可以帮助你检测特征间是否存在多重共线性，以决定是否要删除某些强相关的特征。

#%%
#连续和连续的关系，年龄，bmi，血糖

#所有连续变量（age、BMI、avg_glucose_level）在患者和健康人组中都显著偏离了正态分布
#故用Spearman相关系数
import seaborn as sns
import matplotlib.pyplot as plt

# 连续变量之间的 Spearman 相关系数
continuous_columns = ['age', 'bmi', 'avg_glucose_level']  # 连续变量
corr_matrix = df_encoded[continuous_columns].corr(method='spearman')

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Spearman Correlation Matrix of Continuous Variables')

# 保存为SVG格式的矢量图
plt.savefig('spearman_correlation_matrix.svg', format='svg')

# 显示图形
plt.show()

#三个变量之间的相关性都非常弱，故都保留



#%%
#分类与分类的关系
#Cramér's V 是一种基于卡方检验的相关性度量，用于量化两个分类变量之间的关联强度
import pandas as pd

from scipy.stats import chi2_contingency

# 定义 Cramér's V 计算函数
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2_stat = chi2_contingency(confusion_matrix)[0]  # 使用 chi2_stat 作为变量名
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2_stat / (n * (min(r, k) - 1)))


cat_columns = ['hypertension', 'heart_disease', 'gender_Male', 'ever_married_Yes', 
                'work_type_Self-employed','work_type_children', 'Residence_type_Urban', 
               'smoking_status_never smoked', 'stroke']  # 替换为您的实际分类变量名称

# 计算分类变量之间的 Cramér's V 值
cramers_v_matrix = pd.DataFrame(index=cat_columns, columns=cat_columns)

for col1 in cat_columns:
    for col2 in cat_columns:
        if col1 == col2:
            cramers_v_matrix.loc[col1, col2] = np.nan  # 对角线设置为 NaN
        else:
            cramers_v_matrix.loc[col1, col2] = cramers_v(df_encoded[col1], df_encoded[col2])

# 输出结果
print(cramers_v_matrix)

sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm')
plt.title("Cramér's V Correlation Matrix")

# 保存为SVG格式的矢量图
plt.savefig('Cramérs V Correlation Matrix.svg', format='svg')

plt.show()

#ever_married_Yes和work_type_children的Cramér's V 值=	0.5464257650743133
#So，较强关联
#但是，ever_married_Yes反映婚姻状况，work_type_children反映工作类型，故不删。

#%%
#连续与分类
#print(df_encoded.columns)      
                                               
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

# 分类变量列
binary_columns = ['hypertension', 'heart_disease', 'stroke', 'gender_Male', 
                  'ever_married_Yes', 'work_type_children', 'work_type_Self-employed', 
                  'Residence_type_Urban', 'smoking_status_never smoked']

# 连续变量列
continuous_columns = ['age', 'bmi', 'avg_glucose_level']

# 创建空的相关性结果表
correlation_results = pd.DataFrame(index=continuous_columns, columns=binary_columns)

# 计算相关性并填入表格
for binary_col in binary_columns:
    for continuous_col in continuous_columns:
        corr, _ = pointbiserialr(df_encoded[continuous_col], df_encoded[binary_col])
        correlation_results.loc[continuous_col, binary_col] = corr

# 将结果转为浮点数类型，便于后续绘图
correlation_results = correlation_results.astype(float)

# 可视化相关性热力图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_results, annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar=True, fmt=".2f")

plt.title("Point Biserial Correlation between Continuous and Binary Variables")
# 保存为SVG格式的矢量图
plt.savefig('Point Biserial Correlation between Continuous and Binary Variables.svg', format='svg')

plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_results, annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar=True, fmt=".2f")

#相关性都不太高，不存在共线性



#%%

#直接用组合采样更合适，因为组合采样已有smote过采样，een去噪声

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


# df_encoded 是经过独热编码的数据集
X = df_encoded.drop(columns=['stroke'])  # 自变量
y = df_encoded['stroke']  # 目标变量

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)         # 重要：类别不均衡,所以需要保持类别比例（分类问题需要）添加stratify参数


#SMOTE-ENN组合采样
from imblearn.combine import SMOTEENN  # SMOTE和欠采样组合的方法

# 实例化 SMOTE + 欠采样的组合
smote_enn = SMOTEENN(random_state=42)

# 进行组合采样
X_train_combined, y_train_combined = smote_enn.fit_resample(X_train, y_train)

# 输出新的训练集大小
print("原始训练集大小:", X_train.shape, y_train.value_counts())
print("组合采样后的训练集大小:", X_train_combined.shape, y_train_combined.value_counts())


#%%
# #数据标准化
# 初始化标准化器
scaler = StandardScaler()

# 在组合采样后的训练集上拟合并转换
X_train_combined_scaled = scaler.fit_transform(X_train_combined)

# 在测试集上转换，使用和训练集相同的缩放参数
X_test_scaled = scaler.transform(X_test)

# 检查标准化后的数据
print("标准化后的训练集形状:", X_train_combined_scaled.shape)
print("标准化后的测试集形状:", X_test_scaled.shape)


#%%RandomForest
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

# 定义目标函数
def objective(trial):
    # 定义超参数搜索空间
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50,200),# 树的数量
        'max_depth': trial.suggest_int('max_depth', 10, 40),# 最大树深
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']), # 每个树的最大特征数量
        'min_samples_split': trial.suggest_int('min_samples_split', 1, 20),# 内部节点的最小样本数
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),# 叶子节点的最小样本数
       
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),#Bootstrap 是一种随机采样方法，在训练每棵决策树时，是否从原始训练数据中进行自助采样。
#当 bootstrap=True 时：每棵树的训练数据集是从原始数据集中随机抽取的样本，这些样本可以重复（即自助采样）。这样每棵树会用不同的数据来训练。
#当 bootstrap=False 时：每棵树使用的是整个训练数据集，训练集没有重复的样本。
#为什么使用 bootstrap？
#减少过拟合：
#使用自助采样（bootstrap=True）能够生成不同的训练数据集，使得每棵树的训练数据有所不同。这样可以降低过拟合的风险，因为每棵树的样本不同，减少了模型在特定样本上的依赖。
#提高模型的多样性：
#通过自助采样，模型能够看到不同的训练数据，不同的训练数据集能使得模型的预测能力更强、更具鲁棒性。
#提升准确性：
#通过在每棵树的训练数据中引入随机性，通常可以使得模型更加泛化，在未知数据上的表现更好。
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),  # 处理类别不均衡
        'random_state': 42 # 保证结果可复现
    }
    
    

    # 创建随机森林分类器
    model = RandomForestClassifier(**param)

    # 使用训练集拟合模型
    model.fit(X_train_combined_scaled, y_train_combined)

    # 使用测试集预测概率
    predictions_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算 AUC 分数
    auc_score = roc_auc_score(y_test, predictions_proba)
    
    return auc_score

# 创建一个 Optuna study 对象，优化 AUC
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)

# 打印最佳参数
print("最佳参数:", study.best_trial.params)

# 使用最佳参数重新训练模型
clf_rf = RandomForestClassifier(**study.best_trial.params)
clf_rf.fit(X_train_combined_scaled, y_train_combined)

# 保存最佳模型
joblib.dump(clf_rf, "rf_best_model_optuna.pkl")

# 在测试集上进行预测，获取预测概率
predictions_proba_rf = clf_rf.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率

# 设置新的决策阈值
threshold = 0.05# 可以根据需求调整这个值
predictions_adjusted_rf = (predictions_proba_rf >= threshold).astype(int)  # 使用调整后的阈值进行分类

# 确保 y_test 和 predictions_adjusted_rf 是 NumPy 数组
y_test_array = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
predictions_adjusted_rf_array = predictions_adjusted_rf if isinstance(predictions_adjusted_rf, np.ndarray) else predictions_adjusted_rf.to_numpy()

# 打印模型性能

# 打印训练集的随机森林性能指标

print('随机森林模型在训练集上的性能指标:') 
# 使用训练集的预测结果来评估模型性能
pred_train_rf = clf_rf.predict(X_train_combined_scaled)  # 使用标准化后的训练集进行预测
predictions_train_rf = np.array(pred_train_rf)

# 计算训练集上的性能
train_metrics = performance(y_train_combined, predictions_train_rf)
print(f"归一化代价 (NC): {train_metrics['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {train_metrics['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {train_metrics['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {train_metrics['Accuracy']:.2%}")
print(f"特异性 (Specificity): {train_metrics['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {train_metrics['Sensitivity']:.2%}")
print(f"Youden指数 (Youden Index): {train_metrics['Youden']:.2%}")
print(f"MCC (Matthews Correlation Coefficient): {train_metrics['MCC']:.2%}")  # MCC

# 打印训练集上的 AUC 和混淆矩阵
print("AUC (Training Set):", roc_auc_score(y_train_combined, predictions_train_rf))  # AUC
print("混淆矩阵 (Training Set):\n", confusion_matrix(y_train_combined, predictions_train_rf))  # 混淆矩阵

#随机森林模型在测试集上的性能指标
print('随机森林模型在测试集上的性能指标（调整阈值后）:')
rf_metrics_adjusted = performance(y_test_array, predictions_adjusted_rf)  # 使用测试集的标签进行性能评估
print(f"归一化代价 (NC): {rf_metrics_adjusted['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {rf_metrics_adjusted['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {rf_metrics_adjusted['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {rf_metrics_adjusted['Accuracy']:.2%}")
print(f"特异性 (Specificity): {rf_metrics_adjusted['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {rf_metrics_adjusted['Sensitivity']:.2%}")
print(f"Youden指数 (Youden Index): {rf_metrics_adjusted['Youden']:.2%}")
print(f"MCC (Matthews Correlation Coefficient): {rf_metrics_adjusted['MCC']:.2%}")  # MCC

print("AUC (Testing Set):", roc_auc_score(y_test_array, predictions_adjusted_rf))  # AUC
print("混淆矩阵:\n", confusion_matrix(y_test_array, predictions_adjusted_rf))  # 混淆矩阵

#%%输出rf认为的特征重要性图
import matplotlib.pyplot as plt
import numpy as np

#  clf_rf 是已经训练好的随机森林模型，X 是训练数据的特征矩阵
plot_feature_importances(clf_rf.feature_importances_, 'Feature importance_Random Forest', X.columns)



#%%XGBoost 
#from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# 创建 XGBoost 分类器实例
xgb = XGBClassifier()

# 定义超参数网格
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],  # 学习率
    'max_depth': [3, 5, 7, 9, 11],              # 最大深度
    'n_estimators': [50, 75, 100, 125, 150]  ,     # 树的数量
    'reg_alpha':[ 0.05,0.1,0.2,0.3,0.4, 0.5],  # 添加L1正则化
    'reg_lambda':[ 1, 2] ,   # 添加L2正则化
    'scale_pos_weight': [5, 10, 15, 20, 25]          # 调整类别权重比例
}

# 使用 GridSearchCV 进行超参数调优
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='roc_auc', return_train_score=True)

# 在组合采样后的训练集上训练模型
grid_search_xgb.fit(X_train_combined_scaled, y_train_combined)  # 使用标准化后的训练集

# 获取最佳模型
clf_xgb = grid_search_xgb.best_estimator_

# 输出最佳参数
print("最佳参数:", grid_search_xgb.best_params_)

# 保存最佳模型
joblib.dump(clf_xgb, "xgb_best_model.pkl")

# 在训练集上进行预测
pred_train_xgb = clf_xgb.predict(X_train_combined_scaled)  # 使用标准化后的训练集进行预测
predictions_train_xgb = np.array(pred_train_xgb)

# 在测试集上进行预测
#pred_test_xgb = clf_xgb.predict(X_test_scaled)  # 使用标准化后的测试集进行预测
#predictions_test_xgb = np.array(pred_test_xgb)

# 调整分类阈值
predictions_proba_xgb = clf_xgb.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率
threshold = 0.35  # 调整分类阈值
predictions_adjusted_xgb = (predictions_proba_xgb >= threshold).astype(int)

# 确保 y_test 和 predictions_test_xgb 是 NumPy 数组
y_test_array = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
predictions_adjusted_xgb_array = predictions_adjusted_xgb if isinstance(predictions_adjusted_xgb, np.ndarray) else predictions_adjusted_xgb.to_numpy()

# 打印训练集的 XGBoost 性能指标
print('XGBoost performance metrics on Training Set:')
train_metrics_xgb = performance(y_train_combined, predictions_train_xgb)  # 使用训练集的标签进行性能评估
print(f"归一化代价 (NC): {train_metrics_xgb['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {train_metrics_xgb['False Negative Rate']:.2%}")  # 假阴率
print(f"假阳率 (FPR): {train_metrics_xgb['False Positive Rate']:.2%}")  # 假阳率
print(f"准确率 (Accuracy): {train_metrics_xgb['Accuracy']:.2%}")  # 准确率
print(f"特异性 (Specificity): {train_metrics_xgb['Specificity']:.2%}")  # 特异性
print(f"敏感性 (Sensitivity): {train_metrics_xgb['Sensitivity']:.2%}")  # 敏感性
print(f"Youden指数 (Youden Index): {train_metrics_xgb['Youden']:.2%}")  # Youden 指数
print(f"MCC (Matthews Correlation Coefficient): {train_metrics_xgb['MCC']:.2%}")  # MCC
print("AUC (Training Set):", roc_auc_score(y_train_combined, predictions_train_xgb))  # AUC
print("Confusion Matrix (Training Set):\n", confusion_matrix(y_train_combined, predictions_train_xgb))  # 混淆矩阵

# 打印测试集的 XGBoost 性能指标
print('XGBoost performance metrics on Testing Set:')
test_metrics_xgb = performance(y_test_array, predictions_adjusted_xgb_array)  # 使用测试集的标签进行性能评估
print(f"归一化代价 (NC): {test_metrics_xgb['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {test_metrics_xgb['False Negative Rate']:.2%}")  # 假阴率
print(f"假阳率 (FPR): {test_metrics_xgb['False Positive Rate']:.2%}")  # 假阳率
print(f"准确率 (Accuracy): {test_metrics_xgb['Accuracy']:.2%}")  # 准确率
print(f"特异性 (Specificity): {test_metrics_xgb['Specificity']:.2%}")  # 特异性
print(f"敏感性 (Sensitivity): {test_metrics_xgb['Sensitivity']:.2%}")  # 敏感性
print(f"Youden指数 (Youden Index): {test_metrics_xgb['Youden']:.2%}")  # Youden 指数
print(f"MCC (Matthews Correlation Coefficient): {test_metrics_xgb['MCC']:.2%}")  # MCC
print("AUC (Testing Set):", roc_auc_score(y_test_array, predictions_adjusted_xgb_array))  # AUC
print("Confusion Matrix (Testing Set):\n", confusion_matrix(y_test_array, predictions_adjusted_xgb_array))  # 混淆矩阵

#%%
# 绘制特征重要性图
plot_feature_importances(clf_xgb.feature_importances_, 'Feature importance-XGboost', X.columns)  # 获取特征重要性


#%%
#LightGBM

#optuna超参数优化库

import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import joblib

# 定义目标函数
def objective(trial):
    # 定义超参数搜索空间
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate',  0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 40),
        'max_depth': trial.suggest_int('max_depth', 2,15),
        'min_child_samples': trial.suggest_int('min_child_samples',10,30),
        'subsample': trial.suggest_float('subsample',  0.01,1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7,0.8),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1,0.5),
        'lambda_l2': trial.suggest_float('lambda_l2', 1,2),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1,10),
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42
    }

    model = lgb.LGBMClassifier(**param)

    # 使用验证集预测概率
    model.fit(
        X_train_combined_scaled, y_train_combined, 
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='logloss',               # 设置评价指标
        
    )

    predictions_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算 AUC 分数
    auc_score = roc_auc_score(y_test, predictions_proba)
    
    return auc_score

# 创建一个 Optuna study 对象，优化 AUC
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 打印最佳参数
print("最佳参数:", study.best_trial.params)

# 使用最佳参数重新训练模型
clf_lgb = lgb.LGBMClassifier(**study.best_trial.params)
clf_lgb.fit(X_train_combined_scaled, y_train_combined)

# 保存最佳模型
joblib.dump(clf_lgb, "lgb_best_model_optuna.pkl")

# 在测试集上进行预测，获取预测概率
predictions_proba_lgb = clf_lgb.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率

# 设置新的决策阈值
threshold = 0.48 # 可以根据需求调整这个值
predictions_adjusted = (predictions_proba_lgb >= threshold).astype(int)  # 使用调整后的阈值进行分类

# 确保 y_test 和 predictions_adjusted 是 NumPy 数组
y_test_array = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
predictions_adjusted_array = predictions_adjusted if isinstance(predictions_adjusted, np.ndarray) else predictions_adjusted.to_numpy()

# 打印 LightGBM 模型在训练集上的性能指标
print('LightGBM 模型在训练集上的性能指标（调整阈值后）:')
pred_train_lgb = clf_lgb.predict(X_train_combined_scaled)  # 使用训练集进行预测
predictions_train_lgb = np.array(pred_train_lgb)

# 使用训练集的标签评估模型性能
lgb_metrics_train = performance(y_train_combined, predictions_train_lgb)
print(f"归一化代价 (NC): {lgb_metrics_train['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {lgb_metrics_train['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {lgb_metrics_train['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {lgb_metrics_train['Accuracy']:.2%}")
print(f"特异性 (Specificity): {lgb_metrics_train['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {lgb_metrics_train['Sensitivity']:.2%}")
print(f"Youden指数 (Youden Index): {lgb_metrics_train['Youden']:.2%}")
print(f"MCC (Matthews Correlation Coefficient): {lgb_metrics_train['MCC']:.2%}")  # MCC

# 打印训练集上的 AUC 和混淆矩阵
print("AUC (Training Set):", roc_auc_score(y_train_combined, predictions_train_lgb))  # AUC
print("混淆矩阵 (Training Set):\n", confusion_matrix(y_train_combined, predictions_train_lgb))  # 混淆矩阵

# 打印 LightGBM 模型在测试集上的性能指标
print('LightGBM 模型在测试集上的性能指标:')
lgb_metrics_adjusted = performance(y_test_array, predictions_adjusted)  # 使用测试集的标签进行性能评估
print(f"归一化代价 (NC): {lgb_metrics_adjusted['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {lgb_metrics_adjusted['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {lgb_metrics_adjusted['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {lgb_metrics_adjusted['Accuracy']:.2%}")
print(f"特异性 (Specificity): {lgb_metrics_adjusted['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {lgb_metrics_adjusted['Sensitivity']:.2%}")
print(f"Youden指数 (Youden Index): {lgb_metrics_adjusted['Youden']:.2%}")
print(f"MCC (Matthews Correlation Coefficient): {lgb_metrics_adjusted['MCC']:.2%}")  # MCC

print("AUC (Testing Set):", roc_auc_score(y_test_array, predictions_adjusted))  # AUC
print("混淆矩阵:\n", confusion_matrix(y_test_array, predictions_adjusted))  # 混淆矩阵

#%%
# 绘制特征重要性图
plot_feature_importances(clf_lgb.feature_importances_, 'Feature importance-LightGBM', X.columns)  # 获取特征重要性


#%%
#stacking

#使用逻辑回归元学习器

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# 已训练好的 RandomForest 和 LightGBM 模型
rf_model = joblib.load("rf_best_model_optuna.pkl")
lgb_model = joblib.load("lgb_best_model_optuna.pkl")

# 创建逻辑回归元学习器
meta_model = LogisticRegression(solver='liblinear', random_state=42)

# 使用 StackingClassifier 进行集成
stacking_model = StackingClassifier(
    estimators=[('rf', rf_model), ('lgb', lgb_model)],  # 基学习器：随机森林和LightGBM
    final_estimator=meta_model,  # 元学习器：逻辑回归
    #cv=5  # 交叉验证，帮助选择最佳模型
)

# 训练 Stacking 模型
stacking_model.fit(X_train_combined_scaled, y_train_combined)

# 在测试集上进行预测，获取预测概率
predictions_proba_stacking = stacking_model.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率

# 设置新的决策阈值
threshold = 0.005 # 可以根据需求调整这个值
predictions_adjusted_stacking = (predictions_proba_stacking >= threshold).astype(int)  # 使用调整后的阈值进行分类

# 确保 y_test 和 predictions_adjusted_stacking 是 NumPy 数组
y_test_array = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
predictions_adjusted_stacking_array = predictions_adjusted_stacking if isinstance(predictions_adjusted_stacking, np.ndarray) else predictions_adjusted_stacking.to_numpy()

# 打印模型性能
# 打印训练集的 Stacking 模型性能指标
print('Stacking 模型在训练集上的性能指标:')
# 使用训练集的预测结果来评估模型性能
pred_train_stacking = stacking_model.predict(X_train_combined_scaled)  # 使用标准化后的训练集进行预测
predictions_train_stacking = np.array(pred_train_stacking)

# 计算训练集上的性能
train_metrics_stacking = performance(y_train_combined, predictions_train_stacking)
print(f"归一化代价 (NC): {train_metrics_stacking['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {train_metrics_stacking['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {train_metrics_stacking['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {train_metrics_stacking['Accuracy']:.2%}")
print(f"特异性 (Specificity): {train_metrics_stacking['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {train_metrics_stacking['Sensitivity']:.2%}")
print(f"Youden指数 (Youden Index): {train_metrics_stacking['Youden']:.2%}")
print(f"MCC (Matthews Correlation Coefficient): {train_metrics_stacking['MCC']:.2%}")  # MCC

# 打印训练集上的 AUC 和混淆矩阵
print("AUC (Training Set):", roc_auc_score(y_train_combined, predictions_train_stacking))  # AUC
print("混淆矩阵 (Training Set):\n", confusion_matrix(y_train_combined, predictions_train_stacking))  # 混淆矩阵

#Stacking 模型在测试集上的性能指标
print('Stacking 模型在测试集上的性能指标（调整阈值后）:')
stacking_metrics_adjusted = performance(y_test_array, predictions_adjusted_stacking)  # 使用测试集的标签进行性能评估
print(f"归一化代价 (NC): {stacking_metrics_adjusted['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {stacking_metrics_adjusted['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {stacking_metrics_adjusted['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {stacking_metrics_adjusted['Accuracy']:.2%}")
print(f"特异性 (Specificity): {stacking_metrics_adjusted['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {stacking_metrics_adjusted['Sensitivity']:.2%}")
print(f"Youden指数 (Youden Index): {stacking_metrics_adjusted['Youden']:.2%}")
print(f"MCC (Matthews Correlation Coefficient): {stacking_metrics_adjusted['MCC']:.2%}")  # MCC

print("AUC (Testing Set):", roc_auc_score(y_test_array, predictions_adjusted_stacking))  # AUC
print("混淆矩阵:\n", confusion_matrix(y_test_array, predictions_adjusted_stacking))  # 混淆矩阵





#%%
#stacking
#SVM作为元学习器

import optuna
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# 已训练好的 RandomForest 和 LightGBM 模型
rf_model = joblib.load("rf_best_model_optuna.pkl")
lgb_model = joblib.load("lgb_best_model_optuna.pkl")

# 使用 Optuna 调整 SVM 作为元学习器
def objective(trial):
    # 设定超参数搜索空间
    param = {
        'C': trial.suggest_loguniform('C', 0.001, 0.1),  # SVM 正则化参数
        'gamma': trial.suggest_loguniform('gamma', 0.00001,0.005),  # 核函数gamma参数
        'kernel': trial.suggest_categorical('kernel', ['linear','rbg','poly','sigmoid']),  # 选择核函数
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),  # 处理类别不均衡
        'probability': True,  # 计算概率
        'random_state': 42
    }

    # 训练 SVM 模型
    meta_model_svm = SVC(**param)

    # 使用 StackingClassifier 进行集成
    stacking_model_svm = StackingClassifier(
        estimators=[('rf', rf_model), ('lgb', lgb_model)],  # 基学习器
        final_estimator=meta_model_svm,  # SVM 作为元学习器
        cv=5  # 5折交叉验证
    )

    # 计算交叉验证 AUC
    auc = cross_val_score(stacking_model_svm, X_train_combined_scaled, y_train_combined, 
                          cv=3, scoring='roc_auc').mean()

    return auc  # 目标是最大化 AUC 分数

# 创建 Optuna Study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)  # 

# 获取最佳参数
best_params = study.best_trial.params
print("最佳 SVM 参数:", best_params)

# 使用最优参数训练 SVM
best_svm_model = SVC(**best_params, probability=True, random_state=42)

# 创建 StackingClassifier
stacking_model_svm = StackingClassifier(
    estimators=[('rf', rf_model), ('lgb', lgb_model)],  # 基学习器
    final_estimator=best_svm_model,  # 选择最优 SVM 作为元学习器
    cv=5
)

# 训练 Stacking 模型
stacking_model_svm.fit(X_train_combined_scaled, y_train_combined)

# 保存最佳 Stacking 模型
joblib.dump(stacking_model_svm, "stacking_svm_best_model.pkl")

# 在测试集上进行预测
predictions_proba_stacking_svm = stacking_model_svm.predict_proba(X_test_scaled)[:, 1]

# 调整分类阈值
threshold = 0.003 # 可调整
predictions_adjusted_stacking_svm = (predictions_proba_stacking_svm >= threshold).astype(int)

# 计算测试集上的 AUC
auc_score = roc_auc_score(y_test, predictions_proba_stacking_svm)

# 打印模型性能
stacking_metrics_adjusted_svm = performance(y_test_array, predictions_adjusted_stacking_svm)
print(f"归一化代价 (NC): {stacking_metrics_adjusted_svm['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {stacking_metrics_adjusted_svm['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {stacking_metrics_adjusted_svm['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {stacking_metrics_adjusted_svm['Accuracy']:.2%}")
print(f"特异性 (Specificity): {stacking_metrics_adjusted_svm['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {stacking_metrics_adjusted_svm['Sensitivity']:.2%}")
print("AUC (Testing Set):", roc_auc_score(y_test_array, predictions_adjusted_stacking_svm))  # AUC
print("混淆矩阵:\n", confusion_matrix(y_test_array, predictions_adjusted_stacking_svm))  # 混淆矩阵



#%%

#创建 VotingClassifier（软投票方式）

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

# 载入已优化的 RandomForest /xgb和 LightGBM 模型
# 指定绝对路径，确保加载的是 D:\best_models 下的文件
rf_path = r"D:\best_models\rf_best_model_optuna.pkl"
xgb_path = r"D:\best_models\xgb_best_model.pkl"
lgb_path = r"D:\best_models\lgb_best_model_optuna.pkl"

# 2. 加载模型
rf_model = joblib.load(rf_path)
lgb_model = joblib.load(xgb_path)
xgb_model = joblib.load(lgb_path)


# 创建 VotingClassifier（软投票方式）
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('lgb', lgb_model),('xgb', xgb_model)],  # 组合模型
    voting='soft',  # 软投票，使用 predict_proba()
    weights=[3, 4, 1]  # 可以调整各模型的权重
)

# 训练 VotingClassifier
voting_clf.fit(X_train_combined_scaled, y_train_combined)

# 保存最佳模型
joblib.dump(voting_clf, "voting_clf_new.pkl")


# 在训练集上进行预测
predictions_proba_train = voting_clf.predict_proba(X_train_combined_scaled)[:, 1]  # 获取正类概率
threshold = 0.1 # 调整阈值
predictions_adjusted_train = (predictions_proba_train >= threshold).astype(int)

#  在测试集上进行预测
predictions_proba_test = voting_clf.predict_proba(X_test_scaled)[:, 1]  # 获取正类概率
predictions_adjusted_test = (predictions_proba_test >= threshold).astype(int)

print("y_test 唯一值:", np.unique(y_test))
print("predictions_adjusted_test 唯一值:", np.unique(predictions_adjusted_test))

y_test_array = np.array(y_test).flatten()  # 确保是一维数组
predictions_adjusted_test_array = np.array(predictions_adjusted_test).flatten()


# 计算训练集和测试集的指标
#train_metrics = performance(y_train_combined, predictions_adjusted_train)
test_metrics = performance(y_test_array, predictions_adjusted_test_array)



#  输出评估结果

# 打印测试集上模型性能
print(f"归一化代价 (NC): {test_metrics['Normalized_Cost']:.2%}")
print(f"假阴率 (FNR): {test_metrics['False Negative Rate']:.2%}")
print(f"假阳率 (FPR): {test_metrics['False Positive Rate']:.2%}")
print(f"准确率 (Accuracy): {test_metrics['Accuracy']:.2%}")
print(f"特异性 (Specificity): {test_metrics['Specificity']:.2%}")
print(f"敏感性 (Sensitivity): {test_metrics['Sensitivity']:.2%}")
print("AUC (Testing Set):", roc_auc_score(y_test_array, predictions_adjusted_test))  # AUC
print("混淆矩阵:\n", confusion_matrix(y_test_array, predictions_adjusted_test))  # 混淆矩阵


#%%
#Soft Voting模型在测试集上表现最好

#%%用最佳模型进行预测
import joblib
import pandas as pd

# 保存标准化器
joblib.dump(scaler, 'scaler.pkl')

#将训练时使用的列名保存在文件 features.pkl中
feature_names = X_train_combined.columns.tolist()
joblib.dump(feature_names, 'features.pkl')  # 保存特征列名

import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 输入待测者原始信息：
input_data = {
    'gender': 'Male',
    'age': 18,
    'hypertension': 0,
    'heart_disease': 0,
    'ever_married': 'No',
    'work_type': 'children',
    'Residence_type': 'Urban',
    'avg_glucose_level': 100,
    'bmi': 20,
    'smoking_status': 'never smoked'
}

# 2. 转换为DataFrame
df_input = pd.DataFrame([input_data])

# 3. 类别变量独热编码，保持和训练集一致的列顺序
df_input_encoded = pd.get_dummies(df_input)

# 删除训练时去掉的列
columns_to_drop = ['gender_Other', 'work_type_Never_worked', 'work_type_Private', 'smoking_status_smokes']
df_input_encoded = df_input_encoded.drop(columns=[col for col in columns_to_drop if col in df_input_encoded], errors='ignore')

# 确保与训练时特征一致
expected_columns = joblib.load('features.pkl')  # 包含训练时使用的列顺序
for col in expected_columns:
    if col not in df_input_encoded.columns:
        df_input_encoded[col] = 0  # 缺失的列补0

# 按训练时顺序重新排列列
df_input_encoded = df_input_encoded[expected_columns]

# 4. 加载训练时的标准化器（StandardScaler）并标准化
scaler = joblib.load('scaler.pkl')  # 标准化器
df_input_scaled = scaler.transform(df_input_encoded)

# 5. 加载 Voting 模型
voting_model = joblib.load("voting_clf_new.pkl")

# 6. 预测脑卒中概率
proba = voting_model.predict_proba(df_input_scaled)[:, 1][0]  # 正类概率

# 7. 应用阈值进行判断
threshold = 0.3 #可调整
prediction = int(proba >= threshold)

# 8. 输出结果
print(f"该患者的脑卒中风险概率为: {proba:.2%}")
print(f"模型预测结果（是否脑卒中）: {'是' if prediction == 1 else '否'}")





