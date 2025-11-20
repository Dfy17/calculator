import warnings
warnings.filterwarnings("ignore")
import os
import pickle
from scipy.stats import norm
import numpy as np
import pandas as pd
from tableone import TableOne
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from missforest import MissForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import shap

os.chdir("D://胃肠间质瘤影像-血清学ML预测模型") # 当前工作路径

train_data = pd.read_csv("训练集_final.csv", encoding="gbk")
val_data = pd.read_csv("验证集_final.csv", encoding="gbk")
## 首先训练集连续变量的标准化
#train_data = train_data.drop(columns='group')
continuous_vars = ['PNI', 'MLR', 'SII', 'PLR', 'ALI', 'AFR'] # 连续变量的变量名
train_data[continuous_vars] = StandardScaler().fit_transform(train_data[continuous_vars]) # 对训练集中的连续变量进行标准化
# 保存标准化后的训练集
#train_data.to_csv("数据/train_data_scaled.csv")
## 接下来测试集连续变量的标准化
#val_data = val_data.drop(columns='group')
val_data[continuous_vars] = StandardScaler().fit_transform(val_data[continuous_vars])
# 保存标准化后的测试集
#test_data.to_csv("数据/test_data_scaled.csv")
X_train_all = train_data.iloc[:, 1:]  
Y_train = train_data.iloc[:, 0]  

X_train = train_data.iloc[:, [3,8,13,14,16,18]]  
Y_train = train_data.iloc[:, 0]  
X_val = val_data.iloc[:, [3,8,13,14,16,18]]  
Y_val = val_data.iloc[:, 0]

X_columns=X_train.columns.tolist()
x_columns=X_val.columns.tolist()

dfml_test=pd.read_csv("大连附一汇总表最终版本，全部都有，未更改版本.csv", encoding="gbk")
X_test = dfml_test.iloc[:, [4,8,13,14,17,19]]  
Y_test = dfml_test.iloc[:, 0]
X_test.iloc[:,[4,5]] = StandardScaler().fit_transform(X_test.iloc[:,[4,5]])

###网页计算器
import streamlit as st  # 关键：添加这行解决st未定义的问题

from lime.lime_tabular import LimeTabularExplainer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from lime.lime_tabular import LimeTabularExplainer
##加载训练好的模型
base_learners25 = [
    ("SVM", SVC(C=2, degree=3, gamma=0.05, kernel='rbf', probability=True, random_state=123)),
    ("ANN", MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='logistic', 
                         random_state=123, alpha=0.01, learning_rate_init=0.0015, max_iter=500))
]
# 2. 定义二级学习器
meta_model = LogisticRegression(C=15, penalty='l2', solver='liblinear', max_iter=1000, random_state=123)

# 3. 创建并训练Stacking分类器
stacking_Classifier25 = StackingClassifier(estimators=base_learners25, final_estimator=meta_model, cv=5)
stacking_Classifier25.fit(X_train, Y_train)  # 用你的真实数据训练

# 4. 赋值给sclf_model（你定义的模型变量）
sclf_model = stacking_Classifier25

##加载数据
#X_test
##加载特征(列名)
feathure_names=X_columns
##streamlit 用户界面
## 初始化LIME解释器（需要训练数据作为参考）
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,  # 关键：DataFrame转NumPy数组（如果X_train是DataFrame）
    feature_names=X_columns,
    class_names=["≤5", ">5"],  # 对应预测类别的名称
    mode="classification"
)

## Streamlit 用户界面
st.title("胃肠间质瘤预测模型计算器")  # 设置网页标题

# 肿瘤大小（修正变量名大小写）
size = st.selectbox("肿瘤大小：", options=[1, 2, 3, 4])
# 液化坏死
necrosis = st.selectbox("液化坏死:", options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")
# 粗大血管征（修正变量名空格问题）
vascular_sign = st.selectbox("粗大血管征:", options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")
# 脂肪间隙（修正变量名空格问题）
fat_interval = st.selectbox("脂肪间隙:", options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")
# PLR
plr = st.number_input("PLR:", min_value=-2, max_value=2)
# AFR
afr = st.number_input("AFR:", min_value=-2, max_value=2)

# 处理输入数据并进行预测（修正变量名一致性）
feature_values = [size, necrosis, vascular_sign, fat_interval, plr, afr]
import numpy as np
features = np.array([feature_values])  # 转化为模型输入格式

# 当用户点击“Predict”时执行
if st.button("Predict"):
    # 预测类别（0:≤5，1:>5)
    predicted_class = sclf_model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = sclf_model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class}(1:>5, 0:≤5)")
    st.write(f"**预测概率:** {predicted_proba}")

    # 根据预测结果生成建议（修正语法错误和逻辑错误）
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:  # 高风险
        advice = (
            f"根据模型预测，您的风险较高。\n"
            f"模型预测您的风险概率为 {probability:.1f}%。\n"
            "建议咨询医疗专业人员进行进一步评估和可能的干预。"
        )
    else:  # 低风险（修正else后的语法错误）
        advice = (
            f"根据模型预测，您的风险较低。\n"
            f"模型预测您的风险概率为 {probability:.1f}%。\n"
            "建议保持定期检查和健康生活方式。"
        )
    st.write(advice)

    # 添加LIME模型解释（展示特征重要性）
    st.subheader("模型预测解释（LIME）")
    # 生成解释
    exp = lime_explainer.explain_instance(
        data_row=features[0],
        predict_fn=sclf_model.predict_proba,
        num_features=6  # 展示所有6个特征
    )
    # 显示LIME解释图
    st.pyplot(exp.as_pyplot_figure())

