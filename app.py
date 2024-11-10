import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC

# 生成隨機數據的函數，支持半長軸和半短軸調整
def generate_data(distance_threshold, semi_major_axis, semi_minor_axis):
    # 生成600個隨機點，坐標在以(0,0)為中心的高斯分佈內
    np.random.seed(42)  # 設定隨機種子，保證可重現性
    num_points = 600
    
    # 使用 semi-major 和 semi-minor axis 來調整隨機數據的分佈
    X = np.random.normal(0, 10, (num_points, 2))
    X[:, 0] *= semi_major_axis  # X軸的標準差為半長軸
    X[:, 1] *= semi_minor_axis  # Y軸的標準差為半短軸
    
    # 計算每個點到原點的距離
    distances = np.linalg.norm(X, axis=1)
    
    # 根據距離閾值將點分類
    y = np.where(distances < distance_threshold, 0, 1)
    
    # 使用高斯函數生成第三維度的特徵
    def gaussian_function(x1, x2):
        return np.exp(-(x1**2 + x2**2) / 20)
    
    X3 = np.array([gaussian_function(x1, x2) for x1, x2 in X])
    
    # 將x1, x2, x3結合為特徵矩陣
    X_full = np.column_stack((X, X3))
    
    return X_full, y, X, distances

# 訓練並繪製超平面的函數
def plot_3d(X, y, distance_threshold, semi_major_axis, semi_minor_axis):
    # 訓練SVM模型
    clf = LinearSVC()
    clf.fit(X, y)
    
    # 計算超平面 (coef * X + intercept = 0)
    coef = clf.coef_[0]
    intercept = clf.intercept_
    
    # 創建3D圖形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 設置顏色，Y=0為紅色，Y=1為藍色
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='coolwarm', s=20)
    
    # 繪製超平面
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
    X1, X2 = np.meshgrid(x_range, y_range)
    Z = (-coef[0] * X1 - coef[1] * X2 - intercept) / coef[2]
    
    ax.plot_surface(X1, X2, Z, color='gray', alpha=0.5)
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    
    ax.set_title(f'SVM Classifier\nThreshold={distance_threshold}, Semi-Major Axis={semi_major_axis}, Semi-Minor Axis={semi_minor_axis}')
    
    return fig

# Streamlit介面
def main():
    st.title("SVM Classification with Dynamic Distance Threshold & Elliptical Distribution")
    
    # 距離閾值的滑塊，預設值為4
    distance_threshold = st.slider('Select Distance Threshold', min_value=1, max_value=10, value=4, step=1)
    
    # 半長軸和半短軸的滑塊，預設值為10
    semi_major_axis = st.slider('Select Semi-Major Axis (X-axis scale)', min_value=1, max_value=20, value=10, step=1)
    semi_minor_axis = st.slider('Select Semi-Minor Axis (Y-axis scale)', min_value=1, max_value=20, value=10, step=1)
    
    # 生成數據
    X_full, y, X, distances = generate_data(distance_threshold, semi_major_axis, semi_minor_axis)
    
    # 繪製並顯示結果
    fig = plot_3d(X_full, y, distance_threshold, semi_major_axis, semi_minor_axis)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
