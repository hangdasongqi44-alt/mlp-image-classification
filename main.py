import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# ===============================
# 1. データ読み込み（MNIST使用）
# ===============================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ===============================
# 2. MLPモデル構築
# ===============================
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 3. 学習
# ===============================
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ===============================
# 4. 評価
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ===============================
# 5. 中間層特徴抽出
# ===============================
from tensorflow.keras.models import Model

feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[1].output
)

features = feature_extractor.predict(x_test)

# ===============================
# 6. PCA可視化
# ===============================
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

plt.figure()
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=y_test, s=5)
plt.title("PCA Visualization of Hidden Layer")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.savefig("pca_visualization.png")
plt.show()

print("PCA plot saved as pca_visualization.png")
