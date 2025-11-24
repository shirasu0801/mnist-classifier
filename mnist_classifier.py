import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# --- 1. データセットのロードと準備 ---
def prepare_data():
    """MNISTデータセットをロードし、正規化と準備を行う"""
    print("--- 1. データセットのロードと準備 ---")
    
    # MNISTデータセットをロード
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # データの正規化: 0-255のピクセル値を0-1の範囲に変換
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"訓練データの形状: {x_train.shape}")
    print(f"テストデータの形状: {x_test.shape}")
    return x_train, y_train, x_test, y_test

# --- 2. モデルの構築とコンパイル ---
def build_and_compile_model():
    """シンプルな全結合ニューラルネットワークモデルを構築し、コンパイルする"""
    print("\n--- 2. モデルの構築とコンパイル ---")
    
    model = keras.Sequential([
        # 28x28の画像を784ピクセルに平坦化
        keras.layers.Flatten(input_shape=(28, 28)),
        
        # 隠れ層 (128ニューロン, ReLU活性化関数)
        keras.layers.Dense(128, activation='relu'),
        
        # 出力層 (10ニューロン, Softmaxで確率を出力)
        keras.layers.Dense(10, activation='softmax')
    ])

    # モデルのコンパイル
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

# --- 3. モデルの訓練と評価 ---
def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    """モデルを訓練し、テストデータで評価する"""
    print("\n--- 3. モデルの訓練 ---")
    
    # 訓練の実行
    history = model.fit(
        x_train, y_train, 
        epochs=5,           # 5エポックで学習
        validation_split=0.1 # 訓練データの一部を検証用に使用
    )
    print("モデル訓練が完了しました。")

    print("\n--- 4. モデルの評価 ---")
    # テストデータで評価
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n✅ テストデータの精度 (Accuracy): {test_acc*100:.2f}%")
    print(f"損失 (Loss): {test_loss:.4f}")

# --- 5. 予測結果の可視化 ---
def visualize_predictions(model, x_test, y_test, num_samples=10):
    """モデルの予測結果を画像として可視化する"""
    print(f"\n--- 5. 最初の {num_samples} 枚の画像の予測結果 ---")
    
    # 予測の実行
    predictions = model.predict(x_test[:num_samples])
    
    # 予測結果の可視化用関数
    def plot_image_prediction(i, predictions_array, true_label, img):
        true_label, img = true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        # 最も確率の高い予測ラベルを取得
        predicted_label = np.argmax(predictions_array)
        
        # 正解・不正解に応じて色を変える
        color = 'blue' if predicted_label == true_label else 'red'
        
        # タイトル（予測ラベルと確率）を表示
        plt.xlabel(f"予測: {predicted_label} ({np.max(predictions_array)*100:.1f}%)", 
                   color=color)
        plt.title(f"正解: {true_label}")

    # 最初の num_samples 枚の画像をプロット
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        # 可視化関数に i番目の予測、正解ラベル、画像を渡す
        plot_image_prediction(i, predictions[i], y_test, x_test)

    plt.suptitle(f"最初の {num_samples} 枚の予測結果 (青: 正解, 赤: 不正解)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # supertitleのためのスペース調整
    plt.show()


# --- メイン処理 ---
if __name__ == "__main__":
    # 1. データの準備
    x_train, y_train, x_test, y_test = prepare_data()
    
    # 2. モデルの構築とコンパイル
    model = build_and_compile_model()
    
    # 3 & 4. モデルの訓練と評価
    train_and_evaluate(model, x_train, y_train, x_test, y_test)
    
    # 5. 予測結果の可視化
    visualize_predictions(model, x_test, y_test, num_samples=10)