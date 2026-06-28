import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

TRAIN_DIR = 'images/train'
FEATURES_CSV = 'features_hog_hu_lbp.csv'


def main():
    print("=" * 60)
    print("SVM Evaluation — 80/20 Stratified Split")
    print("=" * 60)

    # ── Load class names ──
    print("\nLoading classes...")
    classes = [d for d in os.listdir(TRAIN_DIR)
               if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    classes.sort()
    print(f"Classes ({len(classes)}): {classes}")

    # ── Load pre-extracted features ──
    print(f"\nLoading features from {FEATURES_CSV}...")
    if not os.path.exists(FEATURES_CSV):
        print(f"Error: {FEATURES_CSV} not found. "
              "Please run handGestures.py first to extract features.")
        return

    dataset = np.loadtxt(FEATURES_CSV, delimiter=',')
    labels = dataset[:, 0].astype(np.int32)
    features = dataset[:, 1:].astype(np.float32)

    print(f"Total samples: {len(labels)}")
    print(f"Feature dimensions: {features.shape[1]}")

    # ── 80/20 stratified train-test split ──
    print("\nSplitting dataset 80/20 (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples:  {len(y_test)}")

    # ── Standardize using TRAINING set statistics only ──
    print("Standardizing features (fit on train, transform both)...")
    scaler_mean = np.mean(X_train, axis=0)
    scaler_std = np.std(X_train, axis=0)
    scaler_std[scaler_std == 0] = 1e-6

    X_train = (X_train - scaler_mean) / scaler_std
    X_test = (X_test - scaler_mean) / scaler_std

    # ── Train a fresh SVM on the 80% training portion ──
    print("Training SVM (Linear, C=1.0) on 80% training split...")
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1.0)
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    print("SVM training complete.")

    # ── Evaluate on the held-out 20% test portion ──
    print("Predicting on 20% test split...")
    _, results = svm.predict(X_test)
    y_pred = results.flatten().astype(np.int32)

    # ── Classification Report ──
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f"  Overall Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    print("\n--- Classification Report (Test Set) ---\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    # ── Confusion Matrix ──
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix — SVM (HOG + LBP + Hu Moments)\n'
              '20% Stratified Test Split', fontsize=16)
    plt.xlabel('Predicted Gesture', fontsize=14)
    plt.ylabel('True Gesture', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = 'confusion_matrix_svm.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nConfusion matrix saved as '{save_path}'")
    plt.show()


if __name__ == "__main__":
    main()
