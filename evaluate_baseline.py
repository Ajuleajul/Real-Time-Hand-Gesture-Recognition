import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

TRAIN_DIR = 'images/train'
BASELINE_FEATURES_CSV = 'features_baseline_raw_pixels.csv'
IMG_SIZE = (32, 32)
K = 5


def main():
    print("=" * 60)
    print("k-NN Baseline Evaluation (Raw Pixel Features)")
    print("=" * 60)

    # ── Load class names ──
    classes = [d for d in os.listdir(TRAIN_DIR)
               if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    classes.sort()
    print(f"Classes ({len(classes)}): {classes}")

    # ── Extract or load cached raw pixel features ──
    if os.path.exists(BASELINE_FEATURES_CSV):
        print(f"\nLoading cached features from {BASELINE_FEATURES_CSV}...")
        dataset = np.loadtxt(BASELINE_FEATURES_CSV, delimiter=',')
        labels = dataset[:, 0].astype(np.int32)
        features = dataset[:, 1:].astype(np.float32)
    else:
        print(f"\nExtracting raw pixel features "
              f"(resize to {IMG_SIZE[0]}x{IMG_SIZE[1]}, grayscale, flatten)...")
        all_features = []
        all_labels = []

        start_time = time.time()
        for label, gesture_name in enumerate(classes):
            gesture_path = os.path.join(TRAIN_DIR, gesture_name)
            if not os.path.isdir(gesture_path):
                continue

            count = 0
            for img_name in os.listdir(gesture_path):
                if img_name.endswith('.db'):
                    continue

                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img_resized = cv2.resize(img, IMG_SIZE)
                pixel_features = img_resized.flatten().astype(np.float32)

                all_features.append(pixel_features)
                all_labels.append(label)
                count += 1

            print(f"  --> {gesture_name}: {count} images")

        elapsed = time.time() - start_time
        print(f"Extraction complete in {elapsed:.1f}s")

        features = np.array(all_features, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        # Cache to CSV so future runs load instantly
        print("Saving features to CSV (this may take a moment)...")
        dataset = np.hstack((labels.reshape(-1, 1), features))
        np.savetxt(BASELINE_FEATURES_CSV, dataset, delimiter=',', fmt='%.6g')
        print(f"Saved to {BASELINE_FEATURES_CSV}")

    print(f"\nTotal samples: {len(labels)}")
    print(f"Feature dimensions: {features.shape[1]} "
          f"(= {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels)")

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

    # ── Normalize pixel values to [0, 1] ──
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # ── Train and evaluate k-NN ──
    print(f"\nTraining k-NN (k={K}, metric=euclidean)...")
    print("(This may take several minutes for large datasets...)")
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean', n_jobs=-1)
    knn.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"k-NN fitting complete in {train_time:.1f}s")

    print("Predicting on test set...")
    start_time = time.time()
    y_pred = knn.predict(X_test)
    predict_time = time.time() - start_time
    print(f"Prediction complete in {predict_time:.1f}s")

    # ── Classification Report ──
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f"  k-NN Baseline Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)
    print("\n--- Classification Report (Test Set) ---\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    # ── Confusion Matrix ──
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix — k-NN Baseline (k={K}, Raw 32x32 Pixels)\n'
              f'20% Stratified Test Split', fontsize=16)
    plt.xlabel('Predicted Gesture', fontsize=14)
    plt.ylabel('True Gesture', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = 'confusion_matrix_knn_baseline.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nConfusion matrix saved as '{save_path}'")
    plt.show()


if __name__ == "__main__":
    main()
