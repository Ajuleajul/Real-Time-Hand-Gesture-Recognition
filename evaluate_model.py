import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

TRAIN_DIR = 'images/train'
MODEL_PATH = 'svm_hog_hu_lbp.xml'
SCALER_PATH = 'scaler_hog_hu_lbp.npz'
FEATURES_CSV = 'features_hog_hu_lbp.csv'

def main():
    print("Loading classes...")
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    classes.sort()

    print(f"Loading features from {FEATURES_CSV}...")
    if not os.path.exists(FEATURES_CSV):
        print(f"Error: {FEATURES_CSV} not found. Please run handGestures.py first to extract features.")
        return

    dataset = np.loadtxt(FEATURES_CSV, delimiter=',')
    labels_true = dataset[:, 0].astype(np.int32)
    features = dataset[:, 1:].astype(np.float32)

    print(f"Loading scaler from {SCALER_PATH}...")
    if not os.path.exists(SCALER_PATH):
        print(f"Error: {SCALER_PATH} not found.")
        return
    scaler_data = np.load(SCALER_PATH)
    scaler_mean = scaler_data['mean']
    scaler_std = scaler_data['std']

    print("Standardizing features...")
    scaler_std_safe = np.copy(scaler_std)
    scaler_std_safe[scaler_std_safe == 0] = 1e-6
    features = (features - scaler_mean) / scaler_std_safe

    print(f"Loading SVM model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    svm = cv2.ml.SVM_load(MODEL_PATH)

    print("Predicting on dataset...")
    _, results = svm.predict(features)
    labels_pred = results.flatten().astype(np.int32)

    print("\n" + "="*50)
    print("--- Classification Report ---")
    print("="*50)
    print(classification_report(labels_true, labels_pred, target_names=classes))

    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(labels_true, labels_pred)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - SVM Model', fontsize=18)
    plt.xlabel('Predicted Gestures', fontsize=14)
    plt.ylabel('True Gestures', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path)
    print(f"\nConfusion matrix plotted and saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    main()
