import cv2
import os
import numpy as np
import time
from skimage.feature import local_binary_pattern

TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'
MODEL_PATH = 'svm_hog_hu_lbp.xml'
SCALER_PATH = 'scaler_hog_hu_lbp.npz'
FEATURES_CSV = 'features_hog_hu_lbp.csv'

IMG_SIZE = (64, 64) 

winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS

classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
classes.sort()

def extract_features(img_gray):
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    
    hog_feat = hog.compute(img_resized).flatten()
    
    moments = cv2.moments(img_resized)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    lbp = local_binary_pattern(img_resized, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS + 3), range=(0, LBP_N_POINTS + 2))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    combined_feat = np.hstack([hog_feat, hu_moments, lbp_hist])
    return combined_feat

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print(f"Loading existing model and scaler from {MODEL_PATH} and {SCALER_PATH}...")
    svm = cv2.ml.SVM_load(MODEL_PATH)
    scaler_data = np.load(SCALER_PATH)
    scaler_mean = scaler_data['mean']
    scaler_std = scaler_data['std']
    print("Model and scaler loaded successfully.")
else:
    print("No existing model/scaler found (or need retrain). Checking for feature CSV...")
    if not os.path.exists(FEATURES_CSV):
        print("Extracting features from images to CSV...")
        all_features = []
        all_labels = []
        
        start_time = time.time()
        for label, gesture_name in enumerate(classes):
            gesture_path = os.path.join(TRAIN_DIR, gesture_name)
            if not os.path.isdir(gesture_path):
                continue
            
            print(f"--> Extracting features for class: {gesture_name}")
            
            for img_name in os.listdir(gesture_path):
                if img_name.endswith('.db'):
                    continue
                    
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                    
                features = extract_features(img)
                all_features.append(features)
                all_labels.append(label)

        print(f"Extraction complete in {round(time.time() - start_time, 2)}s.")
        
        all_features = np.array(all_features, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32).reshape(-1, 1)
        dataset = np.hstack((all_labels, all_features))
        np.savetxt(FEATURES_CSV, dataset, delimiter=',', fmt='%.6g')
        print(f"Saved extracted features to {FEATURES_CSV}")
    
    print("Loading features from CSV...")
    dataset = np.loadtxt(FEATURES_CSV, delimiter=',')
    labels = dataset[:, 0].astype(np.int32)
    features = dataset[:, 1:].astype(np.float32)
    
    print("Standardizing features...")
    scaler_mean = np.mean(features, axis=0)
    scaler_std = np.std(features, axis=0)
    scaler_std[scaler_std == 0] = 1e-6
    features = (features - scaler_mean) / scaler_std
    np.savez(SCALER_PATH, mean=scaler_mean, std=scaler_std)
    print(f"Saved scaler to {SCALER_PATH}")
    
    print("Training SVM with fixed parameters (much faster)...")
    svm = cv2.ml.SVM_create()
    
    svm.setKernel(cv2.ml.SVM_LINEAR) 
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1.0)
    
    svm.train(features, cv2.ml.ROW_SAMPLE, labels)
    svm.save(MODEL_PATH)
    print(f"SVM Trained and saved to {MODEL_PATH}.")




def predict_image(roi_gray):
    features = extract_features(roi_gray)
    features = np.array([features], dtype=np.float32)
    
    features = (features - scaler_mean) / scaler_std
    
    _, result = svm.predict(features)
    _, raw_result = svm.predict(features, flags=cv2.ml.StatModel_RAW_OUTPUT)
    
    predicted_label = int(result[0][0])
    confidence = raw_result[0][0]
    return classes[predicted_label], confidence





# def test_image_directory():
#     print("Processing Test Images...")
#     if os.path.exists(TEST_DIR):
#         for img_name in os.listdir(TEST_DIR):
#             if img_name.endswith('.db'):
#                 continue

#             img_path = os.path.join(TEST_DIR, img_name)
#             img = cv2.imread(img_path)
            
#             if img is None:
#                 continue
                
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
#             prediction, confidence = predict_image(gray)
            
#             img_disp = cv2.resize(img, (0, 0), fx=4.0, fy=4.0)

#             cv2.putText(
#                 img_disp, f"Predict: {prediction} ({confidence:.2f})", (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
#             )

#             cv2.imshow("Sign Language Test - Image", img_disp)
#             cv2.waitKey(0)

#     cv2.destroyAllWindows()

if __name__ == "__main__":
    # test_image_directory()
    
    print("Starting Webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Camera opened. Place your hand in the green box. Press 'Q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]
        box_size = 300
        x1, y1 = w - box_size - 50, int((h - box_size) / 2)
        x2, y2 = x1 + box_size, y1 + box_size
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        predicted_gesture, _ = predict_image(roi_gray)
        
        cv2.putText(
            frame, f"Sign: {predicted_gesture}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )
    
        cv2.imshow("ASL Recognition - Webcam", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()