from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import os
import math
import traceback
import cv2
import numpy as np
from handGestures import predict_image

app = Flask(__name__, template_folder='.')


# ──────────────────────────────────────────────────────────────────────────────
# Live camera stream
# ──────────────────────────────────────────────────────────────────────────────
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
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

        predicted_gesture, confidence = predict_image(roi_gray)

        cv2.putText(
            frame, f"Sign: {predicted_gesture}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/style.css')
def serve_css():
    return send_from_directory(os.path.dirname(__file__), 'style.css')


@app.route('/script.js')
def serve_js():
    return send_from_directory(os.path.dirname(__file__), 'script.js')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    """
    Accepts a multipart/form-data POST with an 'image' field.
    Always returns JSON — never an HTML error page.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename.'}), 400

        # Decode the uploaded image in memory
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({'error': 'Could not decode image. Please upload a valid JPG/PNG/BMP.'}), 400

        # Pre-process the same way as the live feed
        img = cv2.GaussianBlur(img, (5, 5), 0)

        gesture, raw_score = predict_image(img)

        # Convert raw SVM decision score to human-readable percentages.
        # Confidence: sigmoid maps the signed margin to (0, 100).
        sigmoid = lambda x, k=0.5: 1.0 / (1.0 + math.exp(-k * x))
        confidence_pct = round(sigmoid(float(raw_score)) * 100, 2)

        # Accuracy: normalised absolute margin capped at ABS_MAX → (0, 100).
        ABS_MAX = 10.0
        accuracy_pct = round(min(abs(float(raw_score)) / ABS_MAX, 1.0) * 100, 2)

        return jsonify({
            'gesture':        gesture,
            'raw_score':      round(float(raw_score), 6),
            'confidence_pct': confidence_pct,
            'accuracy_pct':   accuracy_pct,
        })

    except Exception as e:
        # Print full traceback to the terminal for debugging
        traceback.print_exc()
        # Always return JSON so the browser never sees an HTML error page
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
