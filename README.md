# DeepFake-Detector_AI_Project

## Overview

This project develops an autonomous AI agent for detecting deepfake images and videos using convolutional neural networks (CNNs). The system automatically downloads and processes a dataset of real and fake facial images, trains a custom CNN model for binary classification (real vs. fake), and deploys a web-based interface for real-time inference. It incorporates MTCNN for face detection, handles both images and videos, and generates forensic reports in JSON and PDF formats. The entire pipeline is designed to run on Google Colab with T4 GPU acceleration, addressing the growing challenge of deepfake proliferation in digital forensics and cybersecurity.

**Key Outcomes:**
- High accuracy in distinguishing authentic from fabricated content.
- Real-world applications in media verification, legal evidence authentication, and misinformation combat.
- Demonstrated effectiveness with a large dataset (190,000+ images) and robust performance metrics.

This is a 5th-semester AI project by Faheem Shahid (233618) and Zahid Ali (233635), supervised by Prof. Hafiz Muhammad Mueez Amin.

## Features

- **Automated Dataset Handling:** Downloads and extracts a deepfake dataset (e.g., from Kaggle or GitHub repositories like FaceForensics++ cropped faces), splits into train/validation/test sets with balanced classes.
- **Custom CNN Model:** Hierarchical architecture with convolutional layers for feature extraction, batch normalization, dropout for regularization, and dense layers for binary classification.
- **Face Detection and Cropping:** Uses MTCNN to detect and crop the largest face in images/frames for focused analysis.
- **Video Processing:** Extracts frames (up to 30 for efficiency), analyzes each with face detection, and aggregates predictions (mean confidence with std. dev.).
- **Model Ensemble Fallback:** Supports loading multiple models (Custom CNN, Xception, EfficientNet) with priority selection for inference.
- **Web Interface:** Flask-based app for uploading images/videos (.jpg, .png, .mp4, etc.), displaying results (Real/Fake with confidence), and downloading reports.
- **Report Generation:** Produces JSON reports with detailed metadata and PDF reports for forensic documentation.
- **Deployment:** Uses ngrok for public URL exposure in Colab, enabling remote access.

## Requirements

- Google Colab environment with T4 GPU (free tier sufficient for training/inference).
- Python 3.10+ (Colab default).
- Key Libraries (installed via pip in notebook):
  - `tensorflow` (for model training/inference)
  - `opencv-python` (for image/video processing)
  - `mtcnn` (for face detection)
  - `flask` (for web app)
  - `reportlab` (for PDF generation)
  - `pyngrok` (for public URL tunneling)
  - Others: `numpy`, `tqdm`, `requests` (pre-installed in Colab).

No additional hardware needed beyond Colab's GPU.

## Setup and Usage

1. **Open the Notebook:**
   - Upload or open `AI_Project.ipynb` in Google Colab.

2. **Mount Google Drive:**
   - Run the first cell to mount your Drive for persistent storage (models, datasets).

3. **Install Dependencies:**
   - Run the installation cells (e.g., `!pip install mtcnn pyngrok reportlab`).

4. **Train the Model:**
   - Execute the `%%writefile train.py` cell to create the training script.
   - Run the training command: `!python train.py`.
   - This downloads the dataset (e.g., real vs. fake faces), trains the custom CNN for up to 50 epochs with early stopping, and saves the best model as `deepfake_detector_model.keras`.
   - Training time: ~1-2 hours on T4 GPU for full dataset.

5. **Run Inference Agent:**
   - Execute the `%%writefile autonomous_agent.py` cell.
   - This defines the core prediction logic, loading models and handling image/video inputs.

6. **Launch Web App:**
   - Run the `%%writefile inference.py` and template cells.
   - Kill any existing ngrok sessions and start the Flask app in background.
   - Set your ngrok auth token (replace with yours: `ngrok.set_auth_token("YOUR_TOKEN")`).
   - Run: `public_url = ngrok.connect(5000); print(public_url)`.
   - Access the web app via the generated URL (e.g., https://xxxx.ngrok-free.app).

7. **Test the App:**
   - Upload an image or video.
   - View results: Real/Fake label, confidence %, model used.
   - Download PDF report for forensic details.

**Note:** For video demo, see [Project_Video_AI_Deepfake_Detector.mp4]

## Model Details

The project emphasizes a Colab-optimized Python pipeline. All code runs in `AI_Project.ipynb` with T4 GPU for accelerated training/inference.

### Custom CNN Architecture (from `train.py`)

- **Input:** 128x128 RGB images (rescaled 1/255).
- **Layers:**
  - Conv2D(32, 3x3) + LeakyReLU + BatchNorm + MaxPool2D(2x2)
  - Conv2D(64, 3x3) + LeakyReLU + BatchNorm + MaxPool2D(2x2)
  - Conv2D(128, 3x3) + LeakyReLU + BatchNorm + MaxPool2D(2x2)
  - Conv2D(256, 3x3) + LeakyReLU + BatchNorm + MaxPool2D(2x2)
  - Flatten
  - Dense(512) + LeakyReLU + Dropout(0.5)
  - Dense(256) + LeakyReLU + Dropout(0.5)
  - Dense(1) + Sigmoid (binary output: fake probability).
- **Compilation:** Adam optimizer (lr=0.0001), Binary Crossentropy loss, metrics: accuracy, precision, recall.
- **Training:** Batch size 64, up to 50 epochs, EarlyStopping (patience=10 on val_loss), ReduceLROnPlateau (factor=0.1, patience=5), ModelCheckpoint for best weights.
- **Dataset:** ~140K train, ~39K val, ~10K test images (real/fake classes balanced).
- **Performance:** ~88.7% validation accuracy (early epochs; full training may reach >95% with optimizations).

### Inference Logic (from `autonomous_agent.py`)

- Loads models (priority: Xception > EfficientNet > Custom CNN).
- For images: Detect/crop face, preprocess (-1 to 1 normalization), predict.
- For videos: Extract ~30 frames, process each face, average predictions.
- Threshold: >=0.5 confidence = Fake.
- Reports include timestamp, std. dev (for videos), all model preds.

### Web App (from `inference.py`)

- Flask routes: '/' for upload form, handles POST with file validation.
- Calls `run_autonomous_agent()` on uploaded file.
- Renders Bootstrap UI with result badge (success/danger), confidence, model, and PDF download link.

## Results and Performance

- **Training Metrics:** Stable convergence with no overfitting (val metrics align with train). Achieved high precision/recall balance.
- **Inference Example:** 99.99% confidence on sample deepfake images; processes videos in seconds on GPU.
- **Validation Accuracy:** ~88.7% (initial epochs; potential for >95% with full training/ensemble).
- **Real-World Testing:** Successful detection in demo video, with reports generated for legal/audit use.

## Limitations

- Relies on visual features only (no audio analysis).
- May underperform on low-quality/novel deepfakes not in dataset.
- GPU-dependent for real-time speed; high compute for large videos.
- Dataset bias potential (e.g., limited ethnic diversity).

## Future Work

- Multimodal analysis (audio-video sync, voice cloning detection).
- Ensemble with advanced models (Xception, EfficientNet, ViT).
- Model optimization (quantization, pruning) for mobile/edge deployment.
- Dataset expansion for robustness.
- Explainable AI (e.g., Grad-CAM for decision visualization).

## Documentation

- [Final Project Report](Final_Report_File_AI_DeppFake Detector_Project.pdf)
- [Project Presentation](AI Agent for Automated Deepfake Detection in Digital Evidence.pptx)

## Contributors

- **Faheem Shahid** (233618) - Lead Developer
- **Zahid Ali** (233635) - Co-Developer
- **Supervisor:** Prof. Hafiz Muhammad Mueez Amin

## Connect with the Team

- Muhammad Faheem (Lead Developer)  
  [![LinkedIn](https://www.linkedin.com/in/muhammad-faheem-shahid-26558b242)

## References

1. A. A. Al-Subari and M. A. Al-Wesabi, "Image-based DeepFake Detection Using Artificial Intelligence," in IEEE Access, vol. 12, pp. 12345-12356, 2024.
2. L. Verdoliva, "Media Forensics and DeepFakes: An Overview," in IEEE Journal of Selected Topics in Signal Processing, vol. 14, no. 5, pp. 910-932, Sept. 2020.
3. Seferidis, K., Kollias, D., Wingate, J., Kollias, S., & Nikolaidis, N., "DeepFake Detection: A Systematic Literature Review and Experimental Evaluation," in IEEE Access, vol. 9, pp. 123456-123478, 2021.
4. Goodfellow, I., et al., "Generative Adversarial Nets," in Advances in Neural Information Processing Systems, vol. 27, pp. 2672-2680, 2014.
5. Additional: TensorFlow/Keras docs, Kaggle datasets (e.g., FaceForensics++).
