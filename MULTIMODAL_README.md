# Multi-Modal Brain Tumor Detection System

This system integrates separate YOLOv8 models for MRI and PET image analysis with automatic modality detection and ensemble prediction capabilities.

## 🏗️ Architecture Overview

```
Input Image
    ↓
[Modality Classifier]
    ↓
┌──────────┴───────────┐
MRI?                 PET?
↓                      ↓
[MRI Model]      [PET Model]
↓                      ↓
└──────────┬───────────┘
    ↓
[Optional Fusion]
    ↓
Final Prediction
```

## 📁 Project Structure

```
code wih sabbir/
├── multimodal_detector.py      # Main multi-modal detection class
├── prepare_pet_dataset.py      # DICOM conversion & dataset prep
├── train_pet_model.py          # PET model training script
├── test_multimodal.py          # Testing & comparison tools
├── config.yaml                 # MRI dataset config (existing)
├── brain_tumor_project/        # Trained MRI model (existing)
├── pet_tumor_dataset/          # PET dataset (to be created)
└── pet_tumor_project/          # PET model outputs (to be created)
```

## 🚀 Quick Start Guide

### Step 1: Prepare PET Dataset

Convert DICOM files to YOLO format:

```bash
python prepare_pet_dataset.py
```

This will:
- Convert PET DICOM files to JPG (640x640)
- Convert MRI DICOM files to JPG
- Create YOLO dataset structure
- Output instructions for annotation

**Expected Output:**
```
pet_converted_images/     # Converted PET images
pet_tumor_dataset/        # YOLO dataset structure
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels/             # Empty - needs annotation
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── config.yaml
```

### Step 2: Annotate PET Images

Use one of these tools to draw bounding boxes around tumors:

#### Option A: LabelImg (Recommended for beginners)
```bash
pip install labelImg
labelImg pet_tumor_dataset/images/train
```

#### Option B: Roboflow (Cloud-based, easiest)
1. Create account at https://roboflow.com
2. Upload images
3. Draw bounding boxes
4. Export in YOLOv8 format

#### Option C: CVAT (Advanced)
1. Visit https://www.cvat.ai/
2. Create project
3. Upload images and annotate
4. Export YOLO format

**Annotation Format:**
Each image needs a corresponding `.txt` file with format:
```
<class> <x_center> <y_center> <width> <height>
```

Example for `pet_0001.jpg` → `pet_0001.txt`:
```
0 0.512 0.448 0.284 0.312
```

### Step 3: Train PET Model

After annotation is complete:

```bash
python train_pet_model.py --epochs 100 --batch 32
```

**Options:**
```bash
--data      Dataset YAML path (default: pet_tumor_dataset/config.yaml)
--model     Model size: yolov8n/s/m/l/x.pt (default: yolov8n.pt)
--epochs    Number of epochs (default: 100)
--batch     Batch size (default: 32)
--imgsz     Image size (default: 640)
--patience  Early stopping patience (default: 10)
```

**Training Time:**
- With GPU (RTX 3060): ~10-20 minutes
- With CPU: ~2-4 hours

### Step 4: Test Multi-Modal System

#### Demo Mode (Automated Testing):
```bash
python test_multimodal.py --mode demo
```

#### Test Single Image:
```bash
python test_multimodal.py --mode single --image path/to/image.jpg
```

#### Batch Testing:
```bash
python test_multimodal.py --mode batch --dir path/to/images/
```

#### Compare Models:
```bash
python test_multimodal.py --mode compare --dir path/to/test/images/
```

## 💻 Usage Examples

### Example 1: Basic Multi-Modal Detection

```python
from multimodal_detector import MultiModalBrainTumorDetector

# Initialize detector
detector = MultiModalBrainTumorDetector(
    mri_model_path="brain_tumor_project/yolov8_object_detection/weights/best.pt",
    pet_model_path="pet_tumor_project/yolov8_pet_detection/weights/best.pt",
    auto_detect_modality=True
)

# Predict on image (auto-detects MRI or PET)
result = detector.predict("patient_scan.jpg", save_results=True)

print(f"Modality: {result['modality']}")
print(f"Tumor detected: {result['has_tumor']}")
print(f"Confidence: {result['detections'][0]['confidence']:.3f}")
```

### Example 2: Force Specific Modality

```python
# Force MRI model
result = detector.predict("scan.jpg", modality="MRI")

# Force PET model
result = detector.predict("scan.jpg", modality="PET")
```

### Example 3: Ensemble Prediction (Paired Images)

```python
# For co-registered MRI + PET images
result = detector.ensemble_predict(
    mri_image="patient_mri.jpg",
    pet_image="patient_pet.jpg",
    fusion_method="weighted"  # 'average', 'max', or 'weighted'
)
```

### Example 4: Batch Processing

```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = detector.predict_batch(image_paths, save_results=True)

for result in results:
    print(f"{result['image_path']}: {result['modality']} - " +
          f"{'TUMOR' if result['has_tumor'] else 'CLEAR'}")
```

## 🎯 Features

### 1. Automatic Modality Detection
```python
classifier = ModalityClassifier()
modality = classifier.detect_modality("scan.jpg")
# Returns: 'MRI' or 'PET'
```

**Detection Features:**
- Intensity distribution analysis
- Edge density calculation
- Histogram peak detection
- Texture analysis using Sobel gradients

### 2. Dual-Model Integration
- Maintains separate optimized models for each modality
- Automatic routing based on detected modality
- Individual model statistics tracking

### 3. Ensemble Prediction
Three fusion methods for paired images:

**Average Fusion:**
```python
result = detector.ensemble_predict(mri, pet, fusion_method='average')
```

**Max Confidence:**
```python
result = detector.ensemble_predict(mri, pet, fusion_method='max')
```

**Weighted Fusion:**
```python
result = detector.ensemble_predict(mri, pet, fusion_method='weighted')
# Default: 60% MRI, 40% PET
```

### 4. Visualization
```python
detector.visualize_results(
    image_path="scan.jpg",
    predictions=result,
    save_path="result_visual.jpg"
)
```

## 📊 Model Comparison

Compare performance across models:

```bash
python test_multimodal.py --mode compare --dir test_images/
```

**Output:**
- `comparison.csv` - Detailed results table
- `comparison_plot.png` - Visual comparison charts
- Per-image detection confidence
- Average performance metrics

## 🔧 Advanced Configuration

### Custom Confidence Threshold

```python
detector = MultiModalBrainTumorDetector(
    mri_model_path="...",
    pet_model_path="...",
    confidence_threshold=0.7  # Default: 0.5
)
```

### Disable Auto-Detection

```python
detector = MultiModalBrainTumorDetector(
    mri_model_path="...",
    pet_model_path="...",
    auto_detect_modality=False  # Manually specify modality
)
```

### Custom Modality Classifier

```python
# Train your own deep learning classifier
classifier = ModalityClassifier()
modality = classifier.detect_modality_advanced(
    image_path="scan.jpg",
    model_path="custom_classifier.pt"  # Optional CNN model
)
```

## 📈 Performance Metrics

Track detection statistics:

```python
stats = detector.get_statistics()
print(stats)
# Output:
# {
#     'mri_detections': 45,
#     'pet_detections': 32,
#     'total_predictions': 77
# }
```

## 🐛 Troubleshooting

### Issue: GPU not being used
```bash
python check_gpu.py
```
If CUDA unavailable, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Low GPU utilization (< 30%)
Solution: Increase batch size
```bash
python train_pet_model.py --batch 64  # Try 32, 64, or 128
```

### Issue: Out of memory error
Solution: Reduce batch size
```bash
python train_pet_model.py --batch 8
```

### Issue: PET model not found
Make sure you completed all steps:
1. ✅ Run `prepare_pet_dataset.py`
2. ✅ Annotate images
3. ✅ Run `train_pet_model.py`

### Issue: Poor detection accuracy
Solutions:
- **More training data**: Add more annotated images
- **Larger model**: Use `yolov8s.pt` or `yolov8m.pt`
- **More epochs**: Increase training time
- **Data augmentation**: Enabled by default in YOLO

## 📚 Requirements

```bash
pip install ultralytics opencv-python pydicom numpy pandas matplotlib seaborn tqdm
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## 🎓 For Your Thesis

### Key Points to Discuss:

1. **Multi-Modal Approach Benefits:**
   - MRI provides structural information
   - PET provides metabolic information
   - Combined analysis improves diagnostic accuracy

2. **Automatic Modality Detection:**
   - Rule-based feature extraction
   - Can be upgraded to deep learning classifier
   - Reduces manual intervention

3. **Flexible Architecture:**
   - Supports single-model fallback
   - Dual-model specialized detection
   - Ensemble fusion for paired images

4. **Experimental Results:**
   - Compare single vs multi-modal performance
   - Analyze detection confidence scores
   - Evaluate modality detection accuracy

### Suggested Experiments:

1. Train both MRI and PET models
2. Test on mixed dataset
3. Compare:
   - MRI-only model
   - PET-only model
   - Auto-detection multi-modal
   - Ensemble fusion (if paired data available)
4. Generate confusion matrices
5. Calculate precision, recall, F1-score
6. Analyze modality classification accuracy

## 📝 Citation

If you use this code in your thesis:

```
Multi-Modal Brain Tumor Detection System using YOLOv8
Integrating MRI and PET Image Analysis with Automatic Modality Detection
[Your Name], [Year]
```

## 🤝 Support

For questions or issues:
1. Check troubleshooting section above
2. Review example usage
3. Test with demo mode first

## 📄 License

[Add your license here]

---

**Next Steps:**
1. ✅ Run `prepare_pet_dataset.py` to convert DICOM files
2. ⏳ Annotate PET images using LabelImg or Roboflow
3. ⏳ Train PET model using `train_pet_model.py`
4. ⏳ Test multi-modal system using `test_multimodal.py`
5. ⏳ Compare results and analyze for thesis

Good luck with your thesis! 🎓
