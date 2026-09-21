# CNN Image Classifier — Mini Project
# Deep Learning Lab | CIFAR-10 | Flask + Keras

## Project Structure

```
cnn_project/
│
├── train_model.py        ← Step 1: Train & save the CNN
├── evaluate.py           ← Step 2: Evaluate model on test set & generate plots
├── app.py                ← Step 3: Run Flask server
├── healthcheck.py        ← Verifies operational status of all scripts & endpoints
├── cnn_model.h5          ← Auto-generated after training (ignored by git)
│
├── tests/                ← Pytest test suite for unit testing features
│
└── templates/
    └── index.html        ← Webpage (served by Flask)
```

---

## Setup (do this once)

Install required libraries:
```
pip install tensorflow flask pillow seaborn pytest scikit-learn matplotlib
```

---

## How to Run

### Step 1 — Train the model
```
python train_model.py
```
- Downloads CIFAR-10 automatically
- Trains for 15+ epochs (supports EarlyStopping and Checkpoints)
- Saves `models/best_model.h5` and `models/training_history.npy`

### Step 2 — Evaluate the model
```
python evaluate.py
```
- Evaluates the model on the CIFAR-10 test set
- Generates a terminal summary report of per-class accuracy
- Saves visualizations to `evaluation_results/`: `confusion_matrix.png`, `per_class_accuracy.png`, and `training_curves.png`

### Step 3 — Start the web server
```
python app.py
```

### Step 4 — Open in browser
```
http://localhost:5000
```
Upload any image → click "Classify Image" → see results!

---

## Testing & Health Checks

Run the automated test suite:
```
python -m pytest
```

Run the project health check (verifies endpoints, files, and dependencies):
```
python healthcheck.py
```

---

## CIFAR-10 Classes
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## CNN Architecture
- 3 Conv blocks (32 → 64 → 128 filters)
- BatchNorm + MaxPooling + Dropout each block
- Dense(256) → Dense(10, softmax)
- Optimizer: Adam | Loss: Categorical Crossentropy

## Expected Accuracy
~75–78% or more on test set after 15-30 epochs (CPU training)
