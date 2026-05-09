# CNN Image Classifier — Mini Project
# Deep Learning Lab | CIFAR-10 | Flask + Keras

## Project Structure

```
cnn_project/
│
├── train_model.py        ← Step 1: Train & save the CNN
├── app.py                ← Step 2: Run Flask server
├── cnn_model.h5          ← Auto-generated after training
│
└── templates/
    └── index.html        ← Webpage (served by Flask)
```

---

## Setup (do this once)

Install required libraries:
```
pip install tensorflow flask pillow
```

---

## How to Run

### Step 1 — Train the model
```
python train_model.py
```
- Downloads CIFAR-10 automatically
- Trains for 15 epochs (~10–15 min on CPU)
- Saves `cnn_model.h5`

### Step 2 — Start the web server
```
python app.py
```

### Step 3 — Open in browser
```
http://localhost:5000
```
Upload any image → click "Classify Image" → see results!

---

## CIFAR-10 Classes
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## CNN Architecture
- 3 Conv blocks (32 → 64 → 128 filters)
- BatchNorm + MaxPooling + Dropout each block
- Dense(256) → Dense(10, softmax)
- Optimizer: Adam | Loss: Categorical Crossentropy

## Expected Accuracy
~75–78% on test set after 15 epochs (CPU training)
