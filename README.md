````markdown
1. Project Overview
-------------------
This repository implements a **Speaker Verification System** based on audio features extracted from Mozilla Common Voice data (version Delta Segment 19.0). The goal is to differentiate between a “target” speaker (reference speaker) and “non-target” speakers using an SVM (Support Vector Machine) classifier. The main steps include:

- **Dataset Preparation**
- **Target & Non-Target Selection**
- **Feature Extraction (MFCC Statistics)**
- **Data Splitting (Train / Test)**
- **Model Training**
- **Evaluation**
- **Model Saving**
- **Prediction**

Detailed descriptions of each step, along with dataset details:

---

2. Dataset Details
------------------
**Source:**  
Audio data is taken from Mozilla’s Common Voice project (version Delta Segment 19.0). You can download it from:

```bash
https://commonvoice.mozilla.org/en/datasets
````

Once downloaded, unzip the archive. The audio files and metadata are stored under:

```bash
data/
```

The main file for this implementation is:

```bash
data/validated.tsv
```

which contains:

* `client_id` – Unique id of each speaker.
* `path` – filename of MP3 audio clip.
* Additional metadata: language, age, gender, client details, etc.

**Data structure:**

1. **`validated.tsv`**

   * important columns for this project: `client_id` and `path`.

2. **Audio files**

   * All audio files are stored under `data//clips`

In our directory structure, once unzipped:

```plain
data
    ├── clips
    │   ├── 0000000.mp3
    │   ├── 0000001.mp3
    │   └── ...
    └── validated.tsv
```

---

3. Selecting Target & Non-Target Speakers data

---

### Choose a “Target” Speaker

1. We scanned the `validated.tsv` file to find which `client_id` has the maximum number of validated audio clips.
2. In our case, the `client_id` with the most clips was `'b87...'` (28 total audio clips).
3. We created a DataFrame `df_target`


### Choose “Non-Target” Speakers

1. All remaining audio clips (one which not belonging to the target speaker) are collected into:

   ```python
   df_non_target_all = df[df["client_id"] != target_id]
   ```
2. From this large pool, we randomly sampled **40** audio clips to represent “non-target” data

### Final DataFrames

* **`df_target`**: Contains 28 audio clips from the chosen reference speaker.
* **`df_non_target`**: Contains 40 randomly selected audio clips from all other speakers.

We concatenated these two DataFrames ( with a binary label) to form our full dataset used for training/testing.

---

4. Feature Extraction: MFCC Statistics

---

We represent each audio clip by extracting **Mel-Frequency Cepstral Coefficients (MFCCs)** and computing simple summary statistics (mean & standard deviation):

1. **Load** the audio file at 16 kHz
2. **Trim** silence from the beginning and end
3. **Compute** MFCC stats

---

5. Data Splitting (Training / Testing)

---

After extracting features, we split the dataset into **training** and **testing** 


* **`x_train, y_train`**: 80% 
* **`x_test, y_test`**: 20%
* We used `stratify=y` to ensure the same proportion of target vs non-target in both the datasetst

---

6. Model Training: Support Vector Classifier (SVC)

---

We opted for an SVM with an RBF kernel due to its effectiveness on moderately low-dimensional feature spaces (26 dimensions here). 

7. Evaluation

---

After training, we evaluated the classifier on the held-out test set:

* **Accuracy:**  correctly classified sample
* **F1-Score:**  mean of precision and recall on the target class.
* **Report:** Precision, recall, F1-score

---

8. Model Extraction

---
we save SVM model to disk for later use (prediction)

* **Output File:** `model.pkl`
* This file can be loaded later to perform speaker verification on new audio clips.

---




````markdown
# Speaker Verification Streamlit App

A simple Streamlit frontend for our Speaker Verification System. Upload an audio file (WAV/MP3) or pick from test clips to predict whether it belongs to the “target” speaker (reference) or a “non-target” speaker, using a pre-trained SVM model.

---

## 1. Prerequisites

- **Python 3.8+**
- Make sure you have the following files/folders in the project root:
  - `model.pkl` (the serialized SVM model)
  - `testing_data/` (a folder containing a few `.wav` or `.mp3` files for quick testing)
  - `app.py` (this Streamlit script)

---

## 2. Setup & Installation

1. **Clone or download** this repository.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
````

3. **Activate** the virtual environment:

   * On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```
   * On Windows (PowerShell):

     ```powershell
     .\venv\Scripts\Activate
     ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
---


## 3. Run the app

With virtual environment activated, simply run:

```bash
streamlit run app.py
```

This will open a browser window (or show a local URL) where you can:

1. **Upload** a WAV/MP3 file from your computer.
2. **Or select** one of the example clips in `testing_data/`.
3. Click “Predict Speaker” to see:

   * **Prediction**: “Target Speaker” or “Non-Target Speaker”
   * **Confidence**: Percentage score

---

## 5. File Structure

```
.
├── model.pkl             # Pre-trained SVM model
├── testing_data/         # Example audio files for prediction
│   ├── sample1.mp3
│   ├── sample2.wav
│   └── ...
├── app.py                # Streamlit application code
├── requirements.txt      # Python package list
└── README.md             # This file
```

---

## 6. Notes

* This app does **not** support real-time microphone input; it only accepts uploaded files or existing test clips.
* Ensure `model.pkl` matches the model used during training 

---

