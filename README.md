# 🎙️ AI Voice Analysis

A lightweight supervised classification pipeline that distinguishes AI-generated speech from human recorded speech using acoustic and spectral features. <br><br> However, this project has a major constraint regarding sample size, with only 79 samples as of August 2025, I will work around these constraints through hyperparameter tuning, and incorporating monte carlo validations on different train test splits.

## 📁 Project Structure

- `AI_Voice_Analysis_Part1_Feature_Extraction_EDA.ipynb`: Extracts prosodic features (e.g., pitch, energy, duration) from audio samples and performs exploratory data analysis.
- `AI_Voice_Analysis_Part2_Modeling.ipynb`: Conducts feature engineering and builds several classification models (Logistic Regression, Naive Bayes, SVM) to distinguish AI voices from human voices.
- `extract_features.py`: Contains a Python function to extract acoustic and spectral features from each audio file programmatically.
- `mc_validation.py`: A script to evaluate model performance over multiple train-test splits using Monte Carlo validation, necessary due to the small dataset size.
- `audio_data.csv`: Dataset containing extracted audio features and binary labels (0 = Human, 1 = AI).
- `requirements.txt`: Required Python libraries and versions.
- `README.md`: This file.

## 🛠️ Feature Extraction Function

To streamline feature extraction, I coded a Python function in the separate `extract_features.py` file. This function takes the path to an audio file as input, processes the audio to extract acoustic and spectral features such as pitch, jitter, shimmer, formants, MFCCs, and spectral centroid, and returns these features as a structured dictionary or DataFrame row.

This modular design allows batch processing of multiple audio files by calling the function iteratively, making the pipeline scalable and easier to maintain. The function is imported and used in the Part 1 notebook to automate feature extraction for the dataset.

## 🎧 Dataset

- **Human Voices**: Collected from Wikimedia Commons, the Harvard Speech Corpus, and original recordings.
- **AI Voices**: Generated using [ElevenLabs](https://www.elevenlabs.io/), known for producing high-quality synthetic speech.

## 📊 Features Extracted

| Category         | Features                                                                 |
|:-----------------|:-------------------------------------------------------------------------|
| **Prosody**      | `pitch_mean`, `pitch_std`, `speaking_rate`                                |
| **Voice Quality**| `jitter`, `shimmer`, `hnr_mean`, `hnr_std`                                |
| **Formants**     | `f1_mean/std/min/max/range`, `f2_mean/...`, `f3_mean/...`, `f2_f1_ratio_mean` |
| **MFCCs**        | `mfcc_1_mean`, Δ-MFCC first coefficient (`delta_mean`)                    |
| **Spectral**     | `centroid_mean`                                                           |

## 🧠 Model Performance

- After tuning, an SVM model showed the highest accuracy in detecting AI voices, outperforming baseline models like Logistic Regression and Naive Bayes.
- The best model is saved using `joblib`.

``` text
CLASSIFICATION REPORT:

              precision    recall  f1-score   support

           0       0.67      0.67      0.67         9
           1       0.73      0.73      0.73        11

    accuracy                           0.70        20
   macro avg       0.70      0.70      0.70        20
weighted avg       0.70      0.70      0.70        20

Monte Carlo validation:
mean accuracy: 0.7797658862876254
standard deviation of scores: 0.08742093632052701

```   
## 🚀 Usage

To run this project:
1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
