# Drowsiness Dection 

## 1. Project Overview

This repository provides a reproducible pipeline for **drowsiness detection** using face-based visual cues extracted from videos. The workflow starts from raw video clips, extracts facial landmarks, derives frame-level features (e.g., EAR/MAR), aggregates them into fixed-length time windows, and finally trains and evaluates multiple machine-learning models for **binary classification** (Alert vs. Drowsy).

The project is designed with two goals in mind:

- **Model comparison (main focus):** train and compare multiple classical ML models on the same processed dataset, and report performance on a held-out test set.
- **Robustness analysis:** evaluate model stability under different windowing/threshold settings (e.g., window size, blink/yawn thresholds) by running a sweep of dataset-construction parameters and summarizing model performance distributions.

## 2. Repository Structure 

The repository is organized into three layers: **reusable source modules**, **runnable entry scripts**, and **input/output directories**.

```text
ComputerVision
├─ config      
├─ scripts       
├─ src          
├─ results
├─ notebooks      
├─ video          
├─ data           
├─ exported_data
├─ processed_data
├─ requirements.txt 
└─ .gitignore     
```


### 2.1 Code layout

- **`src/`** (reusable modules)  
  Implements each pipeline step:
  - `landmarks.py`: extract facial landmarks from videos
  - `frame_features.py`: compute per-frame features from landmarks (e.g., EAR/MAR)
  - `window_features.py`: aggregate frame features into window-level features
  - `dataset_prep.py`: build train/val/test datasets (parquet) from window features
  - `train_models_lib.py`: training + evaluation utilities (multi-model comparison)
  - `experiments_lib.py`: parameter sweep utilities (robustness analysis)

- **`scripts/`** (thin runnable entrypoints)  
  Calls `src/` modules with arguments:
  - `run_dataset.py`: run the data pipeline up to dataset generation (`processed_data/`)
  - `run_models.py`: model training (`train`) and parameter sweep (`sweep`)
  - `summarize_robustness.py`: read-only summarization of sweep outputs into robustness tables
  - `visualize_results.py`: generate plots/tables (confusion matrix, permutation importance, feature distributions) into `results/train/`

- **`notebooks/`**  
  Original exploratory notebooks kept for reference; the reproducible pipeline is implemented in `src/` + `scripts/`.

### 2.2 Inputs (data folders)

These folders contain **large inputs / intermediate artifacts** and are typically **not committed** to Git:

- **`video/`** *(INPUT)*  
  Raw video clips used by the pipeline.  

- **`data/`** *(INTERMEDIATE)*  
  Landmark extraction outputs (e.g., per-video landmark files and related metadata).  This folder is produced by `src/landmarks.py` (usually via `scripts/run_dataset.py`).

- **`exported_data/`** *(INTERMEDIATE)*  
  Exported feature files, produced by `src/frame_features.py` and `src/window_features.py`.
  - frame-level features (e.g., `features_frame_level.csv`)
  - window-level features (e.g., `features_window_level.csv`)

### 2.3 Outputs (dataset + results)

- **`processed_data/`** *(OUTPUT: dataset)*  
  Final datasets used for training/evaluation, produced by `src/dataset_prep.py` (usually via `scripts/run_dataset.py`).
  - `train.parquet`
  - `val.parquet`
  - `test.parquet`  

- **`results/`** *(OUTPUT: training results)*  
  This folder contains **only model-training / evaluation artifacts** (metrics, saved models, reports). It is intentionally separated from `exported_data/` and `processed_data/`.

  - **`results/train/`** *(single-run model comparison)*  
    Produced by `scripts/run_models.py train`. 
    - `metrics_select.csv` — per-model metrics on the selection split (val if available, otherwise test)
    - `metrics_test_all.csv` — per-model metrics on the test set (used for robustness summaries)
    - `metrics_test_best.csv` — test metrics of the selected best model
    - `best_model.joblib` — serialized best model
    - `best_model_name.json` — metadata (best model name, selection metric, feature list, etc.)
    - `classification_report_test.txt` — sklearn classification report on test
    - `confusion_matrix_test.csv` — confusion matrix on test
    - `figures/` — generated figures (confusion matrix, permutation importance, feature distributions)
    - `tables/` — generated tables (sorted test metrics as CSV/MD)

  - **`results/experiments/`** *(robustness sweeps)*  
    Produced by `scripts/run_models.py sweep`. 
    - `results/experiments/<exp_name>/...` — one folder per parameter setting  
      Each `<exp_name>` typically contains the same set of artifacts as `results/train/`
      (e.g., `metrics_test_all.csv`, `best_model.joblib`, reports, etc.).

  - **`results/summary/`** *(aggregated robustness summaries)*  
    Produced by `scripts/summarize_robustness.py`. 
    - `all_models_all_exps.csv` — concatenated table of all models across all experiment settings
    - `model_robustness_summary.csv` — per-model robustness statistics (mean/std/win-rate over experiments)

### 2.4 Configuration templates 

- **`config/`** 
    - `config/example.yaml` - A lightweight configuration template for the pipeline. 
    - `config/landmarks_meta.template.json` - A template describing the expected landmark export metadata

## 3. Installation & Environment Setup

This project is intended to be run from the repository root using Python.

**Install the Python dependencies:**

```bash
pip install -r requirements.txt
```

## 4. Dataset Source (UTA-RLDD)

This project uses the **University of Texas at Arlington Real-Life Drowsiness Dataset (UTA-RLDD / UTA-RLDD)**:

- Official dataset page: https://sites.google.com/view/utarldd/home  
- Reference paper: “A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection” (CVPRW 2019)

### Brief dataset description

UTA-RLDD is a realistic RGB-video dataset for drowsiness detection collected in real-life environments. It contains recordings from **60 participants** and is commonly described as **~30 hours** of videos in total, with **three drowsiness levels** (Alert, Low Vigilance, Drowsy). Each subject contributes one video per level (≈ 180 videos total).

### Labels used in this project

Although the dataset supports 3 stages, **this project focuses on a binary subset**:

- **Label 0** → *Alert*
- **Label 10** → *Drowsy*

> Note: **Label 5 (low vigilant)** videos are not used in our experiments.

## 5. Running the Pipeline

All commands below assume you run them from the repository root.

### 5.1 Build the training dataset (videos → landmarks → features → parquet)

This step generates intermediate artifacts under `data/` and `exported_data/` and the training-ready datasets under `processed_data/`.

```bash
python scripts/run_dataset.py
```

After a successful run, you should have (at minimum):
- `data/` (landmarks outputs; intermediate)
- `exported_data/` (feature CSV exports; intermediate)
- `processed_data/` (final datasets; used for training)
  - `train.parquet`
  - `test.parquet`
  - `val.parquet` 

### 5.2 Train and compare models (main experiment)

Train multiple models on the dataset in `processed_data/` and save all training artifacts to `results/train/`.

```bash
python scripts/run_models.py train --processed_dir processed_data
```

Outputs are written to:
- `results/train/` (metrics tables, best model, reports, confusion matrix, etc.)

### 5.3 Visualize training results

Generate plots and tables for the selected best model (**HGB**) and save them to `results/train/figures/` and `results/train/tables/`:

```bash
python scripts/visualize_results.py --processed_dir processed_data --results_dir results/train
```

### 5.4 Robustness sweep

Run a parameter sweep over dataset-construction settings (e.g., window size, blink/yawn thresholds, min frames).

```bash
python scripts/run_models.py sweep --frame_csv exported_data/features_frame_level.csv
```

Artifacts are separated by design:
- Data artifacts:
  - `exported_data/experiments/<exp_name>/...`
  - `processed_data/experiments/<exp_name>/...`
- Training results artifacts:
  - `results/experiments/<exp_name>/...`

### 5.5 Summarize robustness results

This script does not generate new experiments. It only reads `results/experiments/` and produces aggregated tables for robustness analysis.

```bash
python scripts/summarize_robustness.py --results_root results/experiments
```

Typical outputs:
- `results/summary/all_models_all_exps.csv`
- `results/summary/model_robustness_summary.csv`

## 6. Results Summary (Model Comparison + Robustness)

This section summarizes the outputs produced under `results/`.

### 6.1 Main model comparison (`results/train/`)

**Best model** selected on **val**: `HGB` (HistGradientBoosting) using `f1_macro`.

**i. Test performance**

Although `HGB` was selected as the best model on the validation split, its test-set performance is lower than the linear baselines. This discrepancy suggests that `HGB` is more sensitive to the specific train/validation split and may not generalize as consistently under our current feature set. This observation aligns with the robustness sweep, where `HGB` exhibits higher variance across parameter settings, while linear models (`LogReg` / `LinearSVM`) remain more stable.

```
| Model     | Accuracy | Macro F1 |
|-----------|----------|----------|
| LogReg    | 0.592    | 0.579    |
| LinearSVM | 0.588    | 0.577    |
| HGB       | 0.558    | 0.555    |
| RF        | 0.546    | 0.543    |
```

**ii. Error analysis (Confusion Matrix, HGB)**

From `confusion_matrix_test.png`, HGB makes noticeable errors in both directions:
- A large portion of Alert samples are predicted as Drowsy (false positives).
- Some Drowsy samples are predicted as **Alert** (false negatives).

Overall, the confusion matrix suggests the model is not strongly separable with the current feature set and split,
and the errors are relatively balanced across classes (no class is trivially solved).

**iii. Interpretability (Permutation Importance, HGB)**

**`mar_mean`** is the most influential feature (largestaccuracy drop when permuted). **`ear_mean_mean`** is thesecond most informative feature. The remaining featurescontribute little or are close to zero importance underthis evaluation

This aligns with the intuition that **mouth opening (MAR)** and **eye closure (EAR)** are key cues for drowsiness.

### 6.2 Robustness analysis via sweep (`results/experiments/` → `results/summary/`)

The robustness workflow evaluates how model performance changes when dataset-construction parameters vary
(e.g., `window_size`, blink/yawn thresholds, etc.). 

> Sweep size: 27 experiment settings (3×3×3×1 grid).

**Robustness conclusion:**
- `LinearSVM` has the **highest mean F1** across settings and the **highest win-rate** (most often the best model).
- `LogReg` is a close second (slightly lower mean/win-rate).
- `HGB` shows **higher variance** across settings: it can achieve strong peak performance in some configurations,
  but is less stable overall.
- `RF` is consistently lower across the tested sweep range.

**Best single observed configuration (upper bound within the tested grid):**\
**HGB**, at `ws60_be0.20_ym0.60_mf15` (check `all_models_all_exps.csv`)

**Conclusion:**
- If the goal is **robust performance across dataset definitions**, `LinearSVM` is the most stable choice in this sweep.
- If the goal is **best-case performance under a specific configuration**, `HGB` can reach the highest peak score within the tested parameter grid.
