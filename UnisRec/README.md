# Reproducing and Extending UniSRec

This document provides a comprehensive guide to reproducing the UniSRec model's performance. It covers:
1.  Setting up the environment from scratch.
2.  **Part A: Reproducing the original paper's results** by fine-tuning the provided pre-trained model on the included Amazon `Scientific` and `Instruments` datasets.
3.  **Part B: Adapting the framework** to preprocess and fine-tune on a new, custom dataset (e.g., a Music or TV dataset in a specific JSON format).
4.  Calculating both standard accuracy metrics (NDCG, Recall) and advanced metrics (Gini, ILD, Coverage).

This guide assumes access to an HPC cluster with a Slurm job scheduler. Adapt HPC-specific commands (`module load`, `sbatch`) as needed for your specific environment.

## 1. Initial Setup (One-Time)

### 1.1. Clone This Repository
Navigate to your project space and clone this repository.

```bash
# Example path: /scratch-shared/your_username/
cd /path/to/your/project_space 
git clone <URL_of_your_GitHub_repository>
cd UniSRec # Or your repository name
```

You are now in the main project directory. All necessary data and scripts are included.

### 1.2. Create and Configure Python Virtual Environment

It's crucial to use a virtual environment to manage dependencies.

```bash
# Ensure you are in the project's root directory (e.g., UniSRec/)

# 1. (HPC specific) Purge any currently loaded system modules for a clean start
# module purge # Uncomment if on HPC

# 2. (HPC specific) Load necessary system modules 
# module load 2023 Python/3.11.3-GCCcore-12.3.0 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 
# Adjust module names and versions for your HPC or skip if running locally with conda/pip.
# Ensure Python >= 3.9.7, PyTorch >= 1.11.0, CUDA >= 11.3.1 if using GPU.

# 3. Create a Python virtual environment
python3 -m venv venv_unisrec # Use python3 explicitly if needed

# 4. Activate the virtual environment
source venv_unisrec/bin/activate
# Your command prompt should now start with (venv_unisrec)

# 5. Install required Python packages
pip install --upgrade pip
pip install "recbole>=1.1.1" pandas numpy tqdm transformers
pip install lightgbm
pip install xgboost
```

Note: You must activate the virtual environment (`source venv_unisrec/bin/activate`) every time you start a new terminal session to work on this project.

---

## PART A: Fine-tuning on Provided Amazon Datasets

This section describes how to use the authors' included pre-trained model and processed downstream datasets.

### A.1. Prepare Data

Download the pre-trained model and place it at `saved/UniSRec-FHCKM-300.pth` Then donwload the downstream datasets (Instruments and Scientific for the reproduction) and place then in `dataset/downstream/`. 
### A.2. Configure Evaluation Metrics

Ensure the fine-tuning configuration calculates all desired accuracy metrics. Edit `props/finetune.yaml`:

```bash
nano props/finetune.yaml
```

Verify the metrics line includes Recall, MRR, HIT, and NDCG. The relevant section should look similar to this:

```yaml
# ... (other settings in finetune.yaml) ...
topk: [10, 50]
metrics: [Recall, MRR, HIT, NDCG] # Ensure Recall and MRR are listed
valid_metric: NDCG@10 
# ... (other settings) ...
```

### A.3. Run Fine-tuning Jobs

Use the provided job scripts to fine-tune on the Scientific and Instruments datasets.

```bash
# IMPORTANT: Before submitting, edit the scripts to set the correct PROJECT_DIR path
# and adjust SBATCH directives (partition, time, mem) for your cluster.

sbatch run_finetune_scientific.job
sbatch run_finetune_instruments.job
```

### A.4. Collect Results

- **Accuracy Metrics**: After each job completes, check the corresponding `.out` file in `job_output/`. Find the `test result:` line to get the NDCG, Recall, MRR, and Hit Rate scores.
- **Save Model Path**: Note the path to the best fine-tuned model saved in the `saved/` directory (e.g., `saved/UniSRec-Scientific-[timestamp].pth`).

for gini run gini.job

---

## PART B: Preprocessing and Fine-tuning on a New Custom Dataset

This section describes how to process a new dataset (provided in a generic JSON format) and then fine-tune a model on it. The pre trained model on MIND can be found at `saved\UniSRec-smallMIND.pth` and should be loaded as such

### B.1. Prepare Raw Custom Data

Place your new dataset's raw files (e.g., `meta_data.json`, `train.json`, `val.json`, and an optional `item_id_internal_to_original.json` map) into a new directory inside `dataset/raw/`, e.g.:

```
dataset/raw/NewMusicData/
```

### B.2. Preprocess the Custom Data

This is a two-step process using the scripts in `dataset/preprocessing/`.

#### Step B.2.1: (If needed) Create Item ID Translation Map

If the item IDs in your `train.json` (e.g., `"0"`, `"1"`) are different from the keys in `meta_data.json` (e.g., `"Nxxxxx"`), run `create_item_translation_map.py`:

```bash
python dataset/preprocessing/create_item_translation_map.py \
    --meta_data_filepath dataset/raw/NewMusicData/meta_data.json \
    --session_files_dir dataset/raw/NewMusicData/ \
    --output_map_filepath dataset/raw/NewMusicData/item_id_internal_to_original.json
```

#### Step B.2.2: Convert JSON to UniSRec Format

Use the provided `run_full_preprocess_new_music.job` (or similar) to run `process_generic_json_sessions.py`:

```bash
# Edit run_full_preprocess_new_music.job to ensure paths are correct, then submit:
sbatch run_full_preprocess_new_music.job
```

After this job completes, the processed data for fine-tuning will be in:

```
dataset/downstream/NewMusicFineTune/
```

### B.3. Fine-tune on the New Custom Dataset

This is similar to Step A.3. Use a job script like `run_finetune_new_music.job`.

Example command inside the script:

```bash
python finetune.py \
    -d NewMusicFineTune \
    -p saved/UniSRec-smallMIND.pth \
    train_stage transductive_ft
```

Submit this job:

```bash
sbatch run_finetune_new_music.job
```

---

## 5. Advanced Metrics Evaluation (Gini, ILD, Coverage)

After any fine-tuning run, a `.pth` model file is saved. Use it with `evaluate_advanced_metrics.py` to get diversity and coverage metrics.

Similar process for the TV finetuning

### 5.1. The Evaluation Script

The `evaluate_advanced_metrics.py` script calculates:
- Gini@10
- ILD@10
- Catalog Coverage@10

### 5.2. Run the Advanced Metrics Evaluation

Use `run_evaluate_advanced.job`:

```bash
nano run_evaluate_advanced.job
```

Set the `MODEL_TO_EVALUATE` variable to the exact path of your fine-tuned model, e.g.:

```
saved/UniSRec-NewMusicFineTune-[timestamp].pth
```

Then submit the job:

```bash
sbatch run_evaluate_advanced.job
```

Check the output in the corresponding `.out` file in `job_output/` for Gini, ILD, and Coverage scores.
