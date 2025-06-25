# 🔁 Reproducibility Instructions

This document provides the instructions to get the results of the recbole models (FDSA and SASRec) for all experiments performed in our study.

---

## 🧱 Project Structure

```bash
recbole_baselines/
├── README.md            
├── requirements.txt
│
├── configs/                # Experiment configuration files
│   ├── FDSA_id_text_configs/
│   │   ├── recbole_baseline_config.yaml
│   │   ├── recbole_finetune_config.yaml
│   │   ├── run_baseline.job
│   │   └── run_finetune.job
│   ├── recbole_baseline_config.yaml
│   └── recbole_finetune_config.yaml
│
├── jobs/                   # SLURM job submission scripts
│   ├── cross_inference.job
│   ├── mind_baseline.job
│   ├── mind_finetune.job
│   ├── run_baseline.job
│   └── run_finetune.job
│
├── MIND/                   # MIND dataset and config-related scripts
│   ├── convert_mind_to_recbole.py
│   ├── cross_inference_FDSA.yaml
│   ├── cross_inference_SASRec.yaml
│   ├── mind_config_baseline_FDSA.yaml
│   ├── mind_config_baseline_SASRec.yaml
│   ├── mind_config_finetune_FDSA.yaml
│   ├── mind_config_finetune_SASRec.yaml
│   ├── mind_data_small/
│   └── mind_data_recbole_small/
│
├── python_scripts/         # Main training and conversion logic
│   ├── convert_data_to_recbole.py
│   ├── convert_finetune_data_to_recbole.py
│   ├── finetune_mind.py
│   ├── run_additional_metrics.py
│   └── run.py
│
└── saved_baselines/        # Folder for storing model checkpoints
    ├── SASRec-Jun-01-20....pth
    └── ...


---

## ⚙️ Environment Setup


Setup project by running the following commands:



```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd recformer_reproduction_repo
uv venv recformer_env --python 3.10
source recformer_env/bin/activate
uv pip sync requirements.txt
```

---

## 📂 Download & Prepare Datasets

We used two main sources for our datasets:

- **RecFormer GitHub repository**  
  [https://github.com/AaronHeee/RecFormer](https://github.com/AaronHeee/RecFormer)  
  We downloaded the [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) from this repository.  We used the scripts in `python_scripts' to convert them to the RecBole format.

- **Microsoft MIND Dataset**  
  [https://msnews.github.io/](https://msnews.github.io/)  
  We downloaded:
  - `MIND-small` training set for **pretraining**. specifically on eight slected category subsets (```autos, health, finance, foodanddrink, lifestyle, travel, video, weather```)
  - `MIND-large` training set for **fine-tuning**, specifically on two selected category subsets (```tv``` and ```music```)

### 🛠 Conversion to RecBole Format

To convert the raw MIND datasets into the RecBole-compatible format, we used the script:

```bash
python MIND/convert_mind_to_recbole.py \
    --input_dir MIND/mind_data_large \
    --output_dir MIND/mind_data_recbole_large \
    --pretrain_categories autos health finance foodanddrink lifestyle travel video weather \
    --finetune_categories tv music \
    --split_by_main_keep_sub
```

Before running the script, ensure the following files exist (retrieved from downloading the MIND training datasets):

```bash
MIND/mind_data_large/
├── behaviors.tsv
└── news.tsv
MIND/mind_data_small/
├── behaviors.tsv
└── news.tsv
```


Repeat this process for both the MIND-small and MIND-large dataset. This is required because:
- We pretrain on eight categories in the small dataset (```autos, health, finance, foodanddrink, lifestyle, travel, video, weather```)
- We fine-tune on selected category subsets from the large dataset (```tv``` and ```music```)

We provide the preprocessed recbole finetune and training data for the Amazon and MIND dataset in the folowing [drive](https://drive.google.com/drive/folders/1jj-ynTT8rhZD1yihf7csPTb6E8zRQI2X)

The pretrained baselines on both datasets can be found here:
|              Model              |
|:-------------------------------|
|[Amazon](https://drive.google.com/drive/u/1/folders/1wd0iPhlAgoxvOT3HGKPz1cX9Zpse4pEb)|
|[MIND](https://drive.google.com/drive/u/1/folders/1n_R6prbriDRkD9xLL_W4cVozzciMqCEt)|

---

## 🚀 Training

### Baselines Amazon

For SASRec run the following slurm jobs:

```bash
sbatch run_baseline.job 
sbatch run_finetune.job
```

For FDSA run the same jobs from the folder ```configs/FDSA_id_text_configs/```


The parameters we used are the ones you see in the job scripts and config files.

### Baselines MIND
Run the following slurm files from the ```jobs``` folder:
```bash
sbatch mind_baseline.job 
sbatch mind_finetune.job
```
To change for SASRec/ FDSA, simply change the ```config_file``` in the job files accordingly.

Please note that for finetuning you should set the correct pretrained model path and dataset according to which category you want to finetune the model on.


---

### Cross domain experiment
To reproduce our cross domain experiment, simply run the ```cross_inference.job``` file. To change for SASRec/ FDSA, again simply change the ```config_file``` in the job files accordingly.


## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- https://arxiv.org/pdf/2305.13731
- https://dl.acm.org/doi/pdf/10.1145/3534678.3539381 # the settings for the baselines are taken from this paper

