# Task Execution Guide

This document explains **how to run a task** in this repository and **which files are involved**.  
All tasks (training and evaluation) follow the same execution logic and differ **only by configuration**.
More details can be found in the [report](docs/report.pdf) of the thesis

---

## 🧩 Files You Need to Care About

For any task, execution relies on **four key components**:

1. **Configuration file** (`configs/*.yaml`)
2. **Model Location** (`models/<task>/`)
3. **Python entry point** (`scripts/train_model.py`, `scripts/test_model.py`, etc.)
4. **Execution script** (`*.sh`)

Understanding how these files link together is essential.

---

## 1️⃣ Configuration File (`configs/*.yaml`)

Each task has a dedicated YAML configuration file.

Example:
```text
configs/energy.yaml
```

Configurations files are available for each task and can be manipulated with attributes included. When you want to configure a model, you only need to change inside the prepare model section. You can eventually add new attributes but needs to be careful about [CTLearn](https://github.com/ctlearn-project/ctlearn) syntax and structure.

```bash
prepare_model:
  model_type: 'ResNet' # ['SingleCNN', 'ResNet', 'LoadedModel']
  tasks: ['energy'] # ['type', 'energy', 'cameradirection', 'skydirection']
  input_shape: [96, 96, 2] # Shape to determine
  num_classes: 1 # 2 for type classification, 1 for energy and direction regression
  temp_dir: '{models_path}/models/energy/temp/' # Place to store temporary model (intermediate steps)
  CTLearnModel: # For ResNet Models
    attention_mechanism: null
    attention_reduction_ratio: null
```

The second element that can be modified is related to the training process. You provide the data, the training parameters like the number of epochs or the batch size and you can precise the mode of execution (stereo or mono).

```bash
training_model:
  TrainCTLearnModel:
    output_dir: "{models_path}/models/energy/cnn_batch64/" # Where model is saved
    input_dir_signal: "{data_path}/gammas_diffuse/train/" # Gamma values to use
    file_pattern_signal: ["gamma_*.h5"]
    model_type: 'LoadedModel' # ['SingleCNN', 'ResNet', 'LoadedModel'] (same as in prepare_model)
    reco_tasks: 'energy' # ['type', 'energy', 'direction'] (same as in prepare_model)
    n_epochs: 100
    batch_size: 64
    overwrite: true
    quiet: false
    percentage_per_epoch: 1.0
    log_level: "DEBUG"
#    pruning_model: # Pruning settings to use
#      initial_sparsity: 0.50
#      final_sparsity: 0.90
#      begin_step: 0
    early_stopping:
      monitor: 'val_loss'
      patience: 7
      verbose: 1
      min_delta: 0.001
      restore_best_weights: True
      start_from_epoch: 5
    stack_telescope_images: false # True if stereo mode
    DLImageReader:
      mode: "mono" # ["mono", "stereo"] (Stereo uses images from multiple telescopes 

  CTLearnModel: 
    attention_mechanism: null
    attention_reduction_ratio: null
    load_model_from: {temp_model_path} # Where model is stored
```

There is a last step dedicated to the testing part with the testing data referenced. Also there is the mode used for CTLearn compliance purposes.

```bash
testing_model:
  data_path: {data_path}  # Path to data
  PredictCTLearnModel:
    stack_telescope_images: false # True if stero mode
    DLImageReader:
      mode: "mono" # Same as training
```

Each script has variations depending on the task but globally with comments inside, it shouldn't be an issue to execute task. Be aware that the mode cannot be changed for cameradirection and skydirection task.


## 2️⃣ Model Location (`models/<task>/`)

For **reusability** and a **cleaner repository**, a model structure is proposed to separate the model by tasks. This structure is a **requirement** when the script to **compare models** is used. It will search inside a task folder to compare all models inside. You need to reference it in the configuration file and in parameters of python script.

## 3️⃣ Python Entry point (`scripts/train_model.py`, `scripts/test_model.py`, etc.)

The python entry point are the core execution of a specific task. Each task has a dedicated python script to execute properly. Theses scripts are called by a bash execution script to comply with the SLURM resource manager on the Baobab cluster. The arguments required by the scripts can be found at the end of each of them.

## 4️⃣ Execution script (`*.sh`)

The bash execution scripts are the way to allocate resources to a task and execute it. In order to make it work, a few things needs to be understand. As mentionned earlier, the cluster uses SLURM as a resource manager. It means you need to provide some parameters to SLURM so it can allocate correctly resources necessary for the task you want to execute.

```bash
#!/bin/bash
#SBATCH --job-name=training_test
#SBATCH --time=12:00:00
#SBATCH --partition=shared-gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --mem=48GB
```

Then, you need to provide your python script with the arguments required so it can run correctly. The only thing you need to modify is which script you wanna run and the arguments with it (example here for the training task.)

```bash
apptainer exec --nv \
  --bind /usr/local/cuda/targets/x86_64-linux:/usr/local/cuda/targets/x86_64-linux \
  --bind /usr/local/cuda/lib64:/usr/local/cuda/lib64 \
  ctlearnenv.sif bash -c '
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
python3 scripts/train_model.py --yaml_file {config_path}
'
```

When you want to execute a file, you can directly provide it to the SLURM management with this kind of command.

```bash
sbatch script.sh
```