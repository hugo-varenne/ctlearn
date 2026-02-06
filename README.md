# CTLearn MLOps – Server Usage Guide

This repository provides a **modular training and evaluation framework** designed to run **exclusively on a server environment**.  
It supports multiple learning tasks (regression and classification) using **configuration-driven execution**.

⚠️ **The folder structure must be respected** for the pipeline to work correctly, as paths are resolved dynamically by the scripts. Changing file locations or names without updating the corresponding configs **will break execution**. It applies to the scripts (tasks and tools only) directory and the models directory (except custom template). It is strongly recommended for other folders to keep the structure in place for an easier overview and understanding of the environment.

It contains all information about the thesis and every scripts/code implemented (including exploratory notebooks).
---

## 🎯 Scope and Design Philosophy

- Each **task** (energy, type, sky direction, camera direction) is treated independently
- All experiments are:
  - configurable via YAML files
  - reproducible
  - executable via a single entry point
- No task-specific logic should be hardcoded in scripts

The repository is intentionally structured to:
- scale across multiple tasks
- allow fast experimentation
- support custom model extensions without modifying core logic
- compare different experiences

---

## 📁 Repository Structure

```text
.
│
├── configs/                  # Contains configuration
│   ├── energy.yaml           # Config for energy regression
│   ├── type.yaml             # Config for particle classification
│   ├── cameradirection.yaml  # Config for camera direction regression
│   └── skydirection.yaml     # Config for sky direction regression
│
├── scripts/ 
│   ├── train_model.py             # Training task
│   ├── test_model.py              # Testing task
│   ├── generate_report.py         # Report generation task (evaluation)         
│   ├── compare_models.py          # Comparison task (for same model task)
│   │
│   ├── tools/                     # Tools/Modified libraries files
│   │   ├── train_model.py         # Modified file from CTLearn library (training tool)
│   │   ├── reader.py              # Modified file from DL1_data_handler library (for multiprocessing purposes)        
│   │   └── predict_model.py       # Modified file from CTLearn library (prediction tool)
│   │
│   ├── rerun/                     # Scripts related to rerunning (try to solve Time Limit issue on SLURM)
│   │   └── .... 
│   │         
│   └── data_processing/           # Data processing scripts
│       ├── convert_data.py        # Convertion of data from .simtel.gz to .h5
│       ├── extract_only_images.py # Extract only images from .h5 files      
│       └── run_merge.py           # Merge files .h5 files
│   
├── notebooks/   # Files to experiment/analyze libraries, methodologies, ... (scratch implementation)
│   └── ...  
│
├── reports/        # Space for generated report
│   ├── energy/            
│   ├── type/             
│   ├── skydirection/        
│   ├── cameradirection/              
│   └── compare/ # Contains report dedicated to a comparison (comparison task)
|
├── models/   # Structure for generated models
│   ├── energy/            
│   ├── type/             
│   ├── skydirection/        
│   ├── cameradirection/              
│   └── custom_model_template.py   # Template for custom models
│
├── docs/
│   ├── env_setup.md          # Steps to set up env on the server
│   ├── report.pdf            # Documentation of the thesis
│   ├── poster.pdf            # Poster of the thesis
│   └── task_execution.md     # Steps to train (same for others tasks)
│
├── bash_template.sh          # Bash example to execute scripts
├── ctlearn.yml               # Environment configuration file
└── README.md
```

To prepare the environment, refer to [setup file](docs/env_setup.md).

To [execute a specific task](docs/task_execution.md), refer to the dedicated instructions.

---

Designed by Hugo Varenne

