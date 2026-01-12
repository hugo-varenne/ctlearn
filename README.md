# CTLearn MLOps – Server Usage Guide

This repository provides a **modular training and evaluation framework** designed to run **exclusively on a server environment**.  
It supports multiple learning tasks (regression and classification) using **configuration-driven execution**.

⚠️ **The folder structure must be respected** for the pipeline to work correctly, as paths are resolved dynamically by the scripts. Changing file locations or names without updating the corresponding configs **will break execution**.

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
├── configs/
│   ├── energy.yaml           # Config for energy regression
│   ├── type.yaml             # Config for particle classification
│   ├── cameradirection.yaml  # Config for camera direction regression
│   └── skydirection.yaml     # Config for sky direction regression
│
├── scripts/
│   ├── train_.py             # Training entry point
│   ├── evaluate.py

│
├── models/
│   ├── energy/            
│   ├── type/             
│   ├── skydirection/        
│   ├── energy/              
│   └── template_custom.py/   # Template for custom models
│
├── docs/
│   ├── env_setup.md          # Steps to set up env on the server
│   └── task_execution.md # Steps to train (same for others tasks)
│
├── run_training.sh           # Bash example to execute scripts
├── ctlearn.yml               # Environment configuration
└── README.md
```

To prepare the environment, refer to [setup file](docs/env_setup.md).

To [execute a specific task](docs/task_execution.md), refer to the dedicated instructions.

---

Designed by Hugo Varenne

