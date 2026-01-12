# Environment Setup

In order to have a proper environment to execute tasks on the baobab server, a few things needs to be done

## Data on the server

Data and scripts needs to be stored at the same place to accelerate execution of tasks. The first thing to do is to clone the repository on your personal space on Baobab.

```bash
git clone https://github.com/hugo-varenne/ctlearn
```

Data needs also to be migrated in local. Processed data is already available on a shared directory and just needs to be copied in local, on your personal space (respecting the same architecture).

```bash
cp -r /srv/beegfs/scratch/shares/upeguipa/SST1M/processed_data/* local/repository/data/
```
Now, all the files needed are available on your personal space.

## Container for CTLearn

CTLearn having a lot of dependencies, we create a Singulatity container in order to install all these libraries. The first thing to do is to build the container with the YAML configuration file available on GitHub.

```bash
ml purge
module load GCCcore/13.3.0 cotainr
cotainr build ctlearnenv.sif --base-image=docker://ubuntu:22.04 --accept-licenses --conda-env=ctlearn.yml -v
```

It should take some times to build. Once this is done, you can rapidely test it by running a small python instruction.

```bash
apptainer exec ctlearnenv.sif python3 -c "import numpy; print(numpy.__version__)"
```