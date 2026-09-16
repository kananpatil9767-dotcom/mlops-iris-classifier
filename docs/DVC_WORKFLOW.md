# DVC Workflow

## DVC Remote

The project uses a local DVC remote named `myremote`.

## Dataset Versioning Workflow

The basic workflow is:

dvc add → git add → git commit → dvc push

- `dvc add` tracks the dataset with DVC.
- `git add` stages the DVC metadata file.
- `git commit` versions the metadata using Git.
- `dvc push` uploads the dataset to the DVC remote.

## Comparing Dataset Versions

The command:

dvc diff <commit>

is used to compare the current dataset with a previous Git commit.

## Restoring Dataset Versions

To restore a dataset version:

git checkout <commit> -- data/raw/iris_v1.csv.dvc

followed by:

dvc checkout data/raw/iris_v1.csv.dvc

This restores the actual dataset corresponding to the selected DVC metadata.