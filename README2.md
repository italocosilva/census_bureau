## Prerequisites

1. Conda

## Instructions

1. How to setup environment
    ```console
    conda create -n census_bureau --file requirements.txt python=3.8 -c conda-forge
    ```

2. How to train model
    ```console
    python model/train_model.py
    ```