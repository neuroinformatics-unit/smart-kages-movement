# smart-kages-movement

Analysing data from [Smart-Kages](https://cambridgephenotyping.com/products)
using [movement](https://movement.neuroinformatics.com).

## Installation

First, create a conda environment with some required dependencies:

```bash
conda create -n smart-kages -c conda-forge python=3.13 pytables
conda activate smart-kages
```

Then, navigate to the root directory of the repository and install the package:

```bash
pip install -e .
```

To contribute, make sure to include the `dev` dependencies as well:

```bash
pip install -e '.[dev]'
```
