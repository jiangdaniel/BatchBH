## Setup

This code was developed using python 3.7.3. The code would likely run with anything as new as or newer than python 3.6, but this is untested. For the remainder of the README, `pip` refers to your pip installation for python3, and `python3` refers to your installation of python3.

The required packages are in `requirements.txt` and can be installed through pip with `pip install -r requirements.txt`.
Before installing the packages, you may want to create a virtual environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or another tool in order to avoid overwriting any of your current python packages.


## Usage

The following steps will generate/save the plots and print the tables to standard out.

1. Clone the repo.
2. Download the `creditcard.csv` file from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place the file in the top level directory of the repo.
3. Run `python creditcard.py [--n-cpus CPUS]`, where `CPUS` is the number of cpus you would like to use. If `--n-cpus` is not present, the code will use all available cpus.
4. Run `bash run.sh CPUS`, where `CPUS` is the number of cpus you would like to use. If `CPUS` is not specified, the code will use all available cpus.

### How to interpret outputs

The two tables that are printed to stdout are the two tables that appear in the paper for the credit card fraud experiments.

All of the plots that appear in the paper are saved to `out/imgs`. The plots can be interpreted as:
* `[mean3|mean0]_[bh|sbh|bbh|bsbh]_[pi1s|pi1_1|pi1_5].png`: `mean3` indicates the task where the alternative mean is 3, and `mean0` indicates that the task is where the alternative mean is random and centered at 0. `bh` indicates that the algorithm is BH, `sbh` indicates that the algorithm is Storey-BH, `bbh` indicates that the algorithms are `Batch BH` and `LORD`, and `bsbh` indicates that the algorithms are `Batch St-BH` and `SAFFRON`. `pi1s` indicates that the x-axis represents pi1, `pi1_1` indicates that pi1 = 0.1, and `pi1_5` indicates that pi1 = 0.5.
* `monotone_[mean3|mean0]_[bbh|bsbh].png`: These plots show the empirical percent of trials where an algorithm was monotone. `mean3` indicates the task where the alternative mean is 3, and `mean0` indicates that the task is where the alternative mean is random and centered 0. `bbh` indicates that the algorithms are `Batch BH` and `LORD`, and `bsbh` indicates that the algorithms are `Batch St-BH` and `SAFFRON`
* `rdiff[10|100|1000].png`: These plots show empirical values of R_t^+ - R_t for `Batch BH`.

### Troubleshooting

Depending on how python is installed on your system, you may have to edit `run.sh` to use your desired python installation.
