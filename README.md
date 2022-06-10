![pylint](/cov_utils/coverage.svg)

# KP index prediction research

## Prerequisites

To activate the environment

```shell
conda env create -f tf_env_kp.yml
```

To add git hooks
```shell
git config --local core.hooksPath .git_hooks/
```

## Running models

Default model configuration is set up at /vars/*.yaml
It is recommended that you copy configuration to a separate folder YOUR-FOLDER-NAME to avoid versioning

The extended YAML notation is used in configurations. One can use !include <file>.yaml statements to import a node from another file and ${VARIABLE} to access enviroment variable $VARIABLE.

The folliwing run files are available

*For sklearn compatible models:*

- run.py - to run all models and test them again predefined test set
- run_cv.py - to perform cross-validation on train set and test the best model against test set
- run_hld.py - to select best model with predefined val set and test it against test set

*For keras compatible models:*

- run_nn.py - to run all models and test them again predefined test set
- run_cv_nn.py - to perform cross-validation on train set and test the best model against test set
- run_hld_nn.py - to select best model with predefined val set and test it against test set

All run files have following arguments:

Required:

    --folder A folder to save the results

Optional:
    --vars Path to yaml file with config. It is recommended to use the option

    --data Path to data

    --model A specific model to run. If not specified, scripts run all models in configuration

    --conf Array. Enviroment variable to define in script (--conf "variable1=value1" "variable2=value2")

    --save_models Flag to save the model

## Examples
\
**1) Run all sklearn models, save results in $HOME/folder, also save all models**

```shell
python run.py --folder $HOME/folder --save_models
```

**2) Use environment variable in run file**

Recommended: Copy /vars/ folder to another location (etc. $HOME/folder/vars_init)

In $HOME/folder/vars_init/common/common.yaml

```yaml
backward_steps: $(BACKWARD_STEPS)
forward_steps: 8
random_state: 16
scale: false
shuffle: false
target: category
variables:
- Kp*10
- Dst
- B_x
- B_gsm_y
- B_gsm_z
- B_magn
- SW_spd
- H_den_SWP
- doySin
- hourSin
- doyCos
- hourCos
```
Then run run.py with following commands
```shell
python run.py --folder $HOME/folder --conf "BACKWARD_STEPS=3"
```

<!-- TODO -->

## Creating report

To generate report from folder run:
```shell
python generate_report.py --folder <YOUR-FOLDER-NAME> [--imp]
```
Required:

    --folder A folder to generate report from

Optional:

    --imp The flag to include feature importance report

The report will be available at YOUR-FOLDER-NAME/report
