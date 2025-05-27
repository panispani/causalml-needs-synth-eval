# Setup instructions for causalNF

1. Create new conda environment

```bash
conda create --name causalnf python=3.9.12 --no-default-packages
conda activate causalnf
```

2. Install necessary packages

```bash
# Don't put everything in a single pip install line
pip install torch==1.13.1 torchvision==0.14.1
pip install torch_geometric==2.3.1
pip install torch-scatter==2.1.1
# if this fails due to CUDA mismatch (e.g. "The detected CUDA version (12.6) mismatches the version that was used to compile PyTorch (11.7). Please make sure to use the same CUDA versions." when building torch-scatter), consider just using CPU version of torch depending on whether you really need GPU support, as follows:
# pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# Install the rest of the packages
pip install -r requirements.txt
```

3. Test run the code to ensure everything is setup correctly

3.1 Run tests

```bash
pytest --ignore=test_scm_plots.py --ignore=tests/test_german_preparator.py
# test_scm_plots require latex, test_german_preparator require you to download this dataset from: https://zenodo.org/records/10785677 and place it in "../Data/".
# Note that test_scm.py fails sometimes. You can re-run it individually with
pytest tests/test_scm.py
```

3.2 Run main just to see the code running

```bash
# If you are on Mac M1/M2 (and don't use Docker) this will fail because `causal_nf.yaml` uses `device: auto` which defaults to MPS which is known to have some compatibility issues with certain PyTorch operations (which are used in this project!)
# If you insist on not using Docker, manually change
# device: auto
# to
# device: cpu
# in `causal_nf/configs/causal_nf.yaml`
python main.py --config_file causal_nf/configs/causal_nf.yaml --wandb_mode disabled --project CAUSAL_NF
```

4. Reproduce Table 2 of the CausalNF paper

## CausalNF

```bash
# (a) Create the experiment jobs for CausalNF

# Note that if you have a GPU available you need to manually edit `base.yaml` from `device: [ cpu ]` to `device: [ cuda ]` - however I think the causalNF code doesn't handle this correctly e.g. in main.py the model needs to be moved to cuda + in many other places you will see RuntimeErrors of data being on different devices.
# What does 'generate_jobs.py' do:
# Will create jobs.sh with 90 batch jobs each having 4 runs to main, each is run sequentially. --grid_file specifies where you read the config from
# Neither jobs_per_file nor batch_size mean "How many files to generate" in contrast to what 'generate_jobs.py' claims with --help.
# The 'options' variable is a list of tuples, where each tuple will be the config for a job. If you want to create more or less jobs, edit the options. See helper.py::generate_options
# to run your own experiment create your own directory in place of causal-flows/grids/causal_nf/comparison_x_u with your own base and then base_scm.yaml
python generate_jobs.py --grid_file grids/causal_nf/comparison_x_u/base.yaml --format shell --jobs_per_file 20000 --batch_size 4

# (b) Running the jobs

# See what the previous command says about where the jobs scripts are, they are in a subdirectory <grid_file>/jobs_sh/jobs_1.sh
bash <grid_file>/jobs_sh/jobs_1.sh
bash grids/causal_nf/comparison_x_u/base/jobs_sh/jobs_1_test.sh # for testing if 1 job works
bash grids/causal_nf/comparison_x_u/base/jobs_sh/jobs_1.sh # but you'll need to wait until they terminate. Otherwise, to run and leave the remote machine running:
nohup bash grids/causal_nf/comparison_x_u/base/jobs_sh/jobs_1.sh > output_causal_nf_jobs.log 2>&1 &
# To monitor this job check the contents of `output_causal_nf_jobs.log` e.g.
watch -n1 tail output_causal_nf_jobs.log

# (c) Plotting table 2

# Note that if you didn't run everything (for example you only ran one experiment), you need to edit line 59 from `for exp_folder in exp_folders:` to `for exp_folder in exp_folders[:1]:`
# exp_folders should include all the folders you want (you need to add more if you add more experiments)
python scripts/create_comparison_flows.py
# Print the table
python -c "import pandas as pd; print(pd.read_pickle('results/dataframes/comparison_flows.pickle'))"
## Don't hide columns
python -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None); print(pd.read_pickle('results/dataframes/comparison_flows.pickle'))"
# If it's hard to read in the terminal (it will be), use `display_comparison_table.ipynb` (advised!)
```

## Repeat the same as above for VACA and CAREFL as specified in the README.md (or trust yourself and edit the commands of the previous section manually).

5. Run your own experiment

Add your own SCMs in `causal_nf/sem_equations`. All the SCMs for Experiment 1 live in `triangle.py`.

Create your own directory in place of `causal-flows/grids/causal_nf/comparison_x_u` (& experiment setup in yaml files)

The steps are then the same as before. 1) Generate the jobs 2) bash execute them 3) create comparison flows 4) Print the table

6. Reproducing Experiment 2 of the position paper

```bash
# When assumption violations don't affect performance

## generate sinusoid jobs
python generate_jobs.py --grid_file grids/causal_nf/robust_sinusoid/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
## run it, will save in "robust_sinusoid" directory
nohup bash grids/causal_nf/robust_sinusoid/base/jobs_sh/jobs_1.sh > output_causal_nf_robust_sinusoid_jobs.log 2>&1 &
## generate seg-linear jobs
python generate_jobs.py --grid_file grids/causal_nf/robustness/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
## run it, "robustness" directory
nohup bash grids/causal_nf/robustness/base/jobs_sh/jobs_1.sh > output_causal_nf_robustness_jobs.log 2>&1 &\
## This is part of RealCause. You need to manually edit the "exp_folders" variable in this python file to combine the dataframes/directories that you want (e.g. robust_sinusoid) and the output filename (e.g. comparison_flows_robust_sinusoid.pickle). By default this is "robust_sinusoid".
python scripts/create_comparison_flows.py
## to display the pickle file, see display_comparison_table.ipynb

# When assumption violations deteriorate performance

## generate assumption-violation jobs (ctf1, ctf2, ctf3, ctf4)
python generate_jobs.py --grid_file grids/causal_nf/robustness2/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
## run it
nohup bash grids/causal_nf/robustness2/base/jobs_sh/jobs_1.sh > output_causal_nf_robustness2_jobs.log 2>&1 &
## This is part of RealCause. You need to manually edit the "exp_folders" variable in this python file to combine the dataframes/directories that you want (e.g. robustness2 in this case) and the output filename (e.g., comparison_flows_robustnesss2.pickle). By default this is "robust_sinusoid" (previous experiment).
python scripts/create_comparison_flows.py
## to display the pickle file, see display_comparison_table.ipynb
```

## Misc

```bash
# Load a model and test only
python main.py --config_file grids/causal_nf/robustness/base/configs/1/config_1.yaml --wandb_mode offline --wandb_group robustness --project Test --load_model ./output_causal_nf/robustness/l6b0en36/
# Example run command
python main.py --config_file grids/causal_nf/experim/base/configs/1/config_1.yaml --wandb_mode offline --wandb_group experim --project Test
# See edited files
git diff --name-only ebc5fac..HEAD -- .
```

### Important files

- main.py
- sem_equations/triangle.py - where SCM are defined
- grids/causal_nf/comparison_x_u/base/configs/1/config_1.yaml - configs are defined here after job generation
- grids/causal_nf/comparison_x_u/base/scripts/batch_0.py - exact commands to run causalNF with a specific config after job generation
- causal_nf/modules/causal_nf.py [core]
- causal_nf/models/causal_nf.py [core]
- causal_nf/preparators/scm/scm_preparator.py [core]
- causal_nf/transforms/causal_transform.py [core]
- causal_nf/distributions/scm.py [core]
- intervention_and_counterfactual_logs.txt - to understand the kinds of queries
- scripts/create_comparison_flows.py - After you run training to create the result dataframe
- display_comparison_table.ipynb - results are displayed

### Files edited by us

- display_comparison_table.ipynb [new] - inspect all the experiment results
- generate_jobs.py
- results/dataframes/ [new] - all the experiment results
- intervention_and_counterfactual_logs.txt [new]
- grids/causal_nf/robustness [new]
- grids/causal_nf/robustness2 [new]
- grids/causal_nf/robust_sinusoid [new]
- grids/causal_nf/experim [new]
- triangle.py
