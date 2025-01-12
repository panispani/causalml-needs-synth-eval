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

4. Reproduce Table 2

# CausalNF

```bash
# Create the experiments for CausalNF
# Note that if you have a GPU available you need to manually edit `base.yaml` from `device: [ cpu ]` to `device: [ cuda ]`
# What does 'generate_jobs.py' do:
: <<'EOF'
EOF
python generate_jobs.py --grid_file grids/causal_nf/comparison_x_u/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
```
