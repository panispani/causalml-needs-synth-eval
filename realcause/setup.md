# Setup instructions for RealCause

1. Setup a new conda environment

```bash
conda create --name realcause python=3.9.12 --no-default-packages
conda activate realcause
pip install -r requirements.txt
# Note that they use numpy1.x in this project, so you need to manually downgrade
pip install numpy==1.24.3
# For eval only:
pip install POT
pip install git+git://github.com/josipd/torch-two-sample
```

Setup was relatively easy, but there are no specific instructions about reproducing Table 3.

2. Train a generative model for IHDP

There doesn't seem to be a direct way to load the model. This can be easily implemented by I get different results from the NN in `nonlinear.py::_sample_y::280` before and after I load the model (even if I reset the random seed). Therefore, in an attempt to avoid debugging the saving and loading procedure, we do **all evaluation** in the train script, while the model is still in RAM. Also don't use `early_stop`, it doesn't mean what you think it means.

```bash
# Saves "model.pt", "args.txt" and "log.txt" in --saveroot
# --dist distribution of outcome variable
python train_generator.py --data "ihdp" \
    --saveroot "./ihdp_model" \
    --dist "FactorialGaussian" \
    --model_type "tarnet" \
    --n_hidden_layers 1 \
    --dim_h 128 \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 0.001 \
    --w_transform "Standardize" \
    --y_transform "Normalize" \
    --early_stop False \
    --patience 10 \
    --train True \
    --eval True
```

3. Evaluation

I've created a function `evaluate2` in train_generator. Call the normal `evaluate` function in `main` if you prefer. The new evaluate function computes what we care about in your experiments, the ATE bias and PEHE per realization (the noisy version). Both of these (`ate_bias_noisy` and `pehe_noisy`) are going to be in `summary.txt`. Evaluation is done with the test dataset (which I'm not sure this what what was done before). The results I get seem close enough with Table 3, except PEHE. I can't get a PEHE that bad, not sure what's wrong. I also noticed that in some file they use median instead of mean for PEHE - can't reproduce the huge PEHE for IHDP as in the paper.

Previous key metrics of summary.txt:

- ate_exact: True ATE computed from the generative model.
- ate_noisy: ATE computed with noisy outcomes.
- min_t_pval, min_y_pval: Minimum p-values from hypothesis tests for treatment and outcome distributions.
- q50_t_pval, q50_y_pval: Median p-values (50th percentile) for treatment and outcome.

Specifically the first two are in `train_generator::evaluate`

```python
summary.update(ate_exact=model.ate().item())
summary.update(ate_noisy=model.noisy_ate().item())
```

Example summary.txt:

```python
{
    "nll": 0.5395622849464417,
    "avg_t_pval": 0.4717732934725576,
    "avg_y_pval": 0.2741897185484338,
    "min_t_pval": 0.010850233644951465,
    "min_y_pval": 0.09180866752976602,
    "q30_t_pval": 0.26454555146321435,
    "q30_y_pval": 0.20342390100234867,
    "q50_t_pval": 0.46463541314657086,
    "q50_y_pval": 0.2563214099333352,
    "ate_exact": 4.226516701803807,
    "ate_noisy": 4.223727048638668
}
```

When trying to reproduce the results you should probably use the automation scripts `realism_experiment.py` and `non_id_experiment.py` rather than manually running `train_generator.py`. I include instructions on them in the next section (Misc).

## Misc

```bash
# Example run command
python train_generator.py --data "ihdp" \
    --saveroot "./ihdp_model" \
    --dist "FactorialGaussian" \
    --model_type "tarnet" \
    --n_hidden_layers 1 \
    --dim_h 64 \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 0.001 \
    --w_transform "Standardize" \
    --y_transform "Normalize" \
    --early_stop False \
    --patience 10 \
    --train True \
    --eval True \
    --realization_idx 3

# In reality you should probably use the following automation scripts
# For IHDP:
#   - Given a set of seeds
#   - Create a directory per seed
#   - With each seed, train separate generative model for each of the 100 realizations of IHDP
#   - Compute ATE bias and PEHE at the end
#   - Show global and per-seed mean and std of PEHE and ATE bias
# Note: this tests both different model initializations and different data: Within the same seed you have different data but the same model initialization. Across seeds you can have the same data but different model initializations but also different data and different model initializations. This is why it's valuable to both use different realizations and different seeds.
python realism_experiment.py --seeds 1 42 --out_dir ihdp_experiments
# For non-id experiment:
# For each seed run realcause and get a PEHE and ATE bias. Then aggregate over all seeds into a mean and std. Save all results and models in the --out_dir
python non_id_experiment.py --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 --out_dir non_id_experiments_3

# See edited files
git diff --name-only 82ccf1d..HEAD -- .
git diff 82ccf1d..HEAD -- data/ihdp.py
```

### Important files

- train_generator.py - entry point / main for training and evaluating generative models
- data/ihdp.py - code for downloading/loading and working with IHDP
- models/base.py [core]
- models/nonlinear.py [core]
- models/tarnet.py [core]
- generate_nonid_dataset.py - code for generating a dataset with non-id causal effects
- loading.py
- realism_experiment.py - code for running an experiment similar to table3, multiple seed, multiple realizations, aggregates results
- data/non_id.py - code to generate a non-id dataset in the IHDP format
- datasets/non_id_synthetic_data.npz - non-id generated dataset
- non_id_experiment.py code for running multiple non-id experiments and aggregating results
- ihdp_analysis.py - shows how IHDP looks like

### Files edited by us

- train_generator.py
- generate_nonid_dataset.py [new]
- realism_experiment.py [new]
- data/non_id.py [new]
- datasets/non_id_synthetic_data.npz [new]
- non_id_experiment.py [new]
- ihdp_analysis.py [new]
