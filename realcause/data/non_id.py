import numpy as np

SPLIT_OPTIONS = {"train", "test", "all"}


def load_non_id_synthetic(
    split="all",
    observe_counterfactuals=False,
    return_ites=False,
    return_ate=False,
    file_path="datasets/non_id_synthetic_data.npz",
    seed=42,
):
    """
    Load a single instance of the synthetic "non-identifiable" dataset.

    The .npz file has the following keys:
        - 'x'   : Covariates shape (N, d)
        - 't'   : Treatment assignment shape (N,)
        - 'yF'  : Observed (factual) outcome shape (N,)
        - 'ycf' : Counterfactual outcome shape (N,)
        - 'mu0' : Noiseless potential outcome if T=0 (N,)
        - 'mu1' : Noiseless potential outcome if T=1 (N,)
        - 'ate' : True ATE (shape (1,)) or a scalar

    :param split: 'train', 'test', or 'all' (like IHDP; default='all')
    :param observe_counterfactuals: if True, return a doubled dataset with
        both factual and counterfactual outcomes "observed" (for debugging).
    :param return_ites: if True, return a key 'ites' = mu1 - mu0
    :param return_ate: if True, return a key 'ate' = mean(ites)
    :param file_path: path to the single .npz file
    :param seed: random seed for the train/test split (if split != 'all')
    :return: Dictionary similar to the IHDP format:
        {
          "w":  covariates (N, d),
          "t":  treatment (N,),
          "y":  outcomes (N,),
          "ites": ites array (N, ),
          "ate": scalar or shape (1,)
        }
        If observe_counterfactuals=True, N doubles and t flips in the second half.
    """

    # --------------------------------------------------------
    # 1) Load the dataset from disk
    # --------------------------------------------------------
    data = np.load(file_path)
    x_all = data["x"]  # shape (N, d)
    t_all = data["t"]  # shape (N,)
    yF_all = data["yF"]  # factual outcome, shape (N,)
    ycf_all = data["ycf"]  # counterfactual outcome, shape (N,)
    mu0_all = data["mu0"]  # shape (N,)
    mu1_all = data["mu1"]  # shape (N,)
    ate_all = data["ate"]  # scalar (the true ATE for entire dataset)

    # For convenience, convert ate_all to scalar
    if hasattr(ate_all, "shape") and len(ate_all.shape) == 1 and ate_all.shape[0] == 1:
        ate_all = ate_all.item()

    # We can define ites = mu1 - mu0 for each individual
    ites_all = mu1_all - mu0_all  # shape (N,)

    # --------------------------------------------------------
    # 2) Split into train/test if split != 'all'
    #    Random 90/10 split.
    # --------------------------------------------------------
    N = x_all.shape[0]
    if split not in SPLIT_OPTIONS:
        raise ValueError(f'Invalid split="{split}", choose from {SPLIT_OPTIONS}.')

    if split == "all":
        # Use entire dataset, no splitting
        idx = np.arange(N)
    else:
        # TODO: unused and maybe wrong.
        rng = np.random.default_rng(seed=seed)
        perm = rng.permutation(N)
        train_size = int(0.9 * N)
        if split == "train":
            idx = perm[:train_size]
        else:  # split == "test"
            idx = perm[train_size:]

    # Subset all arrays
    x = x_all[idx]
    t = t_all[idx]
    yF = yF_all[idx]
    ycf = ycf_all[idx]
    mu0 = mu0_all[idx]
    mu1 = mu1_all[idx]
    ites = ites_all[idx]

    # --------------------------------------------------------
    # 3) Optionally "observe" counterfactuals by doubling
    #    the dataset (like IHDP does)
    # --------------------------------------------------------
    # If observe_counterfactuals=True, we create a double-sized dataset:
    #  - first half is the factual, second half is the counterfactual
    #  - T flips in the second half, Y is set to ycf in the second half
    if observe_counterfactuals:
        # Double w
        w_expanded = np.vstack([x, x.copy()])
        # T flips in the second half: if original t=1, second half t=0, etc.
        t_expanded = np.concatenate([t, 1 - t])
        # Y is factual for first half, counterfactual for second
        y_expanded = np.concatenate([yF, ycf])
        # ITEs: we can just duplicate (though typically it's the same for both halves)
        ites_expanded = np.concatenate([ites, ites.copy()])
    else:
        w_expanded = x
        t_expanded = t
        y_expanded = yF
        ites_expanded = ites

    # --------------------------------------------------------
    # 4) Build the final dictionary
    # --------------------------------------------------------
    d = {}
    d["w"] = w_expanded  # covariates
    d["t"] = t_expanded  # treatment
    d["y"] = y_expanded  # observed outcome(s)
    if return_ites:
        d["ites"] = ites_expanded
    if return_ate:
        # IHDP doesn't return the global ATE but the subset-level ATE.
        # We do the same here but it doesn't matter, they are both equal to 2.0 if you use the whole dataset.
        d["ate"] = np.mean(ites_expanded)

    return d
