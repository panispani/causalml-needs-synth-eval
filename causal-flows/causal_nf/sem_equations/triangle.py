import torch
import random
import math

from causal_nf.sem_equations.sem_base import SEM


class Triangle(SEM):
    def __init__(self, sem_name):
        functions = None
        inverses = None
        if sem_name == "linear":
            functions = [
                lambda u1: u1 + 1.0,
                lambda x1, u2: 10 * x1 - u2,
                lambda x1, x2, u3: 0.5 * x2 + x1 + 1.0 * u3,
            ]
            inverses = [
                lambda x1: x1 - 1.0,
                lambda x1, x2: (10 * x1 - x2),
                lambda x1, x2, x3: (x3 - 0.5 * x2 - x1) / 1.0,
            ]
        elif sem_name == "non-linear":
            functions = [
                lambda u1: u1,
                lambda x1, u2: 2 * x1**2 + u2,
                lambda x1, x2, u3: 20.0 / (1 + torch.exp(-(x2**2) + x1)) + u3,
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: x2 - 2 * x1**2,
                lambda x1, x2, x3: x3 - 20.0 / (1 + torch.exp(-(x2**2) + x1)),
            ]
        elif sem_name == "non-linear-2":
            functions = [
                lambda u1: torch.sigmoid(u1),
                lambda x1, u2: 10 * x1**0.5 - u2,
                lambda x1, x2, u3: 0.5 * x2 + 1 / (1.0 + x1) + 1.0 * u3,
            ]
            inverses = [
                lambda x1: torch.logit(x1),
                lambda x1, x2: (10 * x1**0.5 - x2),
                lambda x1, x2, x3: (x3 - 0.5 * x2 - +1 / (1.0 + x1)) / 1.0,
            ]
        # -------------------------------------------------------
        # ctf1
        # x1 = u1
        # x2 = { u2,    if x1 >= 0
        #       u2 - 1, if x1 < 0 }
        # x3 = u3 + x2
        # -------------------------------------------------------
        elif sem_name == "ctf1":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.where(x1 >= 0, u2, u2 - 1),
                lambda x1, x2, u3: x2 + u3,
            ]
            inverses = [
                # Given x1 => u1 = x1
                lambda x1: x1,
                # Given x1, x2 => solve for u2:
                #    if x1 >= 0: x2 = u2 => u2 = x2
                #    else:       x2 = u2 - 1 => u2 = x2 + 1
                lambda x1, x2: torch.where(x1 >= 0, x2, x2 + 1),
                # Given x2, x3 => x3 = x2 + u3 => u3 = x3 - x2
                lambda x1, x2, x3: x3 - x2,
            ]

        # -------------------------------------------------------
        # ctf2
        # x1 = u1
        # x2 = { u2,    if x1 >= 0
        #       -u2,    if x1 < 0 }
        # x3 = u3 + x2
        # -------------------------------------------------------
        elif sem_name == "ctf2":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.where(x1 >= 0, u2, -u2),
                lambda x1, x2, u3: x2 + u3,
            ]
            inverses = [
                # Given x1 => u1 = x1
                lambda x1: x1,
                # Given x1, x2 => solve for u2:
                #    if x1 >= 0: x2 = u2 => u2 = x2
                #    else:       x2 = -u2 => u2 = -x2
                lambda x1, x2: torch.where(x1 >= 0, x2, -x2),
                # Given x2, x3 => x3 = x2 + u3 => u3 = x3 - x2
                lambda x1, x2, x3: x3 - x2,
            ]
        # -------------------------------------------------------
        # ctf3
        # x1 = u1
        # x2 = { u2,    if x1 >= 0
        #       u2 - 1, if x1 < 0 }
        # x3 = { u3,    if x2 >= 0
        #       u3 - 1, if x2 < 0 }
        # -------------------------------------------------------
        elif sem_name == "ctf3":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.where(x1 >= 0, u2, u2 - 1),
                lambda x1, x2, u3: torch.where(x2 >= 0, u3, u3 - 1),
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: torch.where(x1 >= 0, x2, x2 + 1),
                lambda x1, x2, x3: torch.where(x2 >= 0, x3, x3 + 1),
            ]

        # -------------------------------------------------------
        # ctf4
        # x1 = u1
        # x2 = { u2,    if x1 >= 0
        #       -u2,    if x1 < 0 }
        # x3 = { u3,    if x2 >= 0
        #       -u3,    if x2 < 0 }
        # -------------------------------------------------------
        elif sem_name == "ctf4":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.where(x1 >= 0, u2, -u2),
                lambda x1, x2, u3: torch.where(x2 >= 0, u3, -u3),
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: torch.where(x1 >= 0, x2, -x2),
                lambda x1, x2, x3: torch.where(x2 >= 0, x3, -x3),
            ]

        # -------------------------------------------------------
        # ctf5
        # x1 = u1
        # x2 = { u2,    if x1 >= 0
        #       u2 - 1, if x1 < 0 }
        # x3 = { u3,    if x2 >= 0
        #       -u3,    if x2 < 0 }
        # -------------------------------------------------------
        elif sem_name == "ctf5":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.where(x1 >= 0, u2, u2 - 1),
                lambda x1, x2, u3: torch.where(x2 >= 0, u3, -u3),
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: torch.where(x1 >= 0, x2, x2 + 1),
                lambda x1, x2, x3: torch.where(x2 >= 0, x3, -x3),
            ]

        # -------------------------------------------------------
        # ctf6
        # x1 = u1
        # x2 = { u2,    if x1 >= 0
        #       -u2,    if x1 < 0 }
        # x3 = { u3,    if x2 >= 0
        #       u3 - 1, if x2 < 0 }
        # -------------------------------------------------------
        elif sem_name == "ctf6":
            functions = [
                lambda u1: u1,
                lambda x1, u2: torch.where(x1 >= 0, u2, -u2),
                lambda x1, x2, u3: torch.where(x2 >= 0, u3, u3 - 1),
            ]
            inverses = [
                lambda x1: x1,
                lambda x1, x2: torch.where(x1 >= 0, x2, -x2),
                lambda x1, x2, x3: torch.where(x2 >= 0, x3, x3 + 1),
            ]
        elif sem_name.startswith("seg-linear-"):
            """
            "seg-linear-<interval_size>": piecewise linear noise around 0.5,
            applied to the NON-LINEAR structural equations:
                f1(u) = seg_linear(u)
                f2(x1, u) = 2*x1^2 + seg_linear(u)
                f3(x1, x2, u) = 20/(1 + exp(-(x2^2) + x1)) + seg_linear(u)
            """
            # Parse interval_size from the name
            # Example: sem_name = "seg-linear-0.2" => interval_size=0.2
            interval_str = sem_name.replace("seg-linear-", "")
            interval_size = float(interval_str)

            # c1, c2 define the "flat" region around 0.5
            c1 = 0.5 - interval_size / 2.0
            c2 = 0.5 + interval_size / 2.0

            # g(u): segmented linear
            def seg_linear(u):
                """
                g(u) =  u,                  if u < c1
                        c1,                 if c1 <= u < c2
                        u - (c2 - c1),      if u >= c2
                for each element in tensor u.
                """
                out = u.clone()  # copy so we don’t modify in-place

                # mask where u is in [c1, c2)
                mask_middle = (u >= c1) & (u < c2)
                out[mask_middle] = c1

                # mask where u is >= c2
                mask_upper = u >= c2
                out[mask_upper] = u[mask_upper] - (c2 - c1)

                # else, out remains = u (for u < c1)
                return out

            # Helper to invert y = seg_linear(u).
            # We find all u in [0,1] such that seg_linear(u) = y.
            def seg_linear_inverse_vec(y, c1, c2, eps=1e-9):
                """
                Vectorized inverse of seg_linear:
                We find a single valid u for each element of y.

                - If y < c1 => u = y
                - If |y - c1| < eps => u is in [c1, c2), pick the lowest point c1 in the interval (used to be random)
                - If y > c1 => u = y + (c2 - c1)

                We'll produce one chosen solution for each batch element y[i].
                """
                # Prepare output, same shape as y
                out = torch.empty_like(y)

                # (1) mask_low => y < c1 => u = y
                mask_low = y < c1
                out[mask_low] = y[mask_low]

                # (2) mask_eq => y ~ c1 => pick "random" in [c1, c2) - now instead of random, pick lowest
                mask_eq = (y - c1).abs() < eps
                num_eq = mask_eq.sum()
                if num_eq > 0:
                    lower = c1
                    upper = c2
                    # random in [lower, upper)
                    # rand_vals = torch.rand(num_eq, device=y.device, dtype=y.dtype)
                    # out[mask_eq] = lower + (upper - lower) * rand_vals
                    # Set to the lowest point c1 instead of doing it at random.
                    out[mask_eq] = lower

                # (3) mask_high => y > c1 => u = y + (c2 - c1)
                mask_high = y > c1
                out[mask_high] = y[mask_high] + (c2 - c1)

                return out

            # The structural equations (non-linear):
            def f1(u1):
                return seg_linear(u1)

            def f2(x1, u2):
                return 2.0 * x1**2 + seg_linear(u2)

            def f3(x1, x2, u3):
                return 20.0 / (1.0 + torch.exp(-(x2**2) + x1)) + seg_linear(u3)

            functions = [f1, f2, f3]

            # Now their inverses:
            #   f1(x1) = seg_linear(u1) => seg_linear(u1) = x1 => u1 in [0,1]
            def inv_f1(x1):
                # x1 is a Tensor => we do a vector call
                return seg_linear_inverse_vec(x1, c1, c2)

            def inv_f2(x1, x2):
                # seg_linear(u2) = x2 - 2*x1^2
                val = x2 - 2.0 * x1**2
                return seg_linear_inverse_vec(val, c1, c2)

            def inv_f3(x1, x2, x3):
                base = 20.0 / (1.0 + torch.exp(-(x2**2) + x1))
                val = x3 - base
                return seg_linear_inverse_vec(val, c1, c2)

            # Don't think that CausalNF should need this.
            inverses = [inv_f1, inv_f2, inv_f3]
            # inverses = [None, None, None]  # Required by SEM(ABC)

        elif sem_name.startswith("sinusoid-"):
            """
            "sinusoid-<f>": sinusoidal noise
            """
            # Parse frequency f
            freq_str = sem_name.replace("sinusoid-", "")
            f_freq = float(freq_str)

            # g(u) = sin(2 pi f_freq u)
            def sinusoid(u):
                # pi_tensor = torch.tensor(math.pi)  # Constant as a tensor
                # f_freq_tensor = torch.tensor(f_freq)  # Constant as a tensor
                # return torch.sin(2.0 * pi_tensor * f_freq_tensor * u)
                return torch.sin(2.0 * math.pi * f_freq * u)

            # The structural equations (non-linear) with sinusoidal noise:
            def f1(u1):
                return sinusoid(u1)

            def f2(x1, u2):
                return 2.0 * x1**2 + sinusoid(u2)

            def f3(x1, x2, u3):
                return 20.0 / (1.0 + torch.exp(-(x2**2) + x1)) + sinusoid(u3)

            functions = [f1, f2, f3]

            # Helper: invert y = sin(2 pi f_freq u)
            def old_sinusoid_inverse_scalar(y_elem, f_freq):
                """
                Solve sin(2 pi f_freq * u) = y_elem for real u.
                We'll gather solutions for k in [-20..20],
                then pick one at random. This yields a random real solution,
                since there's an infinite number in theory.
                """
                # Must have y in [-1,1] or no real solution:
                if y_elem < -1.0 or y_elem > 1.0:
                    # No real solutions
                    raise "NONONONO"
                    return float("nan")  # or raise ValueError

                alpha = math.asin(y_elem)
                candidates = []
                for k in range(-20, 21):
                    # root1: alpha + 2 pi k => u = (alpha + 2 pi k)/(2 pi f_freq)
                    u1 = (alpha + 2.0 * math.pi * k) / (2.0 * math.pi * f_freq)
                    candidates.append(u1)
                    # root2: pi - alpha + 2 pi k => u = (pi - alpha + 2 pi k)/(2 pi f_freq)
                    beta = math.pi - alpha
                    u2 = (beta + 2.0 * math.pi * k) / (2.0 * math.pi * f_freq)
                    candidates.append(u2)

                # pick a random solution among the unique ones
                candidates = list(set(candidates))
                if not candidates:
                    raise ("NONONONO")
                    return float("nan")
                return random.choice(candidates)

            def old_sinusoid_inverse_vec(y, f_freq):
                """
                Vectorized approach: For each y[i], we find
                sin(2 pi f_freq * u)=y[i] solutions and pick one at random.
                Return a Tensor of the same shape as y.
                """
                out = torch.empty_like(y)
                y_np = y.detach().cpu().numpy()
                for i in range(len(y_np)):
                    out[i] = old_sinusoid_inverse_scalar(y_np[i], f_freq)
                return out.to(y.device)

            def sinusoid_inverse_vec(y, f_freq):
                """
                Vectorized approach: For each y[i], solve sin(2 pi f_freq * u) = y[i].
                We'll gather solutions for k in [-20..20], then pick one at random.

                This is a purely torch-based version that avoids numpy and math.asin,
                instead using torch.asin and broadcasting.
                """
                # 1) y must be in [-1, 1], or else asin is not real
                #    Let's clamp or raise an error if needed
                if torch.any(y < -1.0) or torch.any(y > 1.0):
                    torch.set_printoptions(threshold=100_000)
                    # # print(y)
                    # for i in range(y.shape[0]):
                    #     if y[i] > 1 or y[i] < -1:
                    #         print(y[i])
                    raise ValueError(
                        "Input out of domain [-1, 1]. Cannot invert sin()."
                    )

                # 2) Compute alpha = arcsin(y).
                #    This is a tensor, same shape as y, with requires_grad=True if y does.
                alpha = torch.asin(y)  # shape (N,)

                # 3) Create the integer k values as a 1D tensor: [-20, ..., 20]
                k_values = torch.arange(
                    -20, 21, device=y.device, dtype=y.dtype
                )  # shape (41,)

                # We'll broadcast alpha to shape (N, 1)
                alpha_expanded = alpha.unsqueeze(-1)  # shape (N, 1)

                # 4) root1 = (alpha + 2 pi k) / (2 pi f_freq)
                #    root2 = (pi - alpha + 2 pi k) / (2 pi f_freq)
                two_pi = 2.0 * math.pi
                numerator1 = alpha_expanded + two_pi * k_values  # shape (N, 41)
                root1 = numerator1 / (two_pi * f_freq)  # shape (N, 41)

                beta = math.pi - alpha_expanded  # shape (N, 1)
                numerator2 = beta + two_pi * k_values  # shape (N, 41)
                root2 = numerator2 / (two_pi * f_freq)  # shape (N, 41)

                # 5) Combine them into one big (N, 82) tensor of candidate solutions
                candidates = torch.cat([root1, root2], dim=-1)  # shape (N, 82)

                # 6) Randomly pick exactly one solution from these 82 for each batch element
                #    (If you want a deterministic branch, you'd pick, for example, k=0 root1.)
                num_candidates = candidates.shape[1]  # 82
                N = candidates.shape[0]

                # Generate random indices in [0, num_candidates)
                # shape (N,)
                random_indices = torch.randint(
                    low=0, high=num_candidates, size=(N,), device=y.device
                )

                # Gather the chosen solutions => shape (N,)
                # This is done with advanced indexing:
                idx_rows = torch.arange(N, device=y.device)
                out = candidates[idx_rows, random_indices]

                return out

            def inv_f1(x1):
                return sinusoid_inverse_vec(x1, f_freq)

            def inv_f2(x1, x2):
                val = x2 - 2.0 * x1**2
                return sinusoid_inverse_vec(val, f_freq)

            def inv_f3(x1, x2, x3):
                base = 20.0 / (1.0 + torch.exp(-(x2**2) + x1))
                val = x3 - base
                return sinusoid_inverse_vec(val, f_freq)

            # I think causalNF shouldn't need this
            inverses = [inv_f1, inv_f2, inv_f3]
            # inverses = [None, None, None]

        super().__init__(functions, inverses, sem_name)

    def adjacency(self, add_diag=False):
        adj = torch.zeros((3, 3))

        adj[0, :] = torch.tensor([0, 0, 0])
        adj[1, :] = torch.tensor([1, 0, 0])
        adj[2, :] = torch.tensor([1, 1, 0])
        if add_diag:
            adj += torch.eye(3)

        return adj

    def intervention_index_list(self):
        return [0, 1]
