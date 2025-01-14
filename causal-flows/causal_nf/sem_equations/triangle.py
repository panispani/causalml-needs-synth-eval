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
                - If |y - c1| < eps => u is in [c1, c2), pick random in that interval
                - If y > c1 => u = y + (c2 - c1)

                We'll produce one chosen solution for each batch element y[i].
                """
                # Prepare output, same shape as y
                out = torch.empty_like(y)

                # (1) mask_low => y < c1 => u = y
                mask_low = y < c1
                out[mask_low] = y[mask_low]

                # (2) mask_eq => y ~ c1 => pick random in [c1, c2)
                mask_eq = (y - c1).abs() < eps
                num_eq = mask_eq.sum()
                if num_eq > 0:
                    lower = c1
                    upper = c2
                    # random in [lower, upper)
                    rand_vals = torch.rand(num_eq, device=y.device, dtype=y.dtype)
                    out[mask_eq] = lower + (upper - lower) * rand_vals

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
            #   f1(x1) = seg_linear(u1) => u1 in the set of solutions to seg_linear(u1)=x1
            inv_f1 = lambda x1: seg_linear_inverse(float(x1))

            #   f2(x1, x2) = 2*x1^2 + seg_linear(u2)
            #        => seg_linear(u2) = x2 - 2*x1^2
            def inv_f2(x1, x2):
                val = float(x2 - 2.0 * x1**2)
                return seg_linear_inverse(val)

            #   f3(x1, x2, x3) = 20/(1+exp(-(x2^2)+x1)) + seg_linear(u3)
            #        => seg_linear(u3) = x3 - 20/(1+exp(-(x2^2)+x1))
            def inv_f3(x1, x2, x3):
                val = float(x3 - 20.0 / (1.0 + torch.exp(-(x2**2) + x1)))
                return seg_linear_inverse(val)

            # Don't think that CausalNF should need this.
            # inverses = [inv_f1, inv_f2, inv_f3]
        elif sem_name.startswith("sinusoid_"):
            """
            "sinusoid_<f>": sinusoidal noise
            """
            # Parse frequency f
            freq_str = sem_name.replace("sinusoid_", "")
            f_freq = float(freq_str)

            # g(u) = sin(2 pi f_freq u)
            def sinusoid(u):
                return torch.sin(2.0 * math.pi * f_freq * u)

            # Helper: invert y = sin(2 pi f_freq u), find all solutions in [0,1].
            def sinusoid_inverse(y):
                """
                Solve sin(2 pi f_freq * u) = y for u in [0,1].
                We'll gather all solutions in [0,1], pick one at random.
                If no solutions, raise ValueError.
                """
                # If y not in [-1,1], no solutions
                if y < -1.0 or y > 1.0:
                    raise ValueError(
                        f"sinusoid_inverse: no solutions for y={y} outside [-1,1]."
                    )

                # arcsin(y) in [-pi/2, pi/2]
                alpha = math.asin(y)
                # general solutions: alpha + 2 pi k,  pi - alpha + 2 pi k
                #  => 2 pi f_freq u = alpha + 2 pi k   OR   2 pi f_freq u = pi - alpha + 2 pi k
                #  => u = [alpha + 2 pi k] / [2 pi f_freq] , etc.

                candidates = []
                # We test integer k in some range.  The frequency can be large, so be generarous when picking k
                # These are the general solutions basically.
                for k in range(-20, 21):
                    # first root: alpha + 2 pi k
                    u_candidate1 = (alpha + 2 * math.pi * k) / (2 * math.pi * f_freq)
                    # second root: (pi - alpha) + 2 pi k
                    u_candidate2 = (math.pi - alpha + 2 * math.pi * k) / (
                        2 * math.pi * f_freq
                    )

                    # check if within [0,1]
                    if 0.0 <= u_candidate1 <= 1.0:
                        candidates.append(u_candidate1)
                    if 0.0 <= u_candidate2 <= 1.0:
                        candidates.append(u_candidate2)

                # remove duplicates, sort if you like
                candidates = list(set(candidates))  # unique
                if len(candidates) == 0:
                    raise ValueError(
                        f"No solution u in [0,1] for sin(2 pi * {f_freq} * u)={y}."
                    )

                return random.choice(candidates)

            # The structural equations (non-linear) with sinusoidal noise:
            def f1(u1):
                return sinusoid(u1)

            def f2(x1, u2):
                return 2.0 * x1**2 + sinusoid(u2)

            def f3(x1, x2, u3):
                return 20.0 / (1.0 + torch.exp(-(x2**2) + x1)) + sinusoid(u3)

            functions = [f1, f2, f3]

            # Inverses
            inv_f1 = lambda x1: sinusoid_inverse(float(x1))

            def inv_f2(x1, x2):
                # y = x2 - 2*x1^2
                val = float(x2 - 2.0 * x1**2)
                return sinusoid_inverse(val)

            def inv_f3(x1, x2, x3):
                # y = x3 - 20/(1 + exp(-(x2^2) + x1))
                base = 20.0 / (1.0 + torch.exp(-(x2**2) + x1))
                val = float(x3 - base)
                return sinusoid_inverse(val)

            # I think causalNF shouldn't need this
            # inverses = [inv_f1, inv_f2, inv_f3]

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
