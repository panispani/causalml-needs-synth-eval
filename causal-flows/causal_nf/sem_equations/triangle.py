import torch
import random

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
        elif sem_name.startswith("seg_linear_"):
            """
            "seg_linear_<interval_size>": piecewise linear noise around 0.5
            """
            # Parse interval_size from the name
            # Example: sem_name = "seg_linear_0.2" => interval_size=0.2
            interval_str = sem_name.replace("seg_linear_", "")
            interval_size = float(interval_str)

            # c1, c2 define the "flat" region around 0.5
            c1 = 0.5 - interval_size / 2.0
            c2 = 0.5 + interval_size / 2.0

            # g(u): segmented linear
            def seg_linear(u):
                """
                g(u) =
                  u,                  if u < c1,
                  c1,                 if c1 <= u < c2,
                  u - (c2 - c1),      if u >= c2.
                """
                if u < c1:
                    return u
                elif u < c2:
                    return c1
                else:
                    return u - (c2 - c1)

            # Helper to invert y = seg_linear(u).
            # We find all u in [0,1] such that seg_linear(u) = y.
            def seg_linear_inverse(y):
                """
                Returns one valid u in [0,1] chosen randomly among
                all solutions to seg_linear(u) = y.
                Raises an error if no solutions.
                """
                candidates = []

                # 1) If y < c1, then u = y is a direct solution if it is in [0,1].
                if y < c1 and 0.0 <= y <= 1.0:
                    candidates.append(y)

                # 2) If y == c1, then ANY u in [c1, c2) maps to c1.
                #    We intersect that with [0,1].
                eps = 1e-9
                if abs(y - c1) < eps:
                    lower = max(c1, 0.0)
                    upper = min(c2, 1.0)
                    if lower < upper:
                        # pick randomly from [lower, upper)
                        # (or you might pick from [lower, upper], up to you)
                        candidates.append(random.uniform(lower, upper))

                # 3) If y > c1, then we can solve u = y + (c2 - c1),
                #    must lie in [0,1].
                if y > c1:
                    u_candidate = y + (c2 - c1)
                    if 0.0 <= u_candidate <= 1.0:
                        candidates.append(u_candidate)

                # Remove duplicates, if any.
                candidates = list(set(candidates))

                if len(candidates) == 0:
                    raise ValueError(f"No solution u in [0,1] for seg_linear(u)={y}.")

                # Randomly pick one from the valid set
                return random.choice(candidates)

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
