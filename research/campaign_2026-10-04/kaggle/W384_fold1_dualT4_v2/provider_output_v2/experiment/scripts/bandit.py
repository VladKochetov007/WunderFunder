"""Rank research arms by knowledge-gradient value of information per GPU hour (long-horizon bandit).

Gains stack across arms (an adopted improvement is kept whatever else is adopted), so each arm is valued against
"adopt nothing": V_now = max(0, mu). One pull with measurement noise sd_noise moves the posterior mean by
N(0, s~^2), s~ = sd^2 / sqrt(sd^2 + sd_noise^2); its value is KG = E[max(0, mu')] - max(0, mu) = s~ f(mu/s~) - max(0, mu),
f(z) = z Phi(z) + phi(z). Arms with a known gain (small sd) get KG ~ 0: exploit them late, they need no information.
    python scripts/bandit.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(z: float) -> float:
    return z * 0.5 * (1 + math.erf(z / math.sqrt(2))) + math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def knowledge_gradient(mu: float, sd: float, noise: float) -> float:
    if sd <= 0:
        return 0.0
    s = sd * sd / math.sqrt(sd * sd + noise * noise)
    return s * f(mu / s) - max(0.0, mu)


def main():
    spec = json.loads((ROOT / "research" / "bandit" / "arms.json").read_text())
    rows = []
    for name, a in spec["arms"].items():
        kg = knowledge_gradient(a["mu"], a["sd"], a.get("noise_sd", 4.0))
        rows.append((kg / max(a["cost_h"], 0.25), kg, name, a))
    print(f"{'arm':28s} {'KG/h':>6s} {'KG':>5s} {'mu':>5s} {'sd':>5s} {'noise':>5s} {'cost':>5s}  status")
    for idx, kg, name, a in sorted(rows, key=lambda r: r[0], reverse=True):
        print(f"{name:28s} {idx:6.2f} {kg:5.2f} {a['mu']:5.1f} {a['sd']:5.1f} {a.get('noise_sd', 4.0):5.1f} "
              f"{a['cost_h']:5.1f}  {a['status']}")


if __name__ == "__main__":
    main()
