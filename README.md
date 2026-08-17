# A Comparative Study of Deep Q-Network Variants Across Diverse Gymnasium Environments

An empirical comparison of three DQN variants under standardised training conditions, measuring
sample efficiency, convergence stability, and Q-value overestimation across three environments
with different reward structures.

**Samiya Ali Zaidi · Javeria Azfar · Huzaifah Tariq Ahmed**
School of Science and Engineering, Habib University, Karachi

📄 **[Full report (PDF)](RL_Final_Report.pdf)**

---

## Motivation

Most published DQN benchmarks are run on high-compute Atari suites, which says little about how
these algorithms behave in practical, resource-constrained settings. The variants below have each
been studied individually, but rarely compared against one another under identical conditions.

This study fixes the training setup and varies only the algorithm, so that differences in outcome
are attributable to the variant rather than to tuning.

## What was compared

| Variant | Mechanism |
|---|---|
| **Double DQN** | Decouples action selection from evaluation to reduce overestimation bias |
| **Duelling DQN** | Separates state-value and action-advantage estimation |
| **Elastic Step DQN (ES-DQN)** | Adjusts the multi-step update horizon dynamically via state-similarity clustering |

**Environments:** `CartPole-v1` · `LunarLander-v3` · `FrozenLake-v1`

Chosen to differ in complexity, reward sparsity, and transition dynamics. FrozenLake in particular
provides the sparse-reward case that separates the variants most clearly.

**Metrics:** normalised area under the learning curve (sample efficiency) · average return ·
standard deviation and coefficient of variation (stability) · mean maximum Q-value (overestimation).

**Training setup:** Adam, learning rate 0.001, batch size 32, ε annealed 1.0 → 0.01 over 10,000
steps for CartPole and LunarLander and 500 steps for FrozenLake, reflecting its sparser rewards.

---

## Results

### Sample efficiency (normalised AUC %)

| Environment | Elastic Step | Duelling | Double |
|---|---|---|---|
| CartPole | 3.51% | **43.62%** | 37.96% |
| LunarLander | −32.19% | **8.87%** | 3.02% |
| FrozenLake | **47.20%** | 15.70% | 39.50% |

### Average return and stability

| Environment | Variant | Mean return | Std. dev. | CV |
|---|---|---|---|---|
| CartPole | Elastic Step | 16.66 | 11.37 | 0.68 |
| | Duelling | 218.63 | 172.88 | 0.79 |
| | Double | 76.10 | 80.20 | 1.05 |
| LunarLander | Elastic Step | −64.70 | 153.83 | −2.38 |
| | Duelling | 35.69 | 153.00 | 4.29 |
| | Double | 9.20 | 176.40 | 19.20 |
| FrozenLake | Elastic Step | 0.47 | 0.50 | 1.06 |
| | Duelling | 0.16 | 0.36 | 2.31 |
| | Double | 0.40 | 0.49 | 1.24 |

### Finding

**No variant dominated.** Rankings inverted across environments rather than holding:

- **Duelling DQN** converged fastest on CartPole (43.62% AUC) and was the only variant to approach
  optimal performance there.
- **Elastic Step DQN** recorded the highest AUC on sparse-reward FrozenLake (47.20%), though driven
  by intermittent reward spikes rather than steady policy improvement, and destabilised badly on
  LunarLander (−32.19%), indicating unstable early learning.
- **Double DQN** was never the strongest on any environment but was never the worst either.

The practical implication is that architectural improvements to DQN do not transfer uniformly, and
that variant selection is environment-dependent rather than a matter of picking the most recent
method.

Per-environment learning curves for each variant are in [`Implementation/`](Implementation/).

---

## Repository contents

```
RL_Final_Report.pdf              Full write-up: method, results, analysis, limitations
RL_Proposal.pdf                  Initial project proposal
RL Project Literature Review.docx  Survey of prior work on DQN variants
gymnasium_guide.ipynb            Notes on the Gymnasium API used throughout
Environment Setup/               Environment configuration notebook
Implementation/                  Per-variant notebooks and per-environment result plots
```

## Running the notebooks

```bash
python -m venv venv && source venv/bin/activate
pip install gymnasium[box2d] torch numpy matplotlib
jupyter notebook
```

`LunarLander` requires the Box2D extras. Each implementation notebook is self-contained and can be
run independently.

## Limitations

Reported in full in the report, and worth stating here: single seed per configuration, so the
stability figures describe one run rather than a distribution across seeds; a fixed network
architecture across variants; and no hyperparameter search per variant, which is deliberate — the
point was to hold the setup constant — but means each variant may be some distance from its own best
configuration.

## License

MIT. See [LICENSE](LICENSE).
