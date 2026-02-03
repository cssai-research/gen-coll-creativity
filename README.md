# From Semantic Memory to Collective Creativity — Analysis Notebook

This repository contains the analysis code used to generate the main figures and reported statistics for the paper *From semantic memory to collective creativity: A generative cognitive foundation for social creativity models*. The work implements a socio-cognitive agent-based model in which collective creativity emerges from generative cognitive mechanisms rather than from exogenously assigned traits.

The core artifact in this repository is:

* **`analysis.ipynb`** — an end-to-end Jupyter notebook that reproduces the paper’s quantitative analyses, figures, and metrics.

---

## What the notebook produces (mapped to paper figures)

The notebook is organized into four conceptual sections, each corresponding to results reported in the paper:

1. **Figure 1:** Relationship between Watts–Strogatz rewiring probability (p) and semantic modularity (Q).
2. **Figure 2:** Emergent individual creativity — ideational breadth as a function of semantic modularity under a fixed random-walk retrieval process.
3. **Figure 3:** Dyadic cognitive stimulation — pre-interaction overlap predicting marginal gain, estimated via two-way fixed effects with clustered standard errors.
4. **Figure 4:** Network-level redundancy — shared versus independent inspiration sources in a matched triadic design.

---

## Environment setup

The notebook is compatible with standard scientific Python stacks.

### Option A: `pip`

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install numpy pandas scipy matplotlib statsmodels networkx tqdm
```

### Option B: `conda` (recommended)

```bash
conda create -n cogsci-creativity python=3.10 -y
conda activate cogsci-creativity
pip install numpy pandas scipy matplotlib statsmodels networkx tqdm
```

---

## How to run

1. Place required input files in the `data/` directory.
2. Open `analysis.ipynb` in Jupyter Notebook or VS Code.
3. Run cells sequentially from top to bottom.

All figures and metrics will be written to the `outputs/` directory.

---

## Reproducibility notes

* All simulations and sampling procedures use explicit random seeds where applicable.
* **Figure 3 (stimulation)** is estimated using a two-way fixed-effects specification with:

  * ordered-pair fixed effects,
  * prompt fixed effects,
  * standard errors clustered by ordered pair.
* **Figure 4 (redundancy)** uses matched triad–control contrasts with uncertainty estimated via clustering by inspiration source.

Because cognitive retrieval dynamics are held fixed, all reported effects arise from semantic structure and interaction topology rather than from tuned agent strategies.

---

## Citation

If you use or adapt this code, please cite the associated preprint

> *From semantic memory to collective creativity: A generative cognitive foundation for social creativity models* (arxiv).

---

## License

* **MIT License** 

