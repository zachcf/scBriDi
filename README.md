# scBriDi: an RNA-centered bridging alignment and cross-modal generative framework for single-cell multi-omics


---

## About The Project

Recent advances in single-cell technologies allow us to see how individual cells differ in great detail by measuring multiple omics layers at once. However, measuring all these layers together in the exact same cell is hard to apply widely because the experiments are highly complex, expensive, and often produce incomplete data. Consequently, there is a critical need for computational frameworks capable of bridging disparate modalities to synthesize comprehensive multi-omic views without relying on exhaustive paired measurements. Here, we present scBriDi, a diffusion-based generative framework built on an RNA-centered Bridging Alignment strategy for scalable integration and biologically faithful generation across diverse single-cell omics. The framework leverages a pretrained, fixed RNA encoder as a semantic anchor and progressively aligns additional modalities via transferable contrastive learning. To preserve modality-specific signals and cellular-state topology while avoiding the common pitfall of over-smoothing, we introduce a hierarchical constraint scheme comprising RNA-centered unidirectional contrastive loss, within-modality consistency regularization, and cross-modal deep clustering. Across benchmarks, scBriDi consistently outperforms state-of-the-art methods on cross-modal generation, unsupervised clustering, and cross-batch integration. We further extend the framework to spatial transcriptomics by introducing an adaptive coordinate assignment strategy, enabling high-accuracy alignment between scRNA-seq and spatial data. scBriDi reduces reliance on fully paired training data, supports knowledge transfer across biological contexts, and facilitates spatially constrained analyses. Collectively, scBriDi offers a highly scalable framework for cross-modal generation, facilitating the downstream exploration of cellular heterogeneity and complex regulatory mechanisms.
![Alt text](./data/scBriDi.png?raw=true "scBriDi")
---

## Built With

- Python 3.11.7
- PyTorch 2.3.1
- scanpy / anndata / episcanpy
- scikit-learn / scipy / numpy
- einops / timm
- tqdm

---

## Getting Started

```

### Installation

```bash
pip install torch scanpy anndata episcanpy scikit-learn scipy numpy einops timm tqdm
```

---




### Tutorials

| Notebook | Description |
|----------|-------------|
| `tutorial/BridgingAlignment.ipynb` | Two-stage bridging alignment (RNA–ATAC → RNA–ADT) with cross-modal generation |
| `tutorial/CoordinateAssignment.ipynb` | Spatial coordinate assignment for scRNA-seq cells |

---


## License

Distributed under the MIT License.

---

## Citation

If you find scBriDi useful, please cite:

> scBriDi: an RNA-centered bridging alignment and cross-modal generative framework for single-cell multi-omics

---

## Acknowledgments

- [scanpy](https://scanpy.readthedocs.io/), [anndata](https://anndata.readthedocs.io/), and [episcanpy](https://github.com/colomemaria/epiScanpy) for single-cell data handling
