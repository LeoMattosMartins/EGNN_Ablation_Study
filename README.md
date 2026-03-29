# EGNN Ablation Archaeology: Charged Particle Dynamics

This project presents a systematic ablation study of Equivariant Graph Neural Networks (EGNNs). The research evaluates how various architectural components influence the prediction of long-horizon simulations for charged particles interacting via Coulomb forces.

## Project Overview
EGNNs are designed to bake physical symmetries—specifically E(n) transformations like translation, rotation, and reflection—directly into the model architecture. This study identifies which specific mechanisms are fundamental to maintaining these symmetries and which primarily serve as optimization aids.

## Key Findings
* **Critical Stability:** Removing the tanh bounding on coordinate updates leads to catastrophically exploding trajectories, resulting in the highest observed Mean Squared Error (26.0826).
* **Importance of Geometry:** Models without explicit distance metrics (egnn_no_distance.py) suffer because they cannot accurately represent the inverse-square nature of Coulomb's Law.
* **Equivariance vs. Accuracy:** Breaking explicit rotation and translation symmetry (egnn_no_equivariance.py) causes the Equivariance Error to jump by four orders of magnitude (1.17e-02), making generalization to rotated validation sets nearly impossible.
* **Performance Breakthrough:** Introducing an attention-weighted modification to edge passing (egnn_improved.py) significantly outperformed the baseline, reducing MSE to 0.0092.

---

## Experimental Results

The following table summarizes the performance of each variant after 100 training epochs:

| Model | Validation MSE | Equivariance Error↓ |
| :--- | :--- | :--- |
| **egnn_improved.py** | **0.0092** | 8.45e-07 |
| Full EGNN Baseline | 0.0766 | 8.30e-07 |
| egnn_no_residual.py | 0.0781 | 7.28e-07 |
| egnn_no_velocity.py | 0.0863 | 7.66e-07 |
| egnn_no_equivariance.py | 0.0903 | 1.17e-02 |
| egnn_no_distance.py | 0.1068 | 7.51e-07 |
| egnn_no_tanh.py | 26.0826 | 1.09e-04 |

---

## Ablation Variants
* **Full EGNN:** The complete baseline equivariant network.
* **No Equivariance:** Breaks symmetry by concatenating absolute position vectors into message scalars.
* **No Velocity:** Initializes velocity as zero, forcing the network to approximate momentum from positions alone.
* **No Distance:** Bypasses distance metrics; relies solely on discrete charge embeddings.
* **No Tanh:** Removes the bounded velocity shift predictions, leading to unbounded position corrections.
* **No Residual:** Removes skip connections, preventing features from accumulating sequentially.
* **Improved (Attention):** Uses softmax-scaled logits to introduce dynamic attention weights for localized neighbor updates.

## AI Collaboration & Attribution
This research was developed by Leonardo Mattos Martins (Boston University) using a pipeline of Gemini 3 models.
* **Gemini 3 Pro:** Assisted in architectural brainstorming for the attention mechanism.
* **Gemini 3 Deep Think:** Acted as a judge to select ablation candidates and identify equivariance risks.
* **Gemini 3.1-High (Antigravity IDE):** Facilitated rapid code implementation in Jupyter notebooks.