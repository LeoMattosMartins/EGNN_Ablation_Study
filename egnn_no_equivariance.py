import jax
import jax.numpy as jnp
import flax.linen as nn

D_HIDDEN = 64
D_MSG = 32
N_LAYERS = 4


class EGNN(nn.Module):
    """E(3) Equivariant GNN with velocity for dynamics prediction.
    ABLATION: No equivariance."""

    @nn.compact
    def __call__(self, node_feat, pos, vel, edges):
        """
        Args:
            node_feat: [batch, N, D_feat] node features (charge type)
            pos:       [batch, N, 3] initial positions
            vel:       [batch, N, 3] initial velocities
            edges:     [batch, N, N] adjacency (1 = edge, 0 = no edge)

        Returns:
            predicted positions [batch, N, 3]
        """
        h = nn.Dense(D_HIDDEN)(node_feat)  # embed node features
        x = pos
        v = vel

        for _ in range(N_LAYERS):
            # --- Pairwise geometry ---
            rel = x[:, :, None, :] - x[:, None, :, :]  # [batch, N, N, 3]
            dist_sq = jnp.sum(rel**2, axis=-1, keepdims=True)  # [batch, N, N, 1]

            # --- Messages from sender features, receiver features, and distances ---
            # To break equivariance, we introduce a dependency on the absolute position x.
            # Rotating or translating x will now arbitrarily change the messages.
            m_ij = nn.relu(
                nn.Dense(D_MSG)(h)[:, :, None, :]  # sender:   h_i → [batch, N, 1, D]
                + nn.Dense(D_MSG)(h)[:, None, :, :]  # receiver: h_j → [batch, 1, N, D]
                + nn.Dense(D_MSG)(dist_sq)  # geometry: d²  → [batch, N, N, D]
                + nn.Dense(D_MSG)(x)[:, :, None, :]  # EQUIVARIANCE BROKEN: absolute pos
            )
            m_ij = m_ij * edges[:, :, :, None]  # zero out non-edges

            # --- Velocity update (learned acceleration) ---
            x_weight = jnp.tanh(nn.Dense(1)(m_ij))  # bounded scalar per edge
            coord_shift = (rel * x_weight * edges[:, :, :, None]).sum(axis=2)
            v = v + coord_shift

            # --- Feature update with residual ---
            agg = m_ij.sum(axis=2)  # aggregate messages
            h = h + nn.relu(nn.Dense(D_HIDDEN)(jnp.concatenate([h, agg], axis=-1)))

        # Integrate velocity → final positions
        x = x + v
        return x
