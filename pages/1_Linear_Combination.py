import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import streamlit as st

st.set_page_config(page_title="Linear Combinations Demo", layout="wide")

LIM = 7.0
TOL = 1e-9

st.title("Linear combination of two vectors")
st.markdown("**u = α₁·u₁ + α₂·u₂**  — sliders se coefficients badlo aur dekho vector kaise banta hai.")

# ---------------- session state (so values persist across reruns) ----------------
if "u1" not in st.session_state:
    st.session_state.u1 = np.array([2.0, 1.0])
    st.session_state.u2 = np.array([-1.0, 2.0])
    st.session_state.target = np.array([4.0, 3.0])
    st.session_state.a1 = 1.0
    st.session_state.a2 = 1.0

col_plot, col_controls = st.columns([2, 1])

# ---------------- controls (right column) ----------------
with col_controls:
    st.subheader("Coefficients")
    a1 = st.slider("α₁", -5.0, 5.0, float(st.session_state.a1), 0.05)
    a2 = st.slider("α₂", -5.0, 5.0, float(st.session_state.a2), 0.05)
    st.session_state.a1, st.session_state.a2 = a1, a2

    st.subheader("Vectors")
    u1x = st.number_input("u₁ x", value=float(st.session_state.u1[0]))
    u1y = st.number_input("u₁ y", value=float(st.session_state.u1[1]))
    u2x = st.number_input("u₂ x", value=float(st.session_state.u2[0]))
    u2y = st.number_input("u₂ y", value=float(st.session_state.u2[1]))
    st.session_state.u1 = np.array([u1x, u1y])
    st.session_state.u2 = np.array([u2x, u2y])

    st.subheader("Target")
    tx = st.number_input("target x", value=float(st.session_state.target[0]))
    ty = st.number_input("target y", value=float(st.session_state.target[1]))
    st.session_state.target = np.array([tx, ty])

    show_span = st.checkbox("Shade the span", value=True)
    show_lattice = st.checkbox("Show integer lattice", value=False)

    def det():
        u1, u2 = st.session_state.u1, st.session_state.u2
        return u1[0] * u2[1] - u1[1] * u2[0]

    def independent():
        return abs(det()) > 1e-9

    def solve():
        u1, u2, target = st.session_state.u1, st.session_state.u2, st.session_state.target
        A = np.column_stack([u1, u2])
        if independent():
            return np.linalg.solve(A, target)
        d = u1 if np.linalg.norm(u1) > TOL else u2
        if np.linalg.norm(d) < TOL:
            return None
        cross = d[0] * target[1] - d[1] * target[0]
        if abs(cross) > 1e-9:
            return None
        return np.array([np.dot(target, d) / np.dot(d, d), 0.0])

    if st.button("Solve for α₁, α₂"):
        sol = solve()
        if sol is not None:
            st.session_state.a1 = float(np.clip(sol[0], -5, 5))
            st.session_state.a2 = float(np.clip(sol[1], -5, 5))
            st.rerun()

    if st.button("Make dependent"):
        st.session_state.u2 = -1.5 * st.session_state.u1
        st.rerun()

    if st.button("New random target"):
        st.session_state.target = np.round(np.random.uniform(-6, 6, 2), 1)
        st.rerun()

    if st.button("Reset"):
        st.session_state.u1 = np.array([2.0, 1.0])
        st.session_state.u2 = np.array([-1.0, 2.0])
        st.session_state.target = np.array([4.0, 3.0])
        st.session_state.a1 = 1.0
        st.session_state.a2 = 1.0
        st.rerun()

# ---------------- plot (left column) ----------------
with col_plot:
    u1, u2, target = st.session_state.u1, st.session_state.u2, st.session_state.target
    a1, a2 = st.session_state.a1, st.session_state.a2
    indep = independent()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dde3f2", linewidth=0.8)
    ax.axhline(0, color="#9aa2bc", linewidth=1.0)
    ax.axvline(0, color="#9aa2bc", linewidth=1.0)

    if show_span:
        if indep:
            ax.add_patch(plt.Rectangle((-LIM, -LIM), 2 * LIM, 2 * LIM,
                                        facecolor="#ffb627", alpha=0.10, zorder=0))
            ax.text(-LIM + 0.4, LIM - 0.9, "span{u1,u2} = the whole plane",
                    color="#8a6100", fontsize=11)
        else:
            d = u1 if np.linalg.norm(u1) > TOL else u2
            if np.linalg.norm(d) > TOL:
                d = d / np.linalg.norm(d) * (2 * LIM)
                ax.plot([-d[0], d[0]], [-d[1], d[1]], color="#ffb627",
                        linewidth=7, alpha=0.35, solid_capstyle="round", zorder=0)
            ax.text(-LIM + 0.4, LIM - 0.9, "span is only a line (1D subspace)",
                    color="#8a6100", fontsize=11)

    if show_lattice:
        pts = np.array([i * u1 + j * u2 for i in range(-4, 5) for j in range(-4, 5)])
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color="#ffb627", alpha=0.75, zorder=1)

    p, q = a1 * u1, a2 * u2
    if np.linalg.norm(p) > TOL and np.linalg.norm(q) > TOL:
        ax.add_patch(Polygon([[0, 0], p, p + q, q], closed=True,
                              facecolor="#d62728", alpha=0.07, edgecolor="none", zorder=1))

    def arrow(tail, vec, colour, alpha=1.0, ls="-"):
        if np.linalg.norm(vec) < 1e-12:
            return
        ax.annotate("", xy=tail + vec, xytext=tail,
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=2.2,
                                    linestyle=ls, alpha=alpha, shrinkA=0, shrinkB=0,
                                    mutation_scale=18))

    arrow(np.zeros(2), u1, "#1f77b4", alpha=0.35, ls="--")
    arrow(np.zeros(2), u2, "#2ca02c", alpha=0.35, ls="--")
    arrow(np.zeros(2), p, "#1f77b4")
    arrow(p, q, "#2ca02c")

    u = p + q
    arrow(np.zeros(2), u, "#d62728")

    ax.plot(target[0], target[1], "*", markersize=18, color="#9467bd")
    ax.annotate("target", target + np.array([0.3, 0.3]), color="#9467bd", fontsize=10, fontweight="bold")

    ax.set_title(f"u = {a1:.2f}·u1 + {a2:.2f}·u2 = ({u[0]:.2f}, {u[1]:.2f})", fontsize=12)
    st.pyplot(fig)

    # info panel
    err = np.linalg.norm(u - target)
    sol = solve()
    st.write(f"**determinant [u1 u2] = {det():.3f}**")
    if indep:
        st.success("Vectors are linearly INDEPENDENT — span = whole plane, every point reachable uniquely.")
    else:
        st.warning("Vectors are linearly DEPENDENT — span collapses to a line, most targets unreachable.")
    st.write(f"Distance from u to target: **{err:.4f}**")
    if err < 1e-3:
        st.write("🎯 **TARGET REACHED**")
    if sol is None:
        st.write("This target is **not in the span**: no α₁, α₂ exist.")
    else:
        st.write(f"Exact answer: α₁ = {sol[0]:.4f}, α₂ = {sol[1]:.4f}")
