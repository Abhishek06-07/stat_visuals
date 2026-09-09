import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)
import streamlit as st

st.set_page_config(page_title="Vector Space vs Subspace", layout="wide")

C_U, C_V = "#1f77b4", "#2ca02c"
C_GEN_U, C_GEN_V = "#8a6100", "#4b6b1f"
C_SUM, C_SCAL, C_SET = "#d62728", "#ff7f0e", "#ffb627"
C_OK, C_BAD = "#1f7a5c", "#b33a3a"

LIM = 4.0
TOL = 1e-9
SHIFT = np.array([0.0, 0.0, 2.0])

SETS = ["R3  (whole space)", "plane through 0", "line through 0",
        "{0}  zero subspace", "shifted plane", "first octant"]

P1 = np.array([1.0, 0.5, 0.0])
P2 = np.array([0.0, 1.0, 1.0])

GENERATORS = {
    SETS[0]: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
              np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0])),
    SETS[1]: (P1, P2, P1 + P2, P1 - P2),
    SETS[2]: (P1, 2.0 * P1, -1.0 * P1, 0.5 * P1),
    SETS[3]: (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)),
    SETS[4]: (P1, P2, P1 + P2, P1 - P2),
    SETS[5]: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.5]),
              np.array([0.0, 0.0, 1.0]), np.array([0.5, 1.0, 0.0])),
}

st.title("Vector space versus vector subspace")
st.markdown("**A subspace is a subset you can never leave by adding or by scaling.**")

if "set_name" not in st.session_state:
    st.session_state.set_name = SETS[1]
    st.session_state.a1, st.session_state.a2 = 1.0, 0.5
    st.session_state.b1, st.session_state.b2 = -0.5, 1.0
    st.session_state.k = 2.0

col_plot, col_controls = st.columns([2, 1])

with col_controls:
    st.subheader("Choose a subset of R3")
    set_name = st.radio("Subset", SETS, index=SETS.index(st.session_state.set_name), label_visibility="collapsed")
    st.session_state.set_name = set_name

    st.markdown("**u = a1 u1 + a2 u2**")
    a1 = st.slider("a1", -2.0, 2.0, float(st.session_state.a1), 0.05)
    a2 = st.slider("a2", -2.0, 2.0, float(st.session_state.a2), 0.05)

    st.markdown("**v = b1 v1 + b2 v2**")
    b1 = st.slider("b1", -2.0, 2.0, float(st.session_state.b1), 0.05)
    b2 = st.slider("b2", -2.0, 2.0, float(st.session_state.b2), 0.05)

    st.markdown("**scalar k (for k.u)**")
    k = st.slider("k", -3.0, 3.0, float(st.session_state.k), 0.05)

    st.session_state.a1, st.session_state.a2 = a1, a2
    st.session_state.b1, st.session_state.b2 = b1, b2
    st.session_state.k = k

    c1, c2 = st.columns(2)
    if c1.button("Reset"):
        st.session_state.a1, st.session_state.a2 = 1.0, 0.5
        st.session_state.b1, st.session_state.b2 = -0.5, 1.0
        st.session_state.k = 2.0
        st.rerun()
    if c2.button("Try k = -1"):
        st.session_state.k = -1.0
        st.rerun()

def generators():
    return GENERATORS[st.session_state.set_name]

def base_point():
    return SHIFT if st.session_state.set_name == SETS[4] else np.zeros(3)

def vector_u():
    u1, u2, _, _ = generators()
    a1, a2 = st.session_state.a1, st.session_state.a2
    if st.session_state.set_name == SETS[5]:
        a1, a2 = abs(a1), abs(a2)
    return base_point() + a1 * u1 + a2 * u2

def vector_v():
    _, _, v1, v2 = generators()
    b1, b2 = st.session_state.b1, st.session_state.b2
    if st.session_state.set_name == SETS[5]:
        b1, b2 = abs(b1), abs(b2)
    return base_point() + b1 * v1 + b2 * v2

def inside(x):
    n = st.session_state.set_name
    if n == SETS[0]:
        return True
    if n == SETS[1]:
        return abs(np.dot(np.cross(P1, P2), x)) < 1e-8
    if n == SETS[2]:
        return np.linalg.norm(np.cross(P1, x)) < 1e-8
    if n == SETS[3]:
        return np.linalg.norm(x) < 1e-8
    if n == SETS[4]:
        nrm = np.cross(P1, P2)
        return abs(np.dot(nrm, x) - np.dot(nrm, SHIFT)) < 1e-8
    return bool(np.all(x >= -1e-12))

def contains_zero():
    return inside(np.zeros(3))

def is_subspace():
    return st.session_state.set_name in (SETS[0], SETS[1], SETS[2], SETS[3])

with col_plot:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    for d in (np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])):
        e = d * LIM
        ax.plot([-e[0], e[0]], [-e[1], e[1]], [-e[2], e[2]], color="#9aa2bc", linewidth=0.9, alpha=0.7)

    n = st.session_state.set_name
    g = np.linspace(-2.2, 2.2, 12)
    if n in (SETS[1], SETS[4]):
        S, T = np.meshgrid(g, g)
        base = base_point()
        X = base[0] + S * P1[0] + T * P2[0]
        Y = base[1] + S * P1[1] + T * P2[1]
        Z = base[2] + S * P1[2] + T * P2[2]
        ax.plot_surface(X, Y, Z, color=C_SET, alpha=0.22, edgecolor="none")
    elif n == SETS[2]:
        e = 2.6 * P1
        ax.plot([-e[0], e[0]], [-e[1], e[1]], [-e[2], e[2]], color=C_SET, linewidth=6, alpha=0.45, solid_capstyle="round")
    elif n == SETS[3]:
        ax.scatter([0], [0], [0], color=C_SET, s=180, alpha=0.8)
    elif n == SETS[5]:
        q = np.linspace(0, LIM * 0.8, 8)
        A, B = np.meshgrid(q, q)
        Zr = np.zeros_like(A)
        ax.plot_surface(A, B, Zr, color=C_SET, alpha=0.16, edgecolor="none")
        ax.plot_surface(A, Zr, B, color=C_SET, alpha=0.16, edgecolor="none")
        ax.plot_surface(Zr, A, B, color=C_SET, alpha=0.16, edgecolor="none")

    def arrow(vec, colour, label, ls="-", lw=2.4, base=None):
        base = np.zeros(3) if base is None else base
        if np.linalg.norm(vec) < 1e-9:
            ax.scatter([base[0]], [base[1]], [base[2]], color=colour, s=45, zorder=5)
            return
        tip = base + vec
        ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], color=colour, linewidth=lw, linestyle=ls, zorder=5)
        ax.scatter([tip[0]], [tip[1]], [tip[2]], color=colour, s=32, zorder=6)
        if label:
            ax.text(tip[0], tip[1], tip[2] + 0.18, label, color=colour, fontsize=11.5, fontweight="bold", zorder=7)

    if n != SETS[3]:
        u1, u2, v1, v2 = generators()
        base = base_point()
        for gv, lab, col in ((u1, "u1", C_GEN_U), (u2, "u2", C_GEN_U), (v1, "v1", C_GEN_V), (v2, "v2", C_GEN_V)):
            if np.linalg.norm(gv) < 1e-9:
                continue
            tip = base + gv
            ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], color=col, linewidth=1.5, linestyle="--", alpha=0.85, zorder=4)
            ax.text(tip[0], tip[1], tip[2] + 0.10, lab, color=col, fontsize=9.5, zorder=7)

    u, v = vector_u(), vector_v()
    total = u + v
    scal = st.session_state.k * u

    arrow(u, C_U, "u")
    arrow(v, C_V, None, ls=":", lw=1.4, base=u)
    arrow(v, C_V, "v")
    col_sum = C_OK if inside(total) else C_SUM
    arrow(total, col_sum, "u + v", lw=3.0)
    col_scal = C_SCAL if inside(scal) else C_BAD
    arrow(scal, col_scal, "k.u", ls="--", lw=2.0)

    ax.scatter([0], [0], [0], color="#1e2761", s=30, zorder=8)

    kind = "SUBSPACE" if is_subspace() else "NOT a subspace"
    ax.set_title(f"{n}   -   {kind}", fontsize=13, color="#1e2761", pad=6)

    st.pyplot(fig)

    tick = lambda ok: "YES" if ok else "NO"
    st.markdown(f"""
**u** = ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})  &nbsp;&nbsp; **v** = ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})

**u + v** = ({total[0]:.2f}, {total[1]:.2f}, {total[2]:.2f}) &nbsp;&nbsp; **k.u** = ({scal[0]:.2f}, {scal[1]:.2f}, {scal[2]:.2f})
""")

    st.subheader("The three subspace tests")
    st.write(f"1. Contains the zero vector - **{tick(contains_zero())}**")
    st.write(f"2. Closed under addition - **{tick(inside(total))}**")
    st.write(f"3. Closed under scalar multiplication - **{tick(inside(scal))}**")

    st.divider()
    if n == SETS[0]:
        st.write("R3 is a vector space, and a subspace of itself. Every test passes trivially.")
    elif n == SETS[1]:
        st.write("A plane through the origin. {u1,u2} and {v1,v2} are two DIFFERENT bases of the SAME plane - add or scale, you stay on the plane.")
    elif n == SETS[2]:
        st.write("A line through the origin. All four generators are multiples of one direction, so every combination lands back on the line. Dimension 1.")
    elif n == SETS[3]:
        st.write("The zero subspace. Every generator is 0, so u = v = 0 regardless of coefficients. The smallest subspace of any space.")
    elif n == SETS[4]:
        st.write("The same plane, lifted off the origin. It fails immediately: no zero vector, and adding two points lands on a different plane. Must pass through 0.")
    else:
        st.write("The first octant. Generators have no negative part, so u, v, u+v stay inside - but press 'Try k = -1' and you land in the opposite octant: addition alone is not enough.")
