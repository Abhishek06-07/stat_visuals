import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Linear vs Affine Transformation", layout="wide")

LIM = 6.0
TOL = 1e-9

st.title("Linear Transformation vs Affine Transformation")
st.markdown("**T(x) = A·x + b** — dekho matrix A aur shift b badalne se grid, shapes, aur linearity tests kaise badalte hain.")

if "a11" not in st.session_state:
    st.session_state.a11, st.session_state.a12 = 1.0, 0.0
    st.session_state.a21, st.session_state.a22 = 0.0, 1.0
    st.session_state.bx, st.session_state.by = 0.0, 0.0

col_plot, col_controls = st.columns([2, 1])

with col_controls:
    st.subheader("Matrix A (linear part)")
    c1, c2 = st.columns(2)
    a11 = c1.number_input("A[0,0]", value=float(st.session_state.a11), step=0.1, format="%.2f")
    a12 = c2.number_input("A[0,1]", value=float(st.session_state.a12), step=0.1, format="%.2f")
    a21 = c1.number_input("A[1,0]", value=float(st.session_state.a21), step=0.1, format="%.2f")
    a22 = c2.number_input("A[1,1]", value=float(st.session_state.a22), step=0.1, format="%.2f")

    st.subheader("Shift b (translation part)")
    c3, c4 = st.columns(2)
    bx = c3.slider("b_x", -4.0, 4.0, float(st.session_state.bx), 0.1)
    by = c4.slider("b_y", -4.0, 4.0, float(st.session_state.by), 0.1)

    st.session_state.a11, st.session_state.a12 = a11, a12
    st.session_state.a21, st.session_state.a22 = a21, a22
    st.session_state.bx, st.session_state.by = bx, by

    st.subheader("Quick presets")
    p1, p2, p3 = st.columns(3)
    if p1.button("Identity"):
        st.session_state.a11, st.session_state.a12 = 1.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, 1.0
        st.session_state.bx, st.session_state.by = 0.0, 0.0
        st.rerun()
    if p2.button("Rotate 90"):
        st.session_state.a11, st.session_state.a12 = 0.0, -1.0
        st.session_state.a21, st.session_state.a22 = 1.0, 0.0
        st.session_state.bx, st.session_state.by = 0.0, 0.0
        st.rerun()
    if p3.button("Scale x2"):
        st.session_state.a11, st.session_state.a12 = 2.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, 2.0
        st.session_state.bx, st.session_state.by = 0.0, 0.0
        st.rerun()
    p4, p5 = st.columns(2)
    if p4.button("Pure shift (2,3)"):
        st.session_state.a11, st.session_state.a12 = 1.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, 1.0
        st.session_state.bx, st.session_state.by = 2.0, 3.0
        st.rerun()
    if p5.button("Reset"):
        st.session_state.a11, st.session_state.a12 = 1.0, 0.0
        st.session_state.a21, st.session_state.a22 = 0.0, 1.0
        st.session_state.bx, st.session_state.by = 0.0, 0.0
        st.rerun()

A = np.array([[st.session_state.a11, st.session_state.a12],
              [st.session_state.a21, st.session_state.a22]])
b = np.array([st.session_state.bx, st.session_state.by])

def T(x):
    return A @ x + b

def is_shift_zero():
    return np.linalg.norm(b) < TOL

with col_controls:
    st.subheader("Test u, v, and scalar c")
    u = np.array([st.number_input("u_x", value=1.0), st.number_input("u_y", value=1.0)])
    v = np.array([st.number_input("v_x", value=2.0), st.number_input("v_y", value=-1.0)])
    c = st.slider("scalar c", -3.0, 3.0, 2.0, 0.1)

with col_plot:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax0, ax1 = axes

    grid_pts = np.linspace(-LIM, LIM, 9)
    for g in grid_pts:
        ax0.plot([-LIM, LIM], [g, g], color="#dde3f2", linewidth=0.8)
        ax0.plot([g, g], [-LIM, LIM], color="#dde3f2", linewidth=0.8)
    ax0.axhline(0, color="#9aa2bc", linewidth=1.2)
    ax0.axvline(0, color="#9aa2bc", linewidth=1.2)
    tri = np.array([[0, 0], [2, 0], [1, 2]])
    ax0.add_patch(plt.Polygon(tri, closed=True, facecolor="#1f77b4", alpha=0.25, edgecolor="#1f77b4"))
    ax0.scatter([0], [0], color="#1e2761", s=60, zorder=5, label="origin")
    ax0.set_xlim(-LIM, LIM); ax0.set_ylim(-LIM, LIM)
    ax0.set_aspect("equal"); ax0.set_title("Before: input space")
    ax0.legend(loc="upper left", fontsize=8)

    for g in grid_pts:
        line_h = np.array([[-LIM, g], [LIM, g]])
        line_v = np.array([[g, -LIM], [g, LIM]])
        th = np.array([T(p) for p in line_h])
        tv = np.array([T(p) for p in line_v])
        ax1.plot(th[:, 0], th[:, 1], color="#ffd9a0", linewidth=0.8)
        ax1.plot(tv[:, 0], tv[:, 1], color="#ffd9a0", linewidth=0.8)
    ax1.axhline(0, color="#9aa2bc", linewidth=1.2)
    ax1.axvline(0, color="#9aa2bc", linewidth=1.2)
    tri_t = np.array([T(p) for p in tri])
    ax1.add_patch(plt.Polygon(tri_t, closed=True, facecolor="#d62728", alpha=0.25, edgecolor="#d62728"))
    origin_after = T(np.zeros(2))
    ax1.scatter([origin_after[0]], [origin_after[1]], color="#1e2761", s=60, zorder=5,
                label=f"T(0)=({origin_after[0]:.1f},{origin_after[1]:.1f})")
    ax1.set_xlim(-LIM, LIM); ax1.set_ylim(-LIM, LIM)
    ax1.set_aspect("equal")
    kind = "LINEAR" if is_shift_zero() else "AFFINE (not linear)"
    ax1.set_title(f"After: T(x)=Ax+b  ->  {kind}")
    ax1.legend(loc="upper left", fontsize=8)

    st.pyplot(fig)

    st.subheader("Linearity tests")
    Tu, Tv = T(u), T(v)
    lhs_add, rhs_add = T(u + v), Tu + Tv
    lhs_hom, rhs_hom = T(c * u), c * Tu
    T0 = T(np.zeros(2))

    ok_add = np.allclose(lhs_add, rhs_add)
    ok_hom = np.allclose(lhs_hom, rhs_hom)
    ok_zero = np.linalg.norm(T0) < 1e-6

    st.write(f"**T(0) = ({T0[0]:.2f}, {T0[1]:.2f})** — must be (0,0) for a linear map: **{'YES' if ok_zero else 'NO'}**")
    st.write(f"**T(u+v) = ({lhs_add[0]:.2f}, {lhs_add[1]:.2f})**  vs  **T(u)+T(v) = ({rhs_add[0]:.2f}, {rhs_add[1]:.2f})**  ->  Additivity holds: **{'YES' if ok_add else 'NO'}**")
    st.write(f"**T(c·u) = ({lhs_hom[0]:.2f}, {lhs_hom[1]:.2f})**  vs  **c·T(u) = ({rhs_hom[0]:.2f}, {rhs_hom[1]:.2f})**  ->  Homogeneity holds: **{'YES' if ok_hom else 'NO'}**")

    st.divider()
    if ok_add and ok_hom and ok_zero:
        st.success("This is a valid LINEAR transformation: T(x) = A·x, b = 0.")
    else:
        st.warning("This is only an AFFINE transformation (Linear + shift). Because b != 0, it fails T(0)=0, additivity, and homogeneity.")
