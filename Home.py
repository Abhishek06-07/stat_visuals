import streamlit as st

st.set_page_config(page_title="Math/ML Visuals Hub", layout="wide")

st.title("Math / ML Visuals Hub")
st.markdown("""
Welcome! Use the **sidebar** on the left to pick a topic.

Currently available:
- **Linear Combination** — see how u = α₁·u₁ + α₂·u₂ builds a vector, and explore span/independence.
- **Vector Space and Subspace** — test the 3 subspace rules (zero vector, closure under addition, closure under scaling) on planes, lines, and octants in R³.
- **Linear vs Affine Transformation** — see how T(x) = A·x + b behaves, and test T(0)=0, additivity, and homogeneity live.

More topics will appear here as they're added — just pick them from the sidebar once they're up.
""")