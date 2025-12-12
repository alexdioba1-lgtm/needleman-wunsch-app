import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Needleman-Wunsch Algorithm
def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    n, m = len(seq1), len(seq2)
    matrix = np.zeros((n+1, m+1), dtype=int)

    # Initialisation
    for i in range(1, n+1):
        matrix[i][0] = i * gap
    for j in range(1, m+1):
        matrix[0][j] = j * gap

    # Remplissage
    for i in range(1, n+1):
        for j in range(1, m+1):
            score = match if seq1[i-1] == seq2[j-1] else mismatch
            matrix[i][j] = max(
                matrix[i-1][j-1] + score,
                matrix[i-1][j] + gap,
                matrix[i][j-1] + gap
            )

    # Traceback
    align1, align2 = "", ""
    i, j = n, m
    path = [(i, j)]
    while i > 0 or j > 0:
        current = matrix[i][j]
        if i > 0 and j > 0 and (
            (seq1[i-1] == seq2[j-1] and current == matrix[i-1][j-1] + match) or
            (seq1[i-1] != seq2[j-1] and current == matrix[i-1][j-1] + mismatch)
        ):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif i > 0 and current == matrix[i-1][j] + gap:
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1
        else:
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1
        path.append((i, j))

    return align1, align2, matrix, path


# ---------------- Streamlit Interface ----------------
st.title("🧬 Needleman-Wunsch DNA Alignment")

# Inputs
seq1 = st.text_input("Sequence 1", "ACCGTGAAGCCAATAC")
seq2 = st.text_input("Sequence 2", "AGCGTGCGAGCCAATAC")

match = st.slider("Match score", 1, 5, 1)
mismatch = st.slider("Mismatch penalty", -5, 0, -1)
gap = st.slider("Gap penalty", -5, 0, -2)

if st.button("Run Alignment"):
    align1, align2, matrix, path = needleman_wunsch(seq1, seq2, match, mismatch, gap)

    st.subheader("Aligned Sequences")
    st.text(align1)
    st.text(align2)

    # Calcul du pourcentage de similarité
    matches = sum(a == b for a, b in zip(align1, align2) if a != "-" and b != "-")
    similarity = matches / len(align1) * 100
    st.write(f"Final Score: {matrix[len(seq1)][len(seq2)]}")
    st.write(f"Similarity: {similarity:.2f}%")

    # Plot matrix with traceback
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="coolwarm", origin="upper")
    x, y = zip(*path)
    ax.plot(y, x, color="red", linewidth=2)  # Traceback path
    ax.set_title("Alignment Matrix with Traceback")
    st.pyplot(fig)
