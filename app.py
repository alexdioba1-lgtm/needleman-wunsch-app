# Import necessary libraries
import streamlit as st          
import numpy as np              
import matplotlib.pyplot as plt 
import seaborn as sns           
import pandas as pd             
from Bio import Entrez, SeqIO   
import itertools                

# ---------------- Parameters ----------------
Entrez.email = "alex@example.com"  # Required by NCBI Entrez API (replace with your email)
GENOME_COUNT = 5                   # Number of HIV-1 genomes to download (can set to 10)
WINDOW_SIZE = 1000                 # Block size for splitting genomes
MATCH = 1                          # Match score
MISMATCH = -1                      # Mismatch penalty
GAP = -2                           # Gap penalty

# ---------------- Functions ----------------
def download_hiv_genomes(count):
    """
    Download HIV-1 genomes from NCBI using Entrez.
    Returns a list of genome sequences as strings.
    """
    handle = Entrez.esearch(db="nucleotide", term="HIV-1[Organism] AND complete genome", retmax=count)
    record = Entrez.read(handle)
    ids = record["IdList"]
    genomes = []
    for id in ids:
        fetch = Entrez.efetch(db="nucleotide", id=id, rettype="fasta", retmode="text")
        seq_record = SeqIO.read(fetch, "fasta")
        genomes.append(str(seq_record.seq))
    return genomes

def split_genome(genome, window_size=1000):
    """
    Split a genome into blocks of fixed size (default 1000 bases).
    This makes alignment feasible for long genomes.
    """
    return [genome[i:i+window_size] for i in range(0, len(genome), window_size)]

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Needleman-Wunsch global alignment algorithm.
    Returns the final alignment score between two sequences.
    """
    n, m = len(seq1), len(seq2)
    matrix = np.zeros((n+1, m+1), dtype=int)

    # Initialize first row and column with gap penalties
    for i in range(1, n+1): matrix[i][0] = i * gap
    for j in range(1, m+1): matrix[0][j] = j * gap

    # Fill the matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            score = match if seq1[i-1] == seq2[j-1] else mismatch
            matrix[i][j] = max(
                matrix[i-1][j-1] + score,  # diagonal (match/mismatch)
                matrix[i-1][j] + gap,      # up (gap in seq2)
                matrix[i][j-1] + gap       # left (gap in seq1)
            )
    return matrix[n][m]

def align_genomes(g1, g2):
    """
    Align two genomes block by block.
    Returns total alignment score and similarity percentages per block.
    """
    blocks1 = split_genome(g1, WINDOW_SIZE)
    blocks2 = split_genome(g2, WINDOW_SIZE)
    total_score = 0
    similarities = []
    for b1, b2 in zip(blocks1, blocks2):
        score = needleman_wunsch(b1, b2, MATCH, MISMATCH, GAP)
        total_score += score
        # Calculate similarity percentage for this block
        matches = sum(a == b for a, b in zip(b1, b2))
        similarity = matches / len(b1) * 100
        similarities.append(similarity)
    return total_score, similarities

# ---------------- Streamlit Interface ----------------
st.title("🧬 HIV-1 Genome Pairwise Alignment")

st.write("This app downloads HIV-1 genomes, aligns them block by block, and shows results.")

if st.button("Run Analysis"):
    # Step 1: Download genomes
    genomes = download_hiv_genomes(GENOME_COUNT)
    labels = [f"HIV{i+1}" for i in range(GENOME_COUNT)]
    results = {}
    similarities_dict = {}

    # Step 2: Pairwise alignment
    for (i, g1), (j, g2) in itertools.combinations(enumerate(genomes), 2):
        score, similarities = align_genomes(g1, g2)
        key = (labels[i], labels[j])
        results[key] = score
        similarities_dict[key] = similarities

    # Step 3: Show results in a table
    data = [{"Genome A": a, "Genome B": b, "Score": score} for (a, b), score in results.items()]
    df = pd.DataFrame(data)
    st.subheader("📊 Pairwise Results")
    st.dataframe(df)

    # Step 4: Heatmap of scores
    st.subheader("🔥 Heatmap of Scores")
    matrix = np.zeros((GENOME_COUNT, GENOME_COUNT), dtype=int)
    for (a, b), score in results.items():
        i, j = labels.index(a), labels.index(b)
        matrix[i][j] = score
        matrix[j][i] = score
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Pairwise HIV-1 Genome Alignment Scores")
    st.pyplot(fig)

    # Step 5: Block-by-block similarity bars
    st.subheader("📏 Block-by-block similarity (example)")
    example_pair = list(similarities_dict.keys())[0]
    st.write(f"Alignment {example_pair[0]} vs {example_pair[1]}")
    for idx, sim in enumerate(similarities_dict[example_pair]):
        st.write(f"Block {idx+1}: {sim:.2f}%")
        st.progress(int(sim))
