"""
╔══════════════════════════════════════════════════════════════════╗
║        DATA SCIENCE PROJECT 3 — CUSTOMER SEGMENTATION            ║
║        Unsupervised Learning: PCA + K-Means Clustering           ║
║        DecodeLabs Industrial Training Kit | Batch 2026           ║
╚══════════════════════════════════════════════════════════════════╝

Pipeline:  SCALE → COMPRESS (PCA) → CLUSTER (K-Means) → TRANSLATE (Personas)
"""

# ──────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
print("✅ All libraries imported successfully.\n")


# ──────────────────────────────────────────────────────────────────
# 1. LOAD / GENERATE DATASET
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

n = 200
ages    = np.concatenate([np.random.normal(41, 8, 50), np.random.normal(33, 6, 50),
                           np.random.normal(25, 4, 50), np.random.normal(45, 7, 50)])
incomes = np.concatenate([np.random.normal(88, 10, 50), np.random.normal(87, 9, 50),
                           np.random.normal(26, 6,  50), np.random.normal(26, 7, 50)])
scores  = np.concatenate([np.random.normal(17, 5,  50), np.random.normal(82, 8, 50),
                           np.random.normal(79, 7,  50), np.random.normal(21, 6, 50)])
genders = np.concatenate([
    np.random.choice(['Male','Female'], 50, p=[0.52,0.48]),
    np.random.choice(['Male','Female'], 50, p=[0.46,0.54]),
    np.random.choice(['Male','Female'], 50, p=[0.40,0.60]),
    np.random.choice(['Male','Female'], 50, p=[0.58,0.42]),
])

df = pd.DataFrame({
    'CustomerID':     [f"C{str(i+1).zfill(3)}" for i in range(n)],
    'Gender':         genders,
    'Age':            np.clip(ages.astype(int),    18, 70),
    'Annual_Income':  np.clip(incomes.astype(int) * 1000, 15000, 130000),
    'Spending_Score': np.clip(scores.astype(int),   1, 100),
})

print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(df.head(8).to_string(index=False))
print("\nDescriptive Statistics:")
print(df[['Age','Annual_Income','Spending_Score']].describe().round(1).to_string())


# ──────────────────────────────────────────────────────────────────
# 2. PHASE 1 — SCALE  (StandardScaler → z = (x - μ) / σ)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: PHASE 1 — STANDARDIZATION (SCALE)")
print("=" * 60)

features = ['Age', 'Annual_Income', 'Spending_Score']
X        = df[features].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Original  → Mean: {X.mean(axis=0).round(2)},  Std: {X.std(axis=0).round(2)}")
print(f"Scaled    → Mean: {X_scaled.mean(axis=0).round(4)},  Std: {X_scaled.std(axis=0).round(4)}")
print("✅ All features now on a common scale. Scale-induced bias eliminated.")


# ──────────────────────────────────────────────────────────────────
# 3. PHASE 2 — COMPRESS  (PCA — 95% variance rule)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: PHASE 2 — PCA (COMPRESS)")
print("=" * 60)

pca_full = PCA()
pca_full.fit(X_scaled)
evr      = pca_full.explained_variance_ratio_
cumvar   = np.cumsum(evr)

print("Component | Variance Explained | Cumulative")
for i, (e, c) in enumerate(zip(evr, cumvar)):
    print(f"  PC{i+1:>2}    |      {e*100:6.2f}%        |   {c*100:6.2f}%")

n_components = int(np.argmax(cumvar >= 0.95)) + 1
print(f"\n✅ Components needed for ≥95% variance: {n_components}")

# Apply final PCA
pca   = PCA(n_components=3)          
X_pca = pca.fit_transform(X_scaled)
print(f"✅ Data compressed: {X_scaled.shape[1]}D → {X_pca.shape[1]}D")


# ──────────────────────────────────────────────────────────────────
# 4. PHASE 3 — CLUSTER  (Elbow + Silhouette → optimal K)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: PHASE 3 — K-MEANS CLUSTERING")
print("=" * 60)

K_range    = range(2, 11)
wcss       = []
sil_scores = []

for k in K_range:
    km     = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels))

print("\n K  |   WCSS     | Silhouette")
print("-" * 32)
for k, w, s in zip(K_range, wcss, sil_scores):
    marker = " ← OPTIMAL" if s == max(sil_scores) else ""
    print(f" {k}  | {w:9.2f}  |   {s:.4f}{marker}")

optimal_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"\n✅ Optimal K = {optimal_k}  (Silhouette Score = {max(sil_scores):.4f})")

# Final K-Means
kmeans     = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_pca)


# ──────────────────────────────────────────────────────────────────
# 5. PHASE 4 — TRANSLATE  (Reverse-engineer centroids → Personas)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: PHASE 4 — BUSINESS PERSONAS (TRANSLATE)")
print("=" * 60)


centroids_pca      = kmeans.cluster_centers_
centroids_scaled   = pca.inverse_transform(centroids_pca)
centroids_original = scaler.inverse_transform(centroids_scaled)
centroid_df        = pd.DataFrame(centroids_original, columns=features)

print("\nReverse-Engineered Centroids (original feature space):")
print(centroid_df.round(1).to_string())


summary = df.groupby('Cluster').agg(
    Count         = ('CustomerID',     'count'),
    Avg_Age       = ('Age',            'mean'),
    Avg_Income    = ('Annual_Income',  'mean'),
    Avg_Score     = ('Spending_Score', 'mean'),
    Pct_Female    = ('Gender', lambda x: (x=='Female').mean()*100)
).round(1)

print("\nCluster Profile Summary:")
print(summary.to_string())


persona_map = {
    0: ("The High-Value Trendsetters",    "#B05C5C", "Exclusive perks, early access, experiential marketing"),
    1: ("The Conservative Minimizers",    "#6B7280", "Minimize spend, clear price value, basic utility"),
    2: ("The Affluent Conservatives",     "#5B7FA6", "High-touch support, warranties, loyalty programs"),
    3: ("The Budget-Conscious Explorers", "#C9A84C", "Influencer campaigns, flash sales, buy-now-pay-later"),
}

print("\n" + "=" * 60)
print("STRATEGIC PERSONA MATRIX")
print("=" * 60)
for c in range(optimal_k):
    name, _, action = persona_map.get(c, (f"Cluster {c}", "#999", "TBD"))
    row = summary.loc[c]
    print(f"\nCluster {c}: {name}")
    print(f"  Customers : {int(row['Count'])}")
    print(f"  Avg Age   : {row['Avg_Age']}")
    print(f"  Avg Income: ${row['Avg_Income']/1000:.1f}k")
    print(f"  Avg Score : {row['Avg_Score']}")
    print(f"  % Female  : {row['Pct_Female']:.0f}%")
    print(f"  → Action  : {action}")


# ──────────────────────────────────────────────────────────────────
# 6. VISUALIZATION DASHBOARD
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: GENERATING VISUALIZATION DASHBOARD")
print("=" * 60)

BG     = "#F5F0E8"
DARK   = "#1A1A2E"
COLORS = [persona_map.get(c, ("", "#999999", ""))[1] for c in range(optimal_k)]

fig = plt.figure(figsize=(22, 28), facecolor=BG)
fig.suptitle(
    "Customer Segmentation Pipeline\n"
    "Unsupervised Learning · DecodeLabs Project 3",
    fontsize=22, fontweight='bold', color=DARK, y=0.98
)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       top=0.94, bottom=0.03,
                       left=0.06, right=0.97)

# ── Plot 1: Elbow Method ──────────────────────
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG)
ax1.plot(list(K_range), wcss, 'o-', color=DARK, lw=2.5, ms=6)
ax1.axvline(x=optimal_k, color='#B05C5C', ls='--', lw=2, label=f'Optimal K={optimal_k}')
ax1.fill_between(list(K_range), wcss, alpha=0.08, color=DARK)
ax1.set_title("① Elbow Method (WCSS)", fontweight='bold', color=DARK, fontsize=12)
ax1.set_xlabel("Number of Clusters (K)"); ax1.set_ylabel("WCSS")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
ax1.spines[['top','right']].set_visible(False)

# ── Plot 2: Silhouette Score ──────────────────
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(BG)
bar_colors = ['#B05C5C' if k == optimal_k else '#A0AEC0' for k in K_range]
ax2.bar(list(K_range), sil_scores, color=bar_colors, edgecolor=DARK, lw=0.8)
ax2.set_title("② Silhouette Score by K", fontweight='bold', color=DARK, fontsize=12)
ax2.set_xlabel("Number of Clusters (K)"); ax2.set_ylabel("Silhouette Score")
ax2.annotate(f'Best: {max(sil_scores):.3f}',
             xy=(optimal_k, max(sil_scores)),
             xytext=(optimal_k + 0.4, max(sil_scores) + 0.01),
             color='#B05C5C', fontweight='bold', fontsize=9)
ax2.grid(alpha=0.3, axis='y'); ax2.spines[['top','right']].set_visible(False)

# ── Plot 3: PCA Explained Variance ───────────
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(BG)
ax3.bar(range(1, len(evr)+1), evr*100, color='#5B7FA6', alpha=0.8, edgecolor=DARK, lw=0.8)
ax3t = ax3.twinx()
ax3t.plot(range(1, len(cumvar)+1), cumvar*100, 'o-', color='#B05C5C', lw=2.5, ms=5)
ax3t.axhline(y=95, color='#C9A84C', ls='--', lw=1.5, label='95% threshold')
ax3t.set_ylabel("Cumulative %", color='#B05C5C')
ax3.set_title("③ PCA Explained Variance", fontweight='bold', color=DARK, fontsize=12)
ax3.set_xlabel("Principal Component"); ax3.set_ylabel("Variance Explained (%)")
ax3.spines['top'].set_visible(False)

# ── Plot 4: 2D PCA Scatter ───────────────────
ax4 = fig.add_subplot(gs[1, :2]); ax4.set_facecolor(BG)
for c in range(optimal_k):
    mask = df['Cluster'] == c
    name, color, _ = persona_map.get(c, (f"Cluster {c}", COLORS[c], ""))
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, label=f"C{c}: {name}",
                s=60, alpha=0.8, edgecolors=DARK, linewidths=0.3)
centers = kmeans.cluster_centers_
ax4.scatter(centers[:,0], centers[:,1], c='white', s=250, marker='*',
            edgecolors=DARK, lw=1.5, zorder=5, label='Centroids')
ax4.set_title("④ PCA Space — Cluster Distribution (PC1 vs PC2)",
              fontweight='bold', color=DARK, fontsize=13)
ax4.set_xlabel("Principal Component 1"); ax4.set_ylabel("Principal Component 2")
ax4.legend(fontsize=8, loc='upper right'); ax4.grid(alpha=0.25)
ax4.spines[['top','right']].set_visible(False)

# ── Plot 5: 3D PCA Scatter ───────────────────
ax5 = fig.add_subplot(gs[1, 2], projection='3d'); ax5.set_facecolor(BG)
for c in range(optimal_k):
    mask = df['Cluster'] == c
    _, color, _ = persona_map.get(c, (f"C{c}", COLORS[c], ""))
    ax5.scatter(X_pca[mask,0], X_pca[mask,1], X_pca[mask,2],
                c=color, s=30, alpha=0.7, label=f"C{c}")
ax5.set_title("⑤ 3D PCA View", fontweight='bold', color=DARK, fontsize=12, pad=10)
ax5.set_xlabel("PC1", fontsize=8); ax5.set_ylabel("PC2", fontsize=8)
ax5.set_zlabel("PC3", fontsize=8); ax5.tick_params(labelsize=7)

# ── Plot 6: Feature Profiles ─────────────────
ax6 = fig.add_subplot(gs[2, :]); ax6.set_facecolor(BG)
feat_labels = ['Avg Age', 'Avg Income ($k)', 'Avg Spending Score']
x = np.arange(len(feat_labels)); w = 0.2
for i, c in enumerate(range(optimal_k)):
    row  = summary.loc[c]
    vals = [row['Avg_Age'], row['Avg_Income']/1000, row['Avg_Score']]
    name, color, _ = persona_map.get(c, (f"C{c}", COLORS[c], ""))
    ax6.bar(x + i*w, vals, width=w, color=color,
            label=f"C{c}: {name}", edgecolor=DARK, lw=0.6, alpha=0.9)
ax6.set_title("⑥ Cluster Feature Profiles", fontweight='bold', color=DARK, fontsize=13)
ax6.set_xticks(x + w*(optimal_k-1)/2)
ax6.set_xticklabels(feat_labels, fontsize=11)
ax6.legend(fontsize=8); ax6.grid(alpha=0.25, axis='y')
ax6.spines[['top','right']].set_visible(False)

# ── Plot 7: Persona Cards ────────────────────
gs_cards = gridspec.GridSpecFromSubplotSpec(1, optimal_k,
                                            subplot_spec=gs[3, :], wspace=0.25)
for c in range(optimal_k):
    ax = fig.add_subplot(gs_cards[c]); ax.set_facecolor(BG); ax.axis('off')
    name, color, action = persona_map.get(c, (f"Cluster {c}", COLORS[c], "N/A"))
    row = summary.loc[c]
    card_text = (
        f"Cluster {c}\n{'─'*22}\n{name}\n\n"
        f"n = {int(row['Count'])} customers\n"
        f"Age:    {row['Avg_Age']}\n"
        f"Income: ${row['Avg_Income']/1000:.1f}k\n"
        f"Score:  {row['Avg_Score']}\n"
        f"Female: {row['Pct_Female']:.0f}%\n\n"
        f"→ Action:\n{action}"
    )
    ax.add_patch(plt.Rectangle((0.02,0.02), 0.96, 0.96,
                                facecolor=color, alpha=0.12,
                                transform=ax.transAxes, clip_on=False))
    ax.add_patch(plt.Rectangle((0.02,0.02), 0.96, 0.96,
                                fill=False, edgecolor=color, lw=2,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.95, card_text, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='center', color=DARK, linespacing=1.6)
    ax.set_title(f"Persona {c}", color=color, fontweight='bold', fontsize=10, pad=4)

plt.savefig('customer_segmentation_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("✅ Dashboard saved as 'customer_segmentation_dashboard.png'")

plt.show()

print("\n" + "=" * 60)
print("PROJECT 3 COMPLETE ✅")
print("=" * 60)
print(f"  Dataset       : {df.shape[0]} customers × {df.shape[1]} columns")
print(f"  Scaling       : StandardScaler (z-score)")
print(f"  PCA           : {X_scaled.shape[1]}D → 3D  ({cumvar[2]*100:.1f}% variance retained)")
print(f"  Optimal K     : {optimal_k}  (Silhouette = {max(sil_scores):.4f})")
print(f"  Personas      : {optimal_k} customer segments identified")
print("=" * 60)
