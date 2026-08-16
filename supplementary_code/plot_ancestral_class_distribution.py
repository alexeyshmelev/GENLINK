import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths and switches
# ============================================================
GRAPH_PATH = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/df_anonymized_v2.csv"
TREE_PATH = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/trees_no_dups_v2.csv"
ANCESTRY_PATH = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/samples_2347658_pivot_ancestry_anonymized.csv"
PCA_PATH = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/anonymized_id_PC1-PC20_chiptype.tsv"
OUTPUT_DIR = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/final_datasets"

DATASET_PCA_SUFFIX = "without_pca"  # change to "with_pca" when needed

OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    f"ancestor_expectations_and_class_balance_{DATASET_PCA_SUFFIX}.png",
)
OUTPUT_VALUES = os.path.join(
    OUTPUT_DIR,
    f"ancestor_expectations_{DATASET_PCA_SUFFIX}.csv",
)

CR_THRESHOLD = 80.0
THETA = 4
KNOWN_RELATIVES_MODE = "at_least"  # "at_least" or "exact"
GRAPH_CHUNK_SIZE = 1_000_000
PCA_ID_COL = "anonymized_id"
MASK_LABEL = "masked"
ANNOTATION_DECIMALS = 3

DATASETS = {
    "2nd degree 4/4, 8 cM": os.path.join(
        OUTPUT_DIR,
        "2nd_degree",
        "labeled",
        f"FINAL_2nd_degree_4_of_4_labeled_min8_{DATASET_PCA_SUFFIX}.csv",
    ),
    "2nd degree 3+1, 8 cM": os.path.join(
        OUTPUT_DIR,
        "2nd_degree",
        "labeled",
        f"FINAL_2nd_degree_3_plus_1_labeled_min8_{DATASET_PCA_SUFFIX}.csv",
    ),
    "2nd degree 3+1, 10 cM": os.path.join(
        OUTPUT_DIR,
        "2nd_degree",
        "labeled",
        f"FINAL_2nd_degree_3_plus_1_labeled_min10_{DATASET_PCA_SUFFIX}.csv",
    ),
    "All degrees unanimous, 8 cM": os.path.join(
        OUTPUT_DIR,
        "all_degree",
        "labeled",
        f"FINAL_all_degree_same_labeled_min8_{DATASET_PCA_SUFFIX}.csv",
    ),
}

RELATIONS = ["father", "mother", "fgf", "fgm", "mgf", "mgm"]
N_RELATIVE_SLOTS = len(RELATIONS)

# Canonical names used internally and in the plots.
CLASSES = [
    "Belarusians",
    "Northern Russians",
    "Southern Russians",
    "Ukrainians",
]

# Keep compatibility with the spellings currently present in the source files.
CLASS_ALIASES = {
    "Belorussians": "Belarusians",
    "Belorussian": "Belarusians",
    "Belarusian": "Belarusians",
    "Northen Russians": "Northern Russians",
    "Ukranians": "Ukrainians",
}

FIRST_COLUMN_CLASSES = ["Ukrainians", "Belarusians"]
SECOND_COLUMN_CLASSES = ["Northern Russians", "Southern Russians"]

CLASS_TICK_LABELS = {
    "Belarusians": "Belarusians",
    "Northern Russians": "Northern\nRussians",
    "Southern Russians": "Southern\nRussians",
    "Ukrainians": "Ukrainians",
}

# Use Matplotlib's active default color cycle, but keep class colors consistent
# across the line plots and the class-balance bars.
DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
CLASS_COLORS = {
    class_name: DEFAULT_COLORS[index]
    for index, class_name in enumerate(CLASSES)
}


# ============================================================
# Helpers
# ============================================================
def normalize_classes(series):
    return series.astype("string").str.strip().replace(CLASS_ALIASES)


def eligible_relative_mask(frame):
    if KNOWN_RELATIVES_MODE == "at_least":
        return frame["known_relatives"].ge(THETA)
    if KNOWN_RELATIVES_MODE == "exact":
        return frame["known_relatives"].eq(THETA)
    raise ValueError("KNOWN_RELATIVES_MODE must be 'at_least' or 'exact'.")


def eligibility_latex():
    if KNOWN_RELATIVES_MODE == "at_least":
        return rf"K(n)\geq {THETA}"
    if KNOWN_RELATIVES_MODE == "exact":
        return rf"K(n)={THETA}"
    raise ValueError("KNOWN_RELATIVES_MODE must be 'at_least' or 'exact'.")


def eligibility_text():
    if KNOWN_RELATIVES_MODE == "at_least":
        return f"K >= {THETA}"
    if KNOWN_RELATIVES_MODE == "exact":
        return f"K = {THETA}"
    raise ValueError("KNOWN_RELATIVES_MODE must be 'at_least' or 'exact'.")


def format_float(value):
    text = f"{value:.{ANNOTATION_DECIMALS}f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


# ============================================================
# Source-node filters
# ============================================================
def read_graph_nodes(path):
    nodes = set()

    for chunk in pd.read_csv(
        path,
        usecols=["node_id1", "node_id2"],
        dtype={"node_id1": "int64", "node_id2": "int64"},
        chunksize=GRAPH_CHUNK_SIZE,
    ):
        nodes.update(chunk["node_id1"].tolist())
        nodes.update(chunk["node_id2"].tolist())

    return nodes


def read_cr_nodes(path):
    ancestry = pd.read_csv(
        path,
        usecols=["node_tubeid", "Central-Russia"],
    )

    ancestry["node_tubeid"] = pd.to_numeric(
        ancestry["node_tubeid"], errors="raise"
    ).astype(np.int64)
    ancestry["Central-Russia"] = pd.to_numeric(
        ancestry["Central-Russia"], errors="coerce"
    )

    return set(
        ancestry.loc[
            ancestry["Central-Russia"].gt(CR_THRESHOLD),
            "node_tubeid",
        ]
        .dropna()
        .unique()
    )


def read_pca_nodes(path):
    pca = pd.read_csv(
        path,
        sep=None,
        engine="python",
        usecols=[PCA_ID_COL],
    )

    pca[PCA_ID_COL] = pd.to_numeric(
        pca[PCA_ID_COL], errors="raise"
    ).astype(np.int64)

    return set(pca[PCA_ID_COL].dropna().unique())


# ============================================================
# Final dataset labels and ancestry-relative counts
# ============================================================
def read_final_node_labels(path):
    graph = pd.read_csv(
        path,
        usecols=["node_id1", "node_id2", "label_id1", "label_id2"],
    )

    graph["node_id1"] = pd.to_numeric(
        graph["node_id1"], errors="raise"
    ).astype(np.int64)
    graph["node_id2"] = pd.to_numeric(
        graph["node_id2"], errors="raise"
    ).astype(np.int64)
    graph["label_id1"] = normalize_classes(graph["label_id1"])
    graph["label_id2"] = normalize_classes(graph["label_id2"])

    node_labels = pd.concat(
        [
            graph[["node_id1", "label_id1"]].rename(
                columns={"node_id1": "node", "label_id1": "label"}
            ),
            graph[["node_id2", "label_id2"]].rename(
                columns={"node_id2": "node", "label_id2": "label"}
            ),
        ],
        ignore_index=True,
    )

    non_null_labels = node_labels.dropna(subset=["label"])
    conflicting = non_null_labels.groupby("node")["label"].nunique()
    if conflicting.gt(1).any():
        bad_nodes = conflicting[conflicting.gt(1)].index[:10].tolist()
        raise ValueError(
            f"Conflicting labels found in {path}. Example nodes: {bad_nodes}"
        )

    is_masked = (
        node_labels["label"]
        .str.casefold()
        .eq(MASK_LABEL.casefold())
        .fillna(False)
    )
    node_labels = (
        node_labels.loc[node_labels["label"].notna() & ~is_masked]
        .drop_duplicates("node")
        .copy()
    )

    unexpected = sorted(set(node_labels["label"]) - set(CLASSES))
    if unexpected:
        raise ValueError(
            f"Unexpected non-masked labels in {path}: {unexpected}"
        )

    return node_labels


def build_relative_counts(tree_path, eligible_nodes):
    tree = pd.read_csv(
        tree_path,
        usecols=["node_tubeid", "relation", "group"],
    )

    tree["node_tubeid"] = pd.to_numeric(
        tree["node_tubeid"], errors="raise"
    ).astype(np.int64)
    tree["relation"] = tree["relation"].astype("string").str.strip().str.lower()

    groups = tree["group"].astype("string").str.strip()
    unknown = (
        groups.isna()
        | groups.eq("")
        | groups.str.casefold().eq("unknown").fillna(False)
    )
    tree["ancestry"] = normalize_classes(groups.mask(unknown))

    tree = tree[
        tree["node_tubeid"].isin(eligible_nodes)
        & tree["relation"].isin(RELATIONS)
    ].copy()

    # A node/relation pair must not have conflicting known ancestry values.
    non_null_tree = tree.dropna(subset=["ancestry"])
    conflicts = non_null_tree.groupby(["node_tubeid", "relation"])[
        "ancestry"
    ].nunique()
    if conflicts.gt(1).any():
        bad_pairs = conflicts[conflicts.gt(1)].index[:10].tolist()
        raise ValueError(
            "Conflicting ancestry groups for the same node/relation pair. "
            f"Examples: {bad_pairs}"
        )

    relatives = (
        tree.groupby(["node_tubeid", "relation"], sort=False)["ancestry"]
        .first()
        .unstack()
        .reindex(columns=RELATIONS)
    )

    result = pd.DataFrame(index=relatives.index)
    result["known_relatives"] = relatives.notna().sum(axis=1)

    for class_name in CLASSES:
        result[f"relatives_{class_name}"] = relatives.eq(class_name).sum(axis=1)

    result.index.name = "node"
    return result.reset_index()


def validate_relative_counts(frame, dataset_name):
    if not frame["known_relatives"].between(0, N_RELATIVE_SLOTS).all():
        raise RuntimeError(
            f"{dataset_name}: K(n) is outside [0, {N_RELATIVE_SLOTS}]."
        )

    class_columns = [f"relatives_{class_name}" for class_name in CLASSES]

    for class_name, column in zip(CLASSES, class_columns):
        invalid = frame[column].gt(frame["known_relatives"])
        if invalid.any():
            bad_nodes = frame.loc[invalid, "node"].head(10).tolist()
            raise RuntimeError(
                f"{dataset_name}: R_c(n) > K(n) for {class_name}. "
                f"Example nodes: {bad_nodes}"
            )

    invalid_total = frame[class_columns].sum(axis=1).gt(
        frame["known_relatives"]
    )
    if invalid_total.any():
        bad_nodes = frame.loc[invalid_total, "node"].head(10).tolist()
        raise RuntimeError(
            f"{dataset_name}: summed class counts exceed K(n). "
            f"Example nodes: {bad_nodes}"
        )


# ============================================================
# E_c(k) and plotting
# ============================================================
def compute_expectation_curve(analysis, class_name, x_values):
    """
    E_c(k) = sum_n [R_c(n) / 6] I(R_c(n)=k and K-condition).

    The K-condition is already applied to ``analysis``. Notice that K>=4 does
    not imply R_c>=4; therefore k=1,2,3 may legitimately be non-zero.
    """
    count_column = f"relatives_{class_name}"
    class_counts = analysis[count_column]
    p_ch = class_counts.astype(float) / float(N_RELATIVE_SLOTS)

    expected_values = []
    support_values = []

    for k in x_values:
        indicator = class_counts.eq(k)
        support = int(indicator.sum())
        expected = float(p_ch.where(indicator, 0.0).sum())

        # Cross-check against the closed form E_c(k) = (k/6) N_c(k).
        closed_form = (float(k) / N_RELATIVE_SLOTS) * support
        if not np.isclose(expected, closed_form):
            raise RuntimeError(
                f"E_c(k) mismatch for {class_name}, k={k}: "
                f"{expected} versus {closed_form}."
            )

        expected_values.append(expected)
        support_values.append(support)

    return np.asarray(expected_values), np.asarray(support_values)


def annotate_points(ax, x_values, y_values, color, series_index):
    # Different offsets keep annotations readable when the two curves overlap.
    vertical_offset = 7 + 13 * series_index

    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            format_float(float(y_value)),
            xy=(x_value, y_value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            annotation_clip=False,
        )


def plot_expectation_panel(
    ax,
    analysis,
    dataset_name,
    panel_name,
    panel_classes,
    x_values,
    expectation_rows,
):
    for series_index, class_name in enumerate(panel_classes):
        expected_values, support_values = compute_expectation_curve(
            analysis,
            class_name,
            x_values,
        )
        color = CLASS_COLORS[class_name]

        ax.plot(
            x_values,
            expected_values,
            marker="o",
            linewidth=1.7,
            markersize=5,
            color=color,
            label=class_name,
        )
        annotate_points(
            ax,
            x_values,
            expected_values,
            color,
            series_index,
        )

        for k, support, expected in zip(
            x_values,
            support_values,
            expected_values,
        ):
            expectation_rows.append(
                {
                    "dataset": dataset_name,
                    "class": class_name,
                    "k": int(k),
                    "support_nodes": int(support),
                    "expected_selected_nodes": float(expected),
                    "known_relatives_mode": KNOWN_RELATIVES_MODE,
                    "theta": THETA,
                }
            )

    ax.set_title(f"{dataset_name}\n$E_c(k)$: {panel_name}")
    ax.set_xlabel("Number k of relatives from class c")
    ax.set_ylabel(r"Expected selected nodes, $E_c(k)$")
    ax.set_xticks(x_values)
    ax.set_xlim(-0.3, N_RELATIVE_SLOTS + 0.3)
    ax.grid(alpha=0.3)
    ax.margins(y=0.20)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5)


def plot_class_balance(ax, analysis, dataset_name):
    counts = (
        analysis["label"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .astype(int)
    )
    total = int(counts.sum())
    x_positions = np.arange(len(CLASSES))

    bars = ax.bar(
        x_positions,
        counts.to_numpy(),
        color=[CLASS_COLORS[class_name] for class_name in CLASSES],
    )

    for bar, count in zip(bars, counts.to_numpy()):
        percentage = 100.0 * count / total if total else 0.0
        ax.annotate(
            f"{int(count):,}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(
        f"{dataset_name}\nClass balance after the {eligibility_text()} filter"
    )
    ax.set_ylabel("Number of labeled nodes")
    ax.set_xticks(
        x_positions,
        [CLASS_TICK_LABELS[class_name] for class_name in CLASSES],
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(1.0, float(counts.max()) * 1.22))


def print_low_k_diagnostics(analysis, dataset_name):
    print(f"\n{dataset_name}: nodes with 0 < R_c(n) < {THETA}")
    found = False

    for class_name in CLASSES:
        column = f"relatives_{class_name}"
        low_counts = (
            analysis.loc[analysis[column].between(1, THETA - 1), column]
            .value_counts()
            .sort_index()
        )
        if low_counts.empty:
            continue

        found = True
        summary = ", ".join(
            f"k={int(k)}: {int(count)}"
            for k, count in low_counts.items()
        )
        print(f"  {class_name}: {summary}")

    if not found:
        print("  None.")
    else:
        print(
            "  This is valid because the indicator constrains total K(n), "
            "not class-specific R_c(n)."
        )


# ============================================================
# Main
# ============================================================
def main():
    if not 0 <= THETA <= N_RELATIVE_SLOTS:
        raise ValueError(f"THETA must be between 0 and {N_RELATIVE_SLOTS}.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/4] Reading source graph node IDs...")
    graph_nodes = read_graph_nodes(GRAPH_PATH)

    print(f"[2/4] Applying Central-Russia > {CR_THRESHOLD} filtering...")
    cr_nodes = read_cr_nodes(ANCESTRY_PATH)

    print("[3/4] Reading PCA node IDs from column 'anonymized_id'...")
    pca_nodes = read_pca_nodes(PCA_PATH)

    eligible_nodes = graph_nodes & cr_nodes & pca_nodes

    print("\nGlobal source-node filters:")
    print("Nodes in source graph:", len(graph_nodes))
    print(f"Nodes with Central-Russia > {CR_THRESHOLD}:", len(cr_nodes))
    print("Nodes present in PCA file:", len(pca_nodes))
    print("Nodes passing graph + CR + PCA filters:", len(eligible_nodes))

    print("\n[4/4] Building relative counts and plotting...")
    relative_counts = build_relative_counts(TREE_PATH, eligible_nodes)

    x_values = np.arange(N_RELATIVE_SLOTS + 1)
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=3,
        figsize=(22, 4.8 * len(DATASETS)),
        squeeze=False,
    )

    filtering_rows = []
    expectation_rows = []

    for row, (dataset_name, dataset_path) in enumerate(DATASETS.items()):
        node_labels = read_final_node_labels(dataset_path)
        dataset_nodes = set(node_labels["node"])

        nodes_outside_graph = dataset_nodes - graph_nodes
        nodes_outside_cr = dataset_nodes - cr_nodes
        nodes_without_pca = dataset_nodes - pca_nodes

        if nodes_outside_graph:
            raise RuntimeError(
                f"{dataset_name}: {len(nodes_outside_graph)} final nodes are absent "
                "from the source graph."
            )
        if nodes_outside_cr:
            raise RuntimeError(
                f"{dataset_name}: {len(nodes_outside_cr)} final nodes do not satisfy "
                f"Central-Russia > {CR_THRESHOLD}."
            )
        if nodes_without_pca:
            raise RuntimeError(
                f"{dataset_name}: {len(nodes_without_pca)} final nodes do not occur "
                "in the PCA file."
            )

        analysis = node_labels.merge(relative_counts, on="node", how="left")

        count_columns = [
            "known_relatives",
            *[f"relatives_{class_name}" for class_name in CLASSES],
        ]
        analysis[count_columns] = analysis[count_columns].fillna(0).astype(int)
        validate_relative_counts(analysis, dataset_name)

        nodes_before_relative_filter = len(analysis)
        analysis = analysis.loc[eligible_relative_mask(analysis)].copy()
        nodes_after_relative_filter = len(analysis)

        filtering_rows.append(
            {
                "dataset": dataset_name,
                "final_labeled_dataset_nodes": len(node_labels),
                "nodes_before_known_relative_filter": nodes_before_relative_filter,
                "nodes_after_known_relative_filter": nodes_after_relative_filter,
            }
        )

        print_low_k_diagnostics(analysis, dataset_name)

        plot_expectation_panel(
            axes[row, 0],
            analysis,
            dataset_name,
            "Ukrainians and Belarusians",
            FIRST_COLUMN_CLASSES,
            x_values,
            expectation_rows,
        )
        plot_expectation_panel(
            axes[row, 1],
            analysis,
            dataset_name,
            "Northern and Southern Russians",
            SECOND_COLUMN_CLASSES,
            x_values,
            expectation_rows,
        )
        plot_class_balance(
            axes[row, 2],
            analysis,
            dataset_name,
        )

        print(
            f"{dataset_name}: {len(node_labels)} final labeled nodes; "
            f"{nodes_after_relative_filter} satisfy the K filter."
        )

    fig.suptitle(
        "Ancestor-count expectations and class balance for labeled-only CR datasets\n"
        rf"$E_c(k)=\sum_n [R_c(n)/{N_RELATIVE_SLOTS}]\,"
        rf"I(R_c(n)=k,\ {eligibility_latex()})$. "
        f"All nodes satisfy Central-Russia > {CR_THRESHOLD} and occur in the PCA file.",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.8, w_pad=2.0)
    fig.savefig(OUTPUT_IMAGE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    filtering_stats = pd.DataFrame(filtering_rows)
    expectation_stats = pd.DataFrame(expectation_rows)
    expectation_stats.to_csv(OUTPUT_VALUES, index=False)

    print("\nDataset filtering summary:")
    print(filtering_stats.to_string(index=False))
    print("\nSaved figure:", OUTPUT_IMAGE)
    print("Saved E_c(k) values:", OUTPUT_VALUES)


if __name__ == "__main__":
    main()
