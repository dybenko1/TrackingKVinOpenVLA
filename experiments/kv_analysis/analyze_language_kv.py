"""
Analyze how language-token K/V representations change across
consecutive OpenVLA control steps in a LIBERO trajectory.
"""

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os

from transformers import AutoTokenizer
import numpy as np

SELECTED_LAYERS = [0, 5, 10, 15, 20, 25, 31]

PLOTS_DIR = "experiments/kv_analysis/outputs/plots"

TRAJECTORY_PATH = (
    "experiments/kv_analysis/outputs/"
    "trajectory_task0_episode0.pt"
)

def cosine_similarity(a, b):
    """
    Cosine similarity between two K/V tensors.

    Input shape:
        [1, num_language_tokens, hidden_dim]

    The tensors are flattened so this initially gives one similarity
    value for the entire language representation.
    """
    a = a.float().flatten()
    b = b.float().flatten()

    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
    ).item()

def relative_l2(a, b):
    """
    Relative change between two tensors:

        ||b - a||_2 / ||a||_2
    """
    a = a.float()
    b = b.float()

    return (
        torch.norm(b - a) / torch.norm(a)
    ).item()

def main():

    print("Loading trajectory...")

    data = torch.load(
        TRAJECTORY_PATH,
        weights_only=False,
    )

    prompt_token_ids = data["prompt_token_ids"]
    trajectory_kv = data["trajectory_kv"]

    tokenizer = AutoTokenizer.from_pretrained(
        data["model_id"],
        trust_remote_code=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(
        prompt_token_ids.tolist()
    )

    print("\nPrompt tokens:")
    for i, token in enumerate(tokens):
        print(i, token)

    print("Instruction:")
    print(data["instruction"])

    print("\nNumber of control steps:")
    print(len(trajectory_kv))

    print("\nExample tensor:")
    print(
        "Step 0 Layer 0 K:",
        trajectory_kv[0][0]["k"].shape,
    )

    # Results for each Transformer layer
    results = {}

    for layer_idx in range(32):

        k_cosines = []
        v_cosines = []

        k_l2 = []
        v_l2 = []

        # Compare consecutive control steps
        for step in range(len(trajectory_kv) - 1):

            current = trajectory_kv[step][layer_idx]
            next_step = trajectory_kv[step + 1][layer_idx]

            k_current = current["k"]
            k_next = next_step["k"]

            v_current = current["v"]
            v_next = next_step["v"]

            k_cosines.append(
                cosine_similarity(k_current, k_next)
            )

            v_cosines.append(
                cosine_similarity(v_current, v_next)
            )

            k_l2.append(
                relative_l2(k_current, k_next)
            )

            v_l2.append(
                relative_l2(v_current, v_next)
            )

        results[layer_idx] = {
            "k_cosine": sum(k_cosines) / len(k_cosines),
            "v_cosine": sum(v_cosines) / len(v_cosines),
            "k_relative_l2": sum(k_l2) / len(k_l2),
            "v_relative_l2": sum(v_l2) / len(v_l2),
        }

    # Print layer-wise results
    print("\nLayer-wise language K/V stability:\n")

    print(
        f"{'Layer':<8}"
        f"{'K cosine':<15}"
        f"{'V cosine':<15}"
        f"{'K rel-L2':<15}"
        f"{'V rel-L2':<15}"
    )

    for layer_idx, values in results.items():

        print(
            f"{layer_idx:<8}"
            f"{values['k_cosine']:<15.6f}"
            f"{values['v_cosine']:<15.6f}"
            f"{values['k_relative_l2']:<15.6f}"
            f"{values['v_relative_l2']:<15.6f}"
        )


    layers = np.array(list(results.keys()))

    k_cos = np.array([results[l]["k_cosine"] for l in layers])
    v_cos = np.array([results[l]["v_cosine"] for l in layers])
    k_l2  = np.array([results[l]["k_relative_l2"] for l in layers])
    v_l2  = np.array([results[l]["v_relative_l2"] for l in layers])


    # --------------------------------------------------
    # 1. Relative L2 across layers
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(layers, k_l2, marker="o", markersize=3, label="K")
    plt.plot(layers, v_l2, marker="o", markersize=3, label="V")

    plt.xlabel("Transformer Layer")
    plt.ylabel("Mean Relative L2")
    plt.title("Language K/V Change Across Transformer Layers")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "experiments/kv_analysis/outputs/kv_relative_l2_by_layer.png",
        dpi=300
    )
    plt.close()


    # --------------------------------------------------
    # 2. Cosine similarity across layers
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(layers, k_cos, marker="o", markersize=3, label="K")
    plt.plot(layers, v_cos, marker="o", markersize=3, label="V")

    plt.xlabel("Transformer Layer")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Language K/V Similarity Across Transformer Layers")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "experiments/kv_analysis/outputs/kv_cosine_by_layer.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------
    # Temporal analysis
    # --------------------------------------------------

    os.makedirs(PLOTS_DIR, exist_ok=True)

    temporal_results = {}

    for layer_idx in SELECTED_LAYERS:

        temporal_results[layer_idx] = {
            "k_cosine": [],
            "v_cosine": [],
            "k_relative_l2": [],
            "v_relative_l2": [],
        }

        for step in range(len(trajectory_kv) - 1):

            current = trajectory_kv[step][layer_idx]
            next_step = trajectory_kv[step + 1][layer_idx]

            k_current = current["k"]
            k_next = next_step["k"]

            v_current = current["v"]
            v_next = next_step["v"]

            temporal_results[layer_idx]["k_cosine"].append(
                cosine_similarity(k_current, k_next)
            )

            temporal_results[layer_idx]["v_cosine"].append(
                cosine_similarity(v_current, v_next)
            )

            temporal_results[layer_idx]["k_relative_l2"].append(
                relative_l2(k_current, k_next)
            )

            temporal_results[layer_idx]["v_relative_l2"].append(
                relative_l2(v_current, v_next)
            )


    def plot_temporal_metric(metric, ylabel, filename):

        plt.figure(figsize=(12, 6))

        for layer_idx in SELECTED_LAYERS:

            values = temporal_results[layer_idx][metric]

            # Comparison i represents step i -> i+1
            steps = range(1, len(values) + 1)

            plt.plot(
                steps,
                values,
                label=f"Layer {layer_idx}",
            )

        plt.xlabel("Control-step transition")
        plt.ylabel(ylabel)
        plt.title(f"Language KV Temporal Stability — {ylabel}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(
            PLOTS_DIR,
            filename,
        )

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

        print(f"Saved: {output_path}")


    plot_temporal_metric(
        "k_cosine",
        "K Cosine Similarity",
        "temporal_k_cosine.png",
    )

    plot_temporal_metric(
        "v_cosine",
        "V Cosine Similarity",
        "temporal_v_cosine.png",
    )

    plot_temporal_metric(
        "k_relative_l2",
        "K Relative L2 Change",
        "temporal_k_relative_l2.png",
    )

    plot_temporal_metric(
        "v_relative_l2",
        "V Relative L2 Change",
        "temporal_v_relative_l2.png",
    )


    # --------------------------------------------------
    # Token-wise analysis
    # --------------------------------------------------

    selected_layer = 31

    num_tokens = trajectory_kv[0][selected_layer]["k"].shape[1]

    token_results = []

    for token_idx in range(num_tokens):

        k_cosines = []
        v_cosines = []

        k_l2 = []
        v_l2 = []

        for step in range(len(trajectory_kv) - 1):

            current = trajectory_kv[step][selected_layer]
            next_step = trajectory_kv[step + 1][selected_layer]

            # One token only: [4096]
            k_current = current["k"][0, token_idx, :]
            k_next = next_step["k"][0, token_idx, :]

            v_current = current["v"][0, token_idx, :]
            v_next = next_step["v"][0, token_idx, :]

            k_cosines.append(
                cosine_similarity(k_current, k_next)
            )

            v_cosines.append(
                cosine_similarity(v_current, v_next)
            )

            k_l2.append(
                relative_l2(k_current, k_next)
            )

            v_l2.append(
                relative_l2(v_current, v_next)
            )

        token_results.append(
            {
                "token_idx": token_idx,
                "k_cosine": sum(k_cosines) / len(k_cosines),
                "v_cosine": sum(v_cosines) / len(v_cosines),
                "k_relative_l2": sum(k_l2) / len(k_l2),
                "v_relative_l2": sum(v_l2) / len(v_l2),
            }
        )


    print(f"\nToken-wise stability at layer {selected_layer}:\n")

    print(
        f"{'Idx':<6}"
        f"{'Token':<20}"
        f"{'K cosine':<14}"
        f"{'V cosine':<14}"
        f"{'K rel-L2':<14}"
        f"{'V rel-L2':<14}"
    )

    for result in token_results:

        token_idx = result["token_idx"]

        # language position 0 is BOS,
        # then the remaining language tokens correspond to prompt tokens 1...
        token = tokens[token_idx] if token_idx < len(tokens) else "?"

        print(
            f"{token_idx:<6}"
            f"{token:<20}"
            f"{result['k_cosine']:<14.6f}"
            f"{result['v_cosine']:<14.6f}"
            f"{result['k_relative_l2']:<14.6f}"
            f"{result['v_relative_l2']:<14.6f}"
        )


    # --------------------------------------------------
    # Token × time heatmap
    # --------------------------------------------------


    selected_layer = 31

    num_steps = len(trajectory_kv)
    num_tokens = trajectory_kv[0][selected_layer]["v"].shape[1]

    # rows = tokens
    # columns = transitions: step 0->1, 1->2, ...
    v_l2_heatmap = np.zeros((num_tokens, num_steps - 1))

    for token_idx in range(num_tokens):

        for step in range(num_steps - 1):

            current_v = trajectory_kv[step][selected_layer]["v"][
                0, token_idx, :
            ]

            next_v = trajectory_kv[step + 1][selected_layer]["v"][
                0, token_idx, :
            ]

            v_l2_heatmap[token_idx, step] = relative_l2(
                current_v,
                next_v
            )

    fig, ax = plt.subplots(figsize=(16, 10))

    im = ax.imshow(
        v_l2_heatmap,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Control-step transition")
    ax.set_ylabel("Language token")

    ax.set_title(
        f"Token × Time V Relative L2 — Layer {selected_layer}"
    )

    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(
        [f"{i}: {tokens[i]}" for i in range(num_tokens)]
    )

    phase_boundaries = {
        "reach/align": 30,
        "grasp": 47,
        "lift/transport": 57,
        "approach plate": 75,
        "place/release": 87,
    }

    for label, step in phase_boundaries.items():
        ax.axvline(
            step,
            linestyle="--",
            linewidth=1.2,
        )

        ax.text(
            step + 0.5,
            -0.7,
            label,
            rotation=90,
            va="bottom",
            fontsize=8,
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative L2 change")

    plt.tight_layout()

    heatmap_path = os.path.join(
        PLOTS_DIR,
        f"token_time_v_relative_l2_layer{selected_layer}.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved token × time heatmap to: {heatmap_path}")



   # ============================================================
    # Semantic block × time analysis
    # ============================================================

    layer_idx = 31

    blocks = {
        # "pick up the black bowl"
        "pick_target": [10, 11, 12, 13, 14, 15],

        # "between the plate and the ramekin"
        "spatial_reference": [16, 17, 18, 19, 20, 21, 22, 23],

        # "and place it on the plate"
        "place_target": [24, 25, 26, 27, 28, 29],
    }

    block_l2 = {
        block_name: []
        for block_name in blocks
    }

    for step in range(num_steps - 1):
        for block_name, token_indices in blocks.items():

            current_v = trajectory_kv[step][layer_idx]["v"][
                0, token_indices, :
            ]

            next_v = trajectory_kv[step + 1][layer_idx]["v"][
                0, token_indices, :
            ]

            current_v = current_v.flatten()
            next_v = next_v.flatten()

            change = relative_l2(current_v, next_v)
            block_l2[block_name].append(change)


    # Print summary
    print("\nSemantic block V stability — Layer 31")
    print("-" * 55)

    for block_name, values in block_l2.items():
        print(
            f"{block_name:20s} "
            f"mean={np.mean(values):.6f}  "
            f"std={np.std(values):.6f}"
        )


    # Plot
    phase_boundaries = {
        "reach/align": 30,
        "grasp": 47,
        "lift/transport": 57,
        "approach plate": 75,
        "place/release": 87,
    }

    plt.figure(figsize=(12, 5))

    for block_name, values in block_l2.items():
        plt.plot(values, label=block_name)

    for label, step in phase_boundaries.items():
        plt.axvline(
            step,
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

    plt.xlabel("Control-step transition")
    plt.ylabel("V Relative L2")
    plt.title("Semantic Block V Dynamics vs. Task Phase — Layer 31")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "experiments/kv_analysis/outputs/"
        "semantic_block_v_relative_l2_phases.png",
        dpi=200,
    )

    plt.close() 

    # ============================================================
    # Memory-element analysis across all layers
    # ============================================================

    num_layers = 32

    # Manually defined semantic memory elements
    #
    # 10 pick
    # 11 up
    # 12 the
    # 13 black
    # 14 bow
    # 15 l
    # 16 between
    # 17 the
    # 18 plate
    # 19 and
    # 20 the
    # 21 r
    # 22 ame
    # 23 kin
    # 24 and
    # 25 place
    # 26 it
    # 27 on
    # 28 the
    # 29 plate

    memories = {
        "pick up": [10, 11],
        "black bowl": [12, 13, 14, 15],
        "spatial reference": [16, 17, 18, 19, 20, 21, 22, 23],
        "place it": [25, 26],
        "on the plate": [27, 28, 29],
    }

    memory_names = list(memories.keys())

    # ------------------------------------------------------------
    # Build complete tensor:
    #
    # shape:
    #   [num_memories, num_layers, num_transitions]
    #
    # value:
    #   V relative L2 for one memory, layer, and transition
    # ------------------------------------------------------------

    memory_v_l2 = np.zeros(
        (
            len(memories),
            num_layers,
            num_steps - 1,
        ),
        dtype=np.float32,
    )

    for memory_idx, (memory_name, token_indices) in enumerate(memories.items()):

        for layer_idx in range(num_layers):

            for step in range(num_steps - 1):

                current_v = trajectory_kv[step][layer_idx]["v"][
                    0, token_indices, :
                ]

                next_v = trajectory_kv[step + 1][layer_idx]["v"][
                    0, token_indices, :
                ]

                # Treat the whole token sequence as one memory element
                current_v = current_v.flatten()
                next_v = next_v.flatten()

                memory_v_l2[
                    memory_idx,
                    layer_idx,
                    step,
                ] = relative_l2(
                    current_v,
                    next_v,
                )


    # ============================================================
    # Statistics across control-step transitions
    # ============================================================

    memory_median = np.median(
        memory_v_l2,
        axis=2,
    )

    memory_q25 = np.percentile(
        memory_v_l2,
        25,
        axis=2,
    )

    memory_q75 = np.percentile(
        memory_v_l2,
        75,
        axis=2,
    )

    memory_iqr = memory_q75 - memory_q25

    # ============================================================
    # Plot 1:
    # Memory stability across layers — Median + IQR
    # ============================================================

    layers = np.arange(num_layers)

    plt.figure(figsize=(12, 6))

    for memory_idx, memory_name in enumerate(memory_names):

        median = memory_median[memory_idx]
        q25 = memory_q25[memory_idx]
        q75 = memory_q75[memory_idx]

        line, = plt.plot(
            layers,
            median,
            label=memory_name,
            linewidth=2,
        )

        plt.fill_between(
            layers,
            q25,
            q75,
            alpha=0.15,
            color=line.get_color(),
        )

    plt.xlabel("Layer")
    plt.ylabel("V Relative L2")
    plt.title(
        "Memory-Element V Dynamics Across Layers\n"
        "Median and Interquartile Range Across Control-Step Transitions"
    )

    plt.xticks(np.arange(0, num_layers, 2))
    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        "experiments/kv_analysis/outputs/"
        "memory_v_median_iqr_layers.png",
        dpi=200,
    )

    plt.close()

    # ============================================================
    # Plot 2:
    # Median V dynamics heatmap
    # ============================================================

    fig, ax = plt.subplots(figsize=(13, 4.5))

    im = ax.imshow(
        memory_median,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Memory element")

    ax.set_title(
        "Median V Relative L2 by Memory Element and Layer"
    )

    ax.set_xticks(np.arange(num_layers))
    ax.set_xticklabels(
        np.arange(num_layers),
        fontsize=8,
    )

    ax.set_yticks(
        np.arange(len(memory_names))
    )

    ax.set_yticklabels(
        memory_names
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Median V Relative L2"
    )

    plt.tight_layout()

    plt.savefig(
        "experiments/kv_analysis/outputs/"
        "memory_v_median_heatmap.png",
        dpi=200,
    )

    plt.close()

    # ============================================================
    # Plot 3:
    # Temporal variability heatmap — IQR
    # ============================================================

    fig, ax = plt.subplots(figsize=(13, 4.5))

    im = ax.imshow(
        memory_iqr,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Memory element")

    ax.set_title(
        "Temporal Variability of V Dynamics — IQR by Memory and Layer"
    )

    ax.set_xticks(
        np.arange(num_layers)
    )

    ax.set_xticklabels(
        np.arange(num_layers),
        fontsize=8,
    )

    ax.set_yticks(
        np.arange(len(memory_names))
    )

    ax.set_yticklabels(
        memory_names
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "IQR of V Relative L2"
    )

    plt.tight_layout()

    plt.savefig(
        "experiments/kv_analysis/outputs/"
        "memory_v_iqr_heatmap.png",
        dpi=200,
    )

    plt.close()

    np.save(
        "experiments/kv_analysis/outputs/"
        "memory_v_l2_all_layers.npy",
        memory_v_l2,
    )

    np.save(
        "experiments/kv_analysis/outputs/"
        "memory_v_median.npy",
        memory_median,
    )

    np.save(
        "experiments/kv_analysis/outputs/"
        "memory_v_iqr.npy",
        memory_iqr,
    )

if __name__ == "__main__":
    main()
