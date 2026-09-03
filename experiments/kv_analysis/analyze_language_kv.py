"""
Analyze how language-token K/V representations change across
consecutive OpenVLA control steps in a LIBERO trajectory.
"""

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os

from transformers import AutoTokenizer

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



if __name__ == "__main__":
    main()
