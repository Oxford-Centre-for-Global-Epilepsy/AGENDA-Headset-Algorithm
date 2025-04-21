import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_weights_by_level(model, dataloader, config, level_index, class_labels, class_names):
    print(f"🧠 Plotting attention weights for Level {level_index + 1}...")
    attn_per_class = {label: [] for label in class_labels}

    with torch.no_grad():
        for batch in dataloader:
            x = batch["data"].to(config.device)
            y = batch["labels"].to(config.device)[:, level_index]
            m = batch["label_mask"].to(config.device)[:, level_index].bool()
            attn_mask = batch["attention_mask"].to(config.device)

            out = model(x, attention_mask=attn_mask, return_attn_weights=True)
            weights = out.get("attention_weights")
            if weights is None:
                print("⚠️ No attention weights found in model output.")
                continue
            weights = weights.squeeze(1).detach().cpu().numpy()

            for i in range(len(y)):
                if m[i]:
                    label = y[i].item()
                    if label in class_labels:
                        attn_per_class[label].append(weights[i])

    any_data = False
    for label in class_labels:
        traces = attn_per_class[label]
        if not traces:
            print(f"⚠️ No attention weights collected for class '{class_names[class_labels.index(label)]}'")
            continue
        attn_stack = np.stack(traces)
        mean_attn = attn_stack.mean(axis=0)
        plt.plot(mean_attn, label=class_names[class_labels.index(label)])
        any_data = True

    if not any_data:
        print("❌ No valid attention weights found for any class. Skipping plot.")
        return

    plt.title(f"Attention over time by class (Level {level_index + 1})")
    plt.xlabel("Epochs")
    plt.ylabel("Attention Weight")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(config.output_dir, f"attention_weights_per_class_level{level_index + 1}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved attention plot for Level {level_index + 1} to {plot_path}")

def plot_eeg_with_attention(signal, attention_weights, channel_name="Cz", epoch_len=256, output_path="eeg_with_attention.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    time = np.arange(len(signal))
    ax.plot(time, signal, color='black', label=f"EEG ({channel_name})")

    max_weight = np.max(attention_weights)
    num_epochs = len(attention_weights)
    for i in range(num_epochs):
        start = i * epoch_len
        end = start + epoch_len
        if start >= len(signal):
            break
        alpha = 0.5 * (attention_weights[i] / max_weight) if max_weight > 0 else 0
        ax.axvspan(start, min(end, len(signal)), color='red', alpha=alpha)

    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("EEG Amplitude")
    ax.legend()
    ax.set_xlim(0, len(signal))
    plt.title(f"EEG with Overlaid Attention Weights - {channel_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved EEG with attention overlay to {output_path}")

def plot_multichannel_eeg_with_attention(eeg, attn_weights, channel_names=None, epoch_len=256, output_path="eeg_multi_with_attention.png"):
    eeg = eeg.transpose(1, 0, 2)  # [C, E, T]
    C, E, T = eeg.shape
    time = np.arange(E * T)

    fig, axes = plt.subplots(C, 1, figsize=(14, 2 * C), sharex=True)
    if C == 1:
        axes = [axes]

    max_weight = np.max(attn_weights)
    for i in range(C):
        channel_signal = eeg[i].reshape(-1)
        axes[i].plot(time, channel_signal, color='black', lw=0.7)

        for e in range(E):
            start = e * T
            end = start + T
            if start >= len(channel_signal):
                break
            alpha = 0.5 * (attn_weights[e] / max_weight) if max_weight > 0 else 0
            axes[i].axvspan(start, min(end, len(channel_signal)), color='red', alpha=alpha)

        axes[i].set_ylabel(channel_names[i] if channel_names else f"Ch {i}")
        axes[i].set_xlim(0, E * T)

    axes[-1].set_xlabel("Time (samples)")
    plt.suptitle("EEG Channels with Overlaid Attention Weights")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved multichannel EEG + attention overlay to {output_path}")

def save_attention_and_eeg_plots(eeg_batch, attn_weights, channel_idx=0, output_dir="./", filename_prefix=""):
    if isinstance(eeg_batch, torch.Tensor):
        eeg_batch = eeg_batch.detach().cpu().numpy()
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    signal = eeg_batch[:, channel_idx, :].reshape(-1)
    filename_prefix = filename_prefix or "eeg"

    plot_attention_weights(
        attn_weights,
        title="Attention Over Epochs",
        save_path=os.path.join(output_dir, f"{filename_prefix}_attention_over_epochs.png")
    )

    plot_eeg_with_attention(
        signal,
        attn_weights,
        channel_name=f"Ch {channel_idx}",
        output_path=os.path.join(output_dir, f"{filename_prefix}_attention_overlay_eeg.png")
    )