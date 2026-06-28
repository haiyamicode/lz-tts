#!/usr/bin/env python3
"""Convert a single-speaker MatchaTTS checkpoint to multi-speaker.

Expands all tensors affected by adding spk_emb_dim (64) to channel dimensions,
and vocabs from old_n_vocab to new_n_vocab.
"""
import argparse, sys
from pathlib import Path

TARGET_N_VOCAB = 151  # vocab size for multilingual fused phoneme+text tokens
TARGET_N_SPKS = 22     # number of languages/speakers
SPK_EMB_DIM = 64
N_CHANNELS = 192      # original n_channels (English model)
NEW_CHANNELS = N_CHANNELS + SPK_EMB_DIM  # 256


def expand_tensor(tensor, target_shape):
    """Copy old values into a zero tensor of target_shape."""
    if tensor.shape == target_shape:
        return tensor.clone()
    new = tensor.new_zeros(target_shape)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
    new[slices] = tensor[slices]
    return new


def convert_checkpoint(src_path, dst_path):
    import torch

    REPO = Path(__file__).resolve().parents[2]
    CONFIG = REPO / "local" / "configs" / "starling" / "train_andrew_edge_multilingual.yaml"
    sys.path.insert(0, str(REPO))

    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    cfg = OmegaConf.load(str(CONFIG))
    print(f"Config: n_vocab={cfg.model.n_vocab} n_spks={cfg.model.n_spks} n_channels={cfg.model.encoder.encoder_params.n_channels}")

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    model = instantiate(cfg.model)
    target_shapes = {k: v.shape for k, v in model.state_dict().items()}

    expanded = 0
    for key in list(state.keys()):
        if key not in target_shapes:
            continue
        ts = target_shapes[key]
        if state[key].shape != ts:
            state[key] = expand_tensor(state[key], ts)
            expanded += 1

    # Vocab embedding: copy old (preserve learned weights), random init rest
    old_emb = state["encoder.emb.weight"]
    state["encoder.emb.weight"] = expand_tensor(old_emb, target_shapes["encoder.emb.weight"])
    state["encoder.emb.weight"][: old_emb.shape[0]] = old_emb

    # Speaker embedding: zeros
    state["spk_emb.weight"] = torch.zeros(TARGET_N_SPKS, SPK_EMB_DIM)

    ckpt["hyper_parameters"]["n_vocab"] = TARGET_N_VOCAB
    ckpt["hyper_parameters"]["n_spks"] = TARGET_N_SPKS

    torch.save(ckpt, dst_path)
    print(f"Converted: {src_path} -> {dst_path}")
    print(f"  n_vocab: {TARGET_N_VOCAB}, n_spks: {TARGET_N_SPKS}, expanded: {expanded} tensors")

    # Verify
    model2 = instantiate(cfg.model)
    model2.load_state_dict(torch.load(dst_path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    print("  VERIFIED: strict=True load OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Source single-speaker checkpoint")
    parser.add_argument("dst", help="Destination multi-speaker checkpoint")
    args = parser.parse_args()
    convert_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    main()
