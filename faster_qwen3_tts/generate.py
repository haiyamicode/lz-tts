#!/usr/bin/env python3
"""
Non-streaming generation loop using CUDA graphs for both predictor and talker.
"""
import time
from typing import Optional, Sequence, Tuple, Union

import torch

from .predictor_graph import PredictorGraph
from .sampling import apply_repetition_penalty, sample_logits
from .talker_graph import TalkerGraph


def _normalize_batch_max_new_tokens(
    max_new_tokens: Union[Sequence[int], torch.Tensor],
    batch_size: int,
) -> list[int]:
    if isinstance(max_new_tokens, torch.Tensor):
        values = [int(value) for value in max_new_tokens.flatten().tolist()]
    else:
        if isinstance(max_new_tokens, int):
            raise ValueError("batched max_new_tokens must be specified per item")
        values = [int(value) for value in max_new_tokens]

    if len(values) != batch_size:
        raise ValueError(
            f"max_new_tokens length must match batch size ({batch_size}); got {len(values)}"
        )
    if any(value < 0 for value in values):
        raise ValueError("max_new_tokens must be non-negative")
    return values


def _apply_repetition_penalty_batch(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0 or token_history.numel() == 0:
        return logits
    for batch_idx in range(logits.shape[0]):
        unique_toks = token_history[:, batch_idx].unique()
        tok_logits = logits[batch_idx, unique_toks]
        logits[batch_idx, unique_toks] = torch.where(
            tok_logits > 0,
            tok_logits / repetition_penalty,
            tok_logits * repetition_penalty,
        )
    return logits


@torch.inference_mode()
def fast_generate_batch_graph(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    max_new_tokens: Union[Sequence[int], torch.Tensor],
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> Tuple[list[torch.Tensor], dict]:
    """Batched CUDA-graph generation for a fixed captured batch size."""
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = talker_input_embeds.device
    batch_size = int(talker_input_embeds.shape[0])
    max_new_tokens_values = _normalize_batch_max_new_tokens(max_new_tokens, batch_size)
    max_new_tokens_tensor = torch.tensor(max_new_tokens_values, dtype=torch.long, device=device)
    generation_max_new_tokens = max(max_new_tokens_values, default=0)

    if generation_max_new_tokens <= 0:
        timing = {
            "prefill_ms": 0.0,
            "decode_s": 0.0,
            "steps": 0,
            "ms_per_step": 0.0,
            "steps_per_s": 0.0,
            "batch_size": batch_size,
            "item_steps": [0] * batch_size,
        }
        return [torch.empty((0, config.num_code_groups), dtype=torch.long, device=device) for _ in range(batch_size)], timing

    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    for i in range(suppress_start, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True

    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor_codec_embeds = predictor.get_input_embeddings()

    t_start = time.time()
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )

    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step

    logits = out.logits[:, -1, :]
    token = sample_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        suppress_mask=suppress_mask,
        suppress_tokens=[eos_id] if min_new_tokens > 0 else None,
    )

    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(attention_mask, rope_deltas)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_prefill = time.time() - t_start

    t_decode_start = time.time()
    all_codec_ids = []
    first_eos_steps = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step_idx in range(generation_max_new_tokens):
        cap_done = max_new_tokens_tensor <= step_idx
        newly_done = token == eos_id
        first_eos_steps = torch.where(
            (first_eos_steps < 0) & newly_done,
            torch.full_like(first_eos_steps, step_idx),
            first_eos_steps,
        )
        done |= newly_done | cap_done
        if bool(done.all().item()):
            break

        token_for_step = torch.where(done, torch.full_like(token, eos_id), token)
        last_id_hidden = talker_codec_embed(token_for_step.unsqueeze(1))  # [B, 1, H]
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)  # [B, 2, H]
        codebook_token_ids = predictor_graph.run(pred_input)  # [B, 15]
        if codebook_token_ids.dim() == 1:
            codebook_token_ids = codebook_token_ids.unsqueeze(0)

        all_cb = torch.cat([token_for_step.view(batch_size, 1), codebook_token_ids], dim=1)  # [B, 16]
        all_codec_ids.append(all_cb.detach())

        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[:, i].unsqueeze(1)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            break

        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)
        logits = talker_codec_head(hidden_states[:, -1, :])

        if repetition_penalty != 1.0 and all_codec_ids:
            history = torch.stack([c[:, 0] for c in all_codec_ids], dim=0)
            logits = _apply_repetition_penalty_batch(logits, history, repetition_penalty)

        suppress_eos = len(all_codec_ids) < min_new_tokens
        next_token = sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )
        token = torch.where(done, torch.full_like(next_token, eos_id), next_token)
        past_hidden = hidden_states[:, -1:, :].clone()
        gen_step += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_decode = time.time() - t_decode_start

    if not all_codec_ids:
        timing = {
            "prefill_ms": t_prefill * 1000,
            "decode_s": t_decode,
            "steps": 0,
            "ms_per_step": 0.0,
            "steps_per_s": 0.0,
            "batch_size": batch_size,
            "item_steps": [0] * batch_size,
        }
        return [torch.empty((0, num_code_groups), dtype=torch.long, device=device) for _ in range(batch_size)], timing

    codec_stack = torch.stack(all_codec_ids, dim=1)  # [B, steps, 16]
    total_steps = codec_stack.shape[1]
    lengths = torch.where(
        first_eos_steps >= 0,
        first_eos_steps,
        torch.full_like(first_eos_steps, total_steps),
    )
    lengths = torch.minimum(lengths, torch.minimum(max_new_tokens_tensor, torch.full_like(lengths, total_steps)))
    codec_ids_list = [
        codec_stack[i, : int(length.item()), :].contiguous()
        for i, length in enumerate(lengths)
    ]

    timing = {
        "prefill_ms": t_prefill * 1000,
        "decode_s": t_decode,
        "steps": total_steps,
        "ms_per_step": (t_decode / total_steps * 1000) if total_steps > 0 else 0.0,
        "steps_per_s": (total_steps / t_decode) if t_decode > 0 else 0.0,
        "batch_size": batch_size,
        "item_steps": [int(length.item()) for length in lengths],
    }
    return codec_ids_list, timing


@torch.inference_mode()
def fast_generate(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    subtalker_dosample: Optional[bool] = None,
    subtalker_top_k: Optional[int] = None,
    subtalker_top_p: Optional[float] = None,
    subtalker_temperature: Optional[float] = None,
    parity_mode: bool = False,
) -> Tuple[Optional[torch.Tensor], dict]:
    """
    Fast autoregressive generation with CUDA-graphed predictor and talker.
    """
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = talker_input_embeds.device
    
    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    for i in range(suppress_start, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True

    if parity_mode:
        suppress_tokens = [i for i in range(suppress_start, vocab_size) if i != eos_id]
        t_start = time.time()
        talker_result = talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            suppress_tokens=suppress_tokens,
            subtalker_dosample=subtalker_dosample if subtalker_dosample is not None else do_sample,
            subtalker_top_k=subtalker_top_k if subtalker_top_k is not None else top_k,
            subtalker_top_p=subtalker_top_p if subtalker_top_p is not None else top_p,
            subtalker_temperature=subtalker_temperature if subtalker_temperature is not None else temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        talker_codes = torch.stack(
            [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None],
            dim=1,
        )
        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == eos_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])
        talker_codes_list = [talker_codes[i, :length, :] for i, length in enumerate(effective_lengths)]

        torch.cuda.synchronize()
        total_time = time.time() - t_start
        steps = int(talker_codes_list[0].shape[0]) if talker_codes_list else 0
        timing = {
            'prefill_ms': 0.0,
            'decode_s': total_time,
            'steps': steps,
            'ms_per_step': (total_time / steps * 1000) if steps > 0 else 0.0,
            'steps_per_s': (steps / total_time) if total_time > 0 else 0.0,
        }
        return talker_codes_list[0] if talker_codes_list else None, timing
    
    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor_codec_embeds = predictor.get_input_embeddings()
    
    # === PREFILL (still uses HF forward for variable-length prefill) ===
    t_start = time.time()
    
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )
    
    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step
    
    logits = out.logits[:, -1, :]
    suppress_eos = min_new_tokens > 0
    token = sample_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        suppress_mask=suppress_mask,
        suppress_tokens=[eos_id] if suppress_eos else None,
    )
    
    # Copy prefill KV cache into talker graph's static cache
    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    # Sync padding mask + rope deltas for decode parity
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(attention_mask, rope_deltas)
    
    torch.cuda.synchronize()
    t_prefill = time.time() - t_start
    
    # === DECODE LOOP ===
    t_decode_start = time.time()
    all_codec_ids = []
    
    for step_idx in range(max_new_tokens):
        if token.item() == eos_id:
            break
        
        # --- CUDA-Graphed Code Predictor ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))  # [1, 1, H]
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)  # [1, 2, H]
        codebook_token_ids = predictor_graph.run(pred_input)  # [15] long tensor
        
        # Build full codec: [first_cb, cb1, ..., cb15]
        all_cb = torch.cat([token.view(1), codebook_token_ids])  # [16]
        all_codec_ids.append(all_cb.detach())
        
        # --- Build input embedding for talker ---
        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
        
        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed
        
        # --- CUDA-Graphed Talker decode step ---
        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            # Stop if we exceed max_seq_len
            break
        
        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)
        # hidden_states is the static output buffer - use it immediately
        
        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
        
        if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
            history = torch.stack([c[0] for c in all_codec_ids])
            logits = apply_repetition_penalty(logits, history, repetition_penalty)

        suppress_eos = len(all_codec_ids) < min_new_tokens
        token = sample_logits(
            logits.squeeze(0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )
        past_hidden = hidden_states[:, -1:, :].clone()  # clone since it's the static buffer
        gen_step += 1
    
    torch.cuda.synchronize()
    t_decode = time.time() - t_decode_start
    
    n_steps = len(all_codec_ids)
    timing = {
        'prefill_ms': t_prefill * 1000,
        'decode_s': t_decode,
        'steps': n_steps,
        'ms_per_step': (t_decode / n_steps * 1000) if n_steps > 0 else 0,
        'steps_per_s': (n_steps / t_decode) if t_decode > 0 else 0,
    }
    
    if all_codec_ids:
        return torch.stack(all_codec_ids), timing
    return None, timing
