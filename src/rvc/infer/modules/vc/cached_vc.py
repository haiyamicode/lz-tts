"""
Cached VC wrapper with LRU model management
"""
import logging
import os
from collections import OrderedDict
from typing import Optional

import torch

from infer.modules.vc.modules import VC
from infer.modules.vc.model_cache import ModelCache
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import get_index_path_from_model, load_hubert
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

logger = logging.getLogger(__name__)


class CachedVC(VC):
    """VC class with LRU caching for model management"""
    
    def __init__(self, config, max_cache_size: int = 5):
        """
        Initialize CachedVC with model caching
        
        Args:
            config: RVC config object
            max_cache_size: Maximum number of models to keep in cache
        """
        super().__init__(config)
        self.model_cache = ModelCache(max_size=max_cache_size)
        self.current_model_name: Optional[str] = None
        logger.info(f"Initialized CachedVC with cache size: {max_cache_size}")
    
    def get_vc(self, sid, *to_return_protect):
        """Load model with caching support"""
        logger.info(f"Get sid: {sid}")
        
        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }
        
        if sid == "" or sid == []:
            self._clear_current_model()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        
        # Check if model is already loaded
        if self.current_model_name == sid:
            logger.info(f"Model {sid} already active")
            n_spk = self.cpt["config"][-3]
            index = {"value": get_index_path_from_model(sid), "__type__": "update"}
            return (
                (
                    {"visible": True, "maximum": n_spk, "__type__": "update"},
                    to_return_protect0,
                    to_return_protect1,
                    index,
                    index,
                )
                if to_return_protect
                else {"visible": True, "maximum": n_spk, "__type__": "update"}
            )
        # Try to get from cache
        cached_model = self.model_cache.get(sid)
        if cached_model:
            self._restore_from_cache(sid, cached_model)
        else:
            self._load_model_from_disk(sid)
            self._save_to_cache(sid)
        
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info(f"Select index: {index['value']}")
        
        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )
    
    def _load_from_cpt(self, sid: str, cpt: dict) -> None:
        """
        Initialize network and pipeline from an in-memory checkpoint dict.
        """
        self.cpt = cpt
        self.tgt_sr = self.cpt["config"][-1]
        # Ensure speaker count matches embedding size
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")
        
        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }
        
        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)
        
        # Remove unused encoder to save memory
        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q
        
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        
        self.pipeline = Pipeline(self.tgt_sr, self.config)
        self.n_spk = self.cpt["config"][-3]
        self.current_model_name = sid
    
    def _load_model_from_disk(self, sid: str) -> None:
        """Load model from disk"""
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading from disk: {person}")
        
        cpt = torch.load(person, map_location="cpu")
        self._load_from_cpt(sid, cpt)
    
    def _save_to_cache(self, sid: str) -> None:
        """Save current model state to cache"""
        model_data = {
            'net_g': self.net_g,
            'cpt': self.cpt,
            'tgt_sr': self.tgt_sr,
            'if_f0': self.if_f0,
            'version': self.version,
            'n_spk': self.n_spk,
            'pipeline': self.pipeline,
        }
        self.model_cache.put(sid, model_data)
    
    def _restore_from_cache(self, sid: str, cached_model: dict) -> None:
        """Restore model state from cache"""
        logger.info(f"Restoring model from cache: {sid}")
        self.net_g = cached_model['net_g']
        self.cpt = cached_model['cpt']
        self.tgt_sr = cached_model['tgt_sr']
        self.if_f0 = cached_model['if_f0']
        self.version = cached_model['version']
        self.n_spk = cached_model['n_spk']
        self.pipeline = cached_model['pipeline']
        self.current_model_name = sid
    
    def _clear_current_model(self) -> None:
        """Clear current model references"""
        if self.current_model_name and self.hubert_model is not None:
            logger.info("Clearing current model")
            self.current_model_name = None
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return self.model_cache.get_stats()

    def load_mixed_model(
        self,
        base_sid: str,
        expr_sid: str,
        expr_intensity: float,
    ) -> str:
        """
        Create an in-memory mixed model between a base and an expression checkpoint.

        The resulting weights are:
            (1 - expr_intensity) * base + expr_intensity * expression

        Works with any intensity in [0, 2]:
        - 0: pure base model
        - 0.5: 50% base + 50% expression
        - 1: pure expression model
        - 1.5: extrapolate beyond expression
        - 2: maximum extrapolation

        No checkpoint is written to disk; the mixed model is loaded into this
        CachedVC instance and stored in the in-memory model cache.
        """
        expr_intensity = float(expr_intensity)
        # Handle edge case: intensity 0 means just use base
        if expr_intensity <= 0.0:
            logger.info(
                "Expression intensity <= 0, falling back to base model %s", base_sid
            )
            self.get_vc(base_sid, 0.33, 0.25)
            return base_sid

        mix_sid = f"{base_sid}_x_{expr_sid}_alpha{expr_intensity}"

        # Try to reuse from cache if we mixed this combination before
        cached_model = self.model_cache.get(mix_sid)
        if cached_model:
            logger.info("Restoring mixed model from cache: %s", mix_sid)
            self._restore_from_cache(mix_sid, cached_model)
            return mix_sid

        weight_root = os.getenv("weight_root", "assets/weights")
        base_path = os.path.join(weight_root, base_sid)
        expr_path = os.path.join(weight_root, expr_sid)

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base model not found: {base_path}")
        if not os.path.exists(expr_path):
            raise FileNotFoundError(f"Expression model not found: {expr_path}")

        logger.info(
            "Mixing base model %s and expression model %s (intensity=%.3f)",
            base_sid,
            expr_sid,
            expr_intensity,
        )

        # --- Exact same fusion logic as ckpt Processing tab (process_ckpt.merge) ---
        base_ckpt = torch.load(base_path, map_location="cpu")
        expr_ckpt = torch.load(expr_path, map_location="cpu")

        cfg = base_ckpt["config"]

        def _extract(ckpt_dict):
            if "model" in ckpt_dict:
                a = ckpt_dict["model"]
                opt = OrderedDict()
                opt["weight"] = {}
                for key in a.keys():
                    if "enc_q" in key:
                        continue
                    opt["weight"][key] = a[key]
                return opt["weight"]
            return ckpt_dict["weight"]

        base_weight = _extract(base_ckpt)
        expr_weight = _extract(expr_ckpt)

        if sorted(list(base_weight.keys())) != sorted(list(expr_weight.keys())):
            raise RuntimeError(
                "Fail to merge the models. The model architectures are not the same."
            )

        alpha1 = 1.0 - expr_intensity  # treated as Weight (w) for Model A (base)
        mixed_weight = OrderedDict()
        for key in base_weight.keys():
            if key == "emb_g.weight" and base_weight[key].shape != expr_weight[key].shape:
                min_shape0 = min(base_weight[key].shape[0], expr_weight[key].shape[0])
                mixed_weight[key] = (
                    alpha1 * base_weight[key][:min_shape0].float()
                    + (1.0 - alpha1) * expr_weight[key][:min_shape0].float()
                ).half()
            else:
                mixed_weight[key] = (
                    alpha1 * base_weight[key].float()
                    + (1.0 - alpha1) * expr_weight[key].float()
                ).half()

        mixed_cpt = {
            "weight": mixed_weight,
            "config": cfg,
            "sr": base_ckpt.get("sr", "40k"),
            "f0": base_ckpt.get("f0", 0),
            "version": base_ckpt.get("version", "v2"),
            "info": f"In-memory mixed {base_sid} + {expr_sid} (w={alpha1})",
        }

        # Load into current VC instance and cache it
        self._load_from_cpt(mix_sid, mixed_cpt)
        self._save_to_cache(mix_sid)
        logger.info("Loaded mixed model into VC: %s", mix_sid)

        return mix_sid
