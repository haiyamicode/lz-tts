"""
LRU Cache for RVC Model Management
"""
import logging
import torch
from collections import OrderedDict
from typing import Optional, Tuple, Any
from threading import Lock

logger = logging.getLogger(__name__)


class ModelCache:
    """LRU Cache for managing loaded RVC models"""
    
    def __init__(self, max_size: int = 5):
        """
        Initialize the model cache with LRU eviction policy
        
        Args:
            max_size: Maximum number of models to keep in cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.lock = Lock()
        logger.info(f"Initialized ModelCache with max_size={max_size}")
    
    def get(self, model_name: str) -> Optional[dict]:
        """
        Get a model from cache, moving it to end (most recently used)
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Cached model data or None if not in cache
        """
        with self.lock:
            if model_name in self.cache:
                logger.info(f"Cache HIT for model: {model_name}")
                self.cache.move_to_end(model_name)
                return self.cache[model_name]
            logger.info(f"Cache MISS for model: {model_name}")
            return None
    
    def put(self, model_name: str, model_data: dict) -> None:
        """
        Add or update a model in cache, evicting LRU if needed
        
        Args:
            model_name: Name of the model file
            model_data: Dictionary containing model state (net_g, cpt, etc.)
        """
        with self.lock:
            if model_name in self.cache:
                self.cache.move_to_end(model_name)
                self.cache[model_name] = model_data
                logger.info(f"Updated existing model in cache: {model_name}")
            else:
                if len(self.cache) >= self.max_size:
                    evicted_name, evicted_data = self.cache.popitem(last=False)
                    self._cleanup_model(evicted_name, evicted_data)
                    logger.info(f"Evicted LRU model: {evicted_name}")
                
                self.cache[model_name] = model_data
                logger.info(f"Added new model to cache: {model_name} (cache size: {len(self.cache)}/{self.max_size})")
    
    def _cleanup_model(self, model_name: str, model_data: dict) -> None:
        """
        Clean up GPU/CPU memory when evicting a model
        
        Args:
            model_name: Name of the model being evicted
            model_data: Model data to cleanup
        """
        try:
            if 'net_g' in model_data and model_data['net_g'] is not None:
                del model_data['net_g']
            if 'cpt' in model_data and model_data['cpt'] is not None:
                del model_data['cpt']
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Cleaned up model: {model_name}")
        except Exception as e:
            logger.warning(f"Error cleaning up model {model_name}: {e}")
    
    def clear(self) -> None:
        """Clear all models from cache"""
        with self.lock:
            for model_name, model_data in self.cache.items():
                self._cleanup_model(model_name, model_data)
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'models': list(self.cache.keys())
            }
