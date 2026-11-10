"""ZipEnhancer speech enhancement network, vendored from modelscope 1.40.2
(``modelscope/models/audio/ans``, Apache License 2.0; the Zipformer layers are
Xiaomi's, from icefall). ``model.py`` holds the network; ``generator.py``,
``scaling.py``, ``zipenhancer_layer.py`` and ``zipformer.py`` are copied
unchanged. Used by :mod:`ai_worker.speech_cleanup`.
"""
