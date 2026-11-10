"""SyncNet v2 and the S3FD face detector, vendored for the lipsync validators.

``syncnet.py`` is joonson/syncnet_python's ``SyncNetModel.py`` (MIT); ``s3fd_nets.py`` and
``s3fd_box_utils.py`` are cs-giung/face-detection-pytorch's S3FD (as copied into
the original eval code), unchanged except for the module import. Weights: the
``syncnet_eval`` model artifact. Used by :mod:`ai_worker.lipsync_checks`.
"""
