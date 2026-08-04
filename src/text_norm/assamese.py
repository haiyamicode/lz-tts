"""Assamese spelling cleanup for the eSpeak frontend."""

from __future__ import annotations


def normalize_assamese(text: str) -> str:
    """Rewrite Bengali-style nukta spellings that eSpeak reads as letter names.

    Assamese eSpeak handles the dedicated Assamese letters U+09DF (য়) and
    U+09F0 (ৰ), but spells the combining nukta in decomposed য/ড/ঢ sequences.
    The ড়/ঢ় rewrites are pronunciation-oriented approximations: eSpeak's
    Assamese voice has no reliable retroflex-flap spelling for those letters.
    ৰ্হ retains the aspiration of ঢ় without causing eSpeak to insert a vowel.
    """
    return (
        text.replace("\u09af\u09bc", "\u09df")
        .replace("\u09a1\u09bc", "\u09f0")
        .replace("\u09a2\u09bc", "\u09f0\u09cd\u09b9")
        .replace("\u09dc", "\u09f0")
        .replace("\u09dd", "\u09f0\u09cd\u09b9")
    )
