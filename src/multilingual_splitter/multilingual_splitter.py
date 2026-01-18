"""
Multilingual text splitter for detecting code-switching in text.

Splits text into language-tagged segments by detecting when text switches
between languages, using script analysis and language detection with bias
toward the main language.
"""

from dataclasses import dataclass, field
from typing import Optional

import pycld2 as cld2
import regex
import unicodedataplus as udp
from wordfreq import tokenize


@dataclass
class Segment:
    """A text segment with language and script information."""

    text: str
    start: int
    end: int
    script: str
    language: str


@dataclass
class SplitResult:
    """Result of splitting text into language segments."""

    original_text: str
    main_language: str
    segments: list[Segment]


# =============================================================================
# LANGUAGE DETECTION THRESHOLDS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DetectionThreshold:
    """
    Configuration for language switching thresholds.

    Attributes:
        min_detected_confidence: Minimum confidence for detected language to switch.
        max_main_confidence: Maximum confidence for main language to allow switching.
        description: Human-readable description of why these thresholds exist.
    """

    min_detected_confidence: float
    max_main_confidence: float
    description: str = ""


@dataclass(frozen=True)
class ScriptConfig:
    """
    Configuration for a script shared by multiple languages.

    Attributes:
        languages: Set of ISO 639-1 codes that use this script.
        threshold: Detection threshold for switching between these languages.
    """

    languages: frozenset[str]
    threshold: DetectionThreshold


# Default threshold for scripts without special configuration
DEFAULT_THRESHOLD = DetectionThreshold(
    min_detected_confidence=0.7,
    max_main_confidence=0.1,
    description="Default: switch if detected >70% and main <10%",
)

# Confusable language pairs within the same script.
# These take precedence over script-level configs.
# Maps frozenset of language codes -> threshold
CONFUSABLE_LANGUAGE_PAIRS: dict[frozenset[str], DetectionThreshold] = {
    # Scandinavian languages: Norwegian, Danish, Swedish are nearly identical in writing
    # "jeg har en hund" is valid Norwegian AND Danish
    frozenset({"no", "da", "sv", "nn", "nb"}): DetectionThreshold(
        min_detected_confidence=0.97,
        max_main_confidence=0.03,
        description="Scandinavian: extremely similar written forms",
    ),
    # Dutch and Afrikaans: Afrikaans derived from Dutch, many identical words
    # "het" is 99.92% Dutch but valid in both - effectively never switch
    frozenset({"nl", "af"}): DetectionThreshold(
        min_detected_confidence=1.0,  # impossible to reach
        max_main_confidence=0.0,
        description="Dutch/Afrikaans: never switch, too similar",
    ),
    # Indonesian and Malay: essentially the same language with minor differences
    frozenset({"id", "ms"}): DetectionThreshold(
        min_detected_confidence=0.97,
        max_main_confidence=0.03,
        description="Indonesian/Malay: nearly identical languages",
    ),
    # Spanish, Portuguese, Catalan, Galician: Iberian Romance languages
    frozenset({"es", "pt", "ca", "gl"}): DetectionThreshold(
        min_detected_confidence=0.85,
        max_main_confidence=0.08,
        description="Iberian Romance: significant vocabulary overlap",
    ),
}

# Script-specific configurations for language groups that share scripts
# and are commonly confused by language detectors
SCRIPT_DETECTION_CONFIG: dict[str, ScriptConfig] = {
    # Han script: Chinese and Japanese share characters extensively.
    # Many common words (学生, 大学, 日本, 山, 水) appear in both languages
    # and fasttext cannot reliably distinguish them without context.
    "Han": ScriptConfig(
        languages=frozenset({"zh", "ja"}),
        threshold=DetectionThreshold(
            min_detected_confidence=0.97,
            max_main_confidence=0.03,
            description="Han: extremely conservative, many shared words",
        ),
    ),
    # Cyrillic script: Russian, Ukrainian, Bulgarian, etc. share many words.
    # Common words like "привет", "друг", "москва" are ambiguous.
    # Ukrainian-specific chars (ї,є,ґ) and Bulgarian-specific (ъ) help distinguish.
    "Cyrillic": ScriptConfig(
        languages=frozenset({"ru", "uk", "bg", "sr", "mk", "be", "kk", "ky", "mn", "tg"}),
        threshold=DetectionThreshold(
            min_detected_confidence=0.85,
            max_main_confidence=0.08,
            description="Cyrillic: conservative, Slavic languages share vocabulary",
        ),
    ),
    # Arabic script: Arabic, Persian, Urdu share the script.
    # Words like "سلام", "کتاب" are common across languages.
    # Fasttext often confuses Arabic/Persian especially.
    "Arabic": ScriptConfig(
        languages=frozenset({"ar", "fa", "ur", "ps", "sd", "ug"}),
        threshold=DetectionThreshold(
            min_detected_confidence=0.85,
            max_main_confidence=0.08,
            description="Arabic script: Persian/Arabic/Urdu often confused",
        ),
    ),
    # Devanagari script: Hindi, Marathi, Sanskrit, Nepali share it.
    # "नमस्ते", "भारत" are ambiguous between Hindi/Marathi.
    "Devanagari": ScriptConfig(
        languages=frozenset({"hi", "mr", "ne", "sa", "bh"}),
        threshold=DetectionThreshold(
            min_detected_confidence=0.85,
            max_main_confidence=0.08,
            description="Devanagari: Hindi/Marathi/Nepali share vocabulary",
        ),
    ),
    # Latin script: Many European languages, but generally distinguishable
    # except for very short words. Use moderate thresholds.
    "Latin": ScriptConfig(
        languages=frozenset({
            "en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "sk",
            "hu", "ro", "hr", "sl", "fi", "sv", "no", "da", "is", "et",
            "lv", "lt", "mt", "sq", "tr", "az", "uz", "vi", "id", "ms",
            "tl", "sw", "af", "eu", "ca", "gl", "cy", "ga", "gd", "br",
            "eo", "la",
        }),
        threshold=DetectionThreshold(
            min_detected_confidence=0.7,
            max_main_confidence=0.1,
            description="Latin: moderate thresholds, languages generally distinguishable",
        ),
    ),
}


def get_script_threshold(script: str, main_lang: str, detected_lang: str) -> DetectionThreshold:
    """
    Get the appropriate detection threshold for a script and language pair.

    Args:
        script: Unicode script name (e.g., "Han", "Cyrillic").
        main_lang: Main language ISO code.
        detected_lang: Detected language ISO code.

    Returns:
        DetectionThreshold to use for this combination.
    """
    # Check confusable language pairs first (highest priority)
    for lang_pair, threshold in CONFUSABLE_LANGUAGE_PAIRS.items():
        if main_lang in lang_pair and detected_lang in lang_pair:
            return threshold

    # Fall back to script-level config
    if script in SCRIPT_DETECTION_CONFIG:
        config = SCRIPT_DETECTION_CONFIG[script]
        # Only use special thresholds if BOTH languages are in the confusable group
        if main_lang in config.languages and detected_lang in config.languages:
            return config.threshold
    return DEFAULT_THRESHOLD


# =============================================================================
# DEFINITIVE LANGUAGE MARKERS
# =============================================================================

# Scripts or characters that definitively identify a language.
# If these are found, we can be certain about the language without detection.
DEFINITIVE_MARKERS: dict[str, dict] = {
    # Japanese: Hiragana/Katakana are unique to Japanese
    "ja": {
        "scripts": {"Hiragana", "Katakana"},
        # Kokuji (Japan-made kanji) + yen symbol
        "chars": set("峠畑辻込笹栃榊円"),
    },
    # Korean: Hangul is unique to Korean
    "ko": {"scripts": {"Hangul"}},
    # Thai: Thai script is unique
    "th": {"scripts": {"Thai"}},
    # Ukrainian: specific Cyrillic letters
    "uk": {"chars": set("їєґЇЄҐ")},
    # Serbian: specific Cyrillic letters
    "sr": {"chars": set("ђјљњћџЂЈЉЊЋЏ")},
    # Bulgarian: ъ is characteristic (though not unique)
    "bg": {"chars": set("ъЪ")},
    # Greek: unique script
    "el": {"scripts": {"Greek"}},
    # Hebrew: unique script
    "he": {"scripts": {"Hebrew"}},
    # Various Indic scripts (unique to their languages)
    "ta": {"scripts": {"Tamil"}},
    "te": {"scripts": {"Telugu"}},
    "bn": {"scripts": {"Bengali"}},
    "gu": {"scripts": {"Gujarati"}},
    "kn": {"scripts": {"Kannada"}},
    "ml": {"scripts": {"Malayalam"}},
    "my": {"scripts": {"Myanmar"}},
    "km": {"scripts": {"Khmer"}},
    "lo": {"scripts": {"Lao"}},
    "si": {"scripts": {"Sinhala"}},
    "am": {"scripts": {"Ethiopic"}},
    "ka": {"scripts": {"Georgian"}},
    "hy": {"scripts": {"Armenian"}},
}


# =============================================================================
# SHARED SCRIPTS CONFIGURATION
# =============================================================================

# Scripts that can legitimately appear in multiple languages.
# Used to determine if a script is "valid" for a given main language.
SHARED_SCRIPTS: dict[str, set[str]] = {
    "Latin": set(SCRIPT_DETECTION_CONFIG["Latin"].languages),
    "Cyrillic": set(SCRIPT_DETECTION_CONFIG["Cyrillic"].languages),
    "Arabic": set(SCRIPT_DETECTION_CONFIG["Arabic"].languages),
    "Devanagari": set(SCRIPT_DETECTION_CONFIG["Devanagari"].languages),
    "Han": {"zh", "ja"},  # Chinese and Japanese (Korean rarely uses Hanja now)
    "Common": set(),  # Punctuation, numbers - valid for all languages
}


# =============================================================================
# MAIN SPLITTER CLASS
# =============================================================================


class MultilingualSplitter:
    """
    Splits text into language-tagged segments detecting code-switching.

    Uses script analysis and language detection with bias toward the main
    language to identify when text switches between languages.
    """

    def __init__(
        self,
        languages: Optional[list[str]] = None,
    ):
        """
        Initialize the splitter.

        Args:
            languages: Optional list of ISO 639-1 codes to filter detection to.
                      If None, all languages supported by CLD2 are available.
        """
        self.languages = set(languages) if languages else None

    def _get_char_script(self, char: str) -> str:
        """Get the Unicode script for a character."""
        try:
            return udp.script(char)
        except (ValueError, KeyError):
            return "Unknown"

    def _detect_language(
        self,
        text: str,
        k: int = 1,
        hint_language: Optional[str] = None,
        best_effort: bool = True,
    ) -> tuple[list[tuple[str, float]], bool]:
        """
        Detect language using CLD2.

        Args:
            text: Text to analyze.
            k: Number of top predictions to return (CLD2 returns up to 3).
            hint_language: Optional language hint to bias detection toward.
                          Useful for segment detection where main language is known.
            best_effort: If True, always return a result even when CLD2 is uncertain.
                        Default True to avoid losing useful detection info.

        Returns:
            Tuple of (results, is_reliable) where results is a list of
            (language_code, probability) tuples.
        """
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return [("und", 1.0)], False

        try:
            is_reliable, _, details = cld2.detect(
                clean_text,
                bestEffort=best_effort,
                hintLanguage=hint_language,
            )
        except Exception:
            # CLD2 can fail on certain inputs
            return [("und", 1.0)], False

        results = []
        for name, code, percent, score in details[:k]:
            if code == "un":
                continue
            # Normalize some CLD2 codes (zh-Hant -> zh, etc.)
            if code.startswith("zh"):
                code = "zh"
            # Filter to allowed languages if specified
            if self.languages is None or code in self.languages:
                results.append((code, percent / 100.0))

        if not results:
            return [("und", 1.0)], False

        return results, is_reliable

    def _detect_main_language_heuristic(self, text: str) -> Optional[str]:
        """
        Detect main language using definitive script/character markers.

        Returns ISO 639-1 code if definitive markers found, else None.
        """
        # Collect scripts and characters in text
        scripts_found: set[str] = set()
        chars_found: set[str] = set(text)

        for char in text:
            script = self._get_char_script(char)
            if script not in ("Common", "Inherited", "Unknown"):
                scripts_found.add(script)

        # Check definitive markers
        for lang, markers in DEFINITIVE_MARKERS.items():
            # Check for definitive scripts
            if "scripts" in markers:
                if markers["scripts"] & scripts_found:
                    return lang

            # Check for definitive characters
            if "chars" in markers:
                if markers["chars"] & chars_found:
                    return lang

        return None

    def _has_han_without_kana(self, text: str) -> bool:
        """Check if text contains Han characters but no Hiragana/Katakana.

        Modern Japanese text almost always contains Hiragana for grammatical
        particles (は、が、を、に、で、etc.). Pure Han-only text is extremely
        rare in Japanese but common in Chinese.

        Returns True if Han present but no kana → likely Chinese.
        """
        has_han = bool(regex.search(r"\p{Script=Han}", text))
        has_kana = bool(regex.search(r"[\p{Script=Hiragana}\p{Script=Katakana}]", text))
        return has_han and not has_kana

    def _detect_language_by_script(self, text: str) -> str:
        """
        Fallback language detection based purely on script analysis.

        Used when CLD2 returns unreliable results.
        """
        scripts_found: set[str] = set()
        for char in text:
            script = self._get_char_script(char)
            if script not in ("Common", "Inherited", "Unknown"):
                scripts_found.add(script)

        # Han without Kana → Chinese
        if "Han" in scripts_found:
            if "Hiragana" in scripts_found or "Katakana" in scripts_found:
                return "ja"
            return "zh"

        # Other unique scripts
        script_to_lang = {
            "Hangul": "ko",
            "Thai": "th",
            "Devanagari": "hi",
            "Arabic": "ar",
            "Hebrew": "he",
            "Greek": "el",
            "Cyrillic": "ru",  # Default to Russian for Cyrillic
            "Tamil": "ta",
            "Telugu": "te",
            "Bengali": "bn",
        }
        for script, lang in script_to_lang.items():
            if script in scripts_found:
                return lang

        # Latin → default to English
        if "Latin" in scripts_found:
            return "en"

        return "und"

    def _is_latin_only(self, text: str) -> bool:
        """Check if text contains only Latin script (plus Common/punctuation)."""
        for char in text:
            script = self._get_char_script(char)
            if script not in ("Latin", "Common", "Inherited", "Unknown"):
                return False
        return True

    # Languages that CLD2 commonly misdetects for short Latin text.
    # These cause ridiculous false positives like "def" → Wolof.
    # Includes: obscure minority languages, nearly extinct languages,
    # constructed/fictional languages, and languages that primarily use
    # non-Latin scripts but CLD2 detects anyway.
    _BANNED_LATIN_LANGUAGES = {
        # Fictional/constructed (except Esperanto which some people use)
        "tlh",  # Klingon (!)
        "vo",   # Volapük
        "ia",   # Interlingua
        "ie",   # Interlingue
        # African minority languages
        "wo",   # Wolof - CLD2's favorite for code keywords
        "sn",   # Shona
        "st",   # Sotho
        "xh",   # Xhosa
        "zu",   # Zulu
        "ny",   # Chichewa
        "ig",   # Igbo
        "yo",   # Yoruba
        "ha",   # Hausa
        "rw",   # Kinyarwanda
        "lg",   # Luganda
        "ak",   # Akan/Twi
        "om",   # Oromo
        "aa",   # Afar
        "rn",   # Kirundi
        "ve",   # Venda
        "ss",   # Swati
        "sg",   # Sango
        "nso",  # Northern Sotho
        "nr",   # Southern Ndebele
        "ts",   # Tsonga
        "tn",   # Tswana
        "ln",   # Lingala
        # Pacific/Oceanian minority languages
        "sm",   # Samoan
        "haw",  # Hawaiian
        "to",   # Tongan
        "fj",   # Fijian
        "bi",   # Bislama
        "na",   # Nauru
        "mi",   # Maori
        # Asian minority languages
        "hmn",  # Hmong
        "kha",  # Khasi
        "za",   # Zhuang
        "su",   # Sundanese
        "jw",   # Javanese (has own script)
        # Caribbean/Creole languages
        "ht",   # Haitian Creole
        "mfe",  # Mauritian Creole
        "crs",  # Seychellois Creole
        # South American indigenous
        "gn",   # Guarani
        "qu",   # Quechua
        "ay",   # Aymara
        # European minority/regional (too small for practical TTS)
        "gv",   # Manx (nearly extinct)
        "br",   # Breton
        "rm",   # Romansh
        "gd",   # Scottish Gaelic
        "sco",  # Scots
        "kl",   # Greenlandic
        "fo",   # Faroese
        "oc",   # Occitan
        "co",   # Corsican
        "fy",   # Frisian
        "lb",   # Luxembourgish
        # Other
        "so",   # Somali
        "mg",   # Malagasy
        "ceb",  # Cebuano
        "war",  # Waray
        "sa",   # Sanskrit (dead language)
        "tt",   # Tatar (uses Cyrillic)
        "tk",   # Turkmen
        "uz",   # Uzbek (primarily Cyrillic)
    }

    def _is_common_latin_language(self, lang: str) -> bool:
        """Check if a language is a commonly trusted Latin-script language.

        CLD2 sometimes misdetects Latin-only text (especially code) as
        obscure Latin-script languages like Wolof. This set contains only
        the commonly used Latin-script languages that CLD2 reliably detects.
        """
        # Common Latin-script languages that CLD2 reliably detects
        common_latin_languages = {
            # Major European languages
            "en", "de", "fr", "es", "pt", "it", "nl", "pl", "cs", "sk",
            "hu", "ro", "hr", "sl", "sv", "no", "da", "fi", "et", "lv",
            "lt", "is", "ga", "cy", "eu", "ca", "gl", "sq", "mt",
            # Major non-European Latin-script languages
            "id", "ms", "tl", "vi", "tr", "sw", "af",
            # Constructed languages
            "eo", "la",
        }
        return lang in common_latin_languages

    def detect_main_language(self, text: str) -> str:
        """
        Detect the main language of the text.

        First applies heuristic rules for definitive markers,
        then uses CLD2 with bestEffort mode (which always returns results).

        Args:
            text: Input text to analyze.

        Returns:
            ISO 639-1 language code.
        """
        # Try heuristic detection first (checks for unique scripts like Hiragana)
        heuristic_lang = self._detect_main_language_heuristic(text)
        if heuristic_lang:
            return heuristic_lang

        # Use CLD2 detection with bestEffort=True (default)
        # This always returns a result even for short/ambiguous text
        results, is_reliable = self._detect_language(text)
        detected = results[0][0] if results else "und"

        if detected == "und":
            # CLD2 returned unknown even with bestEffort - use script fallback
            return self._detect_language_by_script(text)

        # Apply sanity checks for known CLD2 biases

        # CLD2 has a Japanese bias for Han-only text (e.g., dates like 2025年9月28日)
        # If it says Japanese but there's no Kana, it's likely Chinese
        if detected == "ja" and self._has_han_without_kana(text):
            return "zh"

        # Filter out banned languages that CLD2 commonly misdetects
        # (e.g., "def hello(): pass" → Wolof)
        if detected in self._BANNED_LATIN_LANGUAGES:
            return self._detect_language_by_script(text)

        # CLD2 sometimes misdetects Latin-only text as other exotic languages.
        # If detected language isn't a common Latin-script language but text
        # is Latin-only, fall back to script-based detection.
        if not self._is_common_latin_language(detected) and self._is_latin_only(text):
            return self._detect_language_by_script(text)

        # For short Latin-only text where CLD2 is unreliable (e.g., "OK" → Dutch,
        # "100kg" → Danish), prefer English as the default Latin language
        if not is_reliable and self._is_latin_only(text):
            return "en"

        return detected

    def _segment_by_script(self, text: str) -> list[tuple[str, int, int, str]]:
        """
        Segment text by Unicode script runs.

        Returns list of (text, start, end, script) tuples.
        """
        if not text:
            return []

        segments: list[tuple[str, int, int, str]] = []

        # Use regex to find script runs
        # Match runs of same-script characters
        pattern = regex.compile(
            r"(\p{Script=Latin}+|\p{Script=Han}+|\p{Script=Hiragana}+|"
            r"\p{Script=Katakana}+|\p{Script=Hangul}+|\p{Script=Cyrillic}+|"
            r"\p{Script=Greek}+|\p{Script=Arabic}+|\p{Script=Hebrew}+|"
            r"\p{Script=Thai}+|\p{Script=Devanagari}+|\p{Script=Bengali}+|"
            r"\p{Script=Tamil}+|\p{Script=Telugu}+|\p{Script=Gujarati}+|"
            r"\p{Script=Kannada}+|\p{Script=Malayalam}+|\p{Script=Myanmar}+|"
            r"\p{Script=Khmer}+|\p{Script=Lao}+|\p{Script=Sinhala}+|"
            r"\p{Script=Ethiopic}+|\p{Script=Georgian}+|\p{Script=Armenian}+|"
            r"\p{Script=Common}+|\p{Script=Inherited}+|.+?)",
            regex.UNICODE,
        )

        for match in pattern.finditer(text):
            segment_text = match.group()
            start = match.start()
            end = match.end()

            # Determine script of segment
            script = "Unknown"
            for char in segment_text:
                char_script = self._get_char_script(char)
                if char_script not in ("Common", "Inherited", "Unknown"):
                    script = char_script
                    break
            else:
                # All characters are Common/Inherited/Unknown
                if segment_text.strip():
                    script = self._get_char_script(segment_text[0])
                else:
                    script = "Common"

            segments.append((segment_text, start, end, script))

        return segments

    def _is_script_valid_for_language(self, script: str, lang: str) -> bool:
        """Check if a script can appear in the given language."""
        if script in ("Common", "Inherited", "Unknown"):
            return True

        # Check shared scripts
        if script in SHARED_SCRIPTS:
            return lang in SHARED_SCRIPTS[script]

        # Check definitive markers
        if lang in DEFINITIVE_MARKERS:
            markers = DEFINITIVE_MARKERS[lang]
            if "scripts" in markers and script in markers["scripts"]:
                return True

        return False

    def _detect_with_bias(
        self, text: str, main_lang: str, script: str = ""
    ) -> str:
        """
        Detect language with bias toward main language.

        Uses CLD2 with hintLanguage to bias detection toward the main language.
        CLD2's hint system naturally handles ambiguous text (biases toward hint)
        while still allowing clear markers (like hiragana for Japanese) to override.

        Args:
            text: Text to analyze.
            main_lang: Main language ISO code.
            script: Script of the text segment (used for sanity checks).

        Returns:
            ISO 639-1 language code.
        """
        # For Han script: CLD2 hints don't work well when main_lang is non-CJK.
        # Apply han-without-kana heuristic: Han without Kana → Chinese
        if script == "Han" and main_lang not in ("zh", "ja"):
            if self._has_han_without_kana(text):
                return "zh"
            else:
                return "ja"

        # Use CLD2 with main_lang as hint - this biases ambiguous text toward
        # main_lang while still allowing definitive markers to override
        results, is_reliable = self._detect_language(
            text, k=3, hint_language=main_lang
        )

        if not results:
            return main_lang

        top_lang, top_prob = results[0]

        # Filter out banned languages (like Wolof) for Latin text
        # CLD2 loves to detect short code/keywords as obscure African languages
        if top_lang in self._BANNED_LATIN_LANGUAGES:
            return main_lang

        # If detection matches main language, keep it
        if top_lang == main_lang:
            return main_lang

        # CLD2 detected a different language despite the hint.
        # This usually means there's a strong signal (like different script).
        # Apply additional sanity checks for confusable language pairs.

        # Get threshold for this script/language pair
        threshold = get_script_threshold(script, main_lang, top_lang)

        # Find main language probability in results
        main_prob = 0.0
        for lang, prob in results:
            if lang == main_lang:
                main_prob = prob
                break

        # For confusable pairs (high thresholds), be more conservative
        # Only switch if detected strongly exceeds threshold AND main is weak
        if threshold.min_detected_confidence > 0.9:
            # Very strict threshold - only switch for extremely confident results
            if top_prob > threshold.min_detected_confidence and main_prob < threshold.max_main_confidence:
                return top_lang
            return main_lang

        # For less confusable pairs, trust CLD2's result with hint
        # The hint already provides bias, so if CLD2 still detects different,
        # it's likely a real code-switch
        return top_lang

    def _merge_adjacent_segments(self, segments: list[Segment]) -> list[Segment]:
        """Merge adjacent segments with the same language.

        Also merges "und" segments into adjacent segments, so that
        Common script text (spaces, numbers, punctuation) inherits the
        language of adjacent language-specific text.

        Strategy:
        - "und" segments merge with preceding segment if it has a real language
        - Leading "und" segments merge with the following segment
        """
        if not segments:
            return []

        # First pass: merge "und" into preceding segments
        merged: list[Segment] = []
        current = segments[0]

        for seg in segments[1:]:
            # Merge if same language OR if this segment is "und" (Common script)
            if seg.language == current.language or seg.language == "und":
                # Merge: extend current segment, keep current's language
                current = Segment(
                    text=current.text + seg.text,
                    start=current.start,
                    end=seg.end,
                    script=current.script
                    if current.script != "Common"
                    else seg.script,
                    language=current.language,
                )
            else:
                merged.append(current)
                current = seg

        merged.append(current)

        # Second pass: merge leading "und" segments into following segments
        if len(merged) > 1 and merged[0].language == "und":
            final: list[Segment] = []
            i = 0
            # Find first non-und segment
            while i < len(merged) and merged[i].language == "und":
                i += 1

            if i < len(merged):
                # Merge all leading "und" segments into the first real segment
                first_real = merged[i]
                leading_text = "".join(seg.text for seg in merged[:i])
                final.append(
                    Segment(
                        text=leading_text + first_real.text,
                        start=merged[0].start,
                        end=first_real.end,
                        script=first_real.script,
                        language=first_real.language,
                    )
                )
                final.extend(merged[i + 1 :])
                return final

        return merged

    def split(self, text: str, main_lang: Optional[str] = None) -> SplitResult:
        """
        Split text into language-tagged segments.

        Args:
            text: Input text to split.
            main_lang: Optional main language ISO code. If None, auto-detected.

        Returns:
            SplitResult with segments and main language.
        """
        if not text:
            return SplitResult(original_text=text, main_language="und", segments=[])

        # Detect main language if not provided
        if main_lang is None:
            main_lang = self.detect_main_language(text)

        # Segment by script
        script_segments = self._segment_by_script(text)

        segments: list[Segment] = []

        for seg_text, start, end, script in script_segments:
            # Common script segments (spaces, numbers, punctuation) get "und"
            # They will be merged with the preceding segment later
            if script in ("Common", "Inherited") or not seg_text.strip():
                segments.append(
                    Segment(
                        text=seg_text,
                        start=start,
                        end=end,
                        script=script,
                        language="und",
                    )
                )
                continue

            # Check if script is valid for main language
            if not self._is_script_valid_for_language(script, main_lang):
                # Script IMPOSSIBLE for main language - detect appropriate language
                # For Han script: apply han-without-kana heuristic
                if script == "Han":
                    if self._has_han_without_kana(seg_text):
                        lang = "zh"
                    else:
                        lang = "ja"
                else:
                    results, is_reliable = self._detect_language(seg_text)
                    if is_reliable and results:
                        lang = results[0][0]
                    else:
                        # Unreliable → use script-based detection
                        lang = self._detect_language_by_script(seg_text)
                segments.append(
                    Segment(
                        text=seg_text, start=start, end=end, script=script, language=lang
                    )
                )
            else:
                # Script valid for main language - tokenize and detect per token
                try:
                    tokens = tokenize(seg_text, main_lang)
                except (ValueError, KeyError):
                    # Fallback if tokenization fails
                    tokens = seg_text.split()

                if not tokens:
                    segments.append(
                        Segment(
                            text=seg_text,
                            start=start,
                            end=end,
                            script=script,
                            language=main_lang,
                        )
                    )
                    continue

                # Build token segments with positions
                token_segments: list[Segment] = []
                pos = 0

                for token in tokens:
                    # Find token position in segment
                    token_start = seg_text.find(token, pos)
                    if token_start == -1:
                        continue

                    token_end = token_start + len(token)

                    # Detect language with bias
                    token_lang = self._detect_with_bias(token, main_lang, script=script)

                    token_segments.append(
                        Segment(
                            text=seg_text[token_start:token_end],
                            start=start + token_start,
                            end=start + token_end,
                            script=script,
                            language=token_lang,
                        )
                    )
                    pos = token_end

                # Merge adjacent same-language tokens
                merged_tokens = self._merge_adjacent_segments(token_segments)

                # Add any remaining text between/around tokens
                if merged_tokens:
                    # Add leading text (spaces/punctuation before first token → "und")
                    if merged_tokens[0].start > start:
                        segments.append(
                            Segment(
                                text=seg_text[: merged_tokens[0].start - start],
                                start=start,
                                end=merged_tokens[0].start,
                                script=script,
                                language="und",
                            )
                        )

                    segments.extend(merged_tokens)

                    # Add trailing text (spaces/punctuation after last token → "und")
                    if merged_tokens[-1].end < end:
                        segments.append(
                            Segment(
                                text=seg_text[merged_tokens[-1].end - start :],
                                start=merged_tokens[-1].end,
                                end=end,
                                script=script,
                                language="und",
                            )
                        )
                else:
                    segments.append(
                        Segment(
                            text=seg_text,
                            start=start,
                            end=end,
                            script=script,
                            language=main_lang,
                        )
                    )

        # Final merge of adjacent same-language segments
        merged_segments = self._merge_adjacent_segments(segments)

        return SplitResult(
            original_text=text, main_language=main_lang, segments=merged_segments
        )


def split_text(
    text: str, main_lang: Optional[str] = None, languages: Optional[list[str]] = None
) -> SplitResult:
    """
    Convenience function to split text into language segments.

    Args:
        text: Input text to split.
        main_lang: Optional main language ISO code.
        languages: Optional list of languages to detect.

    Returns:
        SplitResult with segments and main language.
    """
    splitter = MultilingualSplitter(languages=languages)
    return splitter.split(text, main_lang=main_lang)


if __name__ == "__main__":
    import sys

    # Simple CLI for testing
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "Hello world! これは日本語です。And back to English."

    print(f"Input: {input_text}\n")

    result = split_text(input_text)
    print(f"Main language: {result.main_language}\n")
    print("Segments:")
    for seg in result.segments:
        print(f"  [{seg.start}:{seg.end}] ({seg.script}/{seg.language}): {seg.text!r}")
