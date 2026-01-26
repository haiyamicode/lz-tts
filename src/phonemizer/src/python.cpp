#include <iostream>
#include <optional>
#include <string>
#include <map>
#include <vector>

#include <espeak-ng/speak_lib.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phoneme_ids.hpp"
#include "phonemize.hpp"
#include "tashkeel.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

// True when espeak_Initialize has been called
bool eSpeakInitialized = false;

// Loaded when using Arabic
// https://github.com/mush42/libtashkeel/
std::map<std::string, tashkeel::State> tashkeelStates;

// ----------------------------------------------------------------------------

std::vector<std::vector<piper::Phoneme>>
phonemize_espeak(std::string text, std::string voice, std::string dataPath) {
  if (!eSpeakInitialized) {
    int result =
        espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dataPath.c_str(), 0);
    if (result < 0) {
      throw std::runtime_error("Failed to initialize eSpeak");
    }

    eSpeakInitialized = true;
  }

  piper::eSpeakPhonemeConfig config;
  config.voice = voice;

  std::vector<std::vector<piper::Phoneme>> phonemes;
  piper::phonemize_eSpeak(text, config, phonemes);

  return phonemes;
}

// WordMapping: (textStart, textLength, phonemeStart, phonemeEnd, punctuationLength)
// phonemeStart/End are LOCAL indices into the sentence's phoneme list
using WordMapping = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>;

std::pair<std::vector<std::vector<piper::Phoneme>>, std::vector<std::vector<WordMapping>>>
phonemize_espeak_with_mapping(std::string text, std::string voice, std::string dataPath) {
  if (!eSpeakInitialized) {
    int result =
        espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dataPath.c_str(), 0);
    if (result < 0) {
      throw std::runtime_error("Failed to initialize eSpeak");
    }

    eSpeakInitialized = true;
  }

  // Get phonemes and punctuation data
  piper::eSpeakPhonemeConfig config;
  config.voice = voice;

  std::vector<std::vector<piper::Phoneme>> phonemes;
  std::vector<piper::ClausePunctuation> punctuation;
  piper::phonemize_eSpeak(text, config, phonemes, &punctuation);

  // Get word positions using espeak_Synth callback
  std::vector<piper::WordPosition> wordPositions = piper::get_word_positions(text, voice);

  // Build word mapping PER SENTENCE with LOCAL indices
  // Each sentence gets its own list of word mappings
  std::vector<std::vector<WordMapping>> sentenceWordMappings;

  // Build phoneme groups per sentence (LOCAL indices)
  // Also track which sentence each group belongs to
  struct PhonemeGroup {
    std::size_t sentenceIdx;
    std::size_t localStart;
    std::size_t localEnd;
    std::size_t globalStart;  // For punctuation matching
    std::size_t globalEnd;
  };
  std::vector<PhonemeGroup> allGroups;

  std::size_t globalOffset = 0;
  for (std::size_t sentIdx = 0; sentIdx < phonemes.size(); sentIdx++) {
    const auto &sentence = phonemes[sentIdx];
    std::size_t groupStart = 0;
    bool inGroup = false;

    for (std::size_t i = 0; i < sentence.size(); i++) {
      if (sentence[i] == U' ') {
        if (inGroup) {
          allGroups.push_back({sentIdx, groupStart, i, globalOffset + groupStart, globalOffset + i});
          inGroup = false;
        }
      } else {
        if (!inGroup) {
          groupStart = i;
          inGroup = true;
        }
      }
    }
    // Don't forget the last group in this sentence
    if (inGroup) {
      allGroups.push_back({sentIdx, groupStart, sentence.size(), globalOffset + groupStart, globalOffset + sentence.size()});
    }

    globalOffset += sentence.size();
  }

  // Initialize empty mapping lists for each sentence
  for (std::size_t i = 0; i < phonemes.size(); i++) {
    sentenceWordMappings.emplace_back();
  }

  // Match word positions to phoneme groups by index
  std::size_t numMappings = std::min(wordPositions.size(), allGroups.size());
  for (std::size_t i = 0; i < numMappings; i++) {
    const auto &group = allGroups[i];
    std::size_t punctLen = 0;

    // Check if any punctuation position falls within this word's GLOBAL phoneme range
    for (const auto &punct : punctuation) {
      if (punct.phonemePosition >= group.globalStart && punct.phonemePosition < group.globalEnd) {
        punctLen = punct.phonemeCount;
        break;
      }
    }

    // Add to the appropriate sentence's mapping list with LOCAL indices
    sentenceWordMappings[group.sentenceIdx].push_back({
        wordPositions[i].textStart,
        wordPositions[i].textLength,
        group.localStart,
        group.localEnd,
        punctLen
    });
  }

  return {phonemes, sentenceWordMappings};
}

std::vector<std::vector<piper::Phoneme>>
phonemize_codepoints(std::string text, std::string casing) {
  piper::CodepointsPhonemeConfig config;

  if (casing == "ignore") {
    config.casing = piper::CASING_IGNORE;
  } else if (casing == "lower") {
    config.casing = piper::CASING_LOWER;
  } else if (casing == "upper") {
    config.casing = piper::CASING_UPPER;
  } else if (casing == "fold") {
    config.casing = piper::CASING_FOLD;
  }

  std::vector<std::vector<piper::Phoneme>> phonemes;

  piper::phonemize_codepoints(text, config, phonemes);

  return phonemes;
}

std::pair<std::vector<piper::PhonemeId>, std::map<piper::Phoneme, std::size_t>>
phoneme_ids_espeak(std::vector<piper::Phoneme> &phonemes) {
  piper::PhonemeIdConfig config;
  std::vector<piper::PhonemeId> phonemeIds;
  std::map<piper::Phoneme, std::size_t> missingPhonemes;

  phonemes_to_ids(phonemes, config, phonemeIds, missingPhonemes);

  return std::make_pair(phonemeIds, missingPhonemes);
}

std::pair<std::vector<piper::PhonemeId>, std::map<piper::Phoneme, std::size_t>>
phoneme_ids_codepoints(std::string language,
                       std::vector<piper::Phoneme> &phonemes) {
  if (piper::DEFAULT_ALPHABET.count(language) < 1) {
    throw std::runtime_error("No phoneme/id map for language");
  }

  piper::PhonemeIdConfig config;
  config.phonemeIdMap =
      std::make_shared<piper::PhonemeIdMap>(piper::DEFAULT_ALPHABET[language]);
  std::vector<piper::PhonemeId> phonemeIds;
  std::map<piper::Phoneme, std::size_t> missingPhonemes;

  phonemes_to_ids(phonemes, config, phonemeIds, missingPhonemes);

  return std::make_pair(phonemeIds, missingPhonemes);
}

std::size_t get_max_phonemes() { return piper::MAX_PHONEMES; }

piper::PhonemeIdMap get_espeak_map() { return piper::DEFAULT_PHONEME_ID_MAP; }

std::map<std::string, piper::PhonemeIdMap> get_codepoints_map() {
  return piper::DEFAULT_ALPHABET;
}

std::string tashkeel_run(std::string modelPath, std::string text) {
  if (tashkeelStates.count(modelPath) < 1) {
    tashkeel::State newState;
    tashkeel::tashkeel_load(modelPath, newState);
    tashkeelStates[modelPath] = std::move(newState);
  }

  return tashkeel::tashkeel_run(text, tashkeelStates[modelPath]);
}

// ----------------------------------------------------------------------------

PYBIND11_MODULE(piper_phonemize_cpp, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: piper_phonemize_cpp

        .. autosummary::
           :toctree: _generate

           phonemize_espeak
           phonemize_codepoints
           phoneme_ids_espeak
           phoneme_ids_codepoints
           get_espeak_map
           get_codepoints_map
           get_max_phonemes
           tashkeel_load
           tashkeel_run
    )pbdoc";

  m.def("phonemize_espeak", &phonemize_espeak, R"pbdoc(
        Phonemize text using espeak-ng
    )pbdoc");

  m.def("phonemize_espeak_with_mapping", &phonemize_espeak_with_mapping, R"pbdoc(
        Phonemize text using espeak-ng and return word-to-phoneme mapping per sentence.
        Returns (phonemes, sentence_word_mappings) where:
        - phonemes: List[List[str]] - phonemes per sentence
        - sentence_word_mappings: List[List[tuple]] - word mappings per sentence
        Each word mapping tuple is (textStart, textLength, phonemeStart, phonemeEnd, punctuationLength).
        phonemeStart/End are LOCAL indices into that sentence's phoneme list.
        punctuationLength is the number of trailing punctuation phonemes (0 or 1).
    )pbdoc");

  m.def("phonemize_codepoints", &phonemize_codepoints, R"pbdoc(
        Phonemize text as UTF-8 codepoints
    )pbdoc");

  m.def("phoneme_ids_espeak", &phoneme_ids_espeak, R"pbdoc(
        Get ids for espeak-ng phonemes
    )pbdoc");

  m.def("phoneme_ids_codepoints", &phoneme_ids_codepoints, R"pbdoc(
        Get ids for a language's codepoints
    )pbdoc");

  m.def("get_espeak_map", &get_espeak_map, R"pbdoc(
        Get phoneme/id map for espeak-ng phonemes
    )pbdoc");

  m.def("get_codepoints_map", &get_codepoints_map, R"pbdoc(
        Get codepoint/id map for supported languages
    )pbdoc");

  m.def("get_max_phonemes", &get_max_phonemes, R"pbdoc(
        Get maximum number of phonemes in id maps
    )pbdoc");

  m.def("tashkeel_run", &tashkeel_run, R"pbdoc(
        Add diacritics to Arabic text (must call tashkeel_load first)
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
