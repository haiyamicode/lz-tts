#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <espeak-ng/speak_lib.h>
#include <onnxruntime_cxx_api.h>

#include "phonemize.hpp"
#include "uni_algo.h"

namespace piper {

// language -> phoneme -> [phoneme, ...]
std::map<std::string, PhonemeMap> DEFAULT_PHONEME_MAP = {
    {"pt-br", {{U'c', {U'k'}}}}};

// Thread-local storage for word positions captured by callback
static thread_local std::vector<WordPosition> g_wordPositions;

// Callback for espeak_Synth to capture WORD events
static int wordPositionCallback(short *wav, int numsamples, espeak_EVENT *events) {
  if (events == nullptr) {
    return 0;
  }

  while (events->type != espeakEVENT_LIST_TERMINATED) {
    if (events->type == espeakEVENT_WORD) {
      WordPosition pos;
      pos.textStart = static_cast<std::size_t>(events->text_position);
      pos.textLength = static_cast<std::size_t>(events->length);
      g_wordPositions.push_back(pos);
    }
    events++;
  }

  return 0;
}

PIPERPHONEMIZE_EXPORT std::vector<WordPosition>
get_word_positions(const std::string &text, const std::string &voice) {
  int result = espeak_SetVoiceByName(voice.c_str());
  if (result != 0) {
    throw std::runtime_error("Failed to set eSpeak-ng voice");
  }

  // Clear previous results
  g_wordPositions.clear();

  // Set callback
  espeak_SetSynthCallback(wordPositionCallback);

  // Synthesize to trigger WORD events (no audio output needed)
  result = espeak_Synth(
      text.c_str(),
      text.size() + 1,
      0,                    // position
      POS_CHARACTER,        // position_type
      0,                    // end_position
      espeakCHARS_UTF8,     // flags
      nullptr,              // unique_identifier
      nullptr               // user_data
  );

  if (result != EE_OK) {
    throw std::runtime_error("espeak_Synth failed");
  }

  // Wait for synthesis to complete
  espeak_Synchronize();

  // Clear callback
  espeak_SetSynthCallback(nullptr);

  return g_wordPositions;
}

PIPERPHONEMIZE_EXPORT void
phonemize_eSpeak(std::string text, eSpeakPhonemeConfig &config,
                 std::vector<std::vector<Phoneme>> &phonemes,
                 std::vector<ClausePunctuation> *punctuationOut) {

  auto voice = config.voice;
  int result = espeak_SetVoiceByName(voice.c_str());
  if (result != 0) {
    throw std::runtime_error("Failed to set eSpeak-ng voice");
  }

  std::shared_ptr<PhonemeMap> phonemeMap;
  if (config.phonemeMap) {
    phonemeMap = config.phonemeMap;
  } else if (DEFAULT_PHONEME_MAP.count(voice) > 0) {
    phonemeMap = std::make_shared<PhonemeMap>(DEFAULT_PHONEME_MAP[voice]);
  }

  // Modified by eSpeak
  std::string textCopy(text);

  std::vector<Phoneme> *sentencePhonemes = nullptr;
  const char *inputTextPointer = textCopy.c_str();
  int terminator = 0;

  // Track cumulative phoneme offset across sentences
  std::size_t cumulativeOffset = 0;

  while (inputTextPointer != NULL) {
    // Modified espeak-ng API to get access to clause terminator
    std::string clausePhonemes(espeak_TextToPhonemesWithTerminator(
        (const void **)&inputTextPointer,
        /*textmode*/ espeakCHARS_AUTO,
        /*phonememode = IPA*/ 0x02, &terminator));

    // Decompose, e.g. "ç" -> "c" + "̧"
    auto phonemesNorm = una::norm::to_nfd_utf8(clausePhonemes);
    auto phonemesRange = una::ranges::utf8_view{phonemesNorm};

    if (!sentencePhonemes) {
      // Start new sentence
      phonemes.emplace_back();
      sentencePhonemes = &phonemes[phonemes.size() - 1];
    }

    // Maybe use phoneme map
    std::vector<Phoneme> mappedSentPhonemes;
    if (phonemeMap) {
      for (auto phoneme : phonemesRange) {
        if (phonemeMap->count(phoneme) < 1) {
          // No mapping for phoneme
          mappedSentPhonemes.push_back(phoneme);
        } else {
          // Mapping for phoneme
          auto mappedPhonemes = &(phonemeMap->at(phoneme));
          mappedSentPhonemes.insert(mappedSentPhonemes.end(),
                                    mappedPhonemes->begin(),
                                    mappedPhonemes->end());
        }
      }
    } else {
      // No phoneme map
      mappedSentPhonemes.insert(mappedSentPhonemes.end(), phonemesRange.begin(),
                                phonemesRange.end());
    }

    auto phonemeIter = mappedSentPhonemes.begin();
    auto phonemeEnd = mappedSentPhonemes.end();

    if (config.keepLanguageFlags) {
      // No phoneme filter
      sentencePhonemes->insert(sentencePhonemes->end(), phonemeIter,
                               phonemeEnd);
    } else {
      // Filter out (lang) switch (flags).
      // These surround words from languages other than the current voice.
      bool inLanguageFlag = false;

      while (phonemeIter != phonemeEnd) {
        if (inLanguageFlag) {
          if (*phonemeIter == U')') {
            // End of (lang) switch
            inLanguageFlag = false;
          }
        } else if (*phonemeIter == U'(') {
          // Start of (lang) switch
          inLanguageFlag = true;
        } else {
          sentencePhonemes->push_back(*phonemeIter);
        }

        phonemeIter++;
      }
    }

    // Add appropriate punctuation depending on terminator type
    int punctuationType = terminator & 0x000FFFFF;
    std::size_t punctPosition = cumulativeOffset + sentencePhonemes->size();
    std::size_t punctCount = 0;

    if (punctuationType == CLAUSE_PERIOD) {
      sentencePhonemes->push_back(config.period);
      punctCount = 1;
    } else if (punctuationType == CLAUSE_QUESTION) {
      sentencePhonemes->push_back(config.question);
      punctCount = 1;
    } else if (punctuationType == CLAUSE_EXCLAMATION) {
      sentencePhonemes->push_back(config.exclamation);
      punctCount = 1;
    } else if (punctuationType == CLAUSE_COMMA) {
      sentencePhonemes->push_back(config.comma);
      sentencePhonemes->push_back(config.space);
      punctCount = 1;  // Only count the comma, space becomes separator
    } else if (punctuationType == CLAUSE_COLON) {
      sentencePhonemes->push_back(config.colon);
      sentencePhonemes->push_back(config.space);
      punctCount = 1;  // Only count the colon, space becomes separator
    } else if (punctuationType == CLAUSE_SEMICOLON) {
      sentencePhonemes->push_back(config.semicolon);
      sentencePhonemes->push_back(config.space);
      punctCount = 1;  // Only count the semicolon, space becomes separator
    }

    // Record punctuation if requested
    if (punctuationOut && punctCount > 0) {
      punctuationOut->push_back({punctPosition, punctCount});
    }

    if ((terminator & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE) {
      // End of sentence - update cumulative offset
      cumulativeOffset += sentencePhonemes->size();
      sentencePhonemes = nullptr;
    }

  } // while inputTextPointer != NULL

} /* phonemize_eSpeak */

// ----------------------------------------------------------------------------

PIPERPHONEMIZE_EXPORT void
phonemize_codepoints(std::string text, CodepointsPhonemeConfig &config,
                     std::vector<std::vector<Phoneme>> &phonemes) {

  if (config.casing == CASING_LOWER) {
    text = una::cases::to_lowercase_utf8(text);
  } else if (config.casing == CASING_UPPER) {
    text = una::cases::to_uppercase_utf8(text);
  } else if (config.casing == CASING_FOLD) {
    text = una::cases::to_casefold_utf8(text);
  }

  // Decompose, e.g. "ç" -> "c" + "̧"
  auto phonemesNorm = una::norm::to_nfd_utf8(text);
  auto phonemesRange = una::ranges::utf8_view{phonemesNorm};

  // No sentence boundary detection
  phonemes.emplace_back();
  auto sentPhonemes = &phonemes[phonemes.size() - 1];

  if (config.phonemeMap) {
    for (auto phoneme : phonemesRange) {
      if (config.phonemeMap->count(phoneme) < 1) {
        // No mapping for phoneme
        sentPhonemes->push_back(phoneme);
      } else {
        // Mapping for phoneme
        auto mappedPhonemes = &(config.phonemeMap->at(phoneme));
        sentPhonemes->insert(sentPhonemes->end(), mappedPhonemes->begin(),
                             mappedPhonemes->end());
      }
    }
  } else {
    // No phoneme map
    sentPhonemes->insert(sentPhonemes->end(), phonemesRange.begin(),
                         phonemesRange.end());
  }
} // phonemize_text

} // namespace piper
