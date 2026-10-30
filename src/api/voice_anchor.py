"""Language anchors for cross-language VoxCPM cloning.

A voice cloned into a language its reference doesn't speak has no accent to
copy, so each chunk picks one on its own (northern in one sentence, southern in
the next). The ``voice-anchor`` operation speaks a short fixed text in the
voice and the target language once, with a VoxCPM2 voice instruction that pins
the language's default accent; the API stores that take and sends it with
every later synthesis of the voice in that language, and each chunk continues
from it in the same accent.

Each text is short (5-10 s, so the take has little room to drift) and uses
everyday words, none tied to one region, dense in the sounds where the
language's regional accents differ, so the accent of the take is unmistakable
and every later chunk follows it.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

from pydantic import BaseModel, Field

ANCHOR_SEED = 1


class LanguageAnchor(NamedTuple):
    instruction: str  # VoxCPM2 "(caption)" prefix, never spoken; it names the accent only,
    # since the reference sets the timbre and recording quality
    text: str


# Base language -> instruction + text; the comment names the sounds the text leans on.
LANGUAGE_ANCHORS = {
    # s/x, r/d/gi and v, -n/-ng, -t/-c, -nh, ươ, hỏi/ngã.
    "vi": LanguageAnchor(
        "A native Hanoi speaker with a standard northern Vietnamese accent",
        "Sáng sớm gió mát rượi, những vườn rau xanh mướt, ai cũng vui vẻ rủ nhau ra sông.",
    ),
    # rhotic r, t flapping (better, water), the TRAP/BATH vowel (after, dance, can).
    "en": LanguageAnchor(
        "A native American speaker with a General American English accent",
        "Honestly, it's better to park the car by the water tower after dinner. "
        "Then we can dance for an hour or two.",
    ),
    # zh/ch/sh/r against z/c/s, -n against -ng, f against h, n against l.
    "zh": LanguageAnchor(
        "A native Beijing speaker of standard Mandarin",
        "这周日上午，我们准时出发，先去银行办事，然后到城里找家小店，吃一碗热乎乎的牛肉面。",
    ),
    # c/z (θ or s), ll (ʝ or ʃ), j (x or h), final s.
    "es": LanguageAnchor(
        "A native Madrid speaker with a Castilian Spanish accent",
        "Hace un día precioso: cinco amigos cruzaron la calle "
        "y llegaron juntos a la plaza a las once.",
    ),
    # t/d before i or final e (t or tʃ), final s (s or ʃ), r and rr, reduced vowels.
    "pt": LanguageAnchor(
        "A native Lisbon speaker with a European Portuguese accent",
        "Hoje à tarde, o tio Davi comprou leite e pão quente, "
        "e depois partiu de carro para a cidade.",
    ),
    # t/d before i/u (affricated or not), long vowels (frère, fête), nasal vowels.
    "fr": LanguageAnchor(
        "A native Parisian speaker with a standard French accent",
        "Ce matin, tu m'as dit que ton petit frère viendrait du marché "
        "vers midi, juste avant la fête.",
    ),
    # -ig (ç or k), initial s (z or s), st/sp, r, ch.
    "de": LanguageAnchor(
        "A native German speaker with a standard German accent from Germany",
        "Am Sonntag sind wir zwanzig Minuten zu spät gekommen, "
        "aber eigentlich war das richtig lustig.",
    ),
    # intervocalic s (z or s), open/closed e and o, doubled consonants after a/è, z.
    "it": LanguageAnchor(
        "A native Italian speaker with a standard Italian accent",
        "Stamattina la mia vicina è venuta a casa con una rosa "
        "e una tazza di caffè, che cosa gentile.",
    ),
    # ج (dʒ, g or ʒ), ق (q, g or ʔ), ث (θ, t or s), ذ (ð, d or z), ظ.
    "ar": LanguageAnchor(
        "A native Arabic speaker using Modern Standard Arabic pronunciation",
        "قال جدي إن ثلاثة رجال جاؤوا إلى القرية قبل الظهر، وذهبوا إلى السوق القديم.",
    ),
    # pitch accent on 雨, 橋, 花, ありがとう; long vowels.
    "ja": LanguageAnchor(
        "A native Tokyo speaker of standard Japanese",
        "雨の日に、橋の近くの花屋さんで赤い花を買って、ありがとうと言いました。",
    ),
    # ㅆ against ㅅ, ㅓ against ㅡ, pitch.
    "ko": LanguageAnchor(
        "A native Seoul speaker of standard Korean",
        "어제 쌀을 사러 시장에 갔는데, 생각보다 사람이 많아서 그냥 집으로 돌아왔어요.",
    ),
    # g/ch (hard or soft), w (ʋ or w), ij/ei/ui diphthongs, final -n.
    "nl": LanguageAnchor(
        "A native Dutch speaker with a standard accent from the Netherlands",
        "Gisteren ging ik met mijn goede vriend buiten wandelen langs het water, "
        "en we hebben heerlijk gelachen.",
    ),
    # retroflex ड़/ठ, aspirates (भ, घ, ख), nuqta sounds (ज़, ख़), nasal vowels.
    "hi": LanguageAnchor(
        "A native Delhi speaker with a standard Hindi accent",
        "आज सुबह मेरे बड़े भाई ने घर के पास वाले बाज़ार से ताज़ी सब्ज़ियाँ और मीठे फल ख़रीदे।",
    ),
    # unstressed o (akanye), г (g, not ɣ), ч, palatalized consonants.
    "ru": LanguageAnchor(
        "A native Moscow speaker with a standard Russian accent",
        "Вчера вечером мы с подругой долго гуляли по городу, а потом пили горячий чай с молоком.",
    ),
    # ğ, ı, ö/ü, r, k/g before front vowels.
    "tr": LanguageAnchor(
        "A native Istanbul speaker with a standard Turkish accent",
        "Bugün öğleden sonra kız kardeşimle sahile gittik ve güzel bir çay içtik.",
    ),
    # schwa e against e, trilled r, final k, ny/ng.
    "id": LanguageAnchor(
        "A native Indonesian speaker with a standard Indonesian accent",
        "Kemarin sore saya pergi ke pasar bersama keluarga untuk membeli sayur, ikan, dan buah segar.",
    ),
    # sz/ś, ż/rz/ź, ł, nasal ą/ę, ch.
    "pl": LanguageAnchor(
        "A native Polish speaker with a standard Polish accent",
        "Wczoraj wieczorem moja siostra przyniosła świeży chleb i zrobiła pyszną herbatę z cytryną.",
    ),
    # tones, r against l (อร่อย, ตลาด), aspirates, long vowels.
    "th": LanguageAnchor(
        "A native Bangkok speaker of standard Central Thai",
        "เมื่อวานตอนเย็นไปตลาดกับคุณแม่ ซื้อผลไม้สดและข้าวเหนียวมะม่วงที่อร่อยมากกลับบ้าน",
    ),
    # final a (ə in Malaysia, a in Indonesia), r, final k.
    "ms": LanguageAnchor(
        "A native Malaysian speaker with a standard Malay accent",
        "Semalam kita pergi ke pasar malam bersama keluarga, membeli kuih dan air kelapa yang sejuk.",
    ),
    # ng, glottal stops, o/u and e/i, flapped r.
    "fil": LanguageAnchor(
        "A native Manila speaker with a standard Filipino accent",
        "Kahapon ng hapon, pumunta kami ng nanay ko sa palengke para bumili ng isda at gulay.",
    ),
    # written Cantonese (哋, 咗, 係) keeps it out of Mandarin; -p/-t/-k finals, tones.
    "yue": LanguageAnchor(
        "A native Hong Kong speaker of Cantonese",
        "今日天氣好好，我哋一齊去飲茶，食咗好多點心，真係好開心。",
    ),
    # sj/tj sounds, pitch accent, r, long vowels.
    "sv": LanguageAnchor(
        "A native Stockholm speaker with a standard Swedish accent",
        "I går kväll tog min syster med sig sju kanelbullar, och vi drack kaffe nere vid sjön.",
    ),
    # χ, θ, γ, δ, ψ.
    "el": LanguageAnchor(
        "A native Athens speaker with a standard Greek accent",
        "Χθες το απόγευμα πήγαμε με τους φίλους μας στη θάλασσα και φάγαμε φρέσκο ψάρι δίπλα στο λιμάνι.",
    ),
    # ר, ח against כ, ע.
    "he": LanguageAnchor(
        "A native Israeli speaker with a standard Modern Hebrew accent",
        "אתמול בערב הלכנו עם החברים לים, ואכלנו ארוחה טעימה ליד החוף.",
    ),
    # stød, soft d, ø, r.
    "da": LanguageAnchor(
        "A native Copenhagen speaker with a standard Danish accent",
        "I går aftes gik vi en lang tur ved havnen, og bagefter spiste vi rødgrød med fløde.",
    ),
    # long vowels and consonants, r, ä/ö/y.
    "fi": LanguageAnchor(
        "A native Helsinki speaker with a standard Finnish accent",
        "Eilen illalla kävimme ystävien kanssa saunassa, ja sen jälkeen söimme lämpimiä pullia.",
    ),
    # r, retroflex rd/rt, kj, pitch accent.
    "nb": LanguageAnchor(
        "A native Oslo speaker with a standard Eastern Norwegian accent",
        "I går kveld gikk vi en lang tur rundt fjorden, og etterpå kjøpte vi varme vafler med syltetøy.",
    ),
    # mb/nd/ng prenasals, r against l, vowel length.
    "sw": LanguageAnchor(
        "A native Kenyan speaker with a standard Swahili accent",
        "Jana jioni nilikwenda sokoni pamoja na rafiki yangu kununua matunda, samaki na mboga.",
    ),
    # r clusters (ស្រ, ត្រ), final consonants.
    "km": LanguageAnchor(
        "A native Cambodian speaker with a standard Khmer accent",
        "ម្សិលមិញ ខ្ញុំបានទៅផ្សារជាមួយម្ដាយ ដើម្បីទិញត្រី បន្លែ និងផ្លែឈើស្រស់។",
    ),
    # tones, ຫຼ, final consonants.
    "lo": LanguageAnchor(
        "A native Vientiane speaker with a standard Lao accent",
        "ມື້ວານນີ້ ຂ້ອຍໄປຕະຫຼາດກັບແມ່ ເພື່ອຊື້ປາ ຜັກ ແລະ ໝາກໄມ້ສົດ.",
    ),
    # tones, creaky and checked syllables, ky/gy.
    "my": LanguageAnchor(
        "A native Yangon speaker with a standard Burmese accent",
        "မနေ့က အမေနဲ့အတူ ဈေးကိုသွားပြီး ငါး၊ ဟင်းသီးဟင်းရွက်နဲ့ သစ်သီးတွေ ဝယ်ခဲ့တယ်။",
    ),
    # Shanghainese words (阿拉, 白相, 个) keep it out of Mandarin.
    "wuu": LanguageAnchor(
        "A native Shanghai speaker of Shanghainese",
        "今朝天气老好个，阿拉一道到外滩去白相，后来又吃了一碗小馄饨。",
    ),
}


# Legacy or macro codes for a language whose anchor sits under another code.
_ANCHOR_ALIASES = {"tl": "fil", "no": "nb"}


def anchor_language(language: str) -> str:
    base = language.replace("_", "-").split("-")[0].lower()
    return _ANCHOR_ALIASES.get(base, base)


class VoiceAnchorRequest(BaseModel):
    """Request body for the ``voice-anchor`` operation."""

    model_config = {"extra": "forbid"}

    voice_id: str = Field(..., min_length=1, description="Opaque product voice id")
    reference_url: str = Field(..., description="Reference sample URL of the voice")
    reference_language: Optional[str] = Field(None, description="Language/locale spoken by the reference sample")
    language: str = Field(..., min_length=1, description="Target language/locale; its base language picks the anchor")
    format: Literal["wav", "mp3"] = Field("wav", description="Output audio format (wav or mp3)")
