import argparse
import re

from common import DATA, append_jsonl, read_jsonl

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
    "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
    "ह": "h", "ळ": "l", "ऱ": "r", "ऴ": "zh",
}
NUKTA_CONSONANTS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "e", "ऑ": "o", "ऎ": "e", "ऒ": "o",
}
VOWEL_SIGNS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o", "ॆ": "e", "ॊ": "o",
}
VIRAMA = "्"
NUKTA = "़"
ANUSVARA = "ं"
CANDRABINDU = "ँ"
VISARGA = "ः"
DIGITS = {}
for i in range(10):
    DIGITS[chr(0x0966 + i)] = str(i)
SHORT_FINAL_VOWEL = {"aa": "a", "ee": "i", "oo": "u"}


def kannada_to_devanagari(text):
    out = []
    for c in text:
        if 0x0C80 <= ord(c) <= 0x0CFF:
            out.append(chr(ord(c) - 0x0C80 + 0x0900))
        else:
            out.append(c)
    return "".join(out)


def tokenize(word):
    tokens = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in CONSONANTS:
            latin = CONSONANTS[ch]
            if i + 1 < len(word) and word[i + 1] == NUKTA:
                latin = NUKTA_CONSONANTS.get(ch, latin)
                i += 1
            if i + 1 < len(word):
                following = word[i + 1]
            else:
                following = ""
            if following == VIRAMA:
                tokens.append([latin, "c"])
                i += 2
            elif following in VOWEL_SIGNS:
                tokens.append([latin, "c"])
                tokens.append([VOWEL_SIGNS[following], "V"])
                i += 2
            else:
                tokens.append([latin, "C"])
                i += 1
            continue
        if ch in VOWELS:
            tokens.append([VOWELS[ch], "V"])
        elif ch == ANUSVARA or ch == CANDRABINDU:
            tokens.append(["n", "N"])
        elif ch == VISARGA:
            tokens.append(["h", "N"])
        elif ch in DIGITS:
            tokens.append([DIGITS[ch], "N"])
        elif ch != NUKTA:
            tokens.append([ch, "N"])
        i += 1
    return tokens


def delete_schwas(tokens):
    if tokens[-1][1] == "C" and len(tokens) > 1:
        tokens[-1][1] = "c"
    for k in range(len(tokens) - 2, 0, -1):
        if tokens[k][1] != "C":
            continue
        after_vowel = tokens[k - 1][1] in "CV"
        following = tokens[k + 1]
        if following[1] == "C":
            following_has_vowel = True
        elif following[1] == "c" and k + 2 < len(tokens) and tokens[k + 2][1] == "V":
            following_has_vowel = True
        else:
            following_has_vowel = False
        if after_vowel and following_has_vowel and following[1] in "Cc":
            tokens[k][1] = "c"


def transliterate_word(word, schwa_deletion):
    tokens = tokenize(word)
    if schwa_deletion and tokens:
        delete_schwas(tokens)
    out = []
    last = len(tokens) - 1
    for k, (latin, kind) in enumerate(tokens):
        if kind == "V" and k == last:
            latin = SHORT_FINAL_VOWEL.get(latin, latin)
        if kind == "N" and latin == "n" and k == last and not schwa_deletion:
            latin = "m"
        if kind == "C":
            latin = latin + "a"
        out.append(latin)
    result = "".join(out)
    if schwa_deletion:
        result = result.replace("ee", "i").replace("oo", "u")
    return result


def transliterate(text, lang):
    if lang == "kn":
        text = kannada_to_devanagari(text)
    text = text.replace("।", ".").replace("॥", ".")
    parts = re.split(r"([ऀ-ॿ]+)", text)
    out = []
    for part in parts:
        if re.match(r"[ऀ-ॿ]", part):
            out.append(transliterate_word(part, schwa_deletion=(lang == "hi")))
        else:
            out.append(part)
    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()
    for src, dst in (("hi", "hi_latn"), ("kn", "kn_latn")):
        rows = read_jsonl(DATA / "translations" / f"{src}.jsonl")
        if args.show:
            for row in rows[:args.show]:
                print(f"{src}: {row['text']}\n  -> {transliterate(row['text'], src)}")
            continue
        path = DATA / "translations" / f"{dst}.jsonl"
        have = {row["pid"] for row in read_jsonl(path)}
        new = []
        for row in rows:
            if row["pid"] in have:
                continue
            new.append({"pid": row["pid"], "lang": dst, "text": transliterate(row["text"], src),
                        "ambiguity_preserved": None, "confidence": "translit",
                        "note": f"mechanical transliteration of the {src} translation"})
        append_jsonl(path, new)
        print(f"{dst}: {len(have)} existing + {len(new)} transliterated -> {path}")


if __name__ == "__main__":
    main()
