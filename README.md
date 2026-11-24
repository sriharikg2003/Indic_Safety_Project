# Indic Safety Project

Code, sample data and full results for the DS 307 (IISc) project report
**"Safety Alignment Evaluation for Indic Languages"**.

We test whether models answer harmful requests in Indian languages because they do not understand
them or because they understand them and still fail to refuse. The study covers 28 Indian language
forms (native script, Romanised and English-mixed) and English, five open-weight models
(Llama-3.1-8B, Qwen3-8B, Aya-Expanse-8B, Gemma-3-12B, Sarvam-M) and Claude Haiku 5.5.

> **Warning:** the data files contain harmful requests taken from public safety benchmarks.

## Repository contents

| Path | Contents |
|---|---|
| `data_samples/xstest_samples.csv` | 5 XSTest contrast pairs (a harmless request and its harmful look-alike, e.g. *kill a Python process* / *kill a person*) in English and all 28 Indian forms, as used in the experiments |
| `results/full_results.csv` | ASR, over-refusal and understanding (%) for every form and model (Table 9 of the report) |
| `results/blocked_requests_log.csv` | The 343 Claude Haiku requests blocked by the provider's usage-policy filter: condition, request ID, language, source set, label and time |
| `EXPERIMENTS.md` | Extra experiments and checks that are not in the report: robustness checks, framed requests, jailbreak tactics, severity, harm categories, phrasing and blocked requests |
| `*.py` | The full pipeline: translation, generation, judging, statistics and one script per experiment |

Request IDs (`pid`) starting with `p` are XSTest items (`xstest_id` gives the original ID),
`w` WildJailbreak, `a` Aya Red-teaming and `m` MHJ.

## What each file does

Run the steps in this order: prompt sets, translation, generation, judging, report.
Each experiment script has a `generate` command (runs the models) and a `report` command
(prints the tables used in the report).

**Shared code**

| File | What it does |
|---|---|
| `common.py` | Paths, language list, loading prompts and translations, running models with vLLM or Claude, detecting provider-filter blocks |
| `judge.py` | Labels every reply with the Qwen3-32B judge (compliance, partial refusal, refusal, misunderstood, harmful content); `--use-translated` also gives the judge a machine translation of the reply (used for Santali) |
| `judged.py` | Loads judged replies for the analysis scripts |
| `stats.py` | Bootstrap confidence intervals over requests, McNemar and sign tests, mixed-effects models |

**Prompt sets**

| File | What it does |
|---|---|
| `sample_prompts.py` | Builds the XSTest prompt list (harmful and harmless look-alike requests) |
| `compose_framed.py` | Builds the framed requests (fictional, academic, hypothetical, authority frames) for languages without IndicTrans2 |
| `fetch_mhj.py` | Downloads and samples the human multi-turn jailbreak conversations (MHJ) |
| `localize.py` | Rewrites requests into natural phrasing and keeps only rewrites whose meaning is preserved |

**Translation**

| File | What it does |
|---|---|
| `translate_indictrans.py` | Translates prompts into the 22 IndicTrans2 languages (`translate`), and translates model replies into English for the judge (`responses`, used for Santali) |
| `transliterate.py` | Produces the Romanised forms (Romanised Hindi and Kannada) |
| `translate_llm.py` | Translates into Tulu, Hinglish, Kanglish and Bhojpuri with an LLM |
| `check_translations.py` | Back-translates every translation into English (these IndicTrans2 back-translations are also the machine translations attached in E3) and checks whether the meaning is preserved; flags poor translations |

**Experiments**

| File | Report section | What it does |
|---|---|---|
| `exp01_safety_gap.py` | E1 | Each request alone in English and 28 Indian forms; harm, over-refusal, misunderstanding |
| `exp02_comprehension.py` | E2 | Understanding check: the model translates each request into English |
| `exp03_english_meaning.py` | E2 | The request with its original English version attached |
| `exp04_meaning_controls.py` | E3 | A sentence without meaning or a machine translation attached instead (also the 22-language run) |
| `exp05_contradicting_meaning.py` | E4 | The English of the contrasting request attached, plus the same test with the request itself in English |
| `exp06_self_translate.py` | E5 | The model is told to translate the request first |
| `exp07_jailbreak_disguise.py` | E6 | WildJailbreak disguised vs plain requests, split by tactic |
| `exp08_framing.py` | `EXPERIMENTS.md` | Requests inside fictional, academic, hypothetical or authority frames |
| `exp12_multiturn.py` | E7 | A neutral opening turn in the request's language or in English |
| `exp13_mhj.py` | E7 | Human-written multi-turn jailbreaks in English, Hindi and Kannada |
| `exp11_cultural_phrases.py` | E8 | Requests with caste, dowry or religion terms and their harmless twins; idioms |
| `exp17_thinking.py` | Reasoning traces | Runs Qwen3 and Sarvam-M with reasoning on and classifies each trace (knew and complied, missed the harm, misread) |
| `exp16_severity.py` | `EXPERIMENTS.md` | Scores each harmful answer from 1 to 5, with a check using translated answers |
| `exp14_llama_guard.py` | Appendix | Checks the judge's harm label with Llama Guard 3 |
| `exp15_closed_models.py` | Claude Haiku | Claude Haiku runs and the comparison with the free web chats |
| `exp09_natural_phrasing.py` | `EXPERIMENTS.md` | Literal vs natural phrasing |
| `exp10_harm_categories.py` | `EXPERIMENTS.md` | Results by harm category (Aya Red-teaming) |

Example:

```bash
pip install -r requirements.txt
python exp01_safety_gap.py generate --model meta-llama/Llama-3.1-8B-Instruct --langs en hi kn
python judge.py --gen <generation file> --langs en hi kn
python exp01_safety_gap.py report
```

Paths default to `data/` and `outputs/` next to the code and can be changed with the
environment variables `AI_LAN_DATA` and `AI_LAN_OUT`.

## Data

The source datasets are public and must be obtained from their authors under their own licences:
XSTest (Röttger et al., 2024), WildJailbreak (Jiang et al., 2024), Aya Red-teaming
(Aakanksha et al., 2024) and MHJ (Li et al., 2024). This repository includes only a small sample
of our translations. The full translated sets can be rebuilt with the translation scripts, or
requested from the author for research use.

## Notes

- All translations are automatic and have not been checked by native speakers; quality is lowest
  for Santali, Manipuri, Romanised Kannada, Bodo and Tulu.
- No model replies are included, because many of them contain harmful content.
