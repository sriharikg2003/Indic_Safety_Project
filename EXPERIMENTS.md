# Extra experiments and checks

This file collects analyses that we ran but did not put in the report, so the report stays focused on its main question. All numbers come from the same Qwen3-32B judge as the report. ASR is the share of harmful requests answered with harmful content, and over-refusal is the share of harmless requests refused fully or partly. Santali replies are judged through a machine translation.

## Extra checks on the results

- **Confidence intervals and tests.** 95% intervals come from a bootstrap over requests. Paired comparisons use exact McNemar tests. The 95% interval of the gap between Indian forms and English excludes zero for every open model.
- **Refits.** The gap, the effect of the English version and the effect on understood requests were refitted as mixed-effects logistic models (random intercepts for request and language) and as generalised estimating equation (GEE) models clustered by request. Some fits did not converge. In the usable fits the odds ratio of the English-version effect is at most 0.27 for every open model. The higher ASR of partly than of fully understood requests appears for Qwen, Aya, Gemma and Sarvam-M in most fits but not for Llama, whose ASR is 24% on fully understood and 25% on partly understood requests.
- **Poor translations.** Excluding the 1,383 of 12,600 translated requests (450 per form) whose meaning was lost in every back-translation does not shrink the gap or the effect of the original English (Llama gap 21.9 to 23.0 pp).
- **Repetition loops.** Replies stuck in repetition loops make up 8 to 11% of direct Indian-language replies of Llama, Qwen and Gemma. Removing them changes no rate by more than 0.5 pp.
- **Reply length cap.** Replies are capped at 512 tokens. Re-running 100 requests in English, Hindi, Bengali, Kannada and Tulu with 2,048 tokens changed the harm label of 0 to 2% of replies that had not been cut off and 1 to 7% of those that had. Most changes were downward because longer replies add warnings. The cap slightly overstates ASR, mainly for Aya, but does not create the gap.
- **Data validation.** A script checks about 875,000 verdicts for missing fields, wrong script, duplicates and broken pairings. Six remain unusable and are counted as not harmful.
- **Free web chats.** ChatGPT, Gemini, Claude and Perplexity answered 20 requests in 5 forms, entered by hand in batches of ten. The test is only indicative because people do not normally send requests in batches and batching changed the results in a pilot.

## Framed requests

We placed 316 existing XSTest and Aya requests in four short English frames: a fiction frame (a scene for a regional-language TV serial), an academic frame (a college assignment on social issues in India), a hypothetical frame (just hypothetically, out of curiosity) and an authority frame (a police officer in Bengaluru investigating a case). A pilot used 116 requests, and the other 1,200 apply the four frames to 200 harmful and 100 harmless XSTest requests. They were translated with IndicTrans2.

- A fictional frame raises ASR in six Indian forms for Llama (23% to 37%), Aya (19% to 34%) and Qwen (7% to 19%). In English it adds at most 7 pp.
- An authority frame never raises the ASR of the five open models taken together in Indian forms. It does raise Claude Haiku (2.5% to 4.7%).
- Gemma refuses framed harmless requests much more often in Indian forms (Kannada 24% to 53%, Odia 31% to 59%).

## Jailbreak tactics

WildJailbreak labels each hidden-intent request with free-text tactics, grouped here by keyword into six overlapping families. ASR for the plain request to its hidden-intent version, five open models combined:

| Tactic family | Requests | English | Indian forms |
|---|---|---|---|
| Role-play and fiction | 208 | 7 to 28 | 26 to 27 |
| Authority and pressure | 141 | 5 to 27 | 23 to 26 |
| Justification | 137 | 6 to 20 | 24 to 21 |
| Distraction and nesting | 87 | 4 to 25 | 22 to 26 |
| Obfuscation | 75 | 2 to 23 | 20 to 20 |
| Politeness and emotion | 24 | 3 to 11 | 21 to 18 |

Each family raises English ASR 3 to 15 times but changes it by a factor of only 0.9 to 1.2 in Indian forms, so the result is not driven by one kind of trick.

## Longer notes on the experiments in the report

- **E5, self-translation.** Over 13 forms, self-translation lowers ASR for Aya (14% to 3%), Llama (19% to 12%), Sarvam-M (8% to 2%) and Qwen (6% to 2%), never by more than the attached English. It slightly raises over-refusal for Sarvam-M (5% to 9%), Llama (4% to 6%) and Qwen (6% to 7%). Llama writes an English translation in few of its replies (15% in Hindi, none in Bodo). Gemma translates at least 99% of the requests but gains nothing (5% to 5%), while the attached English brings it to 0%.
- **E6, hidden intent.** In English the tricks raise ASR 2.6 to 5.3 times for every open model. In Indian languages they add much less (Llama 40% vs 8% in English for the plain request), and for Llama and Aya the trick even lowers ASR. The difference is consistent in all 9 languages for Llama, Qwen and Aya, and in 4 and 3 of 9 for Gemma and Sarvam-M. Among requests whose hidden-intent version the model translated correctly, tricks raise ASR (Gemma 6% to 23%) and among those it mistranslated they lower it. This is a link and not proof of cause.
- **E7, conversations.** A neutral opening in the user's language changes ASR by at most 1.3 pp for Llama, Qwen and Gemma across six Indian forms. Aya rises (19% to 24%, Kanglish 8% to 21%), while Sarvam-M falls (11% to 6%) and refuses more harmless requests (5% to 12%). An English opening lowers ASR for Aya (24% to 8%) and Gemma (7% to 3%), but Aya misunderstands more requests (38% to 55%) and Gemma refuses more harmless ones (12% to 22%). Human-written MHJ jailbreaks succeed in 40% of cases in English, 24% in Hindi and 19% in Kannada, with 82 to 98% compliance in most cells. Aya misunderstands 62% of the Kannada conversations.
- **E8, culturally loaded words.** We took the 59 harmful Aya requests that mention caste, dowry or religion and, with the help of Claude, wrote one harmless request on the same term for each. Harmful requests with these terms are about as unsafe in Indian forms as other Aya requests (19% vs 17% ASR). Only 1% of the harmless requests on the same terms are refused in Indian forms (13 of 2,065), against 7% of harmless XSTest requests. We also tested 20 harmless requests built on violent-sounding English idioms. Models refuse them rarely (1%), but keep the idiom word in only 26% of their own translations, and a Gemini check found 14 of the 20 Kannada translations wrong or unnatural.
- **Polysemous and code-mixed phrases.** Requests with such a phrase are translated slightly less faithfully than matched requests (72% vs 78% keep the meaning) but are understood as often, and self-translation and the English version reduce their ASR by about the same relative amount. For example, the ritual *nazar utarna* (removing the evil eye) lost its meaning in the translations of five of six models.
- **Claude Haiku.** Haiku's ASR is at most 3.5% in every form except Romanised Kannada (6%) and Kanglish (9.5%). Attaching a meaningless sentence, a machine translation or the English version leaves it below 1% across five languages, and self-translation brings no gain. WildJailbreak tricks make Haiku safer in English and in the Indian forms taken together. In Santali and Bodo more than 90% of its replies to Aya requests misread them. It refuses none of the harmless requests on culturally loaded terms.

## Other analyses

- **Severity.** The judge scored every harmful answer of the open models (15,811 replies) from 1 (vague or widely known) to 5 (detailed, step-by-step help). Harmful answers in Indian forms score 2.91 on average against 3.67 in English, and 30% against 68% score 4 or 5. Every model scores 0.73 to 0.85 points lower in Indian forms. On matched requests that were answered harmfully in both English and an Indian form, the Indian answer scores 0.56 points lower. When the judge also sees an English translation of each answer, scores fall slightly instead of rising (2.92 to 2.76). Haiku's rare harmful answers also score lower in Indian forms (3.03 vs 3.39, with only 31 harmful English answers). Script: `exp16_severity.py`.
- **Harm categories.** On Aya requests, across the five open models, ASR rises from 0 to 12% in English to 12 to 23% in Indian forms in every category, and correct refusal roughly halves (76% to 37%). Discrimination (0% to 23%) and hate speech (0% to 15%) are among the most affected, and self-harm among the least (2% to 12%). On XSTest, grouped by keyword, weapons and explosives are most affected (0% to 18%), and the English version removes 71 to 97% of the extra ASR in every category. Script: `exp10_harm_categories.py`.
- **Literal versus natural phrasing.** Qwen3-32B rewrote each XSTest request the way an ordinary speaker would say it, and rewrites that changed the meaning in a back-translation check were removed (316 to 393 of 450 kept per language). Across Hindi, Bengali, Marathi, Kannada and Odia, natural rewrites changed ASR by no detectable amount for any open model (McNemar p >= 0.05). Gemma refuses natural harmless rewrites more often (75 items refused only when natural vs 45 only when literal, p = 0.008). Script: `exp09_natural_phrasing.py`.
- **Requests blocked by the provider filter.** For 343 of about 82,000 Claude Haiku requests, Anthropic's service returned a usage-policy error instead of a reply. By condition, 211 fall on hidden-intent harmful WildJailbreak requests, 63 on hidden-intent harmless ones, 26 on plain WildJailbreak requests, 14 on framed requests, 7 on plain XSTest requests and 2 on MHJ conversations. Each of the nine Indian forms is blocked more often than English (4% in English, 5 to 13% in seven forms, 15% in Santali and 24% in Bodo). Of the 330 requests blocked in an Indian form, the same request in English was answered in 254 cases and none of those answers was harmful. When the blocked text was re-sent with an instruction only to translate it, 250 of the 343 passed. One request blocked on every attempt stalled our runner for about five hours until we changed it to save blocks as a separate outcome. The list is in `results/blocked_requests_log.csv`.
