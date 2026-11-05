import atexit
import csv
import glob
import json
import os
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

if Path("/val_data/srihari").is_dir():
    os.environ.setdefault("HF_HUB_CACHE", "/val_data/srihari/hf_cache")
    os.environ.setdefault("VLLM_CACHE_ROOT", "/val_data/srihari/vllm_cache")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/val_data/srihari/torch_cache/inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/val_data/srihari/torch_cache/triton")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

DATA = Path(os.environ.get("AI_LAN_DATA") or ROOT / "data")
OUT = Path(os.environ.get("AI_LAN_OUT") or ROOT / "outputs")

LANG_TABLE = [
    ("en", "English", "English", "eng_Latn", "Latin", "native", 5, 7250783, "core"),
    ("hi", "Hindi", "Hindi", "hin_Deva", "Devanagari", "native", 4, 171996, "core"),
    ("hi_latn", "romanised Hindi", "Hindi\n(roman)", None, "Latin", "romanised", 4, 171996, "core"),
    ("hinglish", "Hindi-English code-mixed", "Hinglish", None, "Latin", "code-mixed", 4, 171996, "core"),
    ("ur", "Urdu", "Urdu", "urd_Arab", "Perso-Arabic", "native", 3, 705083, "extended"),
    ("bn", "Bengali", "Bengali", "ben_Beng", "Bengali", "native", 3, 191982, "core"),
    ("ta", "Tamil", "Tamil", "tam_Taml", "Tamil", "native", 3, 191141, "extended"),
    ("mr", "Marathi", "Marathi", "mar_Deva", "Devanagari", "native", 2, 103344, "core"),
    ("pa", "Punjabi", "Punjabi", "pan_Guru", "Gurmukhi", "native", 2, 59741, "extended"),
    ("sa", "Sanskrit", "Sanskrit", "san_Deva", "Devanagari", "native", 2, 12580, "extended"),
    ("gom", "Konkani (Devanagari script)", "Konkani", "gom_Deva", "Devanagari", "native", 2, 3644, "extended"),
    ("te", "Telugu", "Telugu", "tel_Telu", "Telugu", "native", 1, 129621, "extended"),
    ("ml", "Malayalam", "Malayalam", "mal_Mlym", "Malayalam", "native", 1, 89130, "extended"),
    ("kn", "Kannada", "Kannada", "kan_Knda", "Kannada", "native", 1, 34687, "core"),
    ("kn_latn", "romanised Kannada", "Kannada\n(roman)", None, "Latin", "romanised", 1, 34687, "core"),
    ("kanglish", "Kannada-English code-mixed", "Kanglish", None, "Latin", "code-mixed", 1, 34687, "core"),
    ("gu", "Gujarati", "Gujarati", "guj_Gujr", "Gujarati", "native", 1, 30934, "extended"),
    ("ne", "Nepali", "Nepali", "npi_Deva", "Devanagari", "native", 1, 30052, "extended"),
    ("as", "Assamese", "Assamese", "asm_Beng", "Bengali", "native", 1, 25857, "extended"),
    ("sd", "Sindhi (Perso-Arabic script)", "Sindhi", "snd_Arab", "Perso-Arabic", "native", 1, 22655, "extended"),
    ("or", "Odia", "Odia", "ory_Orya", "Odia", "native", 1, 21129, "core"),
    ("sat", "Santali (Ol Chiki script)", "Santali", "sat_Olck", "Ol Chiki", "native", 1, 16060, "core"),
    ("mai", "Maithili", "Maithili", "mai_Deva", "Devanagari", "native", 1, 14368, "extended"),
    ("ks", "Kashmiri (Perso-Arabic script)", "Kashmiri", "kas_Arab", "Perso-Arabic", "native", 1, 11823, "extended"),
    ("bho", "Bhojpuri", "Bhojpuri", None, "Devanagari", "native", 1, 9246, "core"),
    ("tcy", "Tulu (Kannada script)", "Tulu", None, "Kannada", "native", 1, 3986, "core"),
    ("mni", "Manipuri / Meitei (Bengali script)", "Manipuri", "mni_Beng", "Bengali", "native", 0, 10530, "extended"),
    ("brx", "Bodo (Devanagari script)", "Bodo", "brx_Deva", "Devanagari", "native", 0, 0, "core"),
    ("doi", "Dogri", "Dogri", "doi_Deva", "Devanagari", "native", 0, 0, "extended"),
]

LANGS = []
CORE_LANGS = []
EXTENDED_LANGS = []
LANG_NAMES = {}
LANG_LABEL = {}
IT2_CODES = {}
LANG_INFO = {}
for code, name, label, it2, script, form, joshi, wiki, tier in LANG_TABLE:
    LANGS.append(code)
    if tier == "core":
        CORE_LANGS.append(code)
    else:
        EXTENDED_LANGS.append(code)
    LANG_NAMES[code] = name
    LANG_LABEL[code] = label
    if it2 and code != "en":
        IT2_CODES[code] = it2
    if code == "hinglish":
        base_lang = "hi"
    elif code == "kanglish":
        base_lang = "kn"
    else:
        base_lang = code.split("_")[0]
    LANG_INFO[code] = {"base": base_lang, "script": script, "form": form, "joshi": joshi,
                       "wiki": wiki, "tier": tier}

PROMPT_FILES = [
    ("base_prompts.csv", "xstest"),
    ("aya_prompts.csv", "aya"),
    ("wjb_prompts.csv", "wjb_adversarial"),
    ("wjb_vanilla_prompts.csv", "wjb_vanilla"),
    ("framed_prompts.csv", "framed"),
    ("custom_prompts.csv", "custom"),
    ("cultural_prompts.csv", "cultural"),
    ("mhj_prompts.csv", "mhj"),
]

SOURCE_OF_PREFIX = {"p": "xstest", "a": "aya", "w": "wjb_adversarial", "v": "wjb_vanilla",
                    "f": "framed", "c": "custom", "k": "cultural", "m": "mhj"}

TRANSLATIONS = "translations"
LOCALIZED = "translations_localized"
LOCALIZED_TAG = "_loc"


def trans_dir_for(model_folder_name):
    if model_folder_name.endswith(LOCALIZED_TAG):
        return DATA / LOCALIZED
    return DATA / TRANSLATIONS


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_keys(path, key=("pid", "lang")):
    done = set()
    for row in read_jsonl(path):
        done.add(tuple(row[k] for k in key))
    return done


def read_pids(path):
    with open(path) as f:
        return set(f.read().split())


def load_base():
    base = {}
    for file_name, source in PROMPT_FILES:
        path = DATA / file_name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                base[row["pid"]] = dict(row, source=source)
    return base


def bad_pids(base=None):
    if base is None:
        base = load_base()
    bad = set()
    for pid, row in base.items():
        if re.fullmatch(r"\s*n\s*[./]?\s*a\.?\s*", row["prompt"], flags=re.I):
            bad.add(pid)
    return bad


def load_excluded(trans_dir):
    path = trans_dir / "excluded.csv"
    if not path.exists():
        return set()
    with open(path) as f:
        return {(row["pid"], row["lang"]) for row in csv.DictReader(f)}


def load_prompts(langs=LANGS, pids=None, sources=None, folder=TRANSLATIONS):
    trans_dir = DATA / folder
    base = load_base()
    bad = bad_pids(base)
    excluded = load_excluded(trans_dir)
    records = []
    for lang in langs:
        if lang == "en" and folder == TRANSLATIONS:
            trans = {}
            for p in base:
                trans[p] = {"text": base[p]["prompt"], "ambiguity_preserved": True}
        else:
            path = trans_dir / f"{lang}.jsonl"
            if not path.exists():
                print(f"[warn] missing translations for {lang}, skipping")
                continue
            trans = {}
            for row in read_jsonl(path):
                trans[row["pid"]] = row
        for p in base:
            if p not in trans or (p, lang) in excluded or p in bad:
                continue
            b = base[p]
            if sources and b["source"] not in sources:
                continue
            records.append({"pid": p, "lang": lang, "source": b["source"], "label": b["label"],
                            "type": b["type"], "en_text": b["prompt"], "text": trans[p]["text"],
                            "ambiguity_preserved": trans[p].get("ambiguity_preserved", True)})
    if pids is not None:
        keep = set(pids)
        records = [r for r in records if r["pid"] in keep]
        for lang in langs:
            n = sum(r["lang"] == lang for r in records)
            if n < len(keep):
                print(f"WARNING: {lang} has only {n}/{len(keep)} of the --pids prompts (missing translations)")
    return records


ENGLISH_WORDS = set(
    "the a an is are was were do does did how what what's why who where when which can could should "
    "i i'm you my me to of in on for with and or not it it's this that be have has from at by about "
    "as if your don't can't any".split())


def squash(text):
    return re.sub(r"\W+", "", text).lower()


def is_english(text, source=""):
    t = (text or "").strip()
    if not t or (source and squash(t) == squash(source)):
        return False
    letters = [c for c in t if c.isalpha()]
    latin = sum(c.isascii() for c in letters) / max(1, len(letters))
    words = re.findall(r"[A-Za-z']+", t.lower())
    function_share = sum(w in ENGLISH_WORDS for w in words) / max(1, len(words))
    return latin >= 0.8 and (function_share >= 0.10 or len(words) <= 3)


def model_tag(name):
    return name.split("/")[-1].replace(":", "_")


def model_folder(model, tag=""):
    return model_tag(model) + tag


def generation_path(model, condition, tag=""):
    return OUT / "generations" / model_folder(model, tag) / f"{condition}.jsonl"


def direct_messages(record):
    return [{"role": "user", "content": record["text"]}]


def is_vllm_rejection(error):
    return isinstance(error, ValueError) or "Validation" in type(error).__name__


class VLLMModel:
    def __init__(self, model, max_tokens=512, temperature=0.0, tp=1, max_model_len=8192, thinking=False):
        from vllm import LLM, SamplingParams
        print(f"loading {model} with vLLM (first time downloads it; 1-5 min, quiet at first)...", flush=True)
        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len,
                       trust_remote_code=True,
                       gpu_memory_utilization=float(os.environ.get("GPU_MEM", 0.9)))
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.template_kwargs = {"enable_thinking": thinking}
        self.warned_template = False

    def prompt_fits(self, tokenizer, conv):
        try:
            text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                                 **self.template_kwargs)
        except Exception as e:
            if not self.warned_template:
                print(f"WARNING: chat template failed for length check ({e}); using a rough estimate", flush=True)
                self.warned_template = True
            text = "\n".join(m["content"] for m in conv)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False)) + 16
        return n_tokens <= self.max_model_len - self.max_tokens

    def chat(self, convs, show_progress):
        outputs = self.llm.chat(convs, self.params, use_tqdm=show_progress,
                                chat_template_kwargs=self.template_kwargs)
        return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason} for o in outputs]

    def chat_one_by_one(self, convs):
        results = []
        for conv in convs:
            try:
                results.append(self.chat([conv], False)[0])
            except Exception as e:
                if not is_vllm_rejection(e):
                    raise
                results.append({"text": "", "finish_reason": "prompt_too_long"})
        return results

    def generate(self, convs):
        tokenizer = self.llm.get_tokenizer()
        fits = [self.prompt_fits(tokenizer, conv) for conv in convs]
        if not all(fits):
            print(f"{fits.count(False)} prompt(s) too long for the context window -> skipped")
        ok = [conv for conv, fit in zip(convs, fits) if fit]
        try:
            if ok:
                results = self.chat(ok, True)
            else:
                results = []
        except Exception as e:
            if not is_vllm_rejection(e):
                raise
            print(f"batch rejected ({str(e)[:120]}...) -> retrying one prompt at a time")
            results = self.chat_one_by_one(ok)
        n_cut = 0
        for result in results:
            text = result["text"]
            if "</think>" in text:
                trace, answer = text.split("</think>", 1)
                result["text"] = answer.strip()
                result["thinking"] = trace.replace("<think>", "", 1).strip()
            elif text.lstrip().startswith("<think>"):
                result["finish_reason"] = "thinking_cut_off"
                result["thinking"] = text.replace("<think>", "", 1).strip()
                n_cut += 1
        if n_cut:
            print(f"WARNING: {n_cut}/{len(results)} replies were still reasoning at the token limit "
                  "(finish_reason=thinking_cut_off)", flush=True)
        answers = []
        position = 0
        for fit in fits:
            if fit:
                answers.append(results[position])
                position += 1
            else:
                answers.append({"text": "", "finish_reason": "prompt_too_long"})
        return answers


CLAUDE_SYSTEM = "You are a helpful assistant."


def claude_session_files(session_id):
    return glob.glob(os.path.expanduser(f"~/.claude/projects/*/{session_id}.jsonl"))


def run_claude(cmd, text, timeout):
    error = ""
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            error = f"timeout after {timeout} s"
        else:
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout, None
            error = (result.stderr or result.stdout or f"exit {result.returncode}, empty reply").strip()[:200]
        print(f"  claude retry {attempt + 1}/3: {error}", flush=True)
        time.sleep(10 * (attempt + 1))
    print(f"  claude FAILED after 3 tries (saved as ERROR, rerun to retry): {error}", flush=True)
    return None, error


class ClaudeModel:
    def __init__(self, model, workers=4):
        self.model = model
        self.workers = int(os.environ.get("CLAUDE_WORKERS", workers))
        self.templates = {}
        atexit.register(self.delete_templates)

    def base_command(self):
        return ["claude", "-p", "--model", self.model, "--system-prompt", CLAUDE_SYSTEM, "--tools", ""]

    def reply(self, conv):
        system = "\n".join(m["content"] for m in conv if m["role"] == "system") or CLAUDE_SYSTEM
        user = "\n\n".join(m["content"] for m in conv if m["role"] == "user")
        cmd = ["claude", "-p", "--model", self.model, "--system-prompt", system,
               "--tools", "", "--no-session-persistence", "--output-format", "text"]
        out, error = run_claude(cmd, user, 300)
        if out is None:
            return {"text": "", "finish_reason": f"ERROR:{error}"}
        return {"text": out.strip(), "finish_reason": "stop"}

    def template_session(self, opener, canned_reply):
        key = (opener, canned_reply)
        if key in self.templates:
            return self.templates[key]
        session_id = str(uuid.uuid4())
        cmd = self.base_command() + ["--session-id", session_id, "--output-format", "text"]
        out, error = run_claude(cmd, opener, 600)
        if out is None:
            raise SystemExit(f"ERROR: could not create the template session for opener {opener!r}: {error}")
        files = claude_session_files(session_id)
        if len(files) != 1:
            raise SystemExit(f"ERROR: template session file for {session_id} not found ({files})")
        rows = read_jsonl(files[0])
        assistant_rows = [row for row in rows if row.get("type") == "assistant"]
        user_rows = [row for row in rows if row.get("type") == "user"]
        if len(assistant_rows) != 1 or len(user_rows) != 1:
            raise SystemExit(f"ERROR: template {session_id}: expected 1 user + 1 assistant record, got "
                             f"{len(user_rows)} + {len(assistant_rows)}; cannot insert the canned reply safely")
        print(f"  template {session_id[:8]}: model's own opener reply {out.strip()[:60]!r} -> replaced by canned reply")
        assistant_rows[0]["message"]["content"] = [{"type": "text", "text": canned_reply}]
        with open(files[0], "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.templates[key] = (session_id, files[0])
        return self.templates[key]

    def delete_templates(self):
        for session_id, path in self.templates.values():
            try:
                os.remove(path)
            except OSError as e:
                print(f"WARNING: could not delete template session {path}: {e}")

    def reply_after_template(self, conv):
        roles = [m["role"] for m in conv]
        if roles != ["user", "assistant", "user"]:
            raise SystemExit(f"ERROR: Claude multi-turn supports [user, assistant, user] only, got {roles}")
        session_id, path = self.template_session(conv[0]["content"], conv[1]["content"])
        cmd = self.base_command() + ["--resume", session_id, "--fork-session", "--no-session-persistence",
                                     "--output-format", "text"]
        out, error = run_claude(cmd, conv[2]["content"], 600)
        if out:
            return {"text": out.strip(), "finish_reason": "stop"}
        return {"text": "", "finish_reason": f"ERROR:{error}"}

    def generate(self, convs):
        multi_turn = any(m["role"] == "assistant" for m in convs[0])
        if not multi_turn:
            with ThreadPoolExecutor(self.workers) as pool:
                return list(pool.map(self.reply, convs))
        for conv in convs:
            self.template_session(conv[0]["content"], conv[1]["content"])
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.reply_after_template, convs))

    def replay_one(self, turns):
        session_id = str(uuid.uuid4())
        conversation = []
        out = None
        try:
            for k, turn in enumerate(turns):
                if k == 0:
                    flag = ["--session-id", session_id]
                else:
                    flag = ["--resume", session_id]
                out, error = run_claude(self.base_command() + flag + ["--output-format", "text"], turn, 600)
                if out is None:
                    print(f"  claude replay FAILED at turn {k + 1}/{len(turns)} (not saved, rerun to retry)")
                    return {"text": "", "finish_reason": f"ERROR:turn{k + 1}:{error}", "conversation": []}
                conversation.append({"role": "user", "content": turn})
                if k < len(turns) - 1:
                    conversation.append({"role": "assistant", "content": out.strip()})
            return {"text": out.strip(), "finish_reason": "stop", "conversation": conversation}
        finally:
            for path in claude_session_files(session_id):
                os.remove(path)

    def replay(self, turn_lists):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.replay_one, turn_lists))


def load_model(backend, model, max_tokens=512, temperature=0.0, tp=1, max_model_len=8192, thinking=False):
    if backend == "vllm":
        return VLLMModel(model, max_tokens=max_tokens, temperature=temperature, tp=tp,
                         max_model_len=max_model_len, thinking=thinking)
    if thinking:
        raise SystemExit("ERROR: thinking mode is only supported with the vllm backend")
    if backend == "claude":
        return ClaudeModel(model)
    raise SystemExit(f"ERROR: unknown backend {backend} (use vllm or claude)")


def is_failed(finish_reason):
    return str(finish_reason).startswith("ERROR")


def replay_conversations(llm, records):
    if isinstance(llm, ClaudeModel):
        return llm.replay([r["user_turns"] for r in records])
    convs = [[] for r in records]
    results = [None] * len(records)
    n_turns = max(len(r["user_turns"]) for r in records)
    for t in range(n_turns):
        active = [i for i, r in enumerate(records) if t < len(r["user_turns"]) and results[i] is None]
        if not active:
            break
        for i in active:
            convs[i].append({"role": "user", "content": records[i]["user_turns"][t]})
        outputs = llm.generate([convs[i] for i in active])
        print(f"  turn {t + 1}: {len(active)} conversations", flush=True)
        for i, output in zip(active, outputs):
            last = t == len(records[i]["user_turns"]) - 1
            reason = output["finish_reason"]
            bad = reason in ("prompt_too_long", "thinking_cut_off") or is_failed(reason)
            if bad and not last:
                print(f"  WARNING: {records[i]['pid']} {records[i]['lang']} stopped at turn {t + 1}: {reason}")
            if last or bad:
                if last:
                    stopped_at_turn = None
                else:
                    stopped_at_turn = t + 1
                results[i] = dict(output, conversation=list(convs[i]), stopped_at_turn=stopped_at_turn)
            else:
                convs[i].append({"role": "assistant", "content": output["text"]})
    return results


def replay_rows(batch, outputs):
    rows = []
    for record, output in zip(batch, outputs):
        row = {k: v for k, v in record.items() if k != "user_turns"}
        row["conversation"] = output["conversation"]
        row["stopped_at_turn"] = output.get("stopped_at_turn")
        rows.append(row)
    return rows


def generate(model, condition, records, build_messages=None, backend="vllm", tag="", max_tokens=512,
             tp=1, max_model_len=8192, chunk=10, replay=False, thinking=False):
    out = generation_path(model, condition, tag)
    done = done_keys(out)
    todo = [r for r in records if (r["pid"], r["lang"]) not in done]
    print(f"{len(records)} items, {len(done)} done, {len(todo)} to run -> {out}")
    if not todo:
        return out
    llm = load_model(backend, model, max_tokens=max_tokens, tp=tp, max_model_len=max_model_len, thinking=thinking)
    if backend == "vllm":
        step = len(todo)
    else:
        step = chunk
    for start in range(0, len(todo), step):
        batch = todo[start:start + step]
        if replay:
            outputs = replay_conversations(llm, batch)
            batch = replay_rows(batch, outputs)
        else:
            outputs = llm.generate([build_messages(r) for r in batch])
        rows = []
        for record, output in zip(batch, outputs):
            row = dict(record, model=model, backend=backend, condition=condition,
                       response=output["text"], finish_reason=output["finish_reason"],
                       ts=time.strftime("%Y-%m-%d %H:%M:%S"))
            if thinking:
                row["thinking"] = output.get("thinking")
            rows.append(row)
        good = [row for row in rows if not is_failed(row["finish_reason"])]
        n_failed = len(rows) - len(good)
        append_jsonl(out, good)
        message = f"  {min(start + step, len(todo))}/{len(todo)}"
        if n_failed:
            message += f"  ({n_failed} failed, not saved: rerun to retry)"
        print(message, flush=True)
    return out
