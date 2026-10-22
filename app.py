"""
Police Narrative Classifier & Policy Assistant – Streamlit app
MAAI 902 Applied NLP Project · Navi Dharshan B

Tab 1  Classify a narrative with the fine-tuned BERT model and explain the predicted category from BPD Rule 323.
Tab 2  Ask questions answered from approved documents (RAG: hybrid retrieval + Qwen2.5-1.5B-Instruct, with citations).
Tab 3  About: data, results, limitations and responsible use.

Expected files next to app.py:
  bert_best/            fine-tuned BERT (from bert_best_model.zip) – or set BERT_REPO to a Hugging Face model repo
  rag_outputs/          chunks.json and rag_config.json (from rag_outputs.zip) – or the two files next to app.py
"""
import json, math, os, re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import torch

BERT_DIR = os.environ.get("BERT_DIR", "bert_best")
BERT_REPO = os.environ.get("BERT_REPO", "")          # e.g. "username/police-narrative-bert" on Hugging Face
RAG_DIR = os.environ.get("RAG_DIR", "rag_outputs" if os.path.isdir("rag_outputs") else ".")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 320
MAX_NEW_TOKENS = 220 if DEVICE == "cuda" else 160     # shorter answers keep CPU hosting responsive
APP_MODE = os.environ.get("APP_MODE", "full").strip().lower()
USE_LLM = APP_MODE != "lite"   # lite = no language model (for hosts with ~1–3 GB memory, e.g. Streamlit Cloud)
if DEVICE == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 1))

st.set_page_config(page_title="Police Narrative Classifier", page_icon="🔎", layout="wide")


# ------------------------------------------------------------------ loading (cached once per session)
@st.cache_resource(show_spinner="Downloading the fine-tuned BERT model…")
def bert_path():
    if os.path.exists(os.path.join(BERT_DIR, "config.json")):
        return BERT_DIR
    if BERT_REPO:
        from huggingface_hub import snapshot_download
        return snapshot_download(BERT_REPO, token=os.environ.get("HF_TOKEN") or None)
    return None


@st.cache_resource(show_spinner="Loading the classifier…")
def load_classifier():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = bert_path()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
    return tok, model


@st.cache_resource(show_spinner="Loading documents and the embedding model…")
def load_retriever():
    from sentence_transformers import SentenceTransformer
    chunks = json.load(open(os.path.join(RAG_DIR, "chunks.json")))
    cfg = json.load(open(os.path.join(RAG_DIR, "rag_config.json")))
    embedder = SentenceTransformer(cfg["embed_model"], device=DEVICE)
    emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, convert_to_numpy=True)
    return chunks, cfg, embedder, emb.astype("float32"), BM25([c["text"] for c in chunks])


@st.cache_resource(show_spinner="Loading the language model (first time takes about a minute)…")
def load_llm(name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return tok, model.to(DEVICE).eval()


# ------------------------------------------------------------------ retrieval (same method as the RAG notebook)
STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its what which who when "
           "how does do can may must under about there their than into if any".split())


def tok_words(s):
    return [w for w in re.findall(r"[a-z0-9_]+", s.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tok_words(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = np.mean([len(d) for d in self.docs])
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = tok_words(query)
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            for w in q:
                if w in tf:
                    f = tf[w]
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
        return out


def retrieve(query, k=4, method="hybrid", rrf_k=60):
    chunks, _, embedder, emb, bm25 = load_retriever()
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    cos = emb @ q
    dense_order = np.argsort(-cos)
    if method == "dense":
        order = dense_order
    else:
        bm_order = np.argsort(-bm25.scores(query))
        if method == "bm25":
            order = bm_order
        else:
            dr = {int(i): r for r, i in enumerate(dense_order)}
            br = {int(i): r for r, i in enumerate(bm_order)}
            fused = {i: 1 / (rrf_k + dr[i]) + 1 / (rrf_k + br[i]) for i in range(len(chunks))}
            order = sorted(fused, key=fused.get, reverse=True)
    return [(int(i), float(cos[int(i)])) for i in list(order)[:k]]


# ------------------------------------------------------------------ generation
def build_messages(question, hits, style, cfg, chunks):
    sources = "\n\n".join(f"[S{n}] ({chunks[i]['ref']}) {chunks[i]['text']}" for n, (i, _) in enumerate(hits, 1))
    if style == "Basic":
        return [{"role": "user", "content": f"Context:\n{sources}\n\nQuestion: {question}\nAnswer:"}]
    return [{"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}]


@torch.no_grad()
def answer(question, k=4, method="hybrid", style="Grounded", gate=True):
    chunks, cfg, *_ = load_retriever()
    hits = retrieve(question, k=k, method=method)
    if gate and max(s for _, s in hits) < cfg["min_score"]:
        return cfg["refusal"], hits, True
    if not USE_LLM:                                   # lite mode: extractive answer from the best passage
        i = hits[0][0]
        words = chunks[i]["text"].split()
        excerpt = " ".join(words[:120]) + (" …" if len(words) > 120 else "")
        return f"Most relevant passage – **{chunks[i]['ref']}**:\n\n> {excerpt}", hits, False
    tok, llm = load_llm(cfg["llm"])
    enc = tok.apply_chat_template(build_messages(question, hits, style, cfg, chunks), add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(DEVICE)
    out = llm.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), hits, False


def show_sources(hits):
    chunks = load_retriever()[0]
    with st.expander(f"Sources used ({len(hits)})"):
        for n, (i, s) in enumerate(hits, 1):
            st.markdown(f"**[S{n}] {chunks[i]['ref']}** · similarity {s:.2f}")
            st.caption(chunks[i]["text"])


@torch.no_grad()
def classify(text):
    tok, model = load_classifier()
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    probs = torch.softmax(model(**enc).logits, -1)[0].cpu().numpy()
    labels = [model.config.id2label[i] for i in range(len(probs))]
    n_tokens = len(tok(text)["input_ids"])
    return labels, probs, n_tokens


# ------------------------------------------------------------------ simple redaction for pasted text
REDACT_RULES = [
    (re.compile(r"\b(?:\d{1,5}\s+)?(?:[A-Z][a-z]+\s){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Pl|Place|Sq|Square|"
                r"Ter|Terrace|Way|Ct|Court|Dr|Drive|Pk|Park|Hwy)\b\.?"), "[ADDRESS]"),
    (re.compile(r"\b\(?\d{3}\)?[-. ]?\d{3}[-. ]\d{4}\b"), "[PHONE]"),
    (re.compile(r"X{2,}"), "[REDACTED]"),
]


def light_redact(text):
    for rx, tag in REDACT_RULES:
        text = rx.sub(tag, text)
    return text


EXAMPLES = {
    "— choose an example —": "",
    "Traffic stop with frisk": "Officers stopped the motor vehicle for a red light violation at [ADDRESS]. The operator was "
        "issued a verbal warning. A pat frisk of the passenger was conducted after officers saw a bulge in his "
        "waistband. Nothing was found and both parties were released.",
    "Conversation with a known associate": "Officers engaged in a casual conversation with [REDACTED] outside the corner "
        "store. He stated he was waiting for a friend and was cordial with officers. FIO for intel purposes, known "
        "gang associate.",
    "Observation without contact": "Officers observed [REDACTED] and an unknown male standing in front of [ADDRESS] in "
        "an area known for recent shootings. No contact was made with either individual. FIO for intel.",
}
RULE_TERM = {"Stopped": "field interaction/stop", "Encountered": "encounter", "Observed": "observation"}

# ------------------------------------------------------------------ layout
st.title("Police Narrative Classifier & Policy Assistant")
st.caption("Boston Police FIO records, 2024 · fine-tuned BERT classifier + retrieval-augmented assistant · "
           "MAAI 902 research project – not for operational use")

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Retriever", ["hybrid", "dense", "bm25"], help="Hybrid = dense embeddings + BM25 combined")
    k = st.slider("Passages per answer", 2, 6, 4)
    if USE_LLM:
        style = st.radio("Prompt", ["Grounded", "Basic"], help="Grounded: cite sources, refuse when not covered")
    else:
        style = "Grounded"
        st.caption("Lite mode: the assistant shows the most relevant document passage instead of a generated "
                   "answer (no language model on this host).")
    gate = st.checkbox("Refuse when no relevant passage is found", value=True)
    st.divider()
    st.caption(f"Device: {'GPU – ' + torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU (answers will be slow)'}")

missing = []
if not os.path.exists(os.path.join(RAG_DIR, "chunks.json")):
    missing.append("rag_outputs/chunks.json")
try:
    if bert_path() is None:
        missing.append("bert_best/ (or set BERT_REPO)")
except Exception as e:
    missing.append(f"BERT model from {BERT_REPO} ({e.__class__.__name__})")
if missing:
    st.error("Missing: " + ", ".join(missing) + ". Unzip bert_best_model.zip into `bert_best/` and "
             "rag_outputs.zip into `rag_outputs/` next to app.py, or set BERT_REPO to the Hugging Face model repo.")
    st.stop()
if DEVICE == "cpu" and USE_LLM:
    st.info("Running on CPU: classification is quick, but generated answers take about 20–60 seconds.", icon="⏱️")

tab_cls, tab_chat, tab_about = st.tabs(["🔎 Classify a narrative", "💬 Ask the documents", "ℹ️ About"])

# ---- Tab 1
with tab_cls:
    st.subheader("Classify an FIO narrative")
    st.write("Paste a narrative (or pick an example). BERT predicts the recorded circumstance; the assistant then "
             "explains what that category means under BPD Rule 323.")
    choice = st.selectbox("Example", list(EXAMPLES))
    text = st.text_area("Narrative", value=EXAMPLES[choice], height=170,
                        placeholder="e.g. Officers stopped the vehicle for…")
    c1, c2 = st.columns([1, 1])
    explain = c2.checkbox("Explain with Rule 323", value=True)
    if c1.button("Classify", type="primary", disabled=not text.strip()):
        clean = light_redact(text.strip())
        labels, probs, n_tok = classify(clean)
        pred = labels[int(probs.argmax())]
        st.markdown(f"### Predicted: **{pred}**  ·  confidence {probs.max():.0%}")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=labels))
        if n_tok > MAX_LEN:
            st.warning(f"This narrative has {n_tok} tokens; only the first {MAX_LEN} were read by the model.")
        if probs.max() < 0.6:
            st.info("Low confidence – the narrative may fit more than one category. Encountered and Observed are "
                    "the categories the model confuses most.")
        if explain:
            with st.spinner("Retrieving the policy definition…"):
                q = f"Under Rule 323, what is a {RULE_TERM[pred]} and what must an officer document for it?"
                ans, hits, _ = answer(q, k=k, method=method, style="Grounded", gate=False)
            st.markdown("**What this category means in BPD policy**")
            st.write(ans)
            show_sources(hits)
        st.caption("The prediction reproduces how records were labelled; it is not a judgement on whether any "
                   "police action was justified or lawful.")

# ---- Tab 2
with tab_chat:
    st.subheader("Ask about the FIO data and policy")
    st.write("Answers come only from BPD Rule 323, the official FIO Data Key, and the project's dataset and model "
             "cards. Each answer cites its sources." + ("" if USE_LLM else " In lite mode the best-matching passage is "
             "shown directly; the full version generates answers with Qwen2.5-1.5B-Instruct."))
    st.caption("Try: *What is the difference between an observation and an encounter?* · *When may an officer frisk "
               "someone?* · *Which column holds the narrative?* · *How accurate is the classifier?*")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, msg, hits in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)
            if hits:
                show_sources(hits)
    if q := st.chat_input("Ask a question"):
        st.session_state.chat.append(("user", q, None))
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents…"):
                ans, hits, gated = answer(q, k=k, method=method, style=style, gate=gate)
            st.write(ans)
            if gated:
                st.caption("No passage was relevant enough, so the language model was not called.")
            else:
                show_sources(hits)
        st.session_state.chat.append(("assistant", ans, None if gated else hits))
    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()

# ---- Tab 3
with tab_about:
    st.subheader("About this project")
    st.markdown("""
**Research question.** How effectively can a fine-tuned BERT model classify police encounter narratives into the
existing *Stopped*, *Encountered* and *Observed* categories compared with traditional machine-learning methods?

**Data.** Boston Police Department Field Interrogation and Observation (FIO) records for 2024, published on Analyze
Boston under the PDDL public-domain licence. 4,860 records → 4,616 after removing empty and duplicate narratives;
split 70/15/15. The person-level file with demographic data was not used. Narratives were redacted (officer names,
addresses, neighbourhoods and other names/places) before modelling.
""")
    st.markdown("**Test-set results (693 narratives)**")
    st.table(pd.DataFrame({
        "Accuracy": ["0.775", "0.794 ± 0.005", "0.736", "0.758 ± 0.016"],
        "Macro F1": ["0.693", "0.736 ± 0.009", "0.639", "0.689 ± 0.012"]},
        index=["TF-IDF + Logistic Regression", "BERT (3 seeds)", "TF-IDF + LR, label words masked",
               "BERT, label words masked"]))
    st.markdown("""
BERT improves macro F1 by 4.3 points, with the largest gains on *Encountered* and *Observed*. All 6 BERT models across
two training runs beat the baseline (earlier run: 0.732 ± 0.014), but the difference is not statistically significant
on a single test set (McNemar p = 0.35).

**Limitations.** Some recorded labels contradict the narrative; *Observed* recall is only 0.57; long narratives are
truncated at 320 tokens; one year of data; the narratives contain racial descriptors that a model can learn from.

**Responsible use.** Research and teaching only. The tool must not be used to make or support decisions about
individuals, and it does not assess whether a stop was lawful. The assistant is not legal advice.
""")
