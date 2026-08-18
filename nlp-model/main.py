"""
nlp-model — NanoLM English NLP Model (model #2 for guru-ewm).

A self-contained, CPU-only, lattice-algebra English language model. It
understands English input and responds in English by retrieving from an
ingested knowledge corpus (stored as HLLSets in hllset-next) — no external
LLM, no neural networks, no GPU.

Pipeline (spec NANOLM_NLP_MODEL_SPECIFICATION.md):
    message → intent classify + keyword extract + negation detect
      → keyword index + BSS retrieval (c:nlp:* lattice)
        → response analyzer picks the single most relevant answer (DirectMatch / HighMatch / Match / Fallback)
          → IPFS persistence

Endpoints:
  GET  /health                — model + lattice + IPFS status
  POST /nlp/ingest            — (re)ingest the English seed corpus
  POST /nlp/ingest/document   — multipart file → IPFS → split → ingest cards
  GET  /nlp/status            — corpus count, readiness, last snapshot CID
  POST /chat                  — message → English reply
  POST /nlp/query             — message → ranked matches only
  GET  /nlp/eval              — golden-set evaluation scores
"""

import asyncio
import io
import json
import math
import os
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import httpx

from english_cards import ENGLISH_CARDS

# ── Config ──────────────────────────────────────────────────────────
HLLSET_NEXT_URL = os.environ.get("HLLSET_NEXT_URL", "http://hllset-next:9090")
IPFS_API_URL = os.environ.get("IPFS_API_URL", "http://ipfs:5001")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
KB_SNAPSHOT_FILE = os.environ.get("KB_SNAPSHOT_FILE", "/app/data/nlp_snapshot.json")
# Full self-contained copy of the ingested corpus (cards), used to restore at
# startup even when the IPFS node is empty or unreachable.
KB_LOCAL_BACKUP_FILE = os.environ.get("KB_LOCAL_BACKUP_FILE", "/app/data/nlp_snapshot_cards.json")

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("nlp-model")

app = FastAPI(
    title="NanoLM English NLP Model",
    description="Lattice-algebra English language model (understand + respond), CPU-only",
    version="0.1.0",
)

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return _client


# ═══════════════════════════════════════════════════════════════════════
# Tokenizer / normalization (spec §6.1, G5, G6)
# ═══════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "have", "has", "had", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "into", "about", "over", "under",
    "and", "or", "but", "if", "then", "than", "so", "that", "this", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they", "them",
    "me", "him", "her", "us", "my", "your", "our", "their", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "tell", "explain",
    "define", "describe", "please", "me", "give", "know", "want", "would",
    "ask", "question", "detail", "details", "elaborate", "more", "info", "further",
    "many", "much", "could", "not", "no", "never", "without", "nor", "just", "really", "very",
    "there", "here",
    "something", "anything", "everything", "nothing",
    "s",
}

SYNONYMS = {
    # greeting canonicalisation
    "hi": ["hello"], "hey": ["hello"], "howdy": ["hello"], "greetings": ["hello"],
    # question canonicalisation
    "whats": ["what"], "whos": ["who"], "what's": ["what"], "who's": ["who"],
    # fact paraphrases
    "capital": ["capital"], "capitalcity": ["capital"],
    "creator": ["creator", "created"], "created": ["creator", "created"],
    "create": ["creator", "created"], "invented": ["invent"],
    "inventor": ["invent"], "founded": ["creator", "created"],
    "made": ["creator", "created"], "wrote": ["creator", "created"],
    "formula": ["formula"],
    "discover": ["discover", "discovered"], "found": ["discover", "discovered"],
    # definition / paraphrase canonicalisation
    "meaning": ["meaning", "definition"], "definition": ["meaning", "definition"],
    "define": ["meaning", "definition"],
    # superlatives
    "largest": ["largest", "biggest"], "biggest": ["largest", "biggest"],
    "smallest": ["smallest", "tiniest"], "tiniest": ["smallest", "tiniest"],
    # physics
    "speed": ["speed", "velocity"], "velocity": ["speed", "velocity"],
}

# Country/entity abbreviations expanded to their full names (phrase-level, so
# "usa" becomes "united states" without sprouting spurious "unit"/"state"
# tokens that would also match unrelated cards).
ABBREVIATIONS = {
    "usa": "united states",
    "uk": "united kingdom",
    "uae": "united arab emirates",
    "britain": "united kingdom",
    "tv": "television",
    "pc": "computer",
    "web": "internet",
}

ANTONYMS = {
    "normal": "abnormal", "abnormal": "normal",
    "elevated": "depressed", "depressed": "elevated",
    "high": "low", "low": "high",
    "fast": "slow", "slow": "fast",
    "large": "small", "small": "large",
}

NEGATION_WORDS = {"not", "no", "never", "without", "nor"}

def _words(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _stem(word: str) -> str:
    """Martin Porter's stemming algorithm (1980) — deterministic, dependency-free.

    Replaces naive suffix stripping (which broke words like "speed" → "spe").
    """
    w = word.lower()
    if len(w) <= 2:
        return w

    def is_vowel(i: int) -> bool:
        return w[i] in "aeiou" or (w[i] == "y" and i > 0 and w[i - 1] not in "aeiou")

    def has_vowel(start: int, end: int) -> bool:
        return any(is_vowel(i) for i in range(start, end))

    def measure(start: int = 0, end: int | None = None) -> int:
        if end is None:
            end = len(w)
        m, i = 0, start
        while i < end:
            while i < end and not is_vowel(i):
                i += 1
            if i >= end:
                break
            while i < end and is_vowel(i):
                i += 1
            m += 1
        return m

    def cvc(i: int) -> bool:
        if i < 2:
            return False
        if not (is_vowel(i - 2) is False and is_vowel(i - 1) and is_vowel(i) is False):
            return False
        return w[i] not in "wxy"

    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b
    flag = False
    if w.endswith("eed"):
        if measure(0, len(w) - 3) > 0:
            w = w[:-1]
    elif w.endswith("ed"):
        if has_vowel(0, len(w) - 2):
            w = w[:-2]
            flag = True
    elif w.endswith("ing"):
        if has_vowel(0, len(w) - 3):
            w = w[:-3]
            flag = True
    if flag:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif w.endswith(("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")):
            w = w[:-1]
        elif measure() == 1 and cvc(len(w) - 1):
            w += "e"

    # Step 1c
    if w.endswith("y") and has_vowel(0, len(w) - 1):
        w = w[:-1] + "i"

    # Step 2
    for suf, rep in (
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ("logi", "log"),
    ):
        if w.endswith(suf):
            if measure(0, len(w) - len(suf)) > 0:
                w = w[: -len(suf)] + rep
            break

    # Step 3
    for suf, rep in (
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
        ("ical", "ic"), ("ful", ""), ("ness", ""),
    ):
        if w.endswith(suf):
            if measure(0, len(w) - len(suf)) > 0:
                w = w[: -len(suf)] + rep
            break

    # Step 4
    for suf in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
                "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"):
        if w.endswith(suf):
            if suf == "ion":
                stem = w[:-3]
                if len(stem) >= 1 and stem[-1] in "st" and measure(0, len(stem)) > 1:
                    w = stem
            elif measure(0, len(w) - len(suf)) > 1:
                w = w[: -len(suf)]
            break

    # Step 5a
    if w.endswith("e"):
        m = measure(0, len(w) - 1)
        if m > 1:
            w = w[:-1]
        elif m == 1 and not cvc(len(w) - 1):
            w = w[:-1]

    # Step 5b
    if measure() > 1 and w.endswith("ll"):
        w = w[:-1]

    return w


def _raw_variants(word: str) -> list:
    """Raw word → sorted set of raw synonyms (including itself)."""
    return sorted({word, *SYNONYMS.get(word, [])})


def _is_noise_token(w: str) -> bool:
    """True for numeric-only or digit-leading tokens (trivia noise: years,
    counts, ordinals like '1961', '10kg', '1st'). Letter-leading alphanumerics
    ('h2o', 'co2', 'c3po') are meaningful and kept."""
    if not w:
        return True
    return w.isdigit() or w[0].isdigit()


def _norm_words(text: str) -> list:
    """Normalized word sequence: stopword/noise removal → synonym expansion → stem."""
    out: list = []
    for w in _words(text):
        if w in STOPWORDS or _is_noise_token(w):
            continue
        for v in _raw_variants(w):
            out.append(_stem(v))
    return out


def _norm_tokens(text: str) -> set:
    """Normalized scoring tokens: stemmed unigrams + bigrams (spec §6.1)."""
    ws = _norm_words(text)
    toks: set = set(ws)
    for i in range(len(ws) - 1):
        toks.add(ws[i] + "\x00" + ws[i + 1])
    return toks


def _keywords(text: str) -> list:
    """Extract deterministic content keywords (deduplicated, order-preserved)."""
    seen: set = set()
    out: list = []
    for w in _norm_words(text):
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _raw_keywords(text: str, limit: int = 8) -> list:
    """Extract raw content words (deduped, order-preserved) WITHOUT stemming
    or synonym expansion — the canonical form for stored card keywords.
    Normalization (stem + synonyms) is applied once at index build time."""
    seen: set = set()
    out: list = []
    for w in _words(text):
        if w in STOPWORDS or _is_noise_token(w) or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def _negated_tokens(text: str) -> set:
    """G6 — collect stemmed tokens inside a negation scope.

    Only word-level negation (not/no/never/without/nor) is applied. The old
    un-/non-/dis- PREFIX rule was removed: it falsely negated words where the
    prefix isn't negative — "united", "unit", "universe", "university",
    "discover", "distance", "display" — which suppressed legitimate cards.
    """
    neg: set = set()
    words = _words(text)
    i = 0
    while i < len(words):
        w = words[i]
        if w in NEGATION_WORDS:
            for j in range(i + 1, min(i + 4, len(words))):
                if words[j] in {",", ".", ";", "but", "and"}:
                    break
                neg.add(_stem(words[j]))
        i += 1
    return neg


# ═══════════════════════════════════════════════════════════════════════
# Conversation context management (spec §G7/G8)
# ═══════════════════════════════════════════════════════════════════════

PRONOUNS = {"it", "its", "they", "them", "this", "that", "these", "those", "there", "here"}
FOLLOWUP_FIRST_WORDS = {"and", "also", "then", "now", "so"}


def _resolve_context(message: str, session: Optional[dict]) -> list:
    """Resolve the conversation topic for a message using session context.

    Returns the resolved context keywords (stemmed unigrams) to use for
    retrieval and to persist for the next turn.

    Rules:
      • referential pronoun, follow-up lead, or empty content → carry the
        previous topic forward and merge it with new content words;
      • every new content word already present in context → same-topic
        follow-up (e.g. "Capital of France?" after "What is France?");
      • otherwise → a fresh topic, so reset the context to the new words.
    """
    kws = _keywords(message)
    if not session:
        return kws

    ctx = list(session.get("context") or session.get("keywords") or [])
    if not ctx:
        return kws

    raw = _words(message)
    has_pronoun = any(p in raw for p in PRONOUNS)
    first_word = raw[0] if raw else ""
    is_followup = (
        has_pronoun
        or first_word in FOLLOWUP_FIRST_WORDS
        or not kws
        or all(k in ctx for k in kws)
    )

    if is_followup:
        return list(dict.fromkeys(kws + ctx))
    return kws


ELABORATION_PHRASES = {
    "more", "tell me more", "tell me more about it", "more details", "details",
    "elaborate", "please elaborate", "explain more", "explain further",
    "go on", "continue", "further", "more info", "more information", "expand",
}

ELABORATION_WORDS = {
    "more", "detail", "details", "elaborate", "expand", "explain",
    "continue", "further", "info", "information", "go",
}


def _is_elaboration_request(message: str) -> bool:
    """True when the message asks for more on the current topic rather than a
    new factual question (e.g. 'tell me more', 'elaborate', 'go on')."""
    m = (message or "").strip().lower()
    if not m:
        return False
    if m in ELABORATION_PHRASES:
        return True
    words = _words(m)
    if not words:
        return False
    # No real content words (everything is a stopword or elaboration cue),
    # and at least one elaboration cue present.
    content = [w for w in words if w not in STOPWORDS and w not in ELABORATION_WORDS]
    return not content and any(w in ELABORATION_WORDS for w in words)


# ═══════════════════════════════════════════════════════════════════════
# Intent classifier (spec §6.3)
# ═══════════════════════════════════════════════════════════════════════

GREETINGS = ("hello", "hi", "hey", "good morning", "good afternoon", "good evening")
SMALLTALK = ("how are you", "thank", "thanks", "bye", "goodbye", "see you")


def classify_intent(message: str) -> str:
    m = message.lower().strip()
    if m.startswith(GREETINGS) or m in ("hi", "hello", "hey"):
        return "greeting"
    words = _words(m)
    # Word-boundary matching for single command/courtesy words, so content
    # words like "helper", "helpful", "phelps", "thanksgiving", "thankful"
    # are NOT hijacked into the help/smalltalk templates.
    if ("help" in words or "commands" in words
            or "what can you do" in m or "who are you" in m):
        return "help"
    if ("how are you" in m or "see you" in m
            or any(w in ("thank", "thanks", "bye", "goodbye") for w in words)):
        return "smalltalk"
    if "?" in m or any(w in _words(m) for w in ("what", "who", "when", "where", "why", "how", "which")):
        return "factual"
    return "fallback"


# ═══════════════════════════════════════════════════════════════════════
# Knowledge base (spec §6.2, §6.4, §6.7)
# ═══════════════════════════════════════════════════════════════════════

class NlpKnowledgeBase:
    """Ingest English cards into hllset-next (c:nlp:*) and answer queries by
    keyword-index lookup + exact token-overlap ranking (deterministic)."""

    def __init__(self, base_url: str, cards: list):
        self.base_url = base_url.rstrip("/")
        self.cards = cards
        self.registry: dict[str, dict] = {}       # key → card metadata
        self.card_tokens: dict[str, set] = {}     # key → normalized token set (incl. bigrams)
        self.card_unigrams: dict[str, set] = {}   # key → stemmed unigram set (negation)
        self.kw_index: dict[str, set] = {}        # keyword → set(card keys)
        self.idf: dict[str, float] = {}           # token → inverse document frequency
        self.card_norm: dict[str, float] = {}     # key → L2 norm of idf-weighted vector
        self.ingested = False

    def _key(self, card_id: str) -> str:
        return f"c:nlp:{card_id}"

    async def ingest(self) -> dict:
        """Ingest all cards into the lattice (idempotent) + rebuild local indexes."""
        ingested, errors = await self._post_cards_to_lattice(self.cards)
        self._rebuild_indexes()
        return {"cards": len(self.cards), "ingested": ingested, "errors": errors}

    def _rebuild_indexes(self) -> None:
        """Rebuild local keyword/token indexes from self.cards (in-memory).

        Duplicate card ids are dropped so the registry stays 1:1 with cards.
        """
        seen: set = set()
        deduped = []
        for card in self.cards:
            if card["id"] in seen:
                continue
            seen.add(card["id"])
            deduped.append(card)
        self.cards = deduped

        self.registry = {}
        self.card_tokens = {}
        self.card_unigrams = {}
        self.kw_index = {}
        for card in self.cards:
            key = self._key(card["id"])
            kw_text = " ".join(card.get("keywords", []))
            self.registry[key] = card
            # Index keywords through the same normalization as queries
            # (synonym expansion + Porter stemming) so synonyms match both ways
            # ("speed" card ↔ "velocity" query). Keywords are stored raw.
            norm_words = _norm_words(kw_text)
            self.card_unigrams[key] = set(norm_words)
            toks = set(norm_words)
            for i in range(len(norm_words) - 1):
                toks.add(norm_words[i] + "\x00" + norm_words[i + 1])
            self.card_tokens[key] = toks
            for s in norm_words:
                self.kw_index.setdefault(s, set()).add(key)
        self._compute_idf()
        self.ingested = len(self.registry) == len(self.cards)

    def _compute_idf(self) -> None:
        """Precompute token IDF weights + per-card L2 norms (TF-IDF-style)."""
        n = max(1, len(self.cards))
        df: dict[str, int] = {}
        for toks in self.card_tokens.values():
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1.0 + n / (1.0 + c)) + 1.0 for t, c in df.items()}
        self.all_tokens = set(df.keys())
        self.card_norm = {}
        for key, toks in self.card_tokens.items():
            self.card_norm[key] = math.sqrt(
                sum(self.idf.get(t, 1.0) ** 2 for t in toks)
            ) or 1.0

    async def _post_cards_to_lattice(self, cards: list, concurrency: int = 40) -> tuple:
        """POST the given cards to hllset-next with bounded concurrency.

        If hllset-next is unreachable the POSTs are skipped entirely — the
        local keyword index still powers retrieval and the IPFS snapshot
        provides durability, so the app runs in degraded (but fully
        functional) mode without the lattice.
        """
        client = await get_client()
        try:
            h = await client.get(f"{self.base_url}/api/v1/health")
            h.raise_for_status()
        except Exception as e:
            return 0, [f"hllset-next unreachable: {e}"]
        sem = asyncio.Semaphore(concurrency)

        async def one(card: dict):
            key = self._key(card["id"])
            text = " ".join(card.get("keywords", []))
            async with sem:
                try:
                    r = await client.post(
                        f"{self.base_url}/api/v1/hllset/ingest",
                        json={"key": key, "text": text},
                    )
                    r.raise_for_status()
                    return None
                except Exception as e:
                    return f"{card['id']}: {e}"

        errs = await asyncio.gather(*(one(c) for c in cards))
        errors = [e for e in errs if e]
        return len(cards) - len(errors), errors

    async def add_cards(self, new_cards: list) -> dict:
        """Append new cards, ingest them into the lattice, rebuild indexes.

        Existing card ids are skipped. Returns added/total/ingested/errors.
        """
        existing = {c["id"] for c in self.cards}
        fresh = []
        for c in new_cards:
            cid = c.get("id")
            if cid in existing:
                continue
            existing.add(cid)
            fresh.append(c)
        if not fresh:
            return {"added": 0, "total": len(self.cards), "ingested": 0, "errors": []}
        self.cards = self.cards + fresh
        self._rebuild_indexes()
        ingested, errors = await self._post_cards_to_lattice(fresh)
        return {"added": len(fresh), "total": len(self.cards),
                "ingested": ingested, "errors": errors}

    def _candidates(self, keywords: list) -> list:
        """Gather card keys related to any extracted keyword (spec §6.7)."""
        if not keywords:
            return list(self.registry.keys())
        keys: set = set()
        for kw in keywords:
            keys.update(self.kw_index.get(kw, set()))
        return list(keys) if keys else list(self.registry.keys())

    def query(self, message: str, top_k: int = 8, negated: Optional[set] = None,
              context_keywords: Optional[list] = None) -> dict:
        """Rank cards by IDF-weighted cosine similarity over token sets.

        Rare keywords (e.g. "qubit", "france") weigh far more than common
        ones (e.g. "capital", "what"), so varied phrasings around a specific
        keyword resolve to the right card. When `context_keywords` (resolved
        conversation topic) is supplied, its stemmed unigrams are folded into
        the candidate set and scoring tokens.
        """
        negated = negated or set()
        ctx = list(context_keywords or [])
        combined = list(dict.fromkeys(_keywords(message) + ctx))
        q = _norm_tokens(message) | set(ctx)
        if not q:
            return {"matches": [], "suppressed": [], "top_k": top_k}
        q_norm = math.sqrt(sum(self.idf.get(t, 1.0) ** 2 for t in q)) or 1.0

        # Anchor on the rarest content word the USER actually typed (stemmed),
        # not on synonym-expanded tokens. Cards missing it are discounted, so a
        # specific keyword (e.g. "france", "speed") outweighs generic phrases,
        # and injected synonyms ("velocity" → "veloc") can't hijack the anchor.
        raw_stems = []
        for w in _words(message):
            if w in STOPWORDS:
                continue
            s = _stem(w)
            if s and s not in raw_stems:
                raw_stems.append(s)
        meaningful = [t for t in raw_stems if t in self.all_tokens]
        ctx_meaningful = [t for t in ctx if t in self.all_tokens]
        # A pronoun ("it", "there", ...) refers back to the conversation topic,
        # so the topic's own keywords — not a generic verb the user tacked on
        # (e.g. "invent") — must anchor the match.
        raw = _words(message)
        has_pronoun = any(p in raw for p in PRONOUNS)
        if has_pronoun and ctx_meaningful:
            anchor = max(ctx_meaningful, key=lambda t: self.idf.get(t, 1.0))
        elif meaningful:
            anchor = max(meaningful, key=lambda t: self.idf.get(t, 1.0))
        elif ctx_meaningful:
            anchor = max(ctx_meaningful, key=lambda t: self.idf.get(t, 1.0))
        else:
            anchor = None

        scored = []
        suppressed = []
        for key in self._candidates(combined):
            card_tokens = self.card_tokens.get(key)
            if not card_tokens:
                continue
            overlap = q & card_tokens
            if not overlap:
                continue
            card_unigrams = self.card_unigrams.get(key, set())
            neg_hit = card_unigrams & negated
            # G6 — suppress cards whose required unigram tokens are mostly negated
            if neg_hit and len(neg_hit) >= max(1, len(card_unigrams) * 0.5):
                suppressed.append(key)
                continue
            dot = sum(self.idf.get(t, 1.0) ** 2 for t in overlap)
            score = dot / (q_norm * self.card_norm.get(key, 1.0))
            if anchor is not None and anchor not in card_tokens:
                score *= 0.3
            scored.append({"key": key, "bss": score, "card": self.registry[key],
                           "_ov": len(overlap)})
        scored.sort(key=lambda m: (m["bss"], m["_ov"]), reverse=True)
        return {"matches": scored[:top_k], "suppressed": suppressed, "top_k": top_k}

    def status(self) -> dict:
        return {
            "base_url": self.base_url,
            "cards": len(self.cards),
            "corpus_count": len(self.cards),
            "ingested": len(self.registry),
            "keywords_indexed": len(self.kw_index),
            "ready": self.ingested,
        }


# ═══════════════════════════════════════════════════════════════════════
# Query analyzer — typo correction, keyword/question identification,
# and clarifying questions on ambiguity.
# ═══════════════════════════════════════════════════════════════════════

QUESTION_WORDS = {"what", "who", "whom", "whose", "when", "where", "why", "how", "which"}
FILLERS = {"huh", "um", "umm", "hmm", "uh", "er", "ok", "okay", "yeah", "yep",
           "yes", "nope", "yo", "wow", "eh", "hm", "aha"}


def _damerau_levenshtein(a: str, b: str) -> int:
    """Optimal-string-alignment edit distance (counts adjacent transposition
    as one edit) — the common typo of swapping letters is handled correctly."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 3:
        return 99
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[m][n]


def _question_type(message: str) -> str:
    """Classify the question type: what/who/where/when/why/how/which/yesno/none."""
    words = _words(message)
    if not words:
        return "none"
    first = words[0]
    if first in QUESTION_WORDS:
        return "who" if first in ("who", "whom", "whose") else first
    if first in ("is", "are", "was", "were", "do", "does", "did", "can", "could",
                 "will", "would", "should", "has", "have", "am"):
        return "yesno"
    for w in words:
        if w in QUESTION_WORDS:
            return "who" if w in ("who", "whom", "whose") else w
    return "none"


class QueryAnalyzer:
    """Understands a query: fixes typos against the corpus vocabulary, extracts
    the question type and keyword focus, and asks a clarifying question when the
    query is ambiguous or too vague."""

    def __init__(self, kb: NlpKnowledgeBase):
        self.kb = kb
        self._build_vocab()

    def _build_vocab(self) -> None:
        freq: dict = {}
        for card in self.kb.cards:
            for w in _words(card.get("question", "") or ""):
                freq[w] = freq.get(w, 0) + 1
            for w in _words(card.get("answer", "") or ""):
                freq[w] = freq.get(w, 0) + 1
            for k in card.get("keywords", []):
                freq[str(k)] = freq.get(str(k), 0) + 1
        for w in STOPWORDS:
            freq.setdefault(w, 1)
        self.freq = freq
        self.vocab = set(freq)
        self.vocab_by_len: dict = {}
        for w in self.vocab:
            self.vocab_by_len.setdefault(len(w), set()).add(w)

    def _correct(self, word: str) -> Optional[list]:
        """Return candidate corrections (best edit distance ≤ 2), or None.

        Ties at the best distance are broken by corpus frequency: if the most
        frequent candidate clearly dominates, it's returned alone (auto-correct);
        otherwise all tied candidates are returned (ambiguous → ask the user).
        """
        if len(word) <= 2 or word in self.vocab:
            return None
        candidates: list = []
        for L in range(len(word) - 2, len(word) + 3):
            for v in self.vocab_by_len.get(L, ()):
                d = _damerau_levenshtein(word, v)
                if d <= 2:
                    candidates.append((d, -self.freq.get(v, 0), v))
        if not candidates:
            return None
        candidates.sort()
        best_d = candidates[0][0]
        if best_d > 2:
            return None
        best = [v for d, _, v in candidates if d == best_d]
        if len(best) > 1:
            top = self.freq.get(best[0], 0)
            second = self.freq.get(best[1], 0)
            if top >= 2 * second:
                return [best[0]]
        return sorted(best)

    def _focus(self, words: list) -> Optional[str]:
        """The rarest content word (highest idf) — the query's keyword focus."""
        all_tokens = getattr(self.kb, "all_tokens", set())
        idf = getattr(self.kb, "idf", {})
        best, best_idf = None, 0.0
        for w in words:
            s = _stem(w)
            if s in all_tokens:
                v = idf.get(s, 0.0)
                if v > best_idf:
                    best, best_idf = s, v
        return best

    def analyze(self, message: str, context: Optional[list] = None) -> dict:
        raw = _words(message)
        corrected_words: list = []
        corrections: list = []
        ambiguous: list = []

        for w in raw:
            if w in STOPWORDS or w in self.vocab:
                corrected_words.append(w)
                continue
            cands = self._correct(w)
            if cands is None:
                corrected_words.append(w)
            elif len(cands) == 1:
                corrections.append({"word": w, "corrected": cands[0]})
                corrected_words.append(cands[0])
            else:
                ambiguous.append({"word": w, "candidates": cands})
                corrected_words.append(w)

        corrected = " ".join(corrected_words)
        # expand country abbreviations to full names so they match the corpus
        for abbr, full in ABBREVIATIONS.items():
            corrected = re.sub(rf"\b{re.escape(abbr)}\b", full, corrected,
                               flags=re.IGNORECASE)
        content = [w for w in corrected_words if w not in STOPWORDS]
        qtype = _question_type(corrected)
        focus = self._focus(content) if content else None

        clarification = None
        if ambiguous:
            entry = ambiguous[0]
            opts = " or ".join(f'"{c}"' for c in entry["candidates"][:3])
            clarification = (
                f'I want to be sure I understood correctly — did you mean {opts}?'
            )
        elif (content and all(w in FILLERS for w in content)) or (not content and not context):
            clarification = (
                "I'm not quite sure what you'd like to know. "
                "Could you add a keyword or rephrase your question?"
            )

        return {
            "original": message,
            "corrected": corrected,
            "corrections": corrections,
            "ambiguous": ambiguous,
            "keywords": list(dict.fromkeys(content)),
            "question_type": qtype,
            "focus": focus,
            "intent": classify_intent(corrected),
            "clarification": clarification,
        }


_analyzer: QueryAnalyzer | None = None


def get_analyzer(kb: NlpKnowledgeBase) -> QueryAnalyzer:
    global _analyzer
    if _analyzer is None or _analyzer.kb is not kb:
        _analyzer = QueryAnalyzer(kb)
    return _analyzer


_kb: NlpKnowledgeBase | None = None
_last_snapshot_cid: str = ""
SESSIONS_FILE = os.environ.get("SESSIONS_FILE", "/app/data/sessions.json")
_sessions: dict[str, dict] = {}


def _load_sessions_file() -> None:
    global _sessions
    try:
        with open(SESSIONS_FILE) as f:
            _sessions = json.load(f)
    except Exception:
        _sessions = {}


def _save_sessions_file() -> None:
    try:
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        with open(SESSIONS_FILE, "w") as f:
            json.dump(_sessions, f, default=str)
    except Exception as e:
        logger.warning(f"could not save sessions file: {e}")


def _get_or_create_session(session_id: str) -> dict:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    sid = session_id or uuid.uuid4().hex
    if sid not in _sessions:
        _sessions[sid] = {
            "id": sid,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "title": "",
            "keywords": [],
            "context": [],
            "history": [],
        }
        _save_sessions_file()
    return _sessions[sid]


_load_sessions_file()


def get_kb() -> NlpKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = NlpKnowledgeBase(HLLSET_NEXT_URL, ENGLISH_CARDS)
    return _kb


# ═══════════════════════════════════════════════════════════════════════
# IPFS helpers (spec §6.8)
# ═══════════════════════════════════════════════════════════════════════

async def _store_bytes_to_ipfs(data: bytes, filename: str) -> Optional[str]:
    client = await get_client()
    try:
        files = {"file": (filename, data, "application/octet-stream")}
        r = await client.post(f"{IPFS_API_URL}/api/v0/add", files=files)
        if r.status_code == 200:
            return r.json().get("Hash")
    except Exception as e:
        logger.warning(f"IPFS store failed: {e}")
    return None


async def _fetch_bytes_from_ipfs(cid: str) -> Optional[bytes]:
    client = await get_client()
    try:
        r = await client.post(f"{IPFS_API_URL}/api/v0/cat", params={"arg": cid})
        if r.status_code == 200:
            return r.content
    except Exception as e:
        logger.warning(f"IPFS fetch failed: {e}")
    return None


async def _store_to_ipfs(payload: dict) -> Optional[str]:
    return await _store_bytes_to_ipfs(
        json.dumps(payload, default=str).encode("utf-8"), "nlp_exchange.json"
    )


def _save_snapshot_file(cid: str) -> None:
    try:
        os.makedirs(os.path.dirname(KB_SNAPSHOT_FILE), exist_ok=True)
        with open(KB_SNAPSHOT_FILE, "w") as f:
            json.dump({"cid": cid, "count": len(get_kb().cards),
                       "saved_at": datetime.utcnow().isoformat()}, f)
    except Exception as e:
        logger.warning(f"could not save snapshot file: {e}")


def _load_snapshot_file() -> str:
    try:
        with open(KB_SNAPSHOT_FILE) as f:
            return json.load(f).get("cid", "")
    except Exception:
        return ""


def _save_local_backup(payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(KB_LOCAL_BACKUP_FILE), exist_ok=True)
        with open(KB_LOCAL_BACKUP_FILE, "w") as f:
            json.dump(payload, f, default=str)
    except Exception as e:
        logger.warning(f"could not save local backup: {e}")


def _load_local_backup_cards() -> list:
    try:
        with open(KB_LOCAL_BACKUP_FILE) as f:
            return json.load(f).get("cards", [])
    except Exception:
        return []


async def _snapshot_kb(kb: NlpKnowledgeBase) -> str:
    global _last_snapshot_cid
    payload = {
        "type": "nlp-knowledge-snapshot",
        "engine": "nanolm",
        "count": len(kb.cards),
        "cards": kb.cards,
        "created_at": datetime.utcnow().isoformat(),
    }
    # Always write a self-contained local backup of the ingested corpus so
    # startup can restore it even if the IPFS node is empty or unreachable.
    _save_local_backup(payload)
    cid = await _store_to_ipfs(payload)
    if cid:
        _last_snapshot_cid = cid
        _save_snapshot_file(cid)
        logger.info(f"nlp knowledge snapshot stored at IPFS CID {cid}")
    return cid or ""


# ═══════════════════════════════════════════════════════════════════════
# Response generator (spec §6.5)
# ═══════════════════════════════════════════════════════════════════════

TEMPLATES = {
    "greeting": "Hello! I'm NanoLM, a lattice-based English model. Ask me a question and I'll find the most relevant answer.",
    "help": "I can answer factual questions by retrieving from my ingested knowledge corpus. Try 'What is the capital of France?' or 'What is quantum computing?'.",
    "smalltalk": "I'm doing well, thank you! What would you like to know?",
    "fallback": "I don't have a confident answer for that in my corpus. Try rephrasing, or ingest a related document.",
}


def _format_answer(card: dict) -> str:
    """Render a card's answer self-contained: prepend the question when the
    answer is a bare multiple-choice value (trivia cards store only the correct
    choice, e.g. "True"), so replies don't show meaningless fragments."""
    q = (card.get("question") or "").strip()
    a = (card.get("answer") or "").strip()
    if not a:
        return q or ""
    bare = "not:" in a or (
        len(a) <= 25 and str(card.get("domain", "")).startswith("trivia.")
    )
    return f"{q} — {a}" if (q and bare) else a


META_MARKERS = (
    "what are we talking", "what are you talking", "what is this about",
    "what's this about", "what was this about", "what are we discussing",
    "what is the topic", "what is the current topic", "current topic",
    "what did i ask", "what did you ask", "what was the question",
    "this is all about", "are we talking about", "is this about",
)


def _is_meta_question(message: str) -> bool:
    """True when the user is asking ABOUT the conversation (its topic), rather
    than asking a factual question — e.g. "what are we talking about?" or
    "this is all about paris or france?"."""
    m = message.lower().strip()
    return any(k in m for k in META_MARKERS)


def _meta_reply(session: dict) -> str:
    """Answer a meta-question with the conversation's topic, using the last
    information actually provided (not a canned "we're discussing…" template)."""
    history = session.get("history") or []
    # Skip our own meta/clarification replies so the summary doesn't nest.
    last_bot = next((m["content"] for m in reversed(history)
                     if m.get("role") == "assistant"
                     and m.get("strategy") not in ("Meta", "Clarification")), "").strip()
    if last_bot:
        return f"The current topic is: {last_bot}"
    title = (session.get("title") or "").strip()
    if title:
        return f"We're discussing \"{title}\". What would you like to know?"
    return "We haven't started a specific topic yet — what would you like to ask?"


def _single_keyword_hit(message: str) -> bool:
    """True when the message is a single content keyword.

    A lone keyword the user typed is a strong lexical signal, but its cosine
    is artificially compressed below the usual 0.30 cutoff because the target
    card's OTHER keywords + bigrams inflate its norm. A user who types a bare
    keyword expects SOMETHING about it rather than a refusal, so for any single
    keyword we answer the top match above the retrieval threshold (0.15) rather
    than falling back.
    """
    words = [w for w in _words(message)
             if w not in STOPWORDS and not _is_noise_token(w)]
    return len(words) == 1


EXPLORATION_MARKERS = (
    "tell me about", "let me know about", "know about", "what do you know about",
    "everything about", "all about", "talk about", "more about",
    "information about", "info about", "list about",
)


def _is_exploration_request(message: str) -> bool:
    """True for open-ended "tell me about X" / "let me know about X" queries."""
    return any(m in (message or "").lower() for m in EXPLORATION_MARKERS)


UNION_PAGE_SIZE = 15  # matches per union reply before prompting "more"
MORE_MARKERS = ("more", "next", "continue", "show more")


def _is_more_request(message: str) -> bool:
    """True when the user asks to continue a paginated union reply."""
    return (message or "").strip().lower() in MORE_MARKERS


def _keyword_union_reply(kb: "NlpKnowledgeBase", message: str, intent: str,
                         threshold: float, offset: int = 0,
                         page_size: int = UNION_PAGE_SIZE) -> Optional[dict]:
    """Build a paginated UNION reply over the cards containing the query's
    focus keyword (ranked by cosine), instead of one top hit.

    Applies to (a) lone-keyword queries and (b) exploration requests
    ("tell me about X" / "let me know about X"), where the focus keyword is the
    rarest content word. Returns None otherwise, or when ≤1 card matches.
    """
    words = [w for w in _words(message)
             if w not in STOPWORDS and not _is_noise_token(w)]
    if not words:
        return None
    if len(words) == 1:
        kw = words[0]
    elif _is_exploration_request(message):
        kw = max(words, key=lambda w: kb.idf.get(_stem(w), 0.0))
    else:
        return None

    stem = _stem(kw)
    keys = kb.kw_index.get(stem, set())
    if len(keys) <= 1:
        return None

    result = kb.query(kw, top_k=len(keys), negated=set())
    matches = [m for m in result["matches"] if m.get("card")]
    total = len(matches)
    if total <= 1:
        return None

    page = matches[offset:offset + page_size]
    shown = offset + len(page)
    has_more = shown < total

    if offset == 0:
        lines = [f'About "{kw}" — {total} matches in the corpus:', ""]
    else:
        lines = [f'More about "{kw}" ({offset + 1}–{shown} of {total}):', ""]
    for m in page:
        lines.append("• " + _format_answer(m["card"]))
    if has_more:
        lines.append("")
        lines.append(
            f'Want to know more? Reply "more" to see the next '
            f"{min(page_size, total - shown)}."
        )

    return {
        "reply": "\n".join(lines),
        "confidence": round(page[0]["bss"], 4),
        "strategy": "Union",
        "intent": intent,
        "matched_card": page[0]["key"],
        "total_matches": total,
        "has_more": has_more,
        "pagination": {
            "keyword": kw,
            "offset": shown,
            "page_size": page_size,
            "total": total,
            "has_more": has_more,
        },
    }


def _build_reply(message: str, matches: list, intent: str, threshold: float,
                 keyword_hit: bool = False) -> dict:
    """Select response strategy from ranked matches (spec §6.5)."""
    if intent in TEMPLATES and intent != "factual" and intent != "fallback":
        # greeting / help / smalltalk handled by templates first
        pass

    if intent in ("greeting", "help", "smalltalk"):
        return {
            "reply": TEMPLATES[intent],
            "confidence": 1.0,
            "strategy": "template",
            "intent": intent,
            "matched_card": None,
        }

    valid = [m for m in matches if m.get("card") and m.get("bss", 0.0) >= threshold]
    best = valid[0] if valid else None

    if best is None:
        # Single uniquely-matching candidate below threshold → still answer it
        # (a rare keyword that identifies exactly one card).
        singles = [m for m in matches if m.get("card")]
        if len(singles) == 1 and singles[0].get("bss", 0.0) > 0:
            m = singles[0]
            return {
                "reply": _format_answer(m["card"]),
                "confidence": round(m["bss"], 4),
                "strategy": "SingleHit",
                "intent": intent,
                "matched_card": m["key"],
            }
        return {
            "reply": TEMPLATES["fallback"],
            "confidence": 0.0,
            "strategy": "Fallback",
            "intent": intent,
            "matched_card": None,
        }

    # Response analyzer: only answer when reasonably confident. Below 0.30 the
    # match is too ambiguous to guess — fall back instead of returning wrong info.
    # EXCEPTION: a single keyword (keyword_hit) is a strong lexical signal whose
    # cosine is compressed by the card's other keywords/bigrams, so we answer the
    # top match (already above the retrieval threshold) instead of falling back.
    bss = best["bss"]
    if bss >= 0.30:
        strategy = "DirectMatch" if bss >= 0.55 else "HighMatch"
        reply = _format_answer(best["card"])
    elif keyword_hit:
        strategy = "KeywordHit"
        reply = _format_answer(best["card"])
    else:
        strategy = "Fallback"
        reply = TEMPLATES["fallback"]

    return {
        "reply": reply,
        "confidence": round(bss, 4),
        "strategy": strategy,
        "intent": intent,
        "matched_card": best["key"] if strategy != "Fallback" else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# Startup restore (spec §6.6)
# ═══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def restore_knowledge_base():
    global _last_snapshot_cid
    kb = get_kb()
    builtin_ids = {c["id"] for c in kb.cards}

    cid = _load_snapshot_file()
    if cid:
        _last_snapshot_cid = cid
    snapshot_cards = []
    restore_source = ""
    if cid:
        raw = await _fetch_bytes_from_ipfs(cid)
        if raw:
            try:
                snapshot_cards = json.loads(raw.decode("utf-8-sig")).get("cards", [])
                restore_source = f"snapshot {cid}"
            except Exception as e:
                logger.warning(f"snapshot parse failed: {e}")
    if not snapshot_cards:
        snapshot_cards = _load_local_backup_cards()
        if snapshot_cards:
            restore_source = "local backup"

    if snapshot_cards:
        if not kb.cards:
            kb.cards = snapshot_cards
            logger.info(f"restored {len(snapshot_cards)} cards from {restore_source}")
        else:
            # merge back every card that isn't in the built-in seed (bulk + docs)
            extras = [c for c in snapshot_cards if c.get("id") not in builtin_ids]
            if extras:
                kb.cards = kb.cards + extras
                logger.info(f"restored {len(extras)} cards from {restore_source}")

    if not kb.cards:
        logger.warning("no seed corpus and no snapshot — knowledge base empty")
        return

    if not snapshot_cards:
        # first boot — seed the corpus. If hllset-next is unavailable the
        # local index is still built and the corpus is persisted via the
        # IPFS snapshot, so the app runs in degraded mode without the lattice.
        result = await kb.ingest()
        await _snapshot_kb(kb)
        if result.get("ingested", 0) > 0:
            logger.info(f"nlp knowledge base seeded: {result['ingested']} cards")
        else:
            logger.warning(
                "hllset-next unavailable — using local index only "
                f"({len(result.get('errors', []))} errors)"
            )
        return
    else:
        # restart — the lattice is already persisted; just rebuild local indexes.
        # Refresh the self-contained local backup from the restored cards so it
        # is always current (and restorable) after every startup.
        _save_local_backup({
            "type": "nlp-knowledge-snapshot",
            "engine": "nanolm",
            "count": len(kb.cards),
            "cards": kb.cards,
            "created_at": datetime.utcnow().isoformat(),
        })
        kb._rebuild_indexes()
        logger.info(f"nlp knowledge base ready: {len(kb.cards)} cards (indexes rebuilt)")


# ═══════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    kb = get_kb()
    return {
        "status": "ok",
        "service": "nlp-model",
        "engine": "nanolm",
        "lattice": kb.status(),
        "last_snapshot_cid": _last_snapshot_cid,
        "ipfs_api_url": IPFS_API_URL,
        "disclaimer": "NanoLM English NLP model — retrieval-based, not a general LLM",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/nlp/ingest")
async def nlp_ingest():
    """(Re)ingest the English seed corpus into the c:nlp:* lattice."""
    kb = get_kb()
    result = await kb.ingest()
    if result.get("ingested"):
        await _snapshot_kb(kb)
    return result


@app.post("/nlp/ingest/cards")
async def nlp_ingest_cards(request: Request):
    """Bulk-ingest structured knowledge cards (JSON list) into the KB.

    Body: {"cards": [{"id", "question", "answer", "domain", "intent", "keywords"}, ...]}
    Missing keywords are derived server-side; missing ids are generated.
    """
    body = await request.json()
    raw_cards = body.get("cards") or []
    if not isinstance(raw_cards, list) or not raw_cards:
        raise HTTPException(status_code=400, detail="Missing 'cards' list")

    cards = []
    for i, c in enumerate(raw_cards):
        if not isinstance(c, dict):
            continue
        question = (c.get("question") or "").strip()
        answer = (c.get("answer") or "").strip()
        if not question and not answer:
            continue
        cid = str(c.get("id") or f"bulk.{uuid.uuid4().hex[:10]}.{i}")
        kws = c.get("keywords")
        if not kws:
            kws = _raw_keywords(f"{question} {answer}")[:8]
        elif isinstance(kws, str):
            kws = [kws]
        cards.append({
            "id": cid,
            "question": question[:200],
            "answer": answer,
            "domain": str(c.get("domain") or "general"),
            "intent": str(c.get("intent") or "factual"),
            "keywords": list(kws),
        })

    if not cards:
        raise HTTPException(status_code=400, detail="No valid cards provided")

    kb = get_kb()
    result = await kb.add_cards(cards)
    if result.get("added"):
        await _snapshot_kb(kb)
    result["snapshot_cid"] = _last_snapshot_cid
    return result


@app.get("/nlp/status")
async def nlp_status():
    return {**get_kb().status(), "last_snapshot_cid": _last_snapshot_cid}


@app.post("/nlp/reindex")
async def nlp_reindex():
    """Re-derive raw keywords for all bulk cards and rebuild the index.

    Fixes phantom keywords baked in by earlier ingest-time synonym expansion.
    Curated seed cards (fact./science./tech./gen./smalltalk./help.) are kept.
    """
    kb = get_kb()
    bulk_prefixes = ("trivia.", "wiki.", "doc.", "bulk.")
    for card in kb.cards:
        if card.get("id", "").startswith(bulk_prefixes):
            card["keywords"] = _raw_keywords(
                f"{card.get('question', '')} {card.get('answer', '')}")[:8]
    kb._rebuild_indexes()
    await _snapshot_kb(kb)
    return {"cards": len(kb.cards), "keywords_indexed": len(kb.kw_index),
            "snapshot_cid": _last_snapshot_cid}


@app.post("/session/new")
async def session_new():
    """Create a new chat session."""
    sid = uuid.uuid4().hex
    _sessions[sid] = {
        "id": sid,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "title": "",
        "keywords": [],
        "context": [],
        "history": [],
    }
    _save_sessions_file()
    return {"session_id": sid}


@app.get("/sessions")
async def list_sessions():
    """List all chat sessions (most recently updated first)."""
    items = sorted(_sessions.values(), key=lambda s: s.get("updated_at", ""), reverse=True)
    return {
        "sessions": [
            {
                "id": s["id"],
                "title": s.get("title", "") or "New chat",
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at"),
                "messages": len(s.get("history", [])),
            }
            for s in items
        ]
    }


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    _save_sessions_file()
    return {"deleted": session_id}


@app.post("/nlp/ingest/document")
async def nlp_ingest_document(
    file: UploadFile = File(...),
    keywords: str = Form(""),
):
    """Ingest a related source document via IPFS (spec §6.8, FR-11)."""
    raw = await file.read()
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        except Exception as e:
            return JSONResponse({"error": f"PDF extraction failed: {e}"}, status_code=422)
    elif name.endswith((".txt", ".md")) or not name:
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        return JSONResponse({"error": "Unsupported file type (use .txt, .md, .pdf)"}, status_code=415)

    if not text.strip():
        return JSONResponse({"error": "No text could be extracted from the document"}, status_code=422)

    doc_cid = await _store_bytes_to_ipfs(raw, file.filename or "document.txt")
    if not doc_cid:
        return JSONResponse({"error": "IPFS store failed"}, status_code=502)

    # Split into paragraphs → knowledge cards
    extra_kws = [k for k in keywords.replace(",", " ").split() if k]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    new_cards = []
    for i, para in enumerate(paragraphs[:200]):
        kws = _keywords(para)[:8] or extra_kws or ["document"]
        new_cards.append({
            "id": f"doc.{doc_cid}.{i}",
            "question": para[:120],
            "answer": para,
            "domain": "document",
            "intent": "factual",
            "keywords": kws,
        })

    kb = get_kb()
    kb.cards = kb.cards + new_cards
    result = await kb.ingest()
    if result.get("ingested"):
        await _snapshot_kb(kb)

    return {
        "filename": file.filename,
        "doc_cid": doc_cid,
        "card_count": len(new_cards),
        "ingested": result.get("ingested"),
        "source_text_preview": text[:300],
    }


@app.post("/chat")
async def chat(request: Request):
    """Understand an English message and respond in English (spec §7)."""
    body = await request.json()
    message = (body.get("message") or "").strip()
    session_id = body.get("session_id") or ""
    threshold = float(body.get("threshold", 0.15))

    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")

    kb = get_kb()
    if not kb.ingested:
        await kb.ingest()

    session = _get_or_create_session(session_id) if session_id else None

    # Resolve a pending clarification ("did you mean X or Y?") when the user
    # answers with one of the suggested options — substitute it back into the
    # original question and continue from there.
    if session and session.get("pending"):
        pending = session.pop("pending")
        chosen = next((c for c in pending["candidates"]
                       if c in message.lower()), None)
        if chosen:
            message = re.sub(
                rf"\b{re.escape(pending['word'])}\b", chosen,
                pending["original"], flags=re.IGNORECASE,
            )

    # ── Query analysis: typo correction + keyword/question identification ──
    analysis = get_analyzer(kb).analyze(
        message,
        context=(session.get("context") or session.get("keywords")) if session else None,
    )

    # Ask a clarifying question instead of retrieving when unsure.
    if analysis["clarification"]:
        reply = {
            "reply": analysis["clarification"],
            "needs_clarification": True,
            "strategy": "Clarification",
            "analysis": analysis,
        }
        if session:
            now = datetime.utcnow().isoformat()
            if not session.get("title"):
                session["title"] = message[:80]
            if analysis["ambiguous"]:
                session["pending"] = {
                    "original": message,
                    "word": analysis["ambiguous"][0]["word"],
                    "candidates": analysis["ambiguous"][0]["candidates"],
                }
            session["history"].append({"role": "user", "content": message, "stamp": now})
            session["history"].append({
                "role": "assistant", "content": reply["reply"], "stamp": now,
                "strategy": "Clarification",
            })
            session["updated_at"] = now
            _save_sessions_file()
        return reply

    # Meta-questions about the conversation itself → report the current topic.
    if _is_meta_question(message) and session and session.get("history"):
        reply = {
            "reply": _meta_reply(session),
            "needs_clarification": False,
            "strategy": "Meta",
            "analysis": analysis,
        }
        now = datetime.utcnow().isoformat()
        session["history"].append({"role": "user", "content": message, "stamp": now})
        session["history"].append({
            "role": "assistant", "content": reply["reply"], "stamp": now,
            "strategy": "Meta",
        })
        session["updated_at"] = now
        _save_sessions_file()
        return reply

    corrected = analysis["corrected"]

    # G7/G8 — conversational context management
    resolved = _resolve_context(corrected, session)

    intent = classify_intent(corrected)
    negated = _negated_tokens(corrected)
    keyword_hit = _single_keyword_hit(corrected)

    # Union pagination: "more" continues the truncated list from the last turn.
    # Any other new message clears a pending pagination.
    pagination = session.get("pagination") if session else None
    is_more = bool(pagination) and _is_more_request(corrected)
    if session and not is_more:
        session.pop("pagination", None)

    if is_more:
        kw = pagination["keyword"]
        result = kb.query(kw, top_k=8, negated=negated,
                          context_keywords=_keywords(kw))
        reply = _keyword_union_reply(
            kb, kw, intent, threshold,
            offset=pagination["offset"], page_size=pagination["page_size"],
        )
        if reply is not None:
            pag = reply.pop("pagination", None)
            if pag and pag["has_more"]:
                session["pagination"] = pag
            else:
                session.pop("pagination", None)
        else:
            session.pop("pagination", None)
        resolved = _keywords(kw)
    elif intent in ("greeting", "help", "smalltalk"):
        # templates answer these directly; keep the topic intact for follow-ups
        result = kb.query(corrected, top_k=3, negated=negated)
        reply = _build_reply(corrected, result["matches"], intent, threshold,
                             keyword_hit)
    else:
        # keyword-centric retrieval (spec §6.7) enriched with resolved context
        result = kb.query(corrected, top_k=8, negated=negated, context_keywords=resolved)
        # A lone keyword or an exploration request → union of cards containing
        # the focus keyword, paginated to UNION_PAGE_SIZE per turn.
        reply = None
        if len(resolved) <= 1 or _is_exploration_request(corrected):
            reply = _keyword_union_reply(kb, corrected, intent, threshold)
            if reply is not None:
                pag = reply.pop("pagination", None)
                if pag and pag["has_more"] and session:
                    session["pagination"] = pag
        if reply is None:
            reply = _build_reply(corrected, result["matches"], intent, threshold,
                                 keyword_hit)

    # Elaboration follow-up ("tell me more" / "elaborate" on the current topic):
    # return the card's stored details instead of repeating the one-liner.
    # (Skipped for "more" pagination — that already advances the list.)
    if _is_elaboration_request(corrected) and resolved and not is_more:
        best = result["matches"][0] if result.get("matches") else None
        card = best.get("card") if best else None
        if card and card.get("details"):
            reply = {
                "reply": card["details"],
                "confidence": round(best.get("bss", 0.0), 4),
                "strategy": "Elaboration",
                "intent": intent,
                "matched_card": best.get("key"),
            }

    if session:
        session["context"] = resolved
        session["keywords"] = resolved

    reply.update({
        "keywords": _keywords(corrected),
        "resolved_keywords": resolved,
        "context": resolved,
        "negation_detected": bool(negated),
        "suppressed_cards": result.get("suppressed", []),
        "analysis": analysis,
    })

    cid = await _store_to_ipfs({"message": message, **reply})
    if cid:
        reply["stored_cid"] = cid

    if session:
        now = datetime.utcnow().isoformat()
        if not session.get("title"):
            session["title"] = message[:80]
        session["history"].append({"role": "user", "content": message, "stamp": now})
        session["history"].append({
            "role": "assistant",
            "content": reply.get("reply", ""),
            "stamp": now,
            "strategy": reply.get("strategy"),
        })
        session["updated_at"] = now
        _save_sessions_file()

    return reply


@app.post("/nlp/analyze")
async def nlp_analyze(request: Request):
    """Analyze a query: typo correction, keyword/question extraction, and an
    optional clarifying question."""
    body = await request.json()
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")
    kb = get_kb()
    if not kb.ingested:
        await kb.ingest()
    return get_analyzer(kb).analyze(message)


@app.post("/nlp/query")
async def nlp_query(request: Request):
    """Return ranked matches for a message (no response composition)."""
    body = await request.json()
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")

    kb = get_kb()
    if not kb.ingested:
        await kb.ingest()

    result = kb.query(message, top_k=int(body.get("top_k", 8)),
                      negated=_negated_tokens(message))
    return {
        "message": message,
        "keywords": _keywords(message),
        "matches": [
            {
                "key": m["key"],
                "bss": round(m["bss"], 4),
                "question": m["card"].get("question"),
                "answer": m["card"].get("answer"),
            }
            for m in result["matches"]
        ],
        "suppressed": result.get("suppressed", []),
    }


# ═══════════════════════════════════════════════════════════════════════
# Evaluation (spec §14.6 / G10)
# ═══════════════════════════════════════════════════════════════════════

GOLDEN_SET = [
    ("What is the capital of France?", "fact.capital.france"),
    ("Capital of France?", "fact.capital.france"),
    ("France's capital city", "fact.capital.france"),
    ("Who created Python?", "tech.python.creator"),
    ("What is quantum computing?", "tech.quantum.what"),
    ("What is a qubit?", "tech.quantum.qubit"),
    ("What is the chemical formula of water?", "science.water.formula"),
]


@app.get("/nlp/eval")
async def nlp_eval():
    """Run the golden set and report hit@1 / hit@5."""
    kb = get_kb()
    if not kb.ingested:
        await kb.ingest()

    hit1 = hit5 = 0
    total = len(GOLDEN_SET)
    details = []
    for query, expected in GOLDEN_SET:
        result = kb.query(query, top_k=8, negated=_negated_tokens(query))
        top_keys = [m["key"].replace("c:nlp:", "") for m in result["matches"]]
        hit1 += int(expected in top_keys[:1])
        hit5 += int(expected in top_keys[:5])
        details.append({"query": query, "expected": expected,
                        "top1": top_keys[0] if top_keys else None,
                        "hit1": expected in top_keys[:1]})

    return {
        "total": total,
        "hit@1": round(hit1 / total, 4) if total else 0.0,
        "hit@5": round(hit5 / total, 4) if total else 0.0,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 9095))
    logger.info(f"nlp-model starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
