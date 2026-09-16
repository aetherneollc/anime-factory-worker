"""SiliconFlow CosyVoice2-0.5B hosted TTS. Voice lock. No default-voice fallback."""

from __future__ import annotations

import array
import hashlib
import io
import json
import math
import re
import sqlite3
import struct
import sys
import wave
from pathlib import Path
from typing import Callable, Sequence
from urllib.request import Request, urlopen

from anime_factory.config import load_settings
from anime_factory.db import utcnow
from anime_factory.instrument import Counters
from anime_factory.langs import normalize_langs
from anime_factory.models import H3_MAX_SECONDS, SILICONFLOW_BASE_URL, TTS_MODEL

HttpOpener = Callable[[Request], bytes | dict]

_NATIVE_LITTLE_ENDIAN = sys.byteorder == "little"

# CosyVoice2 renders at 24 kHz. Mix at the native rate and let ffmpeg do the
# final resample; the old 16 kHz nearest-neighbour downsample in mix_clips
# threw away every other sample of the only speech in the episode.
COSYVOICE_SAMPLE_RATE = 24000

# EP001 shipped five line wavs at exactly 24.000s with distinct md5s: that is a
# hard truncation cap on the hosted synthesizer, not five long performances.
# An 8s shot carrying a 24s wav is what drove the FreezePadError loop.
TTS_HARD_CAP_SECONDS = 24.0
CAP_EPSILON_SECONDS = 0.05
# Empty synthesis: 0.16s / 0.28s / 0.36s wavs for full sentences. The absolute
# floor only bites when the text itself warrants a longer take, so a genuinely
# two-word line is still allowed to be short.
MIN_LINE_SECONDS = 1.0
MIN_LINE_RATIO = 0.5
ABSOLUTE_MIN_LINE_SECONDS = 0.05
MAX_LINE_RATIO = 4.0
MAX_LINE_FLOOR_SECONDS = 6.5
MAX_LINE_SLACK_SECONDS = 0.5
# Re-synthesis budget per line before the episode blocks.
TTS_LINE_RETRIES = 2

# CosyVoice clone-form field only. Never pass this as `speech()` input / line wav text.
VOICE_CLONE_TRANSCRIPT = "这是一段音色参考。"
CLONE_TRANSCRIPT_MARKERS = ("这是一段音色参考", "this is a voice reference")

# CosyVoice2 built-ins are language-agnostic. Do not key pools by lang:
# the old zh=(claire,anna,bella) map locked every Chinese male to a female timbre.
FEMALE_STOCK: tuple[str, ...] = ("claire", "anna", "bella", "diana")
MALE_STOCK: tuple[str, ...] = ("alex", "charles", "benjamin", "david")
STOCK_VOICES: dict[str, tuple[str, ...]] = {
    "female": FEMALE_STOCK,
    "male": MALE_STOCK,
    # Age bands inside a gender. Young first; adult lower/steadier.
    "female_young": ("claire", "anna"),
    "female_adult": ("bella", "diana"),
    "male_young": ("alex", "david"),
    "male_adult": ("charles", "benjamin"),
}
_FEMALE_MARKERS = (
    "woman",
    "female",
    "girl",
    "lady",
    "heroine",
    "she",
    "her",
    "女",
    "少女",
    "她",
)
_MALE_MARKERS = ("man", "male", "boy", "guy", "he", "him", "男", "他")
_ROLE_GENDER = {
    "girl": "female",
    "heroine": "female",
    "woman": "female",
    "lady": "female",
    "boy": "male",
    "hero": "male",
    "man": "male",
    "guy": "male",
}
_YOUNG_PERSONALITY = ("bright", "energetic", "cheerful", "soft")
_ADULT_PERSONALITY = ("tired", "quiet", "cool", "weary", "cold", "calm")


class VoiceLockError(RuntimeError):
    def __init__(self, character_id: str, lang: str):
        super().__init__(f"missing voice_uri for {character_id}/{lang}; episode blocked")
        self.character_id = character_id
        self.lang = lang
        self.blocked = True


class VoiceGenderError(RuntimeError):
    """Male lead got a female CosyVoice stock speaker, or langs drifted."""

    def __init__(self, message: str):
        super().__init__(message)
        self.blocked = True


class GenderRequiredError(VoiceGenderError):
    """No explicit cast gender, 1boy/1girl tag, or textual marker for this speaker.

    The old fallback silently called every unmarked speaker male. That is gone:
    an unresolved gender blocks the episode instead of guessing.
    """

    def __init__(self, character_id: str):
        super().__init__(
            f"{character_id}: gender is required (explicit cast gender, 1boy/1girl identity "
            "tag, or an unambiguous textual marker) before a voice can be locked"
        )
        self.character_id = character_id


class GenderConflictError(VoiceGenderError):
    """Explicit cast gender disagrees with the identity's 1boy/1girl tag or markers."""

    def __init__(self, character_id: str, signals: dict[str, str]):
        super().__init__(f"{character_id}: conflicting gender signals {signals}")
        self.character_id = character_id
        self.signals = dict(signals)


class SpeechTextError(ValueError):
    """Empty text or clone-ref transcript leaked into a line-wav speech call."""


class LineDurationError(RuntimeError):
    """A line wav is truncated at the synthesizer cap, empty, or far too long for its text."""

    def __init__(self, message: str):
        super().__init__(message)
        self.blocked = True


def line_expected_seconds(text: str, lang: str) -> float:
    """Rough spoken length of one board line. CosyVoice ~4-5 CJK chars/s.

    Unspaced JA is one English "word", so it must be counted by character or a
    normal 7s line looks like a clone leak.
    """
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    if lang == "zh":
        return max(len(raw) * 0.22, 0.5)
    if lang == "ja":
        return max(len(raw) / 7.0, 0.5)
    return max(len(raw.split()) * 0.28, 0.5)


def assert_line_duration_plausible(
    duration: float,
    text: str,
    lang: str,
    *,
    sid: str = "",
    cap_seconds: float = TTS_HARD_CAP_SECONDS,
) -> None:
    """Reject truncated / empty / runaway line wavs against the text-derived estimate."""
    label = sid or "line"
    seconds = float(duration or 0.0)
    expected = line_expected_seconds(text, lang)
    if seconds <= ABSOLUTE_MIN_LINE_SECONDS:
        raise LineDurationError(f"{label} line wav is {seconds:.2f}s; synthesis produced no audio")
    if cap_seconds > 0 and abs(seconds - cap_seconds) <= CAP_EPSILON_SECONDS and expected < cap_seconds * 0.8:
        raise LineDurationError(
            f"{label} line wav is {seconds:.2f}s, exactly the {cap_seconds:.0f}s synthesizer cap "
            f"for a ~{expected:.1f}s line {text[:24]!r}; the take is truncated, re-synthesize or split"
        )
    if seconds < MIN_LINE_SECONDS and seconds < expected * MIN_LINE_RATIO:
        raise LineDurationError(
            f"{label} line wav is {seconds:.2f}s for a ~{expected:.1f}s line {text[:24]!r}; "
            "empty synthesis, re-synthesize"
        )
    ceiling = max(MAX_LINE_FLOOR_SECONDS, expected * MAX_LINE_RATIO) + MAX_LINE_SLACK_SECONDS
    if seconds > ceiling:
        raise LineDurationError(
            f"{label} line wav is {seconds:.1f}s; too long for line {text[:24]!r} (~{expected:.1f}s)"
        )


_STAGE_DIRECTION_RE = re.compile(r"[（(][^（）()]*[）)]")


def strip_stage_directions(text: str | None) -> str:
    """Drop parenthetical stage directions so CosyVoice never speaks 括号舞台指示."""
    raw = str(text or "")
    prev = None
    while prev != raw:
        prev = raw
        raw = _STAGE_DIRECTION_RE.sub("", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    return raw.strip()


def is_clone_transcript(text: str | None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    compact = raw.rstrip("。．.！!").strip()
    if compact == VOICE_CLONE_TRANSCRIPT.rstrip("。"):
        return True
    lowered = raw.lower()
    return any(marker in raw or marker in lowered for marker in CLONE_TRANSCRIPT_MARKERS)


def assert_speech_text(text: str | None) -> str:
    raw = strip_stage_directions(text)
    if not raw:
        raise SpeechTextError("speech text is empty; board line required")
    if is_clone_transcript(raw):
        raise SpeechTextError(f"clone transcript must not be synthesized as dialogue: {raw[:24]!r}")
    return raw


def _marker_in(blob: str, markers: tuple[str, ...]) -> bool:
    text = str(blob or "").lower()
    for marker in markers:
        token = marker.lower()
        if not token.isascii():
            if token in text:
                return True
            continue
        if re.search(rf"\b{re.escape(token)}\b", text):
            return True
    return False


_GENDER_TAG_RE = re.compile(r"\b(1boy|1girl)\b", re.I)
_GENDER_FIELD_MAP = {
    "female": "female",
    "f": "female",
    "woman": "female",
    "girl": "female",
    "male": "male",
    "m": "male",
    "man": "male",
    "boy": "male",
}


def _explicit_gender_field(gender: str | None) -> str | None:
    raw = str(gender or "").strip().lower()
    return _GENDER_FIELD_MAP.get(raw)


def _identity_gender_tag(identity: str | None) -> str | None:
    match = _GENDER_TAG_RE.search(str(identity or ""))
    if not match:
        return None
    return "male" if match.group(1).lower() == "1boy" else "female"


def resolve_gender(
    character_id: str,
    *,
    gender: str | None = None,
    identity: str | None = None,
    name: str | None = None,
) -> str:
    """Derive a speaker's gender from explicit signals only. Never guess.

    Priority is: explicit cast `gender` field and the identity's 1boy/1girl tag
    (both "structured" signals — if they disagree that is a real authoring bug,
    not a coin flip). Failing those, an unambiguous textual marker in the
    identity/name blob or a literal role id (e.g. `girl`, `hero`) is accepted.
    A character with none of these signals raises `GenderRequiredError`
    instead of defaulting to male, which is the bug this replaces.
    """
    cid = str(character_id or "").strip()
    explicit = _explicit_gender_field(gender)
    tag = _identity_gender_tag(identity)
    blob = " ".join(str(x or "") for x in (identity, name))
    female_hit = _marker_in(blob, _FEMALE_MARKERS)
    male_hit = _marker_in(blob, _MALE_MARKERS)
    role = _ROLE_GENDER.get(cid.lower())

    signals: dict[str, str] = {}
    if explicit:
        signals["gender_field"] = explicit
    if tag:
        signals["identity_tag"] = tag
    if female_hit and not male_hit:
        signals["identity_marker"] = "female"
    elif male_hit and not female_hit:
        signals["identity_marker"] = "male"
    if role:
        signals["role_id"] = role

    distinct = set(signals.values())
    if len(distinct) > 1:
        raise GenderConflictError(cid or "character", signals)
    if distinct:
        return distinct.pop()
    raise GenderRequiredError(cid or "character")


def infer_gender(
    character_id: str,
    *,
    gender: str | None = None,
    identity: str | None = None,
    name: str | None = None,
) -> str:
    """Back-compat alias for `resolve_gender`. Raises instead of defaulting to male."""
    return resolve_gender(character_id, gender=gender, identity=identity, name=name)


def infer_age_band(age: int | float | None, personality: str | None = None) -> str:
    if age is not None:
        try:
            return "young" if int(age) <= 28 else "adult"
        except (TypeError, ValueError):
            pass
    blob = str(personality or "").lower()
    if any(token in blob for token in _YOUNG_PERSONALITY):
        return "young"
    if any(token in blob for token in _ADULT_PERSONALITY):
        return "adult"
    return "young"


def infer_personality(identity: str | None = None, personality: str | None = None) -> str:
    blob = " ".join(str(x or "") for x in (personality, identity)).lower()
    if any(token in blob for token in ("cool", "cold", "冷")):
        return "cool"
    if any(token in blob for token in ("tired", "quiet", "weary", "sad", "累", "疲惫")):
        return "tired"
    return "neutral"


def stock_voice_uri(
    character_id: str,
    lang: str,
    *,
    gender: str | None = None,
    age: int | float | None = None,
    personality: str | None = None,
    identity: str | None = None,
    name: str | None = None,
) -> str:
    """Pick a CosyVoice2 stock speaker from age/gender/personality. Not from language.

    `lang` is kept so callers stay stable; the same character must keep the same
    speaker across zh/en/ja. Hashing `character_id:lang` into a female-only zh
    pool is what put bella on 阿哲.
    """
    del lang  # speakers are cross-lingual; timbre lock is per character
    sex = infer_gender(character_id, gender=gender, identity=identity, name=name)
    band = infer_age_band(age, personality)
    trait = infer_personality(identity, personality)
    pool = STOCK_VOICES.get(f"{sex}_{band}") or STOCK_VOICES[sex]
    digest = hashlib.sha256(f"{character_id}:{sex}:{band}:{trait}".encode("utf-8")).hexdigest()
    speaker = pool[int(digest[:8], 16) % len(pool)]
    return f"{TTS_MODEL}:{speaker}"


def stock_speaker_name(uri: str | None) -> str:
    return str(uri or "").rsplit(":", 1)[-1].strip().lower()


def assert_voice_gender_lock(
    character_id: str,
    uri: str,
    *,
    gender: str | None = None,
    identity: str | None = None,
    name: str | None = None,
    lang: str = "",
) -> None:
    """man/male must not get claire/anna/bella/diana; woman must not get alex/charles/benjamin/david."""
    sex = infer_gender(character_id, gender=gender, identity=identity, name=name)
    speaker = stock_speaker_name(uri)
    if not speaker:
        raise VoiceGenderError(f"empty voice speaker for {character_id}/{lang or '?'}")
    if sex == "male" and speaker in FEMALE_STOCK:
        raise VoiceGenderError(
            f"{character_id} ({name or character_id}/{lang or '?'}) is male but CosyVoice speaker is {speaker}"
        )
    if sex == "female" and speaker in MALE_STOCK:
        raise VoiceGenderError(
            f"{character_id} ({name or character_id}/{lang or '?'}) is female but CosyVoice speaker is {speaker}"
        )


def assert_same_speaker_all_langs(uris: dict[str, str], character_id: str = "") -> None:
    speakers = {stock_speaker_name(uri) for uri in uris.values() if uri}
    if len(speakers) > 1:
        raise VoiceGenderError(
            f"{character_id or 'character'} CosyVoice speaker drifted across langs: {uris}"
        )


def assert_locked_voices(
    uris: dict[str, str],
    character_id: str,
    *,
    gender: str | None = None,
    identity: str | None = None,
    name: str | None = None,
) -> None:
    assert_same_speaker_all_langs(uris, character_id)
    for lang, uri in uris.items():
        assert_voice_gender_lock(
            character_id, uri, gender=gender, identity=identity, name=name, lang=lang
        )


def pcm16_wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    n = max(int(seconds * sample_rate), 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def voiced_dummy_wav(seconds: float = 1.2, sample_rate: int = 16000, freq: float = 140.0) -> bytes:
    """Non-silent PCM for dry-run TTS so compose silence QC can see energy."""
    n = max(int(seconds * sample_rate), 1)
    frames = bytearray()
    for i in range(n):
        t = i / sample_rate
        env = min(1.0, t / 0.04) * min(1.0, (seconds - t) / 0.08)
        s = 0.35 * math.sin(2 * math.pi * float(freq) * t) * env
        frames += struct.pack("<h", int(max(-1.0, min(1.0, s)) * 18000))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def decode_pcm16_mono(data: bytes) -> tuple[int, list[int]]:
    """Read PCM from a WAV, trusting the data chunk bytes — not a bogus nframes."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 16000, []
    pos = 12
    rate = 16000
    nch = 1
    width = 2
    payload = b""
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        declared = struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + 8
        if declared == 0xFFFFFFFF or declared > len(data) - start:
            chunk = data[start:]
            next_pos = len(data)
        else:
            chunk = data[start : start + declared]
            next_pos = start + declared + (declared & 1)
        if cid == b"fmt " and len(chunk) >= 16:
            _fmt, nch, rate, _byte_rate, _block, bits = struct.unpack_from("<HHIIHH", chunk, 0)
            width = max(bits // 8, 1)
        elif cid == b"data":
            payload = chunk
            break
        pos = next_pos
        if pos <= start:
            break
    if width != 2 or len(payload) < 2:
        return rate or 16000, []
    n = len(payload) // 2
    pcm = array.array("h")
    pcm.frombytes(payload[: n * 2])
    if not _NATIVE_LITTLE_ENDIAN:
        pcm.byteswap()
    samples = pcm.tolist()
    if nch == 2:
        samples = samples[::2]
    return int(rate or 16000), samples


def resample_pcm16_mono(samples: list[int], src_rate: int, dst_rate: int) -> list[int]:
    """Linear-interpolation resample between CosyVoice 16k/24k/32k takes.

    Nearest-neighbour dropped whole samples on every 24k→16k line wav.
    """
    src = int(src_rate or 0)
    dst = int(dst_rate or 0)
    if src <= 0 or dst <= 0 or src == dst or len(samples) < 2:
        return list(samples)
    n_out = max(1, int(round(len(samples) * dst / src)))
    step = src / dst
    last = len(samples) - 1
    out: list[int] = []
    for i in range(n_out):
        pos = i * step
        left = int(pos)
        if left >= last:
            out.append(samples[last])
            continue
        frac = pos - left
        out.append(int(round(samples[left] * (1.0 - frac) + samples[left + 1] * frac)))
    return out


def encode_pcm16_mono(samples: list[int], rate: int) -> bytes:
    buf = io.BytesIO()
    try:
        # Bulk C conversion. A 600s 24 kHz master is 14.4M samples; a per-sample
        # struct.pack loop is minutes of the compose stage.
        pcm = array.array("h", samples)
    except (OverflowError, TypeError):
        pcm = array.array("h", (max(-32768, min(32767, int(s))) for s in samples))
    if not _NATIVE_LITTLE_ENDIAN:
        pcm.byteswap()
    frames = pcm.tobytes()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate) or 16000)
        wf.writeframes(frames)
    return buf.getvalue()


def rewrite_pcm_wav(data: bytes) -> bytes:
    rate, samples = decode_pcm16_mono(data)
    if not samples:
        return data
    return encode_pcm16_mono(samples, rate)


def pcm_duration_seconds(data: bytes) -> float:
    rate, samples = decode_pcm16_mono(data)
    if rate <= 0 or not samples:
        return 0.0
    return len(samples) / float(rate)


def wav_duration_seconds(data: bytes) -> float:
    return pcm_duration_seconds(data)


def wav_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CosyVoiceClient:
    def __init__(self, api_keys: list[str], opener: HttpOpener | None = None, live: bool | None = None):
        self.api_keys = api_keys
        self.opener = opener
        self.live = load_settings().live_tts if live is None else live
        self._i = 0

    def _key(self) -> str:
        if not self.api_keys:
            return ""
        k = self.api_keys[self._i % len(self.api_keys)]
        self._i += 1
        return k

    def list_voices(self) -> dict:
        req = Request(
            f"{SILICONFLOW_BASE_URL}/v1/audio/voice/list",
            headers={"Authorization": f"Bearer {self._key()}"},
            method="GET",
        )
        Counters.tts_requests += 1
        if self.opener:
            data = self.opener(req)
            return data if isinstance(data, dict) else json.loads(data)
        if not self.live:
            return {"result": []}
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def upload_voice(self, name: str, audio: bytes, transcript: str = VOICE_CLONE_TRANSCRIPT) -> str:
        """Upload reference audio; returns voice_uri. Clone transcript is the form field only.

        Production line wavs must call `speech(board_line, stock_voice_uri)`.
        SiliconFlow CosyVoice2: POST /v1/uploads/audio/voice
        (not /v1/audio/voice/upload — that 404'd in the 16s MVP and must
        not trigger a Vast destroy).
        """
        boundary = "----AnimeFactoryVoice"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{TTS_MODEL}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"customName\"\r\n\r\n{name}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n{name}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n{transcript}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"ref.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = Request(
            f"{SILICONFLOW_BASE_URL}/v1/uploads/audio/voice",
            data=body,
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        Counters.tts_requests += 1
        if self.opener:
            data = self.opener(req)
            if isinstance(data, dict):
                return data.get("uri") or data.get("voice_uri")
            parsed = json.loads(data)
            return parsed.get("uri") or parsed.get("voice_uri")
        if not self.live:
            raise VoiceLockError(name, "?")
        with urlopen(req, timeout=60) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
            return parsed.get("uri") or parsed.get("voice_uri")

    def speech(self, text: str, voice_uri: str, speed: float = 1.0) -> bytes:
        if not voice_uri:
            raise VoiceLockError("unknown", "?")
        text = assert_speech_text(text)
        payload = json.dumps(
            {"model": TTS_MODEL, "input": text, "voice": voice_uri, "speed": speed, "response_format": "wav"}
        ).encode("utf-8")
        req = Request(
            f"{SILICONFLOW_BASE_URL}/v1/audio/speech",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        Counters.tts_requests += 1
        if self.opener:
            data = self.opener(req)
            blob = data if isinstance(data, (bytes, bytearray)) else voiced_dummy_wav(1.2, freq=_dummy_freq(text))
            return rewrite_pcm_wav(bytes(blob))
        if not self.live:
            seconds = max(0.6, min(float(H3_MAX_SECONDS), 0.22 * max(len(text), 1)))
            return voiced_dummy_wav(seconds, freq=_dummy_freq(text))
        with urlopen(req, timeout=120) as resp:
            return rewrite_pcm_wav(resp.read())


def _dummy_freq(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return 110.0 + (int(digest[:4], 16) % 280)


def require_voice_uri(conn: sqlite3.Connection, character_id: str, lang: str) -> str:
    row = conn.execute(
        "SELECT voice_uri FROM character_voice WHERE character_id = ? AND lang = ?",
        (character_id, lang),
    ).fetchone()
    uri = row["voice_uri"] if row else None
    if not uri:
        raise VoiceLockError(character_id, lang)
    return uri


def lock_voices_or_block(
    conn: sqlite3.Connection, character_ids: list[str], langs: Sequence[str] | None = None
) -> None:
    for cid in character_ids:
        for lang in normalize_langs(langs):
            require_voice_uri(conn, cid, lang)


# ---------------------------------------------------------------------------
# Voice-profile / line fingerprints and cross-episode lock preservation.
#
# A character's locked voice_uri must survive into the next episode untouched:
# recomputing stock_voice_uri from the same identity/gender is deterministic,
# but an author edit (name, personality wording) must not silently reassign
# the speaker mid-series. Preservation is keyed on lock_version: bumping it is
# the only sanctioned way to force a new profile (a real clone swap).
# ---------------------------------------------------------------------------


def profile_fingerprint(
    character_id: str,
    gender: str,
    voice_uri: str,
    model: str = TTS_MODEL,
    lock_version: int = 1,
) -> str:
    """Identity fingerprint for a locked character/lang voice profile.

    Independent of any single line's text — this is what cross-episode reuse
    and lock-preservation checks compare against, not `line_fingerprint`.
    """
    payload = "|".join(
        (
            "profile:v1",
            str(character_id or ""),
            str(gender or ""),
            str(voice_uri or ""),
            str(model or ""),
            str(int(lock_version or 1)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def line_fingerprint(
    character_id: str,
    gender: str,
    voice_uri: str,
    model: str,
    speed: float,
    lang: str,
    cleaned_text: str,
    lock_version: int = 1,
) -> str:
    """Fingerprint over every input that can change what a line wav sounds like.

    Reuse of an on-disk line wav is only valid when this matches exactly:
    character, gender, voice URI, model, speed, lang, the exact cleaned
    (stage-direction-stripped) text, and the voice-lock version. Any drift —
    a voice migration, a text edit, a speed retune — must regenerate.
    """
    payload = "|".join(
        (
            "line:v1",
            str(character_id or ""),
            str(gender or ""),
            str(voice_uri or ""),
            str(model or ""),
            f"{float(speed or 1.0):.4f}",
            str(lang or ""),
            str(cleaned_text or ""),
            str(int(lock_version or 1)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def existing_voice_lock(
    conn: sqlite3.Connection, character_id: str, lang: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT voice_uri, speed, lock_version, profile_fingerprint FROM character_voice "
        "WHERE character_id = ? AND lang = ?",
        (character_id, lang),
    ).fetchone()


def resolve_locked_voice_uri(
    conn: sqlite3.Connection,
    character_id: str,
    lang: str,
    candidate_uri: str,
    *,
    migrate: bool = False,
) -> tuple[str, int]:
    """Preserve an established voice_uri across episodes unless explicitly migrated.

    A non-empty `character_voice.voice_uri` already on disk wins over any
    freshly recomputed `candidate_uri` — recomputing stock_voice_uri from a
    lightly-edited identity string must not reassign an established speaker.
    `migrate=True` is the only way to force `candidate_uri` in and bump
    `lock_version`, e.g. an intentional clone swap.
    Returns (uri, lock_version) to persist.
    """
    row = existing_voice_lock(conn, character_id, lang)
    established = str(row["voice_uri"]) if row and row["voice_uri"] else ""
    current_version = int(row["lock_version"] or 1) if row else 1
    if migrate:
        return candidate_uri, current_version + 1
    if established:
        return established, current_version
    return candidate_uri, current_version


def line_wav_reusable(
    conn: sqlite3.Connection,
    segment_id: str,
    lang: str,
    expected_fingerprint: str,
    wav_path: Path | str,
) -> bool:
    """True only if `wav_path` exists and its recorded fingerprint still matches."""
    path = Path(wav_path)
    if not path.is_file() or path.stat().st_size <= 100:
        return False
    row = conn.execute(
        "SELECT fingerprint FROM line_audio WHERE segment_id = ? AND lang = ?",
        (segment_id, lang),
    ).fetchone()
    return bool(row) and str(row["fingerprint"]) == str(expected_fingerprint)


def record_line_audio(
    conn: sqlite3.Connection,
    segment_id: str,
    lang: str,
    character_id: str,
    fingerprint: str,
    wav_path: str,
    duration: float,
    sample_rate: int,
    content_hash: str,
) -> None:
    conn.execute(
        """
        INSERT INTO line_audio (segment_id, lang, character_id, fingerprint, wav_path,
                                 duration, sample_rate, content_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(segment_id, lang) DO UPDATE SET
            character_id = excluded.character_id,
            fingerprint = excluded.fingerprint,
            wav_path = excluded.wav_path,
            duration = excluded.duration,
            sample_rate = excluded.sample_rate,
            content_hash = excluded.content_hash,
            created_at = excluded.created_at
        """,
        (segment_id, lang, character_id, fingerprint, wav_path, duration, sample_rate, content_hash, utcnow()),
    )
    conn.commit()


def clamp_speed(requested: float, tolerance: float | None = None) -> float:
    tol = load_settings().tts_speed_tolerance if tolerance is None else tolerance
    return max(1.0 - tol, min(1.0 + tol, requested))


def synthesize_line(
    client: CosyVoiceClient,
    conn: sqlite3.Connection,
    character_id: str,
    lang: str,
    text: str,
    target_seconds: float | None = None,
    validate: bool = True,
) -> tuple[bytes, float]:
    """Synthesize one board line, re-synthesizing an implausible take before giving up.

    A truncated (cap-length) or empty wav is never returned: compose cannot
    freeze-pad a 24s wav onto an 8s shot, so the line must be re-synthesized or
    the beat split upstream.
    """
    uri = require_voice_uri(conn, character_id, lang)
    row = conn.execute(
        "SELECT speed FROM character_voice WHERE character_id = ? AND lang = ?",
        (character_id, lang),
    ).fetchone()
    speed = clamp_speed(float(row["speed"] or 1.0) if row else 1.0)
    clean = assert_speech_text(text)
    label = f"{character_id}.{lang}"
    expected = float(target_seconds or 0.0) or line_expected_seconds(clean, lang)
    last_error: LineDurationError | None = None
    audio = b""
    duration = 0.0
    for _attempt in range(TTS_LINE_RETRIES + 1):
        audio = client.speech(clean, uri, speed=speed)
        duration = wav_duration_seconds(audio)
        if not validate:
            return audio, duration
        try:
            assert_line_duration_plausible(duration, clean, lang, sid=label)
        except LineDurationError as exc:
            last_error = exc
            # Nudge speed toward the text estimate so the retry is not a rerun
            # of identical conditions. clamp_speed keeps it inside tolerance.
            if duration > 0 and expected > 0:
                speed = clamp_speed(speed * (duration / expected))
            continue
        return audio, duration
    raise last_error if last_error else LineDurationError(f"{label} synthesis failed")
