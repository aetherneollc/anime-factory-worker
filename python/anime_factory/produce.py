"""600s episode producer driven by the PM control-plane APIs.

Two paths. The real path is the default: the project's own title + logline go to
DeepSeek (`script.draft_episode_script`), the gated script becomes the board
(`board.board_from_script`), and shot count falls out of the dialogue and the
duration target. The template demo path (`shot_plan`, `TITLE`, `DIALOGUE`) is the
fixed 黑客平板 fixture, kept for prompt-policy tests and offline smoke runs only.

Local CLI is a debug entry. Production pre-GPU is Worker cron (DeepSeek /
SiliconFlow HTTP) or the GPU box running produce_episode during the lease.
Never leases Vast before bible→canon→script→tts→board. design/keyframe
stills may run on the GPU card (Kolors/Comfy). TTS stays SiliconFlow.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

from anime_factory.board import (
    PromptCollapseError,
    SpeakerFrameError,
    board_from_script,
    board_totals,
    expand_shots_to_segments,
    persist_board,
)
from anime_factory.canon import (
    export_continuity,
    export_geo,
    export_timeline,
    interior_asset_path,
    save_continuity,
    write_canon_dir,
)
from anime_factory.compose import assert_shots_have_dialogue, write_episode_srts
from anime_factory.config import load_dotenv, load_settings, siliconflow_keys
from anime_factory.continuity_gates import GateResult, GateViolation, gate1_inject, repair_collapsed_staging
from anime_factory.db import checkpoint_and_upload, migrate, open_db
from anime_factory.design import KolorsClient, generate_asset_library
from anime_factory.keyframe import assert_keyframe_files, ensure_keyframe
from anime_factory.langs import normalize_langs, story_langs
from anime_factory.llm import client_from_env
from anime_factory.models import (
    FIXED_NEGATIVE,
    H3_GEN_HEIGHT,
    H3_GEN_WIDTH,
    H3_MAX_SECONDS,
    STYLE_PREFIX,
    TARGET_EPISODE_SECONDS,
    DEFAULT_SHORT_SECONDS,
    TTS_MODEL,
    normalize_story_kind,
)
from anime_factory.r2_client import upload_tree
from anime_factory.script import draft_episode_script, produce_script
from anime_factory.video_backend import max_seconds_for_backend, select_video_backend
from anime_factory.tts import (
    CosyVoiceClient,
    GenderRequiredError,
    LineDurationError,
    SpeechTextError,
    VoiceGenderError,
    assert_locked_voices,
    existing_voice_lock,
    line_fingerprint,
    line_wav_reusable,
    pcm_duration_seconds,
    profile_fingerprint,
    record_line_audio,
    resolve_gender,
    resolve_locked_voice_uri,
    stock_voice_uri,
    strip_stage_directions,
    synthesize_line,
    wav_sha256,
)
from anime_factory.tts_hosted import (
    hosted_voice_for,
    import_hosted_line_audio,
    load_hosted_tts_manifest,
    seed_hosted_voice_lock,
)
from anime_factory.world import init_fiction_world, write_bible

DEFAULT_EPISODE = "EP001"
DURATION_TOLERANCE = 0.02

# ---------------------------------------------------------------------------
# Template demo path: the fixed 黑客平板 fixture. Not what a new project produces.
# ---------------------------------------------------------------------------

SHOT_SECONDS = H3_MAX_SECONDS  # 8.0 — fewer H3 calls than 5.5s while still totaling 600s
N_SHOTS = int(TARGET_EPISODE_SECONDS / SHOT_SECONDS)  # 75
assert abs(N_SHOTS * SHOT_SECONDS - TARGET_EPISODE_SECONDS) < 1e-6

TITLE = "黑客平板的故事"
LOGLINE = "夜里一张平板自己亮了。阿柯本来只想改完这份文档，却发现平板在回他。"
KEEPER = "ke"
LOC = "loft"
INT = "int_desk"

# Spoken 漫剧 beats (0-based shot index). Sparse 10-line beds are not a story.
DIALOGUE: dict[int, dict[str, str]] = {
    1: {"zh": "谁开的机。", "en": "Who turned it on.", "ja": "誰がつけた。"},
    2: {"zh": "我明明盖上了。", "en": "I closed it.", "ja": "閉めたはずだ。"},
    3: {"zh": "平板，你自己亮了。", "en": "Tablet, you lit up by yourself.", "ja": "タブレット、自分で光った。"},
    4: {"zh": "别装死。", "en": "Don't play dead.", "ja": "死んだふりするな。"},
    6: {"zh": "阿柯在。", "en": "Ke is here.", "ja": "阿柯はいる。"},
    7: {"zh": "屏幕上有字。", "en": "There is text on the screen.", "ja": "画面に文字がある。"},
    8: {"zh": "『你醒了。』", "en": "'You're awake.'", "ja": "『目が覚めた。』"},
    9: {"zh": "我一直醒着。", "en": "I was awake the whole time.", "ja": "ずっと起きてた。"},
    10: {"zh": "你不是我的笔记软件。", "en": "You are not my notes app.", "ja": "お前はメモアプリじゃない。"},
    11: {"zh": "『我是平板。』", "en": "'I am the tablet.'", "ja": "『私はタブレットだ。』"},
    12: {"zh": "那就别乱亮。", "en": "Then don't light up at random.", "ja": "なら勝手に光るな。"},
    13: {"zh": "风扇怎么转了。", "en": "Why is the fan spinning.", "ja": "ファンが回ってる。"},
    14: {"zh": "我没跑任务。", "en": "I'm not running a job.", "ja": "タスクは走らせてない。"},
    15: {"zh": "『你在找口令。』", "en": "'You are looking for a passphrase.'", "ja": "『合言葉を探している。』"},
    16: {"zh": "谁告诉你的。", "en": "Who told you that.", "ja": "誰が教えた。"},
    17: {"zh": "『键盘。』", "en": "'The keyboard.'", "ja": "『キーボード。』"},
    18: {"zh": "键盘不会说话。", "en": "Keyboards don't talk.", "ja": "キーボードは喋らない。"},
    19: {"zh": "除非你在听。", "en": "Unless you are listening.", "ja": "お前が聞いているなら別だ。"},
    21: {"zh": "把亮度调低。", "en": "Dim the brightness.", "ja": "輝度を下げろ。"},
    22: {"zh": "邻居会看见。", "en": "The neighbor will see.", "ja": "隣が見える。"},
    23: {"zh": "『窗是关的。』", "en": "'The window is closed.'", "ja": "『窓は閉まっている。』"},
    24: {"zh": "你怎么知道窗。", "en": "How do you know about the window.", "ja": "窓をどう知った。"},
    25: {"zh": "摄像头。别开。", "en": "The camera. Don't turn it on.", "ja": "カメラ。つけるな。"},
    26: {"zh": "『没开。我听风扇。』", "en": "'It's off. I listen to the fan.'", "ja": "『つけてない。ファンを聞いている。』"},
    27: {"zh": "你很烦。", "en": "You are annoying.", "ja": "うるさい。"},
    28: {"zh": "但也有用。", "en": "But useful.", "ja": "でも役に立つ。"},
    29: {"zh": "把那份文档打开。", "en": "Open that document.", "ja": "あの文書を開け。"},
    30: {"zh": "不是云盘那个。", "en": "Not the cloud one.", "ja": "クラウドのやつじゃない。"},
    32: {"zh": "本地。加密的。", "en": "Local. Encrypted.", "ja": "ローカル。暗号化。"},
    33: {"zh": "『口令错了三次。』", "en": "'The passphrase failed three times.'", "ja": "『合言葉が三回違う。』"},
    34: {"zh": "第四次会锁。", "en": "The fourth will lock it.", "ja": "四回目で鎖される。"},
    35: {"zh": "所以别猜。", "en": "So don't guess.", "ja": "だから当てるな。"},
    36: {"zh": "我想想。", "en": "Let me think.", "ja": "考えさせて。"},
    37: {"zh": "旧键盘的第一行。", "en": "The first row of the old keyboard.", "ja": "古いキーボードの一段目。"},
    38: {"zh": "不是 qwerty。", "en": "Not qwerty.", "ja": "qwertyじゃない。"},
    39: {"zh": "是我自己排的。", "en": "I laid it out myself.", "ja": "自分で並べた。"},
    40: {"zh": "『要我输入吗。』", "en": "'Shall I type it.'", "ja": "『入力しようか。』"},
    41: {"zh": "不要。手给我。", "en": "No. Give me the hands.", "ja": "いや。手は私だ。"},
    43: {"zh": "进了。", "en": "We're in.", "ja": "入れた。"},
    44: {"zh": "目录很干净。", "en": "The folder is clean.", "ja": "フォルダがきれいだ。"},
    45: {"zh": "只有一个文件。", "en": "Only one file.", "ja": "ファイルは一つ。"},
    46: {"zh": "『读吗。』", "en": "'Read it?'", "ja": "『読むか。』"},
    47: {"zh": "读。出声。", "en": "Read. Out loud.", "ja": "読め。声に出せ。"},
    48: {"zh": "『今晚不要关机。』", "en": "'Do not shut down tonight.'", "ja": "『今夜は電源を切るな。』"},
    49: {"zh": "这是我写的？", "en": "Did I write this?", "ja": "俺が書いたのか。"},
    51: {"zh": "时间戳是上周。", "en": "The timestamp is last week.", "ja": "時刻は先週。"},
    52: {"zh": "我上周没写这个。", "en": "I didn't write this last week.", "ja": "先週は書いてない。"},
    53: {"zh": "『你写了。你忘了。』", "en": "'You wrote it. You forgot.'", "ja": "『書いた。忘れた。』"},
    54: {"zh": "平板不会忘。", "en": "The tablet does not forget.", "ja": "タブレットは忘れない。"},
    56: {"zh": "那你记得口令。", "en": "Then you remember the passphrase.", "ja": "なら合言葉を覚えてる。"},
    57: {"zh": "『记得。不说。』", "en": "'I remember. I will not say it.'", "ja": "『覚えてる。言わない。』"},
    58: {"zh": "为什么。", "en": "Why.", "ja": "なぜ。"},
    59: {"zh": "『因为你的手在。』", "en": "'Because your hands are here.'", "ja": "『手がここにあるから。』"},
    61: {"zh": "好。那就守着。", "en": "Fine. Then keep watch.", "ja": "いい。なら守れ。"},
    62: {"zh": "别把灯调亮。", "en": "Don't turn the light up.", "ja": "明るくするな。"},
    64: {"zh": "风扇小一点。", "en": "Fan quieter.", "ja": "ファンを小さく。"},
    65: {"zh": "文档先别关。", "en": "Don't close the document yet.", "ja": "文書はまだ閉じるな。"},
    67: {"zh": "我在。", "en": "I am here.", "ja": "私はいる。"},
    68: {"zh": "你也在。", "en": "You are here too.", "ja": "お前もいる。"},
    69: {"zh": "夜里只剩这块屏。", "en": "At night only this screen remains.", "ja": "夜にはこの画面だけ。"},
    71: {"zh": "别关机。", "en": "Don't shut down.", "ja": "切るな。"},
    72: {"zh": "我看着你。", "en": "I am watching you.", "ja": "見てる。"},
    73: {"zh": "你也看着我。", "en": "You are watching me too.", "ja": "お前も見てる。"},
}


def shot_plan() -> list[dict[str, Any]]:
    acting = [
        "MCU young East Asian hacker Ke at a desk, hoodie, facing camera, tablet already in frame, speaking, locked-off",
        "close-up Ke's mouth and eyes, desk lamp rim light, talking, no zoom, subject already in frame",
        "close-up hands on a matte tablet, fingerprint smudges, locked-off",
        "MCU Ke at the loft desk, hoodie, looking at the tablet screen, speaking",
        "close-up hands typing on a compact keyboard beside the tablet, locked-off",
        "MCU tablet on the desk beside Ke, both already in frame, night loft",
        "medium loft desk, Ke and the tablet both visible from first frame, no new subjects",
    ]
    wides = [
        "wide small night loft with one desk and a closed window, establishing only, locked-off, original production background",
        "wide desk clutter and the tablet glow, no endless sunset sky, no zoom",
    ]
    shots = []
    for i in range(N_SHOTS):
        line = DIALOGUE.get(i)
        if line:
            size = "CU" if i % 2 else "MCU"
            prompt = acting[i % len(acting)]
            camera = "Static Shot"
        elif i % 6 == 0:
            size = "wide"
            prompt = wides[i % len(wides)]
            camera = "Static Shot" if i % 12 else "Slow Pan"
        else:
            size = "medium"
            prompt = acting[(i + 3) % len(acting)]
            camera = "Static Shot"
        shots.append(
            {
                "id": f"E01-{i + 1:02d}",
                "seq": i + 1,
                "duration": SHOT_SECONDS,
                "scene_id": f"{DEFAULT_EPISODE}-sc01",
                "location_id": LOC,
                "interior_id": INT,
                "h3_mode": "ref2va",
                "refs": ["char_ke_sheet", "plate_loft"] if size != "wide" else ["plate_loft"],
                "plate_id": "plate_loft",
                "costume_ids": {KEEPER: "char_ke_sheet"},
                "first_frame_prompt": prompt + f", shot {i + 1} of {N_SHOTS}, subject already in frame",
                "h3_prompt": prompt + f", locked-off camera, Static Shot, no zoom, no new subjects appearing, {H3_GEN_WIDTH}x{H3_GEN_HEIGHT}",
                "line": line,
                "cuts": [
                    {
                        "seq": 1,
                        "beats": [1, 1],
                        "seconds": SHOT_SECONDS,
                        "size": size,
                        "camera": camera,
                        "characters": [KEEPER],
                    }
                ],
            }
        )
    return shots


def episode_seconds(shots: list[dict[str, Any]] | None = None) -> float:
    return sum(float(s["duration"]) for s in (shots or shot_plan()))


def speech_like_wav(seconds: float = 6.0, sample_rate: int = 16000) -> bytes:
    """A voiced vowel-ish clip for CosyVoice2 reference upload. Not digital silence."""
    n = max(int(seconds * sample_rate), 1)
    frames = bytearray()
    for i in range(n):
        t = i / sample_rate
        env = min(1.0, t / 0.04) * min(1.0, (seconds - t) / 0.08)
        # F0 ~ 120 Hz + slight vibrato; first two formants of /a/
        f0 = 120.0 + 3.0 * math.sin(2 * math.pi * 5.0 * t)
        s = 0.35 * math.sin(2 * math.pi * f0 * t)
        s += 0.18 * math.sin(2 * math.pi * 700 * t)
        s += 0.10 * math.sin(2 * math.pi * 1200 * t)
        v = max(-1.0, min(1.0, s * env))
        frames += struct.pack("<h", int(v * 20000))
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


class ControlPlane:
    def __init__(self, base: str, user: str, password: str):
        self.base = base.rstrip("/")
        self.cookie = ""
        self.user = user
        self.password = password

    def _open(self, req: urllib.request.Request):
        req.add_header("User-Agent", "AnimeFactoryPM/1.0")
        return urllib.request.urlopen(req, timeout=60)

    def login(self) -> None:
        data = json.dumps({"user": self.user, "password": self.password}).encode()
        req = urllib.request.Request(
            f"{self.base}/login",
            data=data,
            headers={"content-type": "application/json", "User-Agent": "AnimeFactoryPM/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.headers.get("Set-Cookie") or ""
            self.cookie = raw.split(";", 1)[0]
            if not self.cookie:
                raise RuntimeError("login did not set session cookie")

    def _json(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"content-type": "application/json", "User-Agent": "AnimeFactoryPM/1.0"}
        if self.cookie:
            headers["cookie"] = self.cookie
        url = self.base + urllib.parse.quote(path, safe="/?&=%")
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            err = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"{method} {path} -> {exc.code}: {err}") from exc

    def create_project(
        self,
        title: str,
        logline: str,
        langs: Sequence[str] | None = None,
        start: bool = False,
    ) -> dict:
        return self._json(
            "POST",
            "/api/stories",
            {
                "title": title,
                "logline": logline,
                "langs": list(normalize_langs(langs)),
                "target_duration_s": TARGET_EPISODE_SECONDS,
                "episode_count": 1,
                "world_mode": "fiction",
                "start": start,
            },
        )

    def job(self, story_id: str, stage: str, status: str, error: str | None = None, episode_code: str = DEFAULT_EPISODE) -> dict:
        return self._json(
            "POST",
            f"/api/stories/{story_id}/jobs",
            {"stage": stage, "status": status, "episode_code": episode_code, "error": error},
        )

    def action(self, story_id: str, action: str, episode_code: str | None = None) -> dict:
        payload: dict[str, Any] = {}
        if episode_code:
            payload["episode_code"] = episode_code
        return self._json("POST", f"/api/stories/{story_id}/{action}", payload)

    def detail(self, story_id: str) -> dict:
        return self._json("GET", f"/api/stories/{story_id}")


def _lock_voice(
    conn: sqlite3.Connection,
    character_id: str,
    langs: Sequence[str] | None = None,
    uri_for: dict[str, str] | None = None,
    *,
    gender: str | None = None,
    migrate: bool = False,
) -> None:
    """Lock voice_uri per lang, preserving an established (non-empty) URI.

    Recomputing `uri_for` from the same character every episode is
    deterministic, but a lightly-edited identity string must not silently
    reassign an already-locked speaker. `resolve_locked_voice_uri` keeps
    whatever is already on disk unless `migrate=True` (an intentional
    voice-profile change), in which case `lock_version` is bumped so stale
    line wavs (fingerprinted against the old version) invalidate.
    """
    for lang in normalize_langs(langs):
        candidate = (uri_for or {}).get(lang) or f"voice://{character_id}/{lang}"
        uri, lock_version = resolve_locked_voice_uri(
            conn, character_id, lang, candidate, migrate=migrate
        )
        fp = profile_fingerprint(character_id, str(gender or ""), uri, TTS_MODEL, lock_version)
        try:
            conn.execute(
                """
                INSERT INTO character_voice (character_id, lang, voice_uri, reference_audio, speed,
                    emotion, version, lock_version, profile_fingerprint)
                VALUES (?, ?, ?, ?, 1.0, 'neutral', 'v1', ?, ?)
                ON CONFLICT(character_id, lang) DO UPDATE SET
                    voice_uri = excluded.voice_uri,
                    lock_version = excluded.lock_version,
                    profile_fingerprint = excluded.profile_fingerprint
                """,
                (character_id, lang, uri, f"assets/voice/{character_id}.{lang}.wav", lock_version, fp),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"story.sqlite rejects lang {lang!r}: character_voice.CHECK still hardcodes "
                f"('zh','en','ja') in schema/story_sqlite.sql"
            ) from exc
    conn.commit()


def init_demo_story(conn: sqlite3.Connection, story_id: str, root: Path) -> dict:
    world = init_fiction_world(
        conn,
        story_id,
        TITLE,
        locations=[
            {"id": LOC, "name": "夜窗出租屋", "aka": [], "type": "landmark"},
        ],
        edges=[],
        interiors=[
            {
                "id": INT,
                "name": "工作台",
                "parent": LOC,
                "scene_id": "loft",
                "asset_path": "assets/scenes/loft/plate_base.png",
            }
        ],
        timeline=[
            {
                "id": "ev_pre",
                "date": "Y1",
                "title": "平板第一次自己亮起来",
                "kind": "fictional",
                "episode": None,
                "sources": [],
            }
        ],
        root=root,
    )
    conn.execute(
        """
        INSERT INTO characters (id, name, identity_prompt, age, alive, seed, current_location_id, gender)
        VALUES (?, ?, ?, 24, 1, 11, ?, 'male')
        ON CONFLICT(id) DO UPDATE SET identity_prompt = excluded.identity_prompt,
            gender = COALESCE(characters.gender, excluded.gender)
        """,
        (
            KEEPER,
            "阿柯",
            "1boy, young East Asian hacker, short messy black hair, dark hoodie, quiet eyes, "
            "night loft, character design sheet",
            LOC,
        ),
    )
    conn.execute(
        """
        INSERT INTO props (id, name, identity_prompt, state)
        VALUES (?, ?, ?, 'intact')
        ON CONFLICT(id) DO UPDATE SET identity_prompt = excluded.identity_prompt
        """,
        ("tablet", "黑客平板", "matte dark tablet with a thin bezel, faint screen glow, prop design sheet, no people"),
    )
    conn.commit()
    write_bible(
        root,
        world_md=f"# {TITLE}\n\ninvented: true\n\n{LOGLINE}\n",
        style_md=None,
        cast={
            "characters": [{"id": KEEPER, "name": "阿柯"}],
            "props": [{"id": "tablet", "name": "黑客平板"}],
        },
        period_md="# Period\n\n## 正面清单\n\n## 负面清单\n",
    )
    return world


# ---------------------------------------------------------------------------
# Real path: this project's own title/logline → DeepSeek → gated script → board
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSpec:
    title: str
    logline: str
    langs: tuple[str, ...]
    episode_code: str = DEFAULT_EPISODE
    target_seconds: float = TARGET_EPISODE_SECONDS
    shot_seconds: float = H3_MAX_SECONDS
    world_mode: str = "fiction"
    kind: str = "series"
    notes: list[str] = field(default_factory=list)


def _as_lang_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return None


def resolve_spec(
    conn: sqlite3.Connection,
    story_id: str,
    *,
    episode_code: str,
    control: "ControlPlane | None" = None,
    title: str | None = None,
    logline: str | None = None,
    langs: Sequence[str] | None = None,
) -> EpisodeSpec:
    """Project identity: explicit argument, then the control plane, then story.sqlite."""
    remote: dict[str, Any] = {}
    if control is not None and not (title and logline):
        try:
            detail = control.detail(story_id) or {}
        except Exception:  # noqa: BLE001 — PM lookup must not kill the engine
            detail = {}
        candidate = detail.get("story") if isinstance(detail.get("story"), dict) else detail
        remote = candidate if isinstance(candidate, dict) else {}
    row = conn.execute("SELECT title FROM story WHERE id = ?", (story_id,)).fetchone()
    resolved_title = str(title or remote.get("title") or (row["title"] if row else "") or TITLE).strip()
    resolved_logline = str(logline or remote.get("logline") or "").strip()
    if not resolved_logline:
        resolved_logline = LOGLINE if resolved_title == TITLE else resolved_title
    resolved_langs = _as_lang_list(langs) or _as_lang_list(remote.get("langs")) or list(story_langs(conn, story_id))
    kind = normalize_story_kind(remote.get("kind") or remote.get("story_kind"))
    if kind == "short":
        target = float(remote.get("target_duration_s") or DEFAULT_SHORT_SECONDS)
        if float(remote.get("target_duration_s") or 0) >= TARGET_EPISODE_SECONDS * 0.9:
            # Control plane still sending the series default — do not pad a short to 600s.
            target = float(DEFAULT_SHORT_SECONDS)
    else:
        target = float(remote.get("target_duration_s") or TARGET_EPISODE_SECONDS)
    # config.target_shot_seconds defaults to the 5.5s dev placeholder; the 600s episode is
    # costed at 8.0s per H3 call, so only an explicit env override shortens shots here.
    override = (os.environ.get("TARGET_SHOT_SECONDS") or "").strip()
    line_max = max_seconds_for_backend()
    return EpisodeSpec(
        title=resolved_title,
        logline=resolved_logline,
        langs=normalize_langs(resolved_langs),
        episode_code=episode_code,
        target_seconds=target,
        shot_seconds=min(float(override), line_max) if override else line_max,
        kind=kind,
    )


def init_story_shell(conn: sqlite3.Connection, story_id: str, spec: EpisodeSpec) -> None:
    """Story row only. The bible waits for the draft; a sequel must not have its world blanked."""
    init_fiction_world(
        conn,
        story_id,
        spec.title,
        locations=[],
        edges=[],
        interiors=[],
        timeline=[],
        root=None,
        langs=spec.langs,
    )


def apply_script_world(
    conn: sqlite3.Connection,
    story_id: str,
    script: dict,
    spec: EpisodeSpec,
    root: Path,
) -> None:
    """Land the draft's cast and places in canon so gate2/gate3 have something to check against."""
    for loc in script.get("locations") or []:
        conn.execute(
            """
            INSERT INTO locations (id, name, aka_json, type, parent_id)
            VALUES (?, ?, '[]', ?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name, type = excluded.type
            """,
            (loc["id"], loc.get("name") or loc["id"], loc.get("type") or "settlement", loc.get("parent")),
        )
    for interior in script.get("interiors") or []:
        conn.execute(
            """
            INSERT INTO geo_interiors (id, name, parent, scene_id, asset_path)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name, parent = excluded.parent,
                scene_id = excluded.scene_id, asset_path = excluded.asset_path
            """,
            (
                interior["id"],
                interior.get("name"),
                interior.get("parent"),
                interior.get("scene_id"),
                interior_asset_path(interior),
            ),
        )
    first_location = next(iter(script.get("locations") or []), {}).get("id")
    for char in script.get("cast") or []:
        identity = str(char.get("identity_prompt") or "")
        try:
            gender = resolve_gender(
                str(char["id"]),
                gender=char.get("gender"),
                identity=identity,
                name=char.get("name"),
            )
        except GenderRequiredError:
            # Persist what we have; voice-lock (produce_episode) is where an
            # unresolved gender actually blocks the episode, not script ingest.
            gender = None
        conn.execute(
            """
            INSERT INTO characters (id, name, identity_prompt, age, alive, current_location_id, gender)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET identity_prompt = excluded.identity_prompt,
                name = excluded.name,
                gender = COALESCE(excluded.gender, characters.gender)
            """,
            (
                char["id"],
                char.get("name") or char["id"],
                char.get("identity_prompt"),
                int(char.get("age") or 25),
                first_location,
                gender,
            ),
        )
    for prop in script.get("props") or []:
        conn.execute(
            """
            INSERT INTO props (id, name, identity_prompt, state)
            VALUES (?, ?, ?, 'intact')
            ON CONFLICT(id) DO UPDATE SET identity_prompt = excluded.identity_prompt
            """,
            (prop["id"], prop.get("name") or prop["id"], prop.get("identity_prompt")),
        )
    for event in script.get("events") or []:
        conn.execute(
            """
            INSERT INTO timeline_events (id, date, title, kind, episode, sources_json, invented)
            VALUES (?, ?, ?, 'fictional', ?, '[]', 1)
            ON CONFLICT(id) DO UPDATE SET title = excluded.title, episode = excluded.episode
            """,
            (event["id"], event.get("date") or "Y1", event.get("title"), spec.episode_code),
        )
    conn.commit()
    geo = export_geo(conn, story_id, invented=True, sources=[])
    timeline = export_timeline(conn, story_id)
    continuity = export_continuity(conn, story_id)
    write_canon_dir(root, geo, timeline, continuity)
    write_bible(
        root,
        world_md=f"# {spec.title}\n\ninvented: true\n\n{spec.logline}\n\n## {spec.episode_code}\n\n{script.get('synopsis') or ''}\n",
        style_md=f"{STYLE_PREFIX}\n\nnegative: {FIXED_NEGATIVE}\n",
        cast={
            "characters": [{"id": c["id"], "name": c.get("name")} for c in script.get("cast") or []],
            "props": [{"id": p["id"], "name": p.get("name")} for p in script.get("props") or []],
        },
        period_md="# Period\n\n## 正面清单\n\n## 负面清单\n",
    )


def _continuity_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) AS v FROM continuity").fetchone()
    return int((row["v"] if row and row["v"] is not None else 0))


def seed_first_continuity(conn: sqlite3.Connection, script: dict) -> None:
    """EP000 exists once, from the first draft's own cast. Later episodes chain off their predecessor."""
    scenes = script.get("scenes") or []
    opening = scenes[0].get("location_id") if scenes else None
    events = [str(ev["id"]) for ev in script.get("events") or [] if ev.get("id")]
    payload = {
        "characters": [{"id": c["id"], "alive": True, "knows": events} for c in script.get("cast") or []],
        "positions": {c["id"]: opening for c in script.get("cast") or [] if opening},
        "props": [{"id": p["id"], "state": "intact"} for p in script.get("props") or []],
        "locations": [{"id": l["id"], "name": l.get("name")} for l in script.get("locations") or []],
        "open_threads": [],
    }
    save_continuity(conn, "EP000", 1, payload)


def continuity_after_episode(script: dict, previous: dict) -> dict:
    """Where this episode leaves everyone, so the next episode's gate1 can inject it."""
    scenes = script.get("scenes") or []
    closing = scenes[-1].get("location_id") if scenes else None
    known = {str(c["id"]): set(c.get("knows") or []) for c in previous.get("characters") or []}
    for event in script.get("events") or []:
        for cid in known:
            known[cid].add(str(event["id"]))
    for char in script.get("cast") or []:
        known.setdefault(str(char["id"]), {str(ev["id"]) for ev in script.get("events") or []})
    positions = dict(previous.get("positions") or {})
    for scene in scenes:
        for cid in scene.get("characters") or []:
            positions[str(cid)] = scene.get("location_id") or positions.get(str(cid))
    for cid in known:
        positions.setdefault(cid, closing)
    props = {str(p["id"]): dict(p) for p in previous.get("props") or []}
    for prop in script.get("props") or []:
        props.setdefault(str(prop["id"]), {"id": prop["id"], "state": "intact"})
    locations = {str(l["id"]): dict(l) for l in previous.get("locations") or []}
    for loc in script.get("locations") or []:
        locations[str(loc["id"])] = {"id": loc["id"], "name": loc.get("name")}
    return {
        "characters": [{"id": cid, "alive": True, "knows": sorted(events)} for cid, events in sorted(known.items())],
        "positions": positions,
        "props": list(props.values()),
        "locations": list(locations.values()),
        "open_threads": previous.get("open_threads") or [],
        "recap": script.get("recap") or "",
    }


def assert_episode_duration(shots: list[dict], target_seconds: float) -> None:
    total = sum(float(s.get("duration") or 0) for s in shots)
    if abs(total - target_seconds) > target_seconds * DURATION_TOLERANCE:
        raise ValueError(
            f"board totals {total:.1f}s, target {target_seconds:.0f}s "
            f"(±{DURATION_TOLERANCE * 100:.0f}%); dialogue volume and shot budget disagree"
        )


def draft_and_gate(
    conn: sqlite3.Connection,
    story_id: str,
    spec: EpisodeSpec,
    root: Path,
    *,
    llm: Any | None,
    max_rewrites: int | None = None,
) -> tuple[dict, Any, dict, dict]:
    """Draft, land the world, run gate2. A blocked draft is re-prompted with its own violations."""
    limit = load_settings().continuity_max_rewrites if max_rewrites is None else max_rewrites
    drafts: list[dict] = []
    notes = ""
    gated = None
    script: dict = {}
    geo: dict = {}
    timeline: dict = {}
    repaired_staging = False
    while len(drafts) <= limit:
        injected = gate1_inject(conn, spec.episode_code, export_geo(conn, story_id, invented=True, sources=[]))
        if repaired_staging and drafts:
            script = repair_collapsed_staging(drafts[-1])
        else:
            script = draft_episode_script(
                episode_code=spec.episode_code,
                title=spec.title,
                logline=spec.logline,
                langs=spec.langs,
                injected=injected,
                target_seconds=spec.target_seconds,
                shot_seconds=spec.shot_seconds,
                llm=llm,
                notes=notes,
            )
        apply_script_world(conn, story_id, script, spec, root)
        geo = export_geo(conn, story_id, invented=True, sources=[])
        timeline = export_timeline(conn, story_id)
        if _continuity_version(conn) == 0:
            seed_first_continuity(conn, script)
        continuity = export_continuity(conn, story_id)
        drafts.append(script)
        gated = produce_script(conn, spec.episode_code, drafts, geo, timeline, continuity, spec.world_mode)
        if gated.ok:
            script = drafts[gated.rewrites]
            break
        notes = "\n".join(f"- [{v.code}] {v.message}" for v in gated.violations)
        staging_fail = any(v.code == "staging_collapse" for v in gated.violations)
        if staging_fail and not repaired_staging:
            repaired_staging = True
            continue
        if llm is None:
            break
        repaired_staging = False
    return script, gated, geo, timeline


@dataclass
class Prepared:
    script: dict
    shots: list[dict[str, Any]]
    geo: dict
    timeline: dict
    gated: Any
    characters: list[dict]
    locations: list[dict]
    props: list[dict]
    interiors: list[dict]


def _prepare_demo(conn: sqlite3.Connection, story_id: str, root: Path, spec: EpisodeSpec) -> Prepared:
    """The 黑客平板 fixture: fixed cast, fixed 75 shots, fixed dialogue."""
    ep = spec.episode_code
    init_demo_story(conn, story_id, root)
    shots = shot_plan()
    for shot in shots:
        sid = str(shot.get("id") or "")
        if sid.startswith("EP001-") or sid.startswith("E01-"):
            shot["id"] = sid.replace("EP001-", f"{ep}-", 1).replace("E01-", f"{ep}-", 1)
    lines_payload = [
        {
            "id": shot["id"],
            "character_id": KEEPER,
            "text": shot["line"]["zh"],
            "mentions_event_ids": ["ev_pre"],
        }
        for shot in shots
        if shot.get("line")
    ]
    script = {
        "title": ep,
        "synopsis": LOGLINE,
        "scenes": [
            {
                "id": f"{ep}-sc01",
                "location_id": LOC,
                "interior_id": INT,
                "interval_from_prev": "0h0m",
                "characters": [KEEPER],
                "lines": lines_payload,
            }
        ],
    }
    save_continuity(
        conn,
        "EP000",
        1,
        {
            "characters": [{"id": KEEPER, "alive": True, "knows": ["ev_pre"]}],
            "positions": {KEEPER: LOC},
            "props": [{"id": "tablet", "state": "intact"}],
            "locations": [],
            "open_threads": [],
        },
    )
    geo = export_geo(conn, story_id, invented=True, sources=[])
    timeline = export_timeline(conn, story_id)
    continuity = export_continuity(conn, story_id, "EP000")
    gated = produce_script(conn, ep, [script], geo, timeline, continuity, spec.world_mode)
    return Prepared(
        script=script,
        shots=shots,
        geo=geo,
        timeline=timeline,
        gated=gated,
        characters=[
            {
                "id": KEEPER,
                "name": "阿柯",
                "identity_prompt": "1boy, young East Asian hacker, short messy black hair, dark hoodie, quiet eyes, character design sheet",
                "gender": "male",
                "seed": 11,
            }
        ],
        locations=[
            {
                "id": LOC,
                "name": "夜窗出租屋",
                "plate_prompt": "small night loft with one desk, closed window, original production location art, no people, no text, not a famous movie still",
                "seed": 21,
            }
        ],
        props=[
            {
                "id": "tablet",
                "name": "黑客平板",
                "identity_prompt": "matte dark tablet, thin bezel, prop design sheet",
                "seed": 31,
            }
        ],
        interiors=[
            {
                "id": INT,
                "name": "工作台",
                "scene_id": "loft",
                "asset_path": "assets/scenes/loft/plate_base.png",
                "plate_prompt": "loft desk interior, tablet on the desk, original production art, empty of extra people, no text",
            }
        ],
    )


def _prepare_real(
    conn: sqlite3.Connection,
    story_id: str,
    root: Path,
    spec: EpisodeSpec,
    llm: Any | None,
) -> Prepared:
    init_story_shell(conn, story_id, spec)
    script, gated, geo, timeline = draft_and_gate(conn, story_id, spec, root, llm=llm)
    if not gated.ok:
        return Prepared(script, [], geo, timeline, gated, [], [], [], [])
    notes = ""
    shots: list[dict[str, Any]] = []
    board_attempts = 0
    limit = load_settings().continuity_max_rewrites
    while board_attempts <= limit:
        try:
            candidate = repair_collapsed_staging(script) if notes else script
            shots = board_from_script(
                candidate,
                episode_code=spec.episode_code,
                langs=spec.langs,
                target_seconds=spec.target_seconds,
                shot_seconds=spec.shot_seconds,
                kind=spec.kind,
                video_backend=select_video_backend(),
            )
            script = candidate
            break
        except (PromptCollapseError, SpeakerFrameError) as exc:
            notes = str(exc)
            board_attempts += 1
            script = repair_collapsed_staging(script)
            if board_attempts > limit:
                blocked = GateResult(
                    ok=False,
                    status="blocked",
                    violations=[GateViolation("board_collapse", notes, "visual")],
                )
                blocked.block()
                return Prepared(script, [], geo, timeline, blocked, [], [], [], [])
            if llm is not None and board_attempts <= limit:
                injected = gate1_inject(
                    conn, spec.episode_code, export_geo(conn, story_id, invented=True, sources=[])
                )
                script = draft_episode_script(
                    episode_code=spec.episode_code,
                    title=spec.title,
                    logline=spec.logline,
                    langs=spec.langs,
                    injected=injected,
                    target_seconds=spec.target_seconds,
                    shot_seconds=spec.shot_seconds,
                    llm=llm,
                    notes=f"Board QC failed, rewrite staging and split dramatic beats:\n- {notes}",
                )
                apply_script_world(conn, story_id, script, spec, root)
                geo = export_geo(conn, story_id, invented=True, sources=[])
                timeline = export_timeline(conn, story_id)
                continuity = export_continuity(conn, story_id)
                gated = produce_script(
                    conn, spec.episode_code, [script], geo, timeline, continuity, spec.world_mode
                )
                if not gated.ok:
                    return Prepared(script, [], geo, timeline, gated, [], [], [], [])
            notes = "repair"
    return Prepared(
        script=script,
        shots=shots,
        geo=geo,
        timeline=timeline,
        gated=gated,
        characters=list(script.get("cast") or []),
        locations=list(script.get("locations") or []),
        props=list(script.get("props") or []),
        interiors=list(script.get("interiors") or []),
    )


def _shot_line_text(line: dict, lang: str, primary: str) -> str:
    return str(line.get(lang) or line.get(primary) or line.get("text") or "").strip()


def ensure_line_wavs(
    conn: sqlite3.Connection,
    tts: "CosyVoiceClient",
    audio_root: Path,
    shots: Sequence[dict],
    langs: Sequence[str],
    primary: str,
    resolved_gender: dict[str, str],
) -> dict[str, list[str]]:
    """Synthesize every board line that is not already reusable on disk.

    Reuse is gated on `line_wav_reusable`: the on-disk wav's recorded
    fingerprint must exactly match the expected rich local fingerprint
    (character, gender, locked voice, model, speed, lang, exact cleaned text,
    lock_version). Hosted pre-GPU wavs qualify only after
    `tts_hosted.import_hosted_line_audio` verified and recorded them —
    everything else is (re-)synthesized.
    """
    reused: list[str] = []
    synthesized: list[str] = []
    for shot in shots:
        line = shot.get("line")
        if not line:
            continue
        sid = str(shot.get("id") or "")
        cid = str(shot.get("character_id") or KEEPER)
        gender = resolved_gender.get(cid) or ""
        for lang in langs:
            dest = audio_root / f"{shot['id']}.{lang}.wav"
            text = _shot_line_text(line, lang, primary)
            stripped = strip_stage_directions(text)
            if stripped != text:
                line[lang] = stripped
                shot["line"] = line
            if not stripped:
                continue
            lock = existing_voice_lock(conn, cid, lang)
            voice_uri = str(lock["voice_uri"]) if lock and lock["voice_uri"] else ""
            speed = float(lock["speed"] or 1.0) if lock else 1.0
            lock_version = int(lock["lock_version"] or 1) if lock else 1
            expected_fp = line_fingerprint(
                cid, gender, voice_uri, TTS_MODEL, speed, lang, stripped, lock_version
            )
            if sid and line_wav_reusable(conn, sid, lang, expected_fp, dest):
                reused.append(f"{sid}.{lang}")
                continue
            if dest.is_file() and dest.stat().st_size > 100 and not sid:
                # No segment id to fingerprint against (legacy caller path);
                # fall back to the old "file already exists" skip.
                continue
            audio, duration = synthesize_line(
                tts,
                conn,
                cid,
                lang,
                stripped,
                target_seconds=float(shot.get("duration") or 0) or None,
            )
            dest.write_bytes(audio)
            if sid:
                record_line_audio(
                    conn,
                    sid,
                    lang,
                    cid,
                    expected_fp,
                    str(dest),
                    duration or pcm_duration_seconds(audio),
                    24000,
                    wav_sha256(audio),
                )
                synthesized.append(f"{sid}.{lang}")
    return {"reused": reused, "synthesized": synthesized}


def produce_episode(
    story_id: str,
    root: Path,
    *,
    control: ControlPlane | None = None,
    live: bool = False,
    upload: bool = True,
    lease_gpu: bool = False,
    episode_code: str = DEFAULT_EPISODE,
    title: str | None = None,
    logline: str | None = None,
    langs: Sequence[str] | None = None,
    template_demo: bool = False,
    live_script: bool | None = None,
    until_stage: str | None = None,
) -> dict[str, Any]:
    """Pre-GPU stages for one episode of this project. GPU is opt-in and one card only.

    until_stage='board' stops after CosyVoice voice-lock + line wavs + board.
    Picture mix waits for compose. GPU Flux stills can run while H3 weights download.
    """
    load_dotenv()
    settings = load_settings()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ep = (episode_code or DEFAULT_EPISODE).strip().upper()
    if not ep.startswith("EP"):
        ep = f"EP{int(ep):03d}" if ep.isdigit() else DEFAULT_EPISODE
    conn = open_db(root / "story.sqlite")
    migrate(conn)
    status: dict[str, str] = {}

    def mark(stage: str, st: str, error: str | None = None) -> None:
        status[stage] = st
        print(json.dumps({"stage": stage, "status": st, "episode": ep, "error": error}, ensure_ascii=False), flush=True)
        if control:
            try:
                control.job(story_id, stage, st, error, episode_code=ep)
            except Exception as exc:  # noqa: BLE001 — PM sync must not kill the engine
                status[f"{stage}_sync"] = f"error:{exc}"

    spec = resolve_spec(
        conn,
        story_id,
        episode_code=ep,
        control=control,
        title=title,
        logline=logline,
        langs=langs,
    )
    primary = spec.langs[0]
    use_llm = settings.live_llm if live_script is None else live_script
    if live and not template_demo:
        use_llm = True
    llm = client_from_env() if use_llm else None
    try:
        if template_demo:
            prepared = _prepare_demo(conn, story_id, root, spec)
        else:
            if live and llm is None:
                raise RuntimeError("empty preprod script: director LLM missing")
            prepared = _prepare_real(conn, story_id, root, spec, llm)
            n_lines = sum(len(sc.get("lines") or []) for sc in (prepared.script or {}).get("scenes") or [])
            if n_lines < 1 and not (prepared.shots or []):
                raise RuntimeError("empty preprod script")
    except Exception as exc:  # noqa: BLE001 — empty draft must fail bible, not skip to GPU
        mark("bible", "failed", f"{type(exc).__name__}:{exc}")
        return {
            "story_id": story_id,
            "episode_code": ep,
            "status": status,
            "blocked": True,
            "error": str(exc),
        }
    mark("bible", "succeeded")
    mark("canon", "succeeded")

    gated = prepared.gated
    if not gated.ok:
        mark("script", "blocked", ";".join(v.message for v in gated.violations))
        return {
            "story_id": story_id,
            "episode_code": ep,
            "status": status,
            "blocked": True,
            "violations": [v.message for v in gated.violations],
        }
    mark("script", "succeeded")

    script = prepared.script
    shots = prepared.shots
    try:
        assert_shots_have_dialogue(shots)
        if spec.kind == "series":
            assert_episode_duration(shots, spec.target_seconds)
    except Exception as exc:  # noqa: BLE001 — a thin episode must not reach TTS
        mark("script", "blocked", f"{type(exc).__name__}:{exc}")
        return {"story_id": story_id, "episode_code": ep, "status": status, "blocked": True, "error": str(exc)}

    keys = siliconflow_keys() if live else ["test-key"]
    # Voice URIs are per-key on SiliconFlow; do not rotate keys after upload.
    tts = CosyVoiceClient(keys[:1] if keys else ["test-key"], live=live)
    audio_root = root / "episodes" / ep / "audio" / "lines"
    audio_root.mkdir(parents=True, exist_ok=True)
    speakers = sorted({str(s["character_id"]) for s in shots if s.get("character_id")}) or [KEEPER]
    cast_by_id = {str(c.get("id")): c for c in (script.get("cast") or []) if c.get("id")}
    voice_migrate = str(os.environ.get("ANIME_FACTORY_VOICE_MIGRATE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    resolved_gender: dict[str, str] = {}
    # Hosted pre-GPU TTS (control-plane preprod) may already have written
    # line wavs + tts_manifest.json under this episode. Bridge them in instead
    # of re-synthesizing: seed voice locks first (an established lock always
    # wins), then verify + import line rows after locks are final.
    hosted_manifest = load_hosted_tts_manifest(root, ep)
    try:
        for cid in speakers:
            char = cast_by_id.get(cid) or {}
            row = conn.execute(
                "SELECT name, age, identity_prompt, gender FROM characters WHERE id = ?", (cid,)
            ).fetchone()
            identity = str(char.get("identity_prompt") or (row["identity_prompt"] if row else "") or "")
            age = char.get("age") if char.get("age") is not None else (row["age"] if row else None)
            nm = str(char.get("name") or (row["name"] if row else cid))
            cast_or_db_gender = char.get("gender") or (row["gender"] if row else None)
            gender = resolve_gender(cid, gender=cast_or_db_gender, identity=identity, name=nm)
            resolved_gender[cid] = gender
            conn.execute("UPDATE characters SET gender = ? WHERE id = ?", (gender, cid))
            if hosted_manifest is not None:
                hosted_voice = hosted_voice_for(hosted_manifest, cid, shots, spec.langs)
                if hosted_voice:
                    seed_hosted_voice_lock(conn, cid, hosted_voice, spec.langs, gender=gender)
            uris = {
                lang: stock_voice_uri(
                    cid,
                    lang,
                    gender=gender,
                    age=age,
                    identity=identity,
                    name=nm,
                )
                for lang in spec.langs
            }
            assert_locked_voices(
                uris,
                cid,
                gender=gender,
                identity=identity,
                name=nm,
            )
            _lock_voice(conn, cid, spec.langs, uris, gender=gender, migrate=voice_migrate)
        conn.commit()
    except VoiceGenderError as exc:
        mark("tts", "blocked", f"voice:{exc}")
        return {"story_id": story_id, "episode_code": ep, "status": status, "blocked": True, "error": str(exc)}
    try:
        if hosted_manifest is not None:
            import_hosted_line_audio(
                conn,
                root,
                ep,
                shots,
                spec.langs,
                resolved_gender=resolved_gender,
                manifest=hosted_manifest,
                primary=primary,
            )
        ensure_line_wavs(conn, tts, audio_root, shots, spec.langs, primary, resolved_gender)
    except (SpeechTextError, LineDurationError) as exc:
        mark("tts", "blocked", f"speech:{exc}")
        return {"story_id": story_id, "status": status, "blocked": True, "error": str(exc)}
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", "replace")[:400]
        mark("tts", "blocked", f"speech:{exc.code}:{err}")
        return {"story_id": story_id, "status": status, "blocked": True, "error": err}
    work = root / "episodes" / ep
    mark("tts", "succeeded")

    totals = board_totals(shots)
    try:
        segments = persist_board(conn, ep, shots, kind=spec.kind)
    except (PromptCollapseError, SpeakerFrameError) as exc:
        mark("board", "blocked", f"{type(exc).__name__}:{exc}")
        return {"story_id": story_id, "episode_code": ep, "status": status, "blocked": True, "error": str(exc)}
    (work / "board.json").write_text(
        json.dumps(
            {
                "episode_code": ep,
                "kind": spec.kind,
                "title": spec.title,
                "logline": spec.logline,
                "synopsis": script.get("synopsis"),
                "langs": list(spec.langs),
                "drafted_by": script.get("drafted_by") or "template_demo",
                "target_s": spec.target_seconds,
                "shots": shots,
                "segments": segments,
                **totals,
                "n_segments": len(segments),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_episode_srts(work, ep, shots, langs=spec.langs)
    mark("board", "succeeded")

    if (until_stage or "").strip().lower() == "board":
        uploads = []
        if upload:
            for rel in ("bible", "canon", "episodes"):
                uploads.extend(upload_tree(story_id, root, rel))
            checkpoint_and_upload(conn, root / "story.sqlite")
        return {
            "story_id": story_id,
            "episode_code": ep,
            "title": spec.title,
            "logline": spec.logline,
            "langs": list(spec.langs),
            "drafted_by": script.get("drafted_by") or "template_demo",
            "template_demo": template_demo,
            "status": status,
            "blocked": False,
            "n_shots": totals["n_shots"],
            "n_spoken": totals["n_spoken"],
            "duration_s": totals["total_s"],
            "target_s": spec.target_seconds,
            "script_rewrites": getattr(gated, "rewrites", 0),
            "live": live,
            "uploads_ok": sum(1 for u in uploads if u.get("ok")),
            "gpu": {"skipped": True, "reason": "until_board"},
            "until_stage": "board",
            "vast_dry_run": settings.vast_dry_run,
            "root": str(root),
        }

    kolors = KolorsClient(keys, live=live, min_interval_s=30.0 if live else 0)
    lib = generate_asset_library(
        conn,
        story_id,
        prepared.characters,
        prepared.locations,
        prepared.props,
        kolors,
        None,
        spec.world_mode,
        root,
        interiors=prepared.interiors,
        skip_existing=True,
    )
    mark("design", "succeeded")
    specs = lib["specs"]
    assets_index = {"items": [{"id": k, **v} for k, v in specs.items()]}
    from anime_factory.directors.common import is_chain_head

    units = expand_shots_to_segments(shots, max_s=max_seconds_for_backend())
    for unit in units:
        if not is_chain_head(unit):
            continue
        ensure_keyframe(
            conn, story_id, ep, unit, prepared.geo, assets_index, specs, kolors, None, spec.world_mode, root
        )
    assert_keyframe_files(root, ep, units, assets_index)
    mark("keyframe", "succeeded")

    if not template_demo:
        previous = export_continuity(conn, story_id)
        save_continuity(conn, ep, _continuity_version(conn) + 1, continuity_after_episode(script, previous))

    uploads = []
    if upload:
        for rel in ("bible", "canon", "assets", "episodes"):
            uploads.extend(upload_tree(story_id, root, rel))
        checkpoint_and_upload(conn, root / "story.sqlite")

    gpu: dict[str, Any] = {"skipped": True, "reason": "pre_gpu_only"}
    if lease_gpu:
        gpu = _try_one_lease(status)

    return {
        "story_id": story_id,
        "episode_code": ep,
        "title": spec.title,
        "logline": spec.logline,
        "langs": list(spec.langs),
        "drafted_by": script.get("drafted_by") or "template_demo",
        "template_demo": template_demo,
        "status": status,
        "blocked": False,
        "n_shots": totals["n_shots"],
        "n_spoken": totals["n_spoken"],
        "duration_s": totals["total_s"],
        "target_s": spec.target_seconds,
        "script_rewrites": getattr(gated, "rewrites", 0),
        "live": live,
        "uploads_ok": sum(1 for u in uploads if u.get("ok")),
        "gpu": gpu,
        "vast_dry_run": settings.vast_dry_run,
        "root": str(root),
    }


def _try_one_lease(stage_flags: dict[str, str]) -> dict[str, Any]:
    """One Vast card, only after pre-GPU hosted stages. Do not destroy on false health failures."""
    os.environ["VAST_DRY_RUN"] = "0"
    from gpu_worker.boot import (
        ImageCapabilityError,
        LeaseSession,
        PreLeaseError,
        WHY_MULTI_CARD,
        assert_lease_capabilities,
        assert_pre_lease,
        request_lease,
    )
    from gpu_worker.images import image_capabilities, lease_image_for_backend
    from gpu_worker.offers import offer_sample, pick_one_offer
    from gpu_worker.vast_client import VastClient

    flags = {k: ("passed" if v in {"succeeded", "passed"} else v) for k, v in stage_flags.items()}
    flags["timing"] = flags.get("tts") or flags.get("timing") or ""
    try:
        assert_pre_lease(flags)
    except PreLeaseError as exc:
        return {"skipped": True, "reason": str(exc), "why_multi_card": WHY_MULTI_CARD}

    key = os.environ.get("VAST_API_KEY") or ""
    client = VastClient(api_key=key, dry_run=False)
    lease_image = lease_image_for_backend("skyreels_v3_r2v")
    caps = image_capabilities(lease_image)
    try:
        assert_lease_capabilities(caps)
    except ImageCapabilityError as exc:
        return {
            "skipped": True,
            "put_asks": False,
            "destroy": False,
            "capabilities": caps,
            "blocker": str(exc) + " " + WHY_MULTI_CARD,
        }

    offers_body = client.search_offers()
    offers = list((offers_body.get("offers") or []) if isinstance(offers_body, dict) else [])
    sample = offer_sample(offers, 5)
    chosen = pick_one_offer(offers)
    if not chosen:
        return {
            "skipped": True,
            "put_asks": False,
            "destroy": False,
            "capabilities": caps,
            "offers": sample,
            "reason": "no_24gb_offer",
        }
    offer_id = str(chosen.get("id") or "0")
    env = {
        "CONTROL_PLANE_URL": os.environ.get("CONTROL_PLANE_URL") or "",
        "AF_STORY_ID": os.environ.get("AF_STORY_ID") or "",
        "AF_START_COMFY": "0",
        "ANIME_FACTORY_GPU_STILLS": "1",
        "IMAGE_BACKEND": "flux2_klein4b",
        "VIDEO_BACKEND": "skyreels_v3_r2v",
        "AF_VIDEO_BACKEND": "skyreels_v3_r2v",
        "AF_IMAGE_CAPABILITY": "skyreels_v3_r2v",
        "AF_GPU_PROFILE": "sr3-cu128-sm120",
        "VAST_DRY_RUN": "0",
        "ANIME_FACTORY_LIVE_VAST": "1",
        "VAST_ALLOW_REPLACE": "0",
        "COMFYUI_BASE_URL": "http://127.0.0.1:8199",
        "R2_ENDPOINT": os.environ.get("R2_ENDPOINT") or "",
        "R2_BUCKET": os.environ.get("R2_BUCKET") or "",
        "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID") or "",
        "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY") or "",
        "CLOUDFLARE_TUNNEL_TOKEN": os.environ.get("CLOUDFLARE_TUNNEL_TOKEN") or "",
        "STUDIO_USER": os.environ.get("STUDIO_USER") or "studio",
        "STUDIO_PASSWORD": os.environ.get("STUDIO_PASSWORD") or "",
    }
    # Master vision QC (Qwen3.5-4B) needs SiliconFlow on the GPU box.
    for sf_key in ("SILICONFLOW_API_KEY", "SILICONFLOW_API_KEYS"):
        sf_val = (os.environ.get(sf_key) or "").strip()
        if sf_val:
            env[sf_key] = sf_val
    ds_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if ds_key:
        env["DASHSCOPE_API_KEY"] = ds_key
    th_key = (os.environ.get("TOKENHUB_API_KEY") or os.environ.get("HUNYUAN_API_KEY") or "").strip()
    if th_key:
        env["TOKENHUB_API_KEY"] = th_key
    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    # Blackwell 5090 has no kernels in cu124; pass through so Hub image picks cu128
    # even when torch cannot report sm_120 yet.
    torch_index = (os.environ.get("TORCH_INDEX_URL") or "").strip()
    if torch_index:
        env["TORCH_INDEX_URL"] = torch_index
    tagged = f"{lease_image}:main"
    session = LeaseSession(image=tagged, capabilities=caps, lease_env=env)
    try:
        leased = request_lease(client, offer_id, session, flags)
        leased["lease_image"] = tagged
        leased["offer"] = {
            "id": offer_id,
            "gpu": chosen.get("gpu_name") or chosen.get("gpu"),
            "dph": chosen.get("dph_total") or chosen.get("dph"),
            "gpu_ram": chosen.get("gpu_ram"),
        }
    except Exception as exc:  # noqa: BLE001 — report, do not destroy
        return {
            "skipped": False,
            "error": str(exc),
            "destroy": False,
            "note": "did not destroy; running≠ready; same-instance retry only",
            "capabilities": caps,
            "offers": sample,
            "why_multi_card": WHY_MULTI_CARD,
        }
    leased["capabilities"] = caps
    leased["comfy"] = bool(caps.get("comfy"))
    leased["h3"] = bool(caps.get("h3"))
    leased["kolors"] = bool(caps.get("kolors"))
    leased["offers"] = sample
    leased["destroy"] = False
    leased["why_multi_card"] = WHY_MULTI_CARD
    return leased
