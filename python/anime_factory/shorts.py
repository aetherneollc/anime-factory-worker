"""Vertical shorts cut from a finished episode. Pure library; gpu_worker owns the calls.

The picture is rendered once per window. Each language then gets the same picture
with its own dialogue track and its own burned subtitles, so a five-language
episode is five muxes and one scale/crop pass, not five renders.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from anime_factory.compose import mix_clips, shot_line_text, shot_start_times, write_srt
from anime_factory.langs import normalize_langs, spec_for, subtitle_font
from anime_factory.r2_paths import join_story

SHORT_WIDTH = 1080
SHORT_HEIGHT = 1920
SHORT_FPS = 24
# Center-crop the 16:9 source to this aspect so faces are not tiny, then blur-fill above and below.
SUBJECT_CROP_ASPECT = 4 / 3
HOOK_SECONDS = 2.0
OUTRO_SECONDS = 3.0
SAMPLE_RATE = 16000
# libass sizes everything against the script resolution, so shorts ship their own ASS
# rather than force_style on an srt, where FontSize/MarginV land off-frame.
ASS_STYLES = {
    "Hook": {"size": 76, "outline": 5, "align": 8, "margin_v": 190},
    "Line": {"size": 58, "outline": 4, "align": 2, "margin_v": 300},
    "Card": {"size": 64, "outline": 4, "align": 5, "margin_v": 0},
}
# Registry font names resolve through fontconfig; the box must ship Noto CJK or libass draws tofu.
FONTS_DIR_ENV = "ANIME_FACTORY_FONTS_DIR"
# Keyed by the registry's platform_tag_style, so a new language row reuses existing copy.
HOOK_TEXT = {
    "cn": "全片 600 秒，这段是第 {index} 段",
    "intl": "600s episode. Clip {index}.",
    "jp": "本編600秒。第{index}話の一場面。",
}
OUTRO_TEXT = {
    "cn": "全片见 YouTube",
    "intl": "Full episode on YouTube",
    "jp": "本編はYouTubeで",
}
FALLBACK_TAG_STYLE = "intl"


class ShortsSourceError(RuntimeError):
    pass


@dataclass
class ShortPlan:
    index: int
    start_s: float
    end_s: float
    shot_ids: list[str]
    hook: dict[str, str]
    reason: str

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 2)


def _copy(table: dict[str, str], lang: str) -> str:
    style = spec_for(lang).platform_tag_style
    return table.get(style) or table[FALLBACK_TAG_STYLE]


def _shots(board: dict) -> list[dict]:
    shots = list(board.get("shots") or [])
    if not shots:
        raise ValueError("board has no shots")
    return shots


def _hook_langs(board: dict) -> tuple[str, ...]:
    return normalize_langs(board.get("langs"))


def _hook(index: int, langs: Sequence[str], sample: dict | None) -> dict[str, str]:
    """First two seconds. Prefer a real line from the window; a template is the fallback."""
    hook: dict[str, str] = {}
    for lang in langs:
        text = shot_line_text(sample, lang) if sample else ""
        hook[lang] = text or _copy(HOOK_TEXT, lang).format(index=index)
    return hook


def _window_score(window: list[dict]) -> tuple[float, float]:
    """Dialogue density plus a climax bonus for tight framing, which is where the scene peaks."""
    seconds = sum(float(s.get("duration") or 0) for s in window) or 1.0
    spoken = [s for s in window if s.get("line")]
    chars = sum(len(shot_line_text(s, "zh") or shot_line_text(s, "en")) for s in spoken)
    density = len(spoken) / seconds
    tight = sum(1 for s in window if (s.get("cuts") or [{}])[0].get("size") in {"CU", "MCU"})
    return density, density * 100 + chars * 0.4 + tight * 1.5


def plan_shorts(
    board: dict,
    *,
    count: int = 3,
    min_s: float = 20.0,
    max_s: float = 45.0,
) -> list[ShortPlan]:
    """Pick the densest non-overlapping windows. Windows never straddle a scene cut."""
    shots = _shots(board)
    langs = _hook_langs(board)
    starts = shot_start_times(shots)
    candidates: list[tuple[float, int, int, list[dict]]] = []
    for i in range(len(shots)):
        window: list[dict] = []
        seconds = 0.0
        scene = shots[i].get("scene_id")
        for j in range(i, len(shots)):
            if shots[j].get("scene_id") != scene:
                break
            seconds += float(shots[j].get("duration") or 0)
            window.append(shots[j])
            if seconds < min_s:
                continue
            if seconds > max_s:
                break
            density, score = _window_score(window)
            if density > 0:
                candidates.append((score, i, j, list(window)))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    chosen: list[tuple[float, int, int, list[dict]]] = []
    for candidate in candidates:
        if len(chosen) >= count:
            break
        _score, lo, hi, _window = candidate
        if any(not (hi < c_lo or lo > c_hi) for _s, c_lo, c_hi, _w in chosen):
            continue
        chosen.append(candidate)
    chosen.sort(key=lambda item: item[1])

    plans: list[ShortPlan] = []
    for n, (score, lo, hi, window) in enumerate(chosen, start=1):
        spoken = [s for s in window if s.get("line")]
        density, _ = _window_score(window)
        start = starts.get(str(window[0].get("id")), 0.0)
        end = start + sum(float(s.get("duration") or 0) for s in window)
        plans.append(
            ShortPlan(
                index=n,
                start_s=round(start, 2),
                end_s=round(end, 2),
                shot_ids=[str(s.get("id")) for s in window],
                hook=_hook(n, langs, spoken[0] if spoken else None),
                reason=(
                    f"{len(spoken)} 句台词 / {end - start:.0f}s（密度 {density:.2f} 句/秒，评分 {score:.0f}）"
                    f"，{window[0].get('scene_id')} 内不跨场"
                ),
            )
        )
    return plans


def load_board(ep_root: Path) -> dict:
    path = Path(ep_root) / "board.json"
    if not path.is_file():
        raise ShortsSourceError(f"missing board.json under {ep_root}")
    return json.loads(path.read_text(encoding="utf-8"))


def episode_code(ep_root: Path) -> str:
    return Path(ep_root).name


def source_video(ep_root: Path, lang: str | None = None) -> Path:
    """The silent master is preferred: one picture, per-language audio muxed on top."""
    ep = episode_code(ep_root)
    candidates = [Path(ep_root) / f"{ep}.video.mp4", Path(ep_root) / "video.mp4"]
    if lang:
        candidates.append(Path(ep_root) / "final" / f"{ep}.{lang}.mp4")
    candidates.extend(sorted((Path(ep_root) / "final").glob(f"{ep}.*.mp4")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ShortsSourceError(f"no episode video under {ep_root}; expected {ep}.video.mp4")


def window_audio(ep_root: Path, plan: ShortPlan, lang: str, board: dict) -> bytes:
    """Re-lay the window's line wavs on a clip-length bed, with the outro card left silent."""
    starts = shot_start_times(_shots(board))
    clips: list[tuple[float, bytes]] = []
    for shot in _shots(board):
        sid = str(shot.get("id") or "")
        if sid not in plan.shot_ids or not shot_line_text(shot, lang):
            continue
        wav = Path(ep_root) / "audio" / "lines" / f"{sid}.{lang}.wav"
        if wav.is_file():
            clips.append((starts.get(sid, 0.0) - plan.start_s, wav.read_bytes()))
    return mix_clips(plan.duration_s + OUTRO_SECONDS, clips, SAMPLE_RATE)


def window_cues(
    plan: ShortPlan,
    lang: str,
    board: dict,
    youtube_url: str | None = None,
) -> list[tuple[float, float, str, str]]:
    """Hook over the first two seconds, then the window's own lines, then the outro card."""
    starts = shot_start_times(_shots(board))
    cues: list[tuple[float, float, str, str]] = [(0.0, HOOK_SECONDS, plan.hook.get(lang, ""), "Hook")]
    for shot in _shots(board):
        sid = str(shot.get("id") or "")
        if sid not in plan.shot_ids:
            continue
        text = shot_line_text(shot, lang)
        if not text:
            continue
        start = max(starts.get(sid, 0.0) - plan.start_s, HOOK_SECONDS)
        cues.append((start, start + max(float(shot.get("duration") or 4.0) - 0.35, 1.0), text, "Line"))
    outro = _copy(OUTRO_TEXT, lang)
    card = f"{outro}\n{youtube_url}" if youtube_url else outro
    cues.append((plan.duration_s, plan.duration_s + OUTRO_SECONDS, card, "Card"))
    return cues


def _ass_ts(seconds: float) -> str:
    cs = int(round(max(seconds, 0.0) * 100))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, hundredths = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{hundredths:02d}"


def write_ass(path: Path, cues: list[tuple[float, float, str, str]], lang: str) -> None:
    font = subtitle_font(lang)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {SHORT_WIDTH}",
        f"PlayResY: {SHORT_HEIGHT}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,"
        " Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline,"
        " Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    for name, style in ASS_STYLES.items():
        lines.append(
            f"Style: {name},{font},{style['size']},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            f"-1,0,0,0,100,100,0,0,1,{style['outline']},0,{style['align']},80,80,{style['margin_v']},1"
        )
    lines += [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text, style in cues:
        if not text:
            continue
        body = text.replace("\\", "").replace("\n", r"\N")
        lines.append(f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end)},{style},,0,0,0,,{body}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def vertical_filter(ass_path: Path, hold_from_s: float, fonts_dir: Path | None = None) -> str:
    """Center-crop the subject, blur-fill the bars, hold the last frame for the outro card."""
    inner_h = int(round(SHORT_WIDTH / SUBJECT_CROP_ASPECT))
    escaped = str(ass_path).replace("\\", "/").replace(":", r"\:")
    subtitles = f"ass='{escaped}'"
    if fonts_dir is not None:
        subtitles += f":fontsdir='{str(fonts_dir).replace(':', r'\:')}'"
    return (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={SHORT_WIDTH}:{SHORT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={SHORT_WIDTH}:{SHORT_HEIGHT},gblur=sigma=28[bgb];"
        f"[fg]crop=ih*{SUBJECT_CROP_ASPECT:.4f}:ih:(iw-ih*{SUBJECT_CROP_ASPECT:.4f})/2:0,"
        f"scale={SHORT_WIDTH}:{inner_h}[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps={SHORT_FPS},"
        f"tpad=stop_mode=clone:stop_duration={OUTRO_SECONDS:.3f},"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.72:t=fill:enable='gte(t,{hold_from_s:.3f})',"
        f"{subtitles}[v]"
    )


def fonts_dir() -> Path | None:
    raw = (os.environ.get(FONTS_DIR_ENV) or "").strip()
    return Path(raw) if raw and Path(raw).is_dir() else None


def build_render_command(
    *,
    source: Path,
    audio: Path,
    subtitles: Path,
    out: Path,
    plan: ShortPlan,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{plan.start_s:.3f}",
        "-t",
        f"{plan.duration_s:.3f}",
        "-i",
        str(source),
        "-i",
        str(audio),
        "-filter_complex",
        vertical_filter(subtitles, plan.duration_s, fonts_dir()),
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]


def build_cover_command(video: Path, out: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{HOOK_SECONDS + 1.0:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(out),
    ]


def short_paths(ep_root: Path, index: int, lang: str) -> tuple[Path, Path]:
    shorts = Path(ep_root) / "shorts"
    return shorts / f"{index:02d}.{lang}.mp4", shorts / f"{index:02d}.{lang}.jpg"


def short_r2_key(story_id: str, episode_code: str, index: int, lang: str, suffix: str = "mp4") -> str:
    return join_story(story_id, f"episodes/{episode_code}/shorts/{index:02d}.{lang}.{suffix}")


def render_short(
    plan: ShortPlan,
    ep_root: Path,
    lang: str,
    *,
    youtube_url: str | None = None,
) -> Path:
    """Cut, verticalise and mux one language of one window. Returns the mp4 path."""
    ep_root = Path(ep_root)
    board = load_board(ep_root)
    if lang not in normalize_langs(board.get("langs")):
        raise ValueError(f"{lang} is not one of this episode's languages: {board.get('langs')}")
    if shutil.which("ffmpeg") is None:
        raise ShortsSourceError("ffmpeg not on PATH")
    source = source_video(ep_root, lang)
    out, cover = short_paths(ep_root, plan.index, lang)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / ".work"
    work.mkdir(parents=True, exist_ok=True)
    audio = work / f"{plan.index:02d}.{lang}.wav"
    audio.write_bytes(window_audio(ep_root, plan, lang, board))
    cues = window_cues(plan, lang, board, youtube_url)
    subtitles = work / f"{plan.index:02d}.{lang}.ass"
    write_ass(subtitles, cues, lang)
    # Portable sidecar next to the mp4 for platforms that want an upload-time subtitle file.
    write_srt(out.with_suffix(".srt"), [(s, e, t) for s, e, t, _style in cues])
    subprocess.run(
        build_render_command(source=source, audio=audio, subtitles=subtitles, out=out, plan=plan),
        check=True,
        capture_output=True,
    )
    subprocess.run(build_cover_command(out, cover), check=True, capture_output=True)
    return out
