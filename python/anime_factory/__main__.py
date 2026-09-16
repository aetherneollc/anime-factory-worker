"""CLI: python -m anime_factory demo-offline <dir> | assets --story-id …"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anime_factory")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo-offline", help="Run a fixture episode through non-GPU stages (no Vast lease)")
    demo.add_argument("dir", type=Path, help="Story working directory")
    demo.add_argument("--title", default="Demo Harbor")

    assets = sub.add_parser("assets", help="Generate the standard Kolors asset library (no Vast)")
    assets.add_argument("--story-id", default="story-dcc831dfc9bc-3e6e67")
    assets.add_argument("--root", type=Path, default=None, help="Story working directory")
    assets.add_argument("--upload-r2", action="store_true", help="Upload assets/ to configured R2 bucket")
    assets.add_argument("--live", action="store_true", help="Force live Kolors even if env is off")

    produce = sub.add_parser("produce", help="DEBUG: produce a 600s episode locally. Production is Worker cron + GPU session.")
    produce.add_argument("--story-id", default="", help="Existing PM story id (created if empty + --create)")
    produce.add_argument("--create", action="store_true", help="POST /api/stories first")
    produce.add_argument("--control-plane", default=os.environ.get("CONTROL_PLANE_URL") or "")
    produce.add_argument("--root", type=Path, default=None)
    produce.add_argument("--live", action="store_true", help="Live Kolors + CosyVoice2")
    produce.add_argument("--no-upload", action="store_true")
    produce.add_argument("--lease-gpu", action="store_true", help="After pre-GPU, lease one Vast card (VAST_DRY_RUN=0 for this process)")
    produce.add_argument("--resume-gpu", action="store_true", help="Do not rerun bible→keyframe; lease one card and assign this story")
    produce.add_argument("--episode", default="EP001", help="Episode code, e.g. EP001 / EP002")
    produce.add_argument("--title", default="", help="Project title; falls back to the control plane, then story.sqlite")
    produce.add_argument("--logline", default="", help="One-line story the director writes from")
    produce.add_argument("--langs", default="", help="Comma-separated project languages; must be in config/languages.json")
    produce.add_argument("--template-demo", action="store_true", help="Run the fixed 黑客平板 fixture instead of generating an episode")
    produce.add_argument("--live-script", action="store_true", help="Call DeepSeek for the script instead of the offline draft")
    args = parser.parse_args(argv)

    if args.cmd == "demo-offline":
        from anime_factory.pipeline import run_local_episode

        result = run_local_episode(args.dir, title=args.title)
        print(json.dumps({k: v for k, v in result.items() if k != "violations"}, ensure_ascii=False, indent=2))
        return 1 if result.get("blocked") else 0

    if args.cmd == "assets":
        return _cmd_assets(args)
    if args.cmd == "produce":
        return _cmd_produce(args)
    return 2


def _cmd_assets(args) -> int:
    from anime_factory.config import load_dotenv, load_settings, siliconflow_keys
    from anime_factory.db import migrate, open_db
    from anime_factory.design import (
        KolorsClient,
        generate_asset_library,
        harbor_mvp_cast,
        library_from_db,
    )
    from anime_factory.world import init_fiction_world, write_bible

    load_dotenv()
    if args.live:
        os.environ["ANIME_FACTORY_LIVE_KOLORS"] = "1"
    settings = load_settings()
    live = bool(settings.live_kolors or args.live)
    root = args.root
    if root is None:
        guessed = Path("var/scratch/mvp_episode/story")
        root = guessed if guessed.exists() else Path("work") / args.story_id
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "story.sqlite"
    conn = open_db(db_path)
    migrate(conn)
    chars, locs, props, interiors = harbor_mvp_cast()
    story_row = conn.execute("SELECT id FROM story WHERE id = ?", (args.story_id,)).fetchone()
    if story_row is None:
        init_fiction_world(
            conn,
            args.story_id,
            "暮潮码头",
            locations=[{"id": loc["id"], "name": loc["name"], "aka": [], "type": "settlement"} for loc in locs],
            edges=[],
            interiors=interiors,
            timeline=[
                {"id": "ev_pre", "date": "Y1", "title": "Founding", "kind": "fictional", "episode": None, "sources": []}
            ],
            root=root,
        )
    if conn.execute("SELECT COUNT(*) AS n FROM characters").fetchone()["n"] == 0:
        for ch in chars:
            conn.execute(
                """
                INSERT INTO characters (id, name, identity_prompt, age, alive, seed)
                VALUES (?, ?, ?, 28, 1, ?)
                ON CONFLICT(id) DO UPDATE SET identity_prompt = excluded.identity_prompt
                """,
                (ch["id"], ch["name"], ch.get("identity_prompt"), ch.get("seed")),
            )
        for p in props:
            conn.execute(
                """
                INSERT INTO props (id, name, identity_prompt, state)
                VALUES (?, ?, ?, 'intact')
                ON CONFLICT(id) DO UPDATE SET identity_prompt = excluded.identity_prompt
                """,
                (p["id"], p["name"], p.get("identity_prompt")),
            )
        conn.commit()
        write_bible(
            root,
            world_md="# 暮潮码头\n\ninvented: true\n",
            style_md=None,
            cast={"characters": chars, "props": props},
            period_md="# Period\n\n## 正面清单\n\n## 负面清单\n",
        )
    keys = siliconflow_keys() if live else ["test-key"]
    client = KolorsClient(keys, live=live)
    n_chars = conn.execute("SELECT COUNT(*) AS n FROM characters").fetchone()["n"]
    if n_chars == 0 or story_row is None:
        result = generate_asset_library(
            conn,
            args.story_id,
            chars,
            locs,
            props,
            client,
            None,
            "fiction",
            root,
            interiors=interiors,
            skip_existing=not live,
        )
    else:
        result = library_from_db(conn, args.story_id, client, root, skip_existing=not live)

    uploads = []
    if args.upload_r2 or live:
        from anime_factory.r2_client import upload_tree

        uploads = upload_tree(args.story_id, root, "assets")
    summary = {
        "story_id": args.story_id,
        "root": str(root),
        "live": live,
        "count": result.get("count"),
        "created": result.get("created"),
        "index": result.get("index"),
        "vast_for_stills": False,
        "r2_uploads": [u for u in uploads if u.get("ok")],
        "r2_skipped": [u for u in uploads if u.get("skipped")],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_produce(args) -> int:
    from anime_factory.config import load_dotenv
    from anime_factory.langs import normalize_langs
    from anime_factory.produce import LOGLINE, TITLE, ControlPlane, produce_episode

    load_dotenv()
    if args.live:
        os.environ["ANIME_FACTORY_LIVE_KOLORS"] = "1"
        os.environ["ANIME_FACTORY_LIVE_TTS"] = "1"
    title = (args.title or "").strip()
    logline = (args.logline or "").strip()
    langs = normalize_langs([p for p in (args.langs or "").split(",") if p.strip()] or None)
    story_id = (args.story_id or "").strip()
    control = None
    user = os.environ.get("STUDIO_USER") or "studio"
    password = os.environ.get("STUDIO_PASSWORD") or ""
    if args.control_plane and password:
        control = ControlPlane(args.control_plane, user, password)
        try:
            control.login()
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"pm_login:{exc}"}), file=sys.stderr)
            return 1
        if args.create or not story_id:
            created = control.create_project(
                title or TITLE,
                logline or (LOGLINE if not title else title),
                langs=langs,
                start=True,
            )
            story_id = created["story_id"]
            print(
                json.dumps(
                    {"created": True, "story_id": story_id, "next_gate": created.get("next_gate")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            if not getattr(args, "resume_gpu", False):
                try:
                    control.action(story_id, "start")
                except RuntimeError as exc:
                    # MAX_CONCURRENT_STORIES=2 must not block the engine; jobs still sync.
                    print(json.dumps({"start_skipped": True, "error": str(exc)}, ensure_ascii=False), flush=True)
    if not story_id:
        print(json.dumps({"ok": False, "error": "story-id required (or --create with control plane)"}))
        return 2
    if getattr(args, "resume_gpu", False):
        os.environ["AF_STORY_ID"] = story_id
        os.environ["VAST_DRY_RUN"] = "0"
        from anime_factory.produce import _try_one_lease

        flags = {
            "bible": "passed",
            "canon": "passed",
            "script": "passed",
            "tts": "passed",
            "timing": "passed",
            "board": "passed",
            "design": "passed",
            "keyframe": "passed",
        }
        if control:
            try:
                control.job(story_id, "anim", "waiting_gpu")
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"job_sync": str(exc)}, ensure_ascii=False), flush=True)
        gpu = _try_one_lease(flags)
        print(json.dumps({"story_id": story_id, "resume_gpu": True, "gpu": gpu}, ensure_ascii=False, indent=2))
        return 0 if gpu.get("put_asks") or gpu.get("action") == "lease" or gpu.get("instance_id") else 1
    root = args.root or Path("work") / story_id
    result = produce_episode(
        story_id,
        root,
        control=control,
        live=bool(args.live),
        upload=not args.no_upload,
        lease_gpu=bool(args.lease_gpu),
        episode_code=str(getattr(args, "episode", None) or "EP001"),
        title=title or None,
        logline=logline or None,
        langs=langs if args.langs else None,
        template_demo=bool(args.template_demo),
        live_script=True if args.live_script else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
