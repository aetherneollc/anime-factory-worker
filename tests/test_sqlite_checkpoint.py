import sqlite3

from anime_factory.db import checkpoint_and_upload, open_db


def test_checkpoint_keeps_live_writer_and_reopened_database_in_sync(tmp_path):
    path = tmp_path / "story.sqlite"
    writer = open_db(path)
    writer.execute("CREATE TABLE progress (shot TEXT)")
    writer.execute("INSERT INTO progress VALUES ('s001')")
    writer.commit()
    inode = path.stat().st_ino
    assert checkpoint_and_upload(writer, path) == path
    assert path.stat().st_ino == inode
    writer.execute("INSERT INTO progress VALUES ('s002')")
    writer.commit()
    checkpoint_and_upload(writer, path)
    writer.close()
    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT shot FROM progress").fetchall() == [('s001',), ('s002',)]


def test_checkpoint_can_export_to_a_different_file(tmp_path):
    writer = open_db(tmp_path / "live.sqlite")
    writer.execute("CREATE TABLE progress (shot TEXT)")
    writer.execute("INSERT INTO progress VALUES ('s001')")
    writer.commit()
    dest = tmp_path / "export.sqlite"
    checkpoint_and_upload(writer, dest)
    writer.execute("INSERT INTO progress VALUES ('s002')")
    writer.commit()
    with sqlite3.connect(dest) as reader:
        assert reader.execute("SELECT shot FROM progress").fetchall() == [('s001',)]
    writer.close()
