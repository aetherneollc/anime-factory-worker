"""HunyuanImage client helpers."""

from __future__ import annotations

import pytest

from anime_factory.hunyuan_image import (
    HunyuanImageError,
    assert_tokenhub_area,
    parse_wxh,
)


def test_parse_wxh_and_area_gate():
    assert parse_wxh("768x1344") == (768, 1344)
    assert parse_wxh("768*1024") == (768, 1024)
    assert_tokenhub_area(768, 1344)
    with pytest.raises(HunyuanImageError):
        assert_tokenhub_area(1024, 1536)
