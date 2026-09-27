"""Unused on the single-GPU R2V path. Import-only."""


def get_sp_group(*_args, **_kwargs):
    raise RuntimeError("xfuser USP is not installed; single-GPU R2V does not call it")
