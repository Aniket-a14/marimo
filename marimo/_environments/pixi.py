# Copyright 2026 Marimo. All rights reserved.
"""Provision script environments with pixi.

The pixi backend mirrors the uv backend's contract through pixi's
script commands: `pixi install --script` synchronizes and reports the
environment, `pixi add --script --pypi` edits the manifest. pixi has no
`--with` overlay, so marimo is pinned into the manifest (an editable
path dependency from a development checkout) rather than layered onto
the launch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

from marimo import _loggers
from marimo._environments.errors import EnvironmentManagerError
from marimo._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from marimo._environments.environment import Environment, ProcessPlan

LOGGER = _loggers.marimo_logger()


class PixiError(EnvironmentManagerError):
    """Base class for pixi invocation errors."""


class PixiNotFoundError(PixiError):
    """No pixi executable was found."""

    def __init__(self) -> None:
        super().__init__(
            "pixi must be installed to use --sandbox=pixi. "
            "Install pixi from https://pixi.sh"
        )


class PixiUnsupportedVersionError(PixiError):
    """The installed pixi predates script environments."""

    def __init__(self) -> None:
        super().__init__(
            "--sandbox=pixi requires a pixi with `pixi install --script` "
            "support. Upgrade with `pixi self-update`."
        )


class PixiCommandError(PixiError):
    """A pixi command exited with a failure."""

    def __init__(
        self, command: Sequence[str], returncode: int, stderr: str
    ) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"`{' '.join(self.command)}` failed with exit code "
            f"{returncode}.\n{stderr.strip()}"
        )


def find_pixi_bin() -> str | None:
    """Path to the pixi executable, or None if not found."""
    return shutil.which("pixi")


def is_pixi_available() -> bool:
    return find_pixi_bin() is not None


def require_pixi_bin() -> str:
    """Path to the pixi executable; raises `PixiNotFoundError`."""
    pixi_bin = find_pixi_bin()
    if pixi_bin is None:
        raise PixiNotFoundError()
    return pixi_bin


def ensure_supported_pixi() -> None:
    """Raise unless the invoked pixi understands `install --script`.

    Probes `pixi install --help` rather than a version floor: the verb
    is not in a released pixi yet, so no floor exists to compare
    against.
    """
    completed = subprocess.run(
        [require_pixi_bin(), "install", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0 or "--script" not in completed.stdout:
        raise PixiUnsupportedVersionError()


# `pixi install --script` reports the environment on stderr:
#   ✔ The script environment has been installed at '<prefix>'.
# A `--json` report is the upstream ask that retires this parse.
_INSTALLED_AT = re.compile(r"installed at '([^']+)'")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sync(
    script: str,
    *,
    cwd: str | None = None,
    on_output: Callable[[str], None] | None = None,
    on_command: Callable[[Sequence[str]], None] | None = None,
) -> Environment:
    """Makes the script's environment match its metadata.

    Runs `pixi install --script` from `cwd`, normally the notebook's
    directory. Returns the installed environment; pixi does not report
    what changed, so the action is always `updated` and restarts hinge
    on interpreter identity. Raises `PixiCommandError` on failure and
    never mutates `script`.
    """
    from marimo._environments.environment import Environment

    args = [
        require_pixi_bin(),
        "install",
        "--script",
        os.path.abspath(script),
    ]
    if on_command is not None:
        on_command(args)
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        # pixi must never hang waiting for input.
        stdin=subprocess.DEVNULL,
        env=command_env(),
        cwd=cwd,
        start_new_session=os.name != "nt",
    )
    if on_output is not None:
        for line in (completed.stdout + completed.stderr).splitlines(True):
            on_output(line)
    if completed.returncode != 0:
        raise PixiCommandError(
            args, completed.returncode, completed.stderr or completed.stdout
        )
    report = _ANSI.sub("", completed.stderr)
    match = _INSTALLED_AT.search(report)
    if match is None:
        raise PixiError(
            "pixi installed the script environment but did not report "
            f"its location.\n{report.strip()}"
        )
    root = match.group(1)
    python = _env_python(root)
    if python is None:
        raise PixiError(f"No interpreter found in the environment at {root}")
    return Environment(python=python, root=root, action="updated")


def add(
    script: str,
    package: str,
    *,
    cwd: str,
    upgrade: bool = False,
    on_output: Callable[[str], None] | None = None,
    on_command: Callable[[Sequence[str]], None] | None = None,
) -> None:
    """Add a PyPI dependency to a script, or refresh it when upgrading.

    `pixi update` takes package names, not requirements; a constrained
    upgrade rewrites the manifest entry through `pixi add` instead, and
    the next solve advances within the new constraint.
    """
    if upgrade and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", package):
        args = [
            require_pixi_bin(),
            "update",
            "--script",
            os.path.abspath(script),
            package,
        ]
    else:
        args = [
            require_pixi_bin(),
            "add",
            "--script",
            os.path.abspath(script),
            "--pypi",
            package,
        ]
    _run(
        args,
        cwd=cwd,
        on_output=on_output,
        on_command=on_command,
    )


def remove(
    script: str,
    package: str,
    *,
    cwd: str,
    on_output: Callable[[str], None] | None = None,
    on_command: Callable[[Sequence[str]], None] | None = None,
) -> None:
    """Remove a direct PyPI dependency from a script."""
    args = [
        require_pixi_bin(),
        "remove",
        "--script",
        os.path.abspath(script),
        "--pypi",
        package,
    ]
    _run(
        args,
        cwd=cwd,
        on_output=on_output,
        on_command=on_command,
    )


def list_script_packages(script: str, *, cwd: str) -> list[dict[str, Any]]:
    """Return pixi's structured resolved package records for a script."""
    args = [
        require_pixi_bin(),
        "list",
        "--script",
        os.path.abspath(script),
        "--json",
    ]
    completed = _run(args, cwd=cwd)
    import json

    try:
        records = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PixiError("pixi returned an unreadable package list") from error
    if not isinstance(records, list):
        raise PixiError("pixi returned an invalid package list")
    return [record for record in records if isinstance(record, dict)]


def tree_script_packages(script: str, *, cwd: str) -> str:
    """Return pixi's resolved dependency tree for a script."""
    args = [
        require_pixi_bin(),
        "tree",
        "--no-install",
        "--color",
        "never",
        "--script",
        os.path.abspath(script),
    ]
    return _run(args, cwd=cwd).stdout


def _run(
    args: Sequence[str],
    *,
    cwd: str,
    on_output: Callable[[str], None] | None = None,
    on_command: Callable[[Sequence[str]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    if on_command is not None:
        on_command(args)
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        # pixi must never hang waiting for input.
        stdin=subprocess.DEVNULL,
        env=command_env(),
        cwd=cwd,
        start_new_session=os.name != "nt",
    )
    if on_output is not None:
        for line in (completed.stdout + completed.stderr).splitlines(True):
            on_output(line)
    if completed.returncode != 0:
        raise PixiCommandError(
            args, completed.returncode, completed.stderr or completed.stdout
        )
    return completed


def ensure_marimo(
    path: str,
    *,
    on_command: Callable[[Sequence[str]], None] | None = None,
) -> None:
    """Pin marimo into the script metadata if not already a dependency.

    pixi cannot overlay marimo onto a launch, so the manifest must
    carry it: the running version, or an editable path dependency when
    marimo runs from a development checkout. Creates the metadata block
    if the file has none. No-op for non-`.py` targets and for missing
    or empty files, whose block is the notebook serializer's to create.
    """
    if not path.endswith(".py"):
        return
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return

    from marimo._utils.inline_script_metadata import (
        has_marimo_in_script_metadata,
    )

    if has_marimo_in_script_metadata(path) is True:
        return

    from marimo._environments.script_metadata import ensure_metadata_block

    ensure_metadata_block(path)

    from marimo._environments.overlay import marimo_dir
    from marimo._utils.versions import is_editable

    args = [require_pixi_bin(), "add", "--script", os.path.abspath(path)]
    if is_editable("marimo"):
        args.extend(
            ["--pypi", "marimo", "--path", str(marimo_dir()), "--editable"]
        )
    else:
        args.extend(["--pypi", f"marimo=={__version__}"])
    if on_command is not None:
        on_command(args)
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        # pixi must never hang waiting for input.
        stdin=subprocess.DEVNULL,
        env=command_env(),
        cwd=os.path.dirname(os.path.abspath(path)),
        timeout=60,
        start_new_session=os.name != "nt",
    )
    if completed.returncode != 0:
        raise PixiCommandError(
            args, completed.returncode, completed.stderr or completed.stdout
        )


def launch(
    environment: Environment,
    args: Sequence[str],
    *,
    base_env: Mapping[str, str] | None = None,
) -> ProcessPlan:
    """Plans running `python <args...>` inside the environment.

    A pixi script environment is a conda environment, so the plan
    activates it the way `pixi run` would: `CONDA_PREFIX` and the
    environment's bin directory first on `PATH`. `VIRTUAL_ENV` is
    dropped; a conda prefix is not a virtualenv.
    """
    from marimo._environments.environment import ProcessPlan

    env = dict(os.environ if base_env is None else base_env)
    env.pop("VIRTUAL_ENV", None)
    env["CONDA_PREFIX"] = environment.root
    env["CONDA_DEFAULT_ENV"] = os.path.basename(environment.root)
    bin_dir = os.path.join(environment.root, "bin")
    path = env.get("PATH")
    env["PATH"] = bin_dir if not path else bin_dir + os.pathsep + path
    return ProcessPlan(argv=(environment.python, *args), env=env)


def command_env() -> dict[str, str]:
    """Environment for pixi script commands.

    Drops activation state from any enclosing pixi, conda, or uv
    environment so the script's own manifest decides everything.
    """
    env = os.environ.copy()
    for variable in (
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "PIXI_PROJECT_MANIFEST",
        "PIXI_PROJECT_ROOT",
        "PIXI_ENVIRONMENT_NAME",
        "PIXI_IN_SHELL",
    ):
        env.pop(variable, None)
    return env


def _env_python(root: str) -> str | None:
    """The environment's interpreter within a conda prefix layout."""
    if os.name == "nt":
        candidates = (
            os.path.join(root, "python.exe"),
            os.path.join(root, "Scripts", "python.exe"),
        )
    else:
        candidates = (
            os.path.join(root, "bin", "python"),
            os.path.join(root, "bin", "python3"),
        )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None
