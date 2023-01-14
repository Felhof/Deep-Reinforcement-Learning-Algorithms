import tempfile
from typing import Any

import nox
from nox.sessions import Session

nox.options.sessions = "lint", "mypy", "tests"
lint_annotations_locations = "agents", "noxfile.py", "utilities"
style_locations = (
    "agents",
    "noxfile.py",
    "utilities",
    "train_VPG_for_cartpole.py",
    "train_DQN_for_cartpole.py",
    "train_TRPG_for_cartpole.py",
    "tests",
)
typing_locations = "agents", "noxfile.py", "utilities"


@nox.session(python=["3.10"])
def black(session: Session) -> None:
    args = session.posargs or style_locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python=["3.10"])
def tests(session: Session) -> None:
    args = session.posargs
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    with tempfile.NamedTemporaryFile() as tmp:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={tmp.name}",
            external=True,
        )
        # remove lines with extra constrains e.g. gymnasium[accept-rom-license]
        # as they cause a pip error
        requirements = tmp.readlines()
        tmp.seek(0)
        for line in requirements:
            if "[" not in str(line):
                tmp.write(line)
        tmp.truncate()
        session.install(f"--constraint={tmp.name}", *args, **kwargs)


@nox.session(python=["3.10"])
def lint(session: Session) -> None:
    args = session.posargs or style_locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python=["3.10"])
def lint_annotations(session: Session) -> None:
    args = session.posargs or lint_annotations_locations
    install_with_constraints(
        session,
        "flake8-annotations",
    )
    session.run("flake8", *args)


@nox.session(python=["3.10"])
def mypy(session: Session) -> None:
    args = session.posargs or typing_locations
    install_with_constraints(session, "mypy", "numpy", "torch")
    session.run("mypy", *args)
