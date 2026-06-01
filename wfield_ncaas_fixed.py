"""Launch wfield NeuroCAAS GUI with compatibility patches.

The released wfield 0.4.2 GUI can crash on newer PyQt when it passes a
numpy.float64 to QProgressBar.setValue during AWS uploads. It also reads AWS
temporary credentials incompletely: STS access keys that start with "ASIA"
require aws_session_token, but wfield's S3 helper only passes access/secret.

This launcher keeps the installed package unchanged and patches both behaviors
only for the current process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QProgressBar


_original_set_value = QProgressBar.setValue


def _set_value_int(self, value):
    return _original_set_value(self, int(value))


def _read_default_aws_credentials() -> dict[str, str]:
    credentials_file = Path(
        os.environ.get("AWS_SHARED_CREDENTIALS_FILE", Path.home() / ".aws" / "credentials")
    )
    profile = os.environ.get("AWS_PROFILE", "default")
    values: dict[str, str] = {}
    active_profile = None

    if not credentials_file.exists():
        return values

    for raw_line in credentials_file.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            active_profile = line[1:-1].strip()
            continue
        if active_profile != profile or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def _patch_wfield_s3_connect() -> None:
    import boto3
    import wfield.ncaas_utils as ncaas_utils

    def s3_connect_with_session_token():
        aws = _read_default_aws_credentials()
        session_kwargs = {
            "aws_access_key_id": aws.get("aws_access_key_id", ""),
            "aws_secret_access_key": aws.get("aws_secret_access_key", ""),
        }
        token = aws.get("aws_session_token")
        if token:
            session_kwargs["aws_session_token"] = token

        session = boto3.session.Session(**session_kwargs)
        return session.resource("s3"), session.client("s3")

    ncaas_utils.s3_connect = s3_connect_with_session_token


def main() -> None:
    QProgressBar.setValue = _set_value_int
    _patch_wfield_s3_connect()

    import wfield.ncaas_gui as ncaas_gui

    ncaas_gui.QProgressBar.setValue = _set_value_int

    # ncaas_gui imports s3_connect into its module namespace via import *.
    # Patch that copy too so all GUI paths use the session-token-aware helper.
    import wfield.ncaas_utils as ncaas_utils

    ncaas_gui.s3_connect = ncaas_utils.s3_connect

    print("Using Widefield_DAQ_recorder wfield NeuroCAAS compatibility launcher.")
    print("Patch active: QProgressBar values are coerced to int; AWS session token is supported.")

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    ncaas_gui.main(folder=folder)


if __name__ == "__main__":
    main()
