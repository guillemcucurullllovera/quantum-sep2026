"""AWS/Braket credential handling — environment-only, mirrors
`core.config.IBMQuantumCredentials` exactly, but lives here rather than in
`core/config.py` to avoid modifying shared platform infrastructure from an
adaptation-workstream fork.

ABSOLUTE RULE: no hardcoded fallback value for any credential, ever.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from core.errors import CredentialError


@dataclass(frozen=True)
class AWSCredentials:
    access_key_id: str | None
    secret_access_key: str | None
    region: str

    @property
    def is_present(self) -> bool:
        return bool(self.access_key_id) and bool(self.secret_access_key)

    def require(self) -> "AWSCredentials":
        """Return self if credentials are present, else raise CredentialError.

        Call this only at the point of actually submitting to a real AWS
        Braket QPU/managed simulator — never earlier, so that the local
        simulator path never needs AWS credentials at all.
        """
        if not self.is_present:
            raise CredentialError(
                "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not both set. "
                "Real Braket QPU or managed-simulator execution requires them "
                "(see .env.example). The local simulator path "
                "(hardware.braket.backend_adapter.run_local) needs no AWS "
                "credentials at all — use that if you only meant to simulate."
            )
        return self

    @classmethod
    def from_env(cls) -> "AWSCredentials":
        return cls(
            access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
            secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )


_aws_credentials: AWSCredentials | None = None


def get_aws_credentials(refresh: bool = False) -> AWSCredentials:
    global _aws_credentials
    if _aws_credentials is None or refresh:
        _aws_credentials = AWSCredentials.from_env()
    return _aws_credentials
