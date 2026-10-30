"""
Automated security, credentials, and data privacy tests.
"""

import os
import re

PATTERNS_TO_FLAG = [
    r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    r"(?i)secret\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
    r"(?i)password\s*=\s*['\"][^\n\r'\"]{6,}['\"]",
    r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}",
]


def test_no_hardcoded_secrets_in_source_code(workspace_root):
    """Scan all Python files in the workspace to verify no hardcoded secrets or API keys exist."""
    suspicious_matches = []
    
    for root, dirs, files in os.walk(workspace_root):
        if any(d in root for d in [".git", ".pytest_cache", "__pycache__", ".venv", "venv"]):
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for pattern in PATTERNS_TO_FLAG:
                        matches = re.findall(pattern, content)
                        if matches:
                            suspicious_matches.append((path, matches))
                            
    assert len(suspicious_matches) == 0, f"Found potential secret keys: {suspicious_matches}"


def test_no_patient_identifiable_information(covid_csv_path):
    """
    Verify that COVID-19 dataset contains only country/aggregated statistical metrics
    and no patient-identifying fields (e.g. patient_name, ssn, national_id, patient_address).
    """
    import pandas as pd
    df_head = pd.read_csv(covid_csv_path, nrows=10)
    phi_columns = [
        "name", "patient_name", "first_name", "last_name",
        "ssn", "national_id", "address", "phone", "email"
    ]
    for col in df_head.columns:
        assert col.lower() not in phi_columns, f"Potential PHI column found: {col}"
