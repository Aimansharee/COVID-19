# 📖 Testing & Development Guide — COVID-19 AI

This guide explains how to configure the environment, run the application, execute automated tests, and inspect code coverage across Windows, VS Code, and Git Bash.

---

## 1. Environment Setup

### 1.1 Prerequisites
- **Python**: 3.9 or higher (tested on Python 3.11)
- **Git**

### 1.2 Creating a Virtual Environment

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv .venv

# Allow script execution if restricted
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

#### On Git Bash / Linux / macOS:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/Scripts/activate  # on Windows Git Bash
# or: source .venv/bin/activate # on Linux/macOS
```

---

## 2. Installing Dependencies

Install both production and development/testing requirements:

```bash
# Production dependencies
pip install -r requirements.txt

# Development and testing dependencies
pip install -r requirements-dev.txt
```

---

## 3. Running the Streamlit Application

To start the interactive dashboard:

```bash
streamlit run main.py
```

The app will open automatically in your default browser at:
`http://localhost:8501`

---

## 4. Executing Automated Tests

### 4.1 Run the Full Test Suite
```bash
python -m pytest tests/ -v
```

### 4.2 Run with Code Coverage Report
```bash
python -m pytest tests/ --cov=main --cov-report=term-missing
```

### 4.3 Run a Specific Test Module
```bash
# Example: Run only regression tests
python -m pytest tests/test_regression.py -v

# Example: Run only data validation tests
python -m pytest tests/test_data_validation.py -v

# Example: Run only application tests
python -m pytest tests/test_application.py -v
```

### 4.4 Run Specific Test by Name Expression
```bash
python -m pytest -k "adjusted_smape" -v
```

---

## 5. VS Code Integration

1. Open the repository folder in VS Code (`code .`).
2. Press `Ctrl + Shift + P` and select **Python: Select Interpreter**, then choose your `.venv` environment.
3. Open the **Testing** tab (flask icon) in the left sidebar.
4. VS Code will automatically discover all 45 tests in `tests/`.
5. You can run or debug individual tests directly with a single click.

---

## 6. Troubleshooting Common Issues

### Issue 1: `missing ScriptRunContext!` warnings during pytest
- **Cause**: Streamlit outputs internal warnings when bare functions or `st.cache_data` are initialized without a browser session.
- **Solution**: These warnings are completely normal during unit testing and do not impact test assertions or correctness.

### Issue 2: `FileNotFoundError: cleaned_covid_data_last.csv`
- **Cause**: Running Python from outside the project directory without absolute path resolution.
- **Solution**: `main.py` has been updated with `os.path.abspath(__file__)` base directory resolution, allowing execution from any directory.

### Issue 3: PowerShell Script Execution Policy error
- **Cause**: Windows blocks unsigned PowerShell scripts by default.
- **Solution**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` before activating the virtual environment.
