# BCG Data Challenge

## Project Description

This project analyzes barley yield and climate data.

---

## Getting Started

Follow these steps **in order** to set up the project on your computer.

### Step 1: Prerequisites

Before you begin, make sure you have:

1. **Python 3.12 or higher** installed
   - Check if you have Python: Open a terminal and type `python3 --version` or `python --version`
   - If you don't have Python, download it from [python.org](https://www.python.org/downloads/)

2. **Git** installed
   - Check if you have Git: Open a terminal and type `git --version`
   - If you don't have Git, download it from [git-scm.com](https://git-scm.com/downloads)

### Step 2: Install uv

This project uses `uv` for managing Python packages. Install it by running this command in your terminal:

**On Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, **close and reopen your terminal** (or restart your computer) for the changes to take effect.

Verify installation by running:
```bash
uv --version
```

You should see a version number. If you get an error, make sure you restarted your terminal.

### Step 3: Clone the Repository

If you haven't already, clone this repository to your computer:

```bash
git clone https://github.com/Aymeric314/bcg-data-challenge.git
cd bcg-data-challenge
```

### Step 4: Install Project Dependencies

Run this command to install all required packages:

```bash
uv sync --extra dev
```

This will:
- Create a virtual environment (`.venv/`)
- Install all project dependencies (pandas, numpy, etc.)
- Install development tools (black, ruff, mypy, pre-commit)

**Wait for this to finish** - it may take a few minutes the first time.

### Step 5: Activate the Virtual Environment

**On Linux/Mac:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

You should see `(bcg-data-challenge)` appear at the beginning of your terminal prompt, indicating the environment is active.

### Step 6: Install Pre-commit Hooks

Pre-commit hooks automatically check your code before you commit changes. Install them:

```bash
uv run pre-commit install
```

You should see: `pre-commit installed at .git/hooks/pre-commit`

### Step 7: Verify Everything Works

Test that everything is set up correctly:

```bash
# Check Python version (should be 3.12+)
python --version

# Check that packages are installed
python -c "import pandas; import numpy; print('All packages installed!')"

# Test pre-commit (should run without errors)
uv run pre-commit run --all-files
```

If all commands work without errors, you're all set! 🎉

---

## Using the Project

### Running Python Scripts

1. Make sure your virtual environment is activated (you should see `(bcg-data-challenge)` in your terminal)

2. Run your Python scripts:
   ```bash
   python your_script.py
   ```

   Or use `uv run`:
   ```bash
   uv run python your_script.py
   ```

### Working with Git

When you make changes and want to commit them:

1. **Stage your changes:**
   ```bash
   git add .
   ```

2. **Commit your changes:**
   ```bash
   git commit -m "Your commit message"
   ```

   The pre-commit hooks will automatically run and check your code. If there are issues, they will be fixed automatically or you'll see error messages.

3. **Push to GitHub:**
   ```bash
   git push
   ```

### Project Structure

```
bcg-data-challenge/
├── src/                    # Your source code goes here
│   ├── constants/          # Constants and configuration
│   ├── data_pipelines/     # Data processing pipelines
│   └── model_pipelines/    # Model training pipelines
├── data/                   # Data files
│   ├── bronze_datasets/    # Raw/initial data
│   └── silver_datasets/    # Processed data
├── EDA/                    # Exploratory Data Analysis notebooks/scripts
├── constants/              # Project-wide constants (file paths, etc.)
├── pyproject.toml          # Project configuration and dependencies
└── README.md               # This file
```
