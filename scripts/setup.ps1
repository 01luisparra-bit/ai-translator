Write-Host "Setting up local development environment..."

# --- Python virtual environment ---
if (!(Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

.\.venv\Scripts\activate

Write-Host "Upgrading pip and installing requirements..."
pip install --upgrade pip
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
}

# --- Node dependencies ---
Write-Host "Installing Node dependencies..."
npm install

# --- Pre-commit hooks (optional) ---
if (Test-Path ".pre-commit-config.yaml") {
    Write-Host "Installing pre-commit hooks..."
    pre-commit install
}

Write-Host ""
Write-Host 'Setup complete. You can now run: scripts\run.ps1'
Write-Host ""
