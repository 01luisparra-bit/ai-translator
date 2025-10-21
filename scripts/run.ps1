Write-Host "Running AI Translator..."
.\.venv\Scripts\activate

# Create a temporary Python runner script
$tempPy = Join-Path $env:TEMP "run_ai_translator.py"

@"
import sys
import runpy
from pathlib import Path

# Force ROOT to project directory, not PowerShell temp
ROOT = Path(r'C:\Users\Lp\Desktop\ai-translator')
SRC_PYTHON = ROOT / 'src' / 'python'
MAIN_FILE = SRC_PYTHON / 'ai_translator' / 'main.py'

if not MAIN_FILE.exists():
    raise FileNotFoundError(f'main.py not found at {MAIN_FILE}')

if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

print(f'Executing: {MAIN_FILE}')
runpy.run_path(str(MAIN_FILE), run_name='__main__')
"@ | Out-File -FilePath $tempPy -Encoding utf8

python $tempPy
Remove-Item $tempPy -Force
