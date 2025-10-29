----Macos


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
-----
----Microsoft
powershell


python -m venv .venv
python --version
.venv\Scripts\Activate.ps1
if error
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
pip install fastapi uvicorn pydantic
uvicorn app:app --reload --port 8000
