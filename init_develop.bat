@echo off
rem Create environment
py -m venv .venv

rem Install requirements
call .venv/scripts/pip.exe install -r requirements.txt

rem When using local apread, add the following line to the end of the file
rem     "python.analysis.extraPaths": ["../APReader"]

rem Create .vscode/settings.json
mkdir .vscode
rem Add to settings.json
echo { >> .vscode/settings.json
echo     "python.analysis.extraPaths": ["../APReader"] >> .vscode/settings.json
echo } >> .vscode/settings.json
