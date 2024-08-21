@echo off
set "args=%*"
cd /d %~dp0/..
call .venv\scripts\python.exe -W ignore -m fracsuite %args%