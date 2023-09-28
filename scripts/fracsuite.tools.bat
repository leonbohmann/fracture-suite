@echo off
cd /d %~dp0/..
call .venv\scripts\python.exe -m fracsuite.tools %*