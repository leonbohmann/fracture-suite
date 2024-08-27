@echo off
cd /d %~dp0/..
call .venv\scripts\python -m fracsuite %*