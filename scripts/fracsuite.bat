@echo off
cd /d %~dp0/..
call .venv\scripts\python.exe -W ignore -m fracsuite %*