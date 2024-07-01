@echo off
cd /d %~dp0/..
call fracsuite splinters compare-manual test1 --x-range 10:9000 --y-range 0:90
call fracsuite splinters compare-manual test2 --x-range 10:9000 --y-range 0:90 
call fracsuite splinters compare-manual test3 --x-range 10:9000 --y-range 0:90 