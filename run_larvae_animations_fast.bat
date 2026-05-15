@echo off
REM Quick launcher: same as python create_larvae_animations.py --fast ...
set "SCRIPT=%~dp0create_larvae_animations.py"
python "%SCRIPT%" --fast %*
