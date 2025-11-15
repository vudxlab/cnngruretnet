@echo off
REM Script để chạy training với conda environment tf
REM Sử dụng: run.bat [models] [epochs]
REM Ví dụ: run.bat conv1d_gru 10
REM        run.bat "conv1d_gru gru" 100

setlocal

REM Activate conda environment
call conda activate tf

REM Default values
set MODELS=%1
set EPOCHS=%2

if "%MODELS%"=="" set MODELS=conv1d_gru
if "%EPOCHS%"=="" set EPOCHS=1000

echo ========================================
echo Training Models: %MODELS%
echo Epochs: %EPOCHS%
echo ========================================

REM Run training
python main.py --models %MODELS% --epochs %EPOCHS% --batch_size 64

REM Deactivate
call conda deactivate

pause
