@echo off
REM Script chay analysis trong conda environment tf

echo ====================================================================================================
echo  RUNNING ANALYSIS IN TF ENVIRONMENT
echo ====================================================================================================

REM Activate conda environment
call conda activate tf

REM Run analysis
python -X utf8 analyze_existing_results.py --plot_predictions --num_samples 10

echo.
echo ====================================================================================================
echo  DONE!
echo ====================================================================================================

pause
