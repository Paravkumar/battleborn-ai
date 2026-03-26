@echo off
cd /d "C:\Users\parav\Documents\New project 2"

echo Step 1: Generate synthetic tickets...
.venv\Scripts\python.exe scripts\generate_synthetic_tickets.py --access-count 1 --billing-count 1 --integration-count 1 --escalation-count 1 --output artifacts\generated_tickets_more.json
if %errorlevel% neq 0 (
    echo Error in step 1
    exit /b %errorlevel%
)

echo.
echo Step 2: Generate teacher runs...
.venv\Scripts\python.exe scripts\generate_teacher_runs.py --all-tickets --tickets-file artifacts\generated_tickets_more.json --model llama3.1:70b --require-ollama-plan --require-ollama-draft --output-jsonl artifacts\teacher_runs_more.jsonl
if %errorlevel% neq 0 (
    echo Error in step 2
    exit /b %errorlevel%
)

echo.
echo Step 3: Append non-empty lines to teacher_runs.jsonl...
powershell -Command "^
Get-Content 'artifacts\teacher_runs_more.jsonl' ^| Where-Object { $_ -match '\S' } ^| Add-Content 'artifacts\teacher_runs.jsonl'
"

echo.
echo Step 4: Print counts...
powershell -Command "^
$more_count = (Get-Content 'artifacts\teacher_runs_more.jsonl' ^| Where-Object { $_ -match '\S' } ^| Measure-Object -Line).Lines; ^
$total_count = (Get-Content 'artifacts\teacher_runs.jsonl' ^| Where-Object { $_ -match '\S' } ^| Measure-Object -Line).Lines; ^
Write-Host ('Lines in artifacts\teacher_runs_more.jsonl: ' + $more_count); ^
Write-Host ('Total lines in artifacts\teacher_runs.jsonl: ' + $total_count)
"

echo Pipeline completed!
