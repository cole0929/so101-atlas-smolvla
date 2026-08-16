@echo off
rem ============================================================
rem  restart dsh web — 杀掉占用 3080 端口的旧实例并重新启动
rem  用法: 双击运行；启动后浏览器打开 http://127.0.0.1:3080
rem ============================================================

echo [1/3] Stopping old dsh web on port 3080 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":3080" ^| findstr "LISTENING"') do (
  taskkill /PID %%p /F >nul 2>&1
)

timeout /t 2 /nobreak >nul

echo [2/3] Starting dsh web ...
cd /d C:\Users\98384
node "C:\Users\98384\AppData\Local\npm-cache\_npx\1e7f6d9597241db0\node_modules\@deepseek-ai\dsh\lib\bin.js" web

echo.
echo [3/3] dsh web stopped. If this was an error, copy the text above.
pause
