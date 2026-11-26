@echo off
chcp 65001 >nul
title MF Tracker - coletor de dados
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente Python...
  py -3.10 -m venv .venv 2>nul || py -3 -m venv .venv
)
echo Instalando/atualizando bibliotecas (ccxt, yfinance)...
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
echo.
echo Coletor rodando. Deixe esta janela aberta. Ctrl+C para parar.
".venv\Scripts\python.exe" coletor.py %*
pause
