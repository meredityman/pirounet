@echo off
set CUDA_VISIBLE_DEVICES=0

:conda
set cur_dir=%CD%
echo .. conda :: env choreo :: py 3.8 .. cuda 11.3 .. torch 1.11
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\libnvvp;%PATH%
call C:\ProgramData\Miniconda3\Scripts\activate.bat C:\ProgramData\Miniconda3
call activate choreo
cd %cur_dir%

if %1=set goto end

:work
call %1 %2 %3 %4 %5 %6 %7 %8 %9

:fin
echo .. conda exit ..
call conda deactivate
call conda deactivate
goto end

:end
