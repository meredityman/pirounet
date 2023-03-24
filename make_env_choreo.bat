@echo off
set CUDA_VISIBLE_DEVICES=0

set cur_dir=%CD%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\libnvvp;%PATH%
call C:\ProgramData\Miniconda3\Scripts\activate.bat C:\ProgramData\Miniconda3

call conda update -n base -c defaults conda
call conda env create -f pirounet/environment.yml
call conda activate choreo

call conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt 
