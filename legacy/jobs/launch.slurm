#!/bin/bash

#SBATCH --job-name=an-GraphKAno
#SBATCH --partition=normal
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL    # Type can be BEGIN, END, FAIL, ALL(any statchange).
#SBATCH --output=job-%j.out
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mail-user=jerry.lacmou.zeutouo@u-picardie.fr

#pip3 install --no-cache-dir --upgrade -r requirements.txt
#source $HOME/supports/ange/bashrc
ml load compiler/gnu11
#ml load compiler/python/3.12.1
ml load compiler/python/3.11.2
ml load libffi
#source /users/jlacmou/supports/ange/ange-packs/bin/activate
pip3 install --no-cache-dir -r requirements.txt
echo "--- BEGIN ---"
date
python script.py
echo "--- END ---"
date
