#!/bin/bash

python finetuneRegr_k_fold.py --name delaney --study_name delaney --data_path delaney/ --drp 0.0

python finetuneRegr_k_fold.py --name delaney --study_name delaney --data_path delaney/ --drp 0.05

python finetuneRegr_k_fold.py --name delaney --study_name delaney --data_path delaney/ --drp 0.1

python finetuneRegr_k_fold.py --name delaney --study_name delaney --data_path delaney/ --drp 0.17