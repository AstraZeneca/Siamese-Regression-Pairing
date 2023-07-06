#!/bin/bash
python finetuneRegr_k_fold.py --name lipo --study_name lipo --data_path lipo --random 1 --drp 0.0
python finetuneRegr_k_fold.py --name freesolv --study_name freesolv --data_path freesolv --random 1 --drp 0.0
python finetuneRegr_k_fold.py --name delaney --study_name delaney --data_path delaney --random 1  --drp 0.0

exec bash
