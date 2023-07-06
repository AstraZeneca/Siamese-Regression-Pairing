#!/bin/bash

python finetuneRegr_k_fold.py --name freesolv --study_name freesolv --data_path freesolv --drp 0.0

python finetuneRegr_k_fold.py --name freesolv --study_name freesolv --data_path freesolv --drp 0.05

python finetuneRegr_k_fold.py --name freesolv --study_name freesolv --data_path freesolv/ --drp 0.1



python finetuneRegr_k_fold.py --name freesolv --study_name freesolv --data_path freesolv/ --drp 0.17
