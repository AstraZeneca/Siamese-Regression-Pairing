#!/bin/bash
python finetuneRegr_k_fold.py --name lipo --study_name lipo --data_path lipo --drp 0.0


python finetuneRegr_k_fold.py --name lipo --study_name lipo --data_path lipo --drp 0.05


python finetuneRegr_k_fold.py --name lipo --study_name lipo --data_path lipo --drp 0.1


python finetuneRegr_k_fold.py --name lipo --study_name lipo --data_path lipo --drp 0.17
