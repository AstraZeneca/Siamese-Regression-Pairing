#!/bin/bash
python mlp.py -s freesolv_top1.yml -st 1 -f freesolv
python mlp.py -s lipo_top1.yml -st 1 -f lipo
python mlp.py -s esol_top1.yml -st 1 -f delaney
python mlp.py -s freesolv_all.yml -st 0 -f freesolv
python mlp.py -s lipo_all.yml -st 0 -f lipo
python mlp.py -s esol_all.yml -st 0 -f delaney
python mlp_fp.py -s freesolv_mlp.yml -f freesolv
python mlp_fp.py -s lipo_mlp.yml -f lipo
python mlp_fp.py -s esol_mlp.yml -f delaney
exec bash 
