#!/bin/bash
python mlp_snn.py -s freesolv_mlp_snn_top1.yml -st 1 -f freesolv
python mlp_snn.py -s lipo_mlp_snn_top1.yml -st 1 -f lipo
python mlp_snn.py -s esol_mlp_snn_top1.yml -st 1 -f delaney
python mlp_snn.py -s freesolv_mlp_snn_all.yml -st 0 -f freesolv
python mlp_snn.py -s lipo_mlp_snn_all.yml -st 0 -f lipo
python mlp_snn.py -s esol_mlp_snn_all.yml -st 0 -f delaney
exec bash 
