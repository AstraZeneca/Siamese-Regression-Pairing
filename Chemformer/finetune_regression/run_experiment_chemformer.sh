#!/bin/bash
python finetuneRegr.py --name delaney --data_path delaney/
python finetuneRegr.py --name freesolv --data_path freesolv/
python finetuneRegr.py --name lipo --data_path lipo/
exec bash
