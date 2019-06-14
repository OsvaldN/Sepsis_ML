#!/bin/sh
source /pkgs/scripts/use-anaconda3.sh
source activate diagnostics
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 0 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 0 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 0 -lr 1 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 16 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 16 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 16 -lr 1 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 32 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 32 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 32 -layer 2 -hidden 64 -fcl 32 -lr 1 -mom 0.9
wait

python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 0 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 0 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 0 -lr 1 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 16 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 16 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 16 -lr 1 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 32 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 32 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 64 -fcl 32 -lr 1 -mom 0.9
wait

python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 0 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 0 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 0 -lr 1 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 16 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 16 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 16 -lr 1 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 32 -lr 1 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 32 -lr 1 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/LSTM_train.py -emb 64 -layer 2 -hidden 96 -fcl 32 -lr 1 -mom 0.9
wait

python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 -fcl 16 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 -fcl 16 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 -fcl 16 -lr 0.001 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 32 32 -fcl 16 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 32 32 -fcl 16 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 32 32 -fcl 16 -lr 0.001 -mom 0.9
wait

python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 16 -fcl 16 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 16 -fcl 16 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 16 -fcl 16 -lr 0.001 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 32 32 16 -fcl 16 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 32 32 16 -fcl 16 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 32 32 16 -fcl 16 -lr 0.001 -mom 0.9
wait

python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 32 32 -fcl 16 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 32 32 -fcl 16 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 32 -layer 32 32 32 32 -fcl 16 -lr 0.001 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 64 64 64 -fcl 32 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 64 64 64 -fcl 32 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 64 64 64 -fcl 32 -lr 0.001 -mom 0.9
wait

python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 96 96 -fcl 32 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 96 96 -fcl 32 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 96 96 -fcl 32 -lr 0.001 -mom 0.9
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 128 128 -fcl 32 -lr 0.001 -mom 0.1
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 128 128 -fcl 32 -lr 0.001 -mom 0.5
wait
python /home/osvald/Projects/Diagnostics/Sepsis/Sepsis_2019_PhysioNet/TCN_train.py -emb 64 -layer 128 128 -fcl 32 -lr 0.001 -mom 0.9
wait