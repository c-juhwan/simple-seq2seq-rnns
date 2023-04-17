clear

#python main.py --job=preprocessing

# Training
python main.py --job=training --model_type=gru --reverse_input=True --desc="GRU Reverse T" --device=cuda:3

# Testing
python main.py --job=testing --model_type=gru --reverse_input=True --decoding_strategy=greedy --desc="GRU Reverse G" --device=cuda:3
python main.py --job=testing --model_type=gru --reverse_input=True --decoding_strategy=beam --desc="GRU Reverse B5" --device=cuda:3
