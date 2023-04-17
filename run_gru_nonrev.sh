clear

#python main.py --job=preprocessing

# Training
python main.py --job=training --model_type=gru --reverse_input=False --desc="GRU Basic T" --device=cuda:3

# Testing
python main.py --job=testing --model_type=gru --reverse_input=False --decoding_strategy=greedy --desc="GRU Basic G" --device=cuda:3
python main.py --job=testing --model_type=gru --reverse_input=False --decoding_strategy=beam --desc="GRU Basic B5" --device=cuda:3
