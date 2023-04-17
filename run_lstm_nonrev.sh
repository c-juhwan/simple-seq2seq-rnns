clear

#python main.py --job=preprocessing

# Training
python main.py --job=training --model_type=lstm --reverse_input=False --desc="LSTM Basic T" --device=cuda:2

# Testing
#python main.py --job=testing --model_type=lstm --reverse_input=False --decoding_strategy=greedy --desc="LSTM Basic G" --device=cuda:2
#python main.py --job=testing --model_type=lstm --reverse_input=False --decoding_strategy=beam --desc="LSTM Basic B5" --device=cuda:2
