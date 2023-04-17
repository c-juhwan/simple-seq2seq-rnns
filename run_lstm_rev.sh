clear

#python main.py --job=preprocessing

# Training
python main.py --job=training --model_type=lstm --reverse_input=True --desc="LSTM Reverse T" --device=cuda:2

# Testing
python main.py --job=testing --model_type=lstm --reverse_input=True --decoding_strategy=greedy --desc="LSTM Reverse G" --device=cuda:2
python main.py --job=testing --model_type=lstm --reverse_input=True --decoding_strategy=beam --desc="LSTM Reverse B5" --device=cuda:2
