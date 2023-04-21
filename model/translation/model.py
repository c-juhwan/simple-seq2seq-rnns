# Standard Library Modules
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
from collections import defaultdict
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class TranslationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(TranslationModel, self).__init__()
        self.args = args

        # Embedding
        self.src_embedding = nn.Embedding(args.src_vocab_size, args.embed_size)
        self.tgt_embedding = nn.Embedding(args.tgt_vocab_size, args.embed_size)

        # Encoder & Decoder
        self.model_type = args.model_type
        if self.model_type == 'lstm':
            self.encoder = nn.LSTM(input_size=args.embed_size, hidden_size=args.hidden_size,
                                   num_layers=args.encoder_rnn_nlayers,
                                   bidirectional=False, batch_first=True)
            self.decoder = nn.LSTM(input_size=args.embed_size, hidden_size=args.hidden_size,
                                   num_layers=args.decoder_rnn_nlayers,
                                   bidirectional=False, batch_first=True)
        elif self.model_type == 'gru':
            self.encoder = nn.GRU(input_size=args.embed_size, hidden_size=args.hidden_size,
                                  num_layers=args.encoder_rnn_nlayers,
                                  bidirectional=False, batch_first=True)
            self.decoder = nn.GRU(input_size=args.embed_size, hidden_size=args.hidden_size,
                                  num_layers=args.decoder_rnn_nlayers,
                                  bidirectional=False, batch_first=True)

        # Output Layer
        self.out = nn.Sequential(
            nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(in_features=args.hidden_size * 4, out_features=args.tgt_vocab_size)
        )

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # Embedding
        src_embed = self.src_embedding(source_ids)
        tgt_embed = self.tgt_embedding(target_ids)

        if self.args.model_type == 'lstm':
            # Encoder
            _, (encoder_hidden, encoder_cell) = self.encoder(src_embed) # encoder_outputs: (batch_size, max_seq_len, hidden_size)
            # Decoder
            decoder_outputs, _ = self.decoder(tgt_embed, (encoder_hidden, encoder_cell)) # decoder_outputs: (batch_size, max_seq_len, hidden_size)
        elif self.args.model_type == 'gru':
            # Encoder
            _, encoder_hidden = self.encoder(src_embed)
            # Decoder
            decoder_outputs, _ = self.decoder(tgt_embed, encoder_hidden)

        # Output Layer
        target_logits = self.out(decoder_outputs) # target_logits: (batch_size, max_seq_len, tgt_vocab_size)
        return target_logits

    def greedy_generate(self, source_ids: torch.Tensor) -> torch.Tensor:
        # Greedy decoding
        batch_size = source_ids.size(0)

        # Encoder
        src_embed = self.src_embedding(source_ids)
        if self.args.model_type == 'lstm':
            _, (encoder_hidden, encoder_cell) = self.encoder(src_embed)
        elif self.args.model_type == 'gru':
            _, encoder_hidden = self.encoder(src_embed)

        # Initialize decoder input
        decoder_input = torch.tensor([self.args.bos_token_id] * batch_size, dtype=torch.long, device=self.args.device).unsqueeze(1) # decoder_input: (batch_size, 1)

        for step in range (self.args.max_seq_len - 1): # -1 for <bos> token
            # Embedding
            tgt_embed = self.tgt_embedding(decoder_input) # tgt_embed: (batch_size, cur_seq_len, embed_size)

            # Decoder
            if self.args.model_type == 'lstm':
                decoder_outputs, _ = self.decoder(tgt_embed, (encoder_hidden, encoder_cell))
            elif self.args.model_type == 'gru':
                decoder_outputs, _ = self.decoder(tgt_embed, encoder_hidden)

            # Output Layer
            target_logits = self.out(decoder_outputs) # target_logits: (batch_size, cur_seq_len, tgt_vocab_size)
            next_token_logits = target_logits[:, -1, :] # next_token_logits: (batch_size, tgt_vocab_size)
            # Avoid generating <s> and <pad> tokens
            next_token_logits[:, self.args.bos_token_id] = -float('inf')
            next_token_logits[:, self.args.pad_token_id] = -float('inf')
            # Generate the next token
            next_token = torch.argmax(next_token_logits, dim=1).unsqueeze(1) # (batch_size, 1)
            # Concatenate next token to decoder_input
            decoder_input = torch.cat([decoder_input, next_token], dim=1) # (batch_size, cur_seq_len + 1)

        # Remove <bos> token from the output
        best_seq = decoder_input[:, 1:]

        return best_seq

    def beam_generate(self, source_ids: torch.Tensor) -> torch.Tensor:
        # Beam search
        batch_size = source_ids.size(0)
        assert batch_size == 1, 'Beam search only supports batch size of 1'
        beam_size = self.args.beam_size

        # Initialize the decoder input with <bos> token
        decoder_input = torch.tensor([self.args.bos_token_id] * beam_size, device=source_ids.device).unsqueeze(1) # (beam_size, 1)

        # Initialize beam search variables
        current_beam_scores = torch.zeros(beam_size, device=source_ids.device) # (beam_size)
        final_beam_scores = torch.zeros(beam_size, device=source_ids.device) # (beam_size)
        final_beam_seqs = torch.zeros(beam_size, self.args.max_seq_len, device=source_ids.device).long() # (beam_size, max_seq_len-1)
        beam_complete = torch.zeros(beam_size, device=source_ids.device).bool() # (beam_size)

        # Encoder
        src_embed = self.src_embedding(source_ids)
        if self.args.model_type == 'lstm':
            _, (encoder_hidden, encoder_cell) = self.encoder(src_embed)
            encoder_cell = encoder_cell.repeat(1, beam_size, 1) # Cell is LSTM specific
        elif self.args.model_type == 'gru':
            _, encoder_hidden = self.encoder(src_embed)

        encoder_hidden = encoder_hidden.repeat(1, beam_size, 1)

        # Beam search
        for step in range(self.args.max_seq_len - 1): # -1 for <bos> token
            # Embedding
            tgt_embed = self.tgt_embedding(decoder_input) # tgt_embed: (beam_size, cur_seq_len, embed_size)

            # Decoder
            if self.args.model_type == 'lstm':
                decoder_outputs, _ = self.decoder(tgt_embed, (encoder_hidden, encoder_cell))
            elif self.args.model_type == 'gru':
                decoder_outputs, _ = self.decoder(tgt_embed, encoder_hidden)

            # Output Layer
            target_logits = self.out(decoder_outputs) # target_logits: (batch_size, cur_seq_len, tgt_vocab_size)
            target_score = F.log_softmax(target_logits[:, -1, :], dim=1) # target_score: (beam_size, tgt_vocab_size)

            target_score[:, self.args.bos_token_id] = -float('inf') # Avoid generating <s> token
            target_score[:, self.args.pad_token_id] = -float('inf') # Avoid generating <pad> token
            if step == 0:
                target_score[:, self.args.eos_token_id] = -float('inf') # Avoid generating <eos> token at the first step

                # As we are using the same decoder input for all beams, we need to make sure that the first token of each beam is different
                # Get the top-k tokens for first beam
                topk_score, topk_token = target_score[0, :].topk(beam_size, dim=0, largest=True, sorted=True) # (beam_size)
                topk_beam_idx = torch.arange(beam_size, device=source_ids.device) # (beam_size)
                topk_token_idx = topk_token # (beam_size)
            else:
                next_token_score = current_beam_scores.unsqueeze(1) + target_score # (beam_size, tgt_vocab_size)
                next_token_score = next_token_score.view(-1) # (beam_size * tgt_vocab_size)

                # Get the top k tokens but avoid getting the same token across different beams
                topk_score, topk_token = torch.topk(next_token_score, beam_size, dim=0, largest=True, sorted=True) # (beam_size)
                topk_beam_idx = topk_token // self.args.tgt_vocab_size # (beam_size)
                topk_token_idx = topk_token % self.args.tgt_vocab_size # (beam_size)

            # Update the current beam tokens and scores
            current_beam_scores = topk_score # (beam_size)

            # Update the beam sequences - attach the new word to the end of the current beam sequence
            # load the top beam_size sequences for each batch
            # and attach the new word to the end of the current beam sequence
            cur_beam_seq = decoder_input.view(beam_size, -1) # (beam_size, cur_seq_len)
            new_beam_seq = cur_beam_seq[topk_beam_idx, :] # (beam_size, cur_seq_len) - topk_beam_idx is broadcasted to (beam_size, cur_seq_len)
            decoder_input = torch.cat([new_beam_seq, topk_token_idx.unsqueeze(1)], dim=1) # (beam_size, cur_seq_len + 1)

            # If the <eos> token is generated,
            # set the score of the <eos> token to -inf so that it is not generated again
            # and save the sequence
            for beam_idx, token_idx in enumerate(topk_token_idx):
                if beam_complete[beam_idx]: # If the beam has already generated the <eos> token, skip
                    continue
                if token_idx == self.args.eos_token_id:
                    final_beam_scores[beam_idx] = current_beam_scores[beam_idx] # Save the sequence score
                    current_beam_scores[beam_idx] = -float('inf') # Set the score of the <eos> token to -inf so that it is not generated again
                    final_beam_seqs[beam_idx, :decoder_input.size(1)] = decoder_input[beam_idx, :] # Save the sequence
                    beam_complete[beam_idx] = True

            # If all the sequences have generated the <eos> token, break
            if beam_complete.all():
                break

        # If there are no completed sequences, save current sequences
        if not beam_complete.any():
            final_beam_seqs = decoder_input
            final_beam_scores = current_beam_scores

        # Beam Length Normalization
        each_seq_len = torch.sum(final_beam_seqs != self.args.pad_token_id, dim=1).float() # (beam_size)
        length_penalty = (((each_seq_len + beam_size) ** self.args.beam_alpha) / ((beam_size +1) ** self.args.beam_alpha))
        final_beam_scores = final_beam_scores / length_penalty

        # Find the best sequence
        best_seq_idx = torch.argmax(final_beam_scores).item()
        best_seq = final_beam_seqs[best_seq_idx, 1:] # Remove the <bos> token

        return best_seq.unsqueeze(0) # (1, max_seq_len - 1) - remove the <bos> token
