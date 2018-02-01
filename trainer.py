import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from CONSTANTS import LEARNING_RATE, EPOCHS, DEBUG_LENGTH, HIDDEN_SIZE, DEBUG, use_cuda
from data_handling import process_train_test_datasets, showPlot
from attention_nmt import EncoderRNN, Attention, DecoderRNN

def train_step(in_sentence, out_sentence, model_sections, optimizing_params):
    # pass input sentence through encoder
    encoder_hidden = model_sections['encoder'].initHidden()
    input = Variable(torch.LongTensor(in_sentence).view(1,-1))
    input = input.cuda() if use_cuda else input
    encoder_output, encoder_hidden = model_sections['encoder'](input, encoder_hidden)
    
    # initialize decoder hidden layer with final encoder hidden layer
    decoder_hidden = encoder_hidden[0].clone().view(1,1,-1)
    
    # initialize loss to 0
    loss = 0 
    
    # pass encoder output through attention + decoder
    for word_index in out_sentence:
        decoder_input = Variable(torch.LongTensor([[int(word_index)]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        input_context = model_sections['attention'](decoder_hidden, encoder_output) 
        decoder_output, decoder_hidden = model_sections['decoder'](input_context, decoder_hidden, decoder_input)
        loss += optimizing_params['loss_function'](decoder_output.view(1,-1), decoder_input.view(-1))
    loss.backward()
    optimizing_params['enc_optimizer'].step()
    optimizing_params['att_optimizer'].step()
    optimizing_params['dec_optimizer'].step()
    
    return loss.data[0] / len(out_sentence)

def test_step(in_sentence, out_sentence, model_sections, optimizing_params):
    # pass input sentence through encoder
    encoder_hidden = encoder.initHidden()
    input = Variable(torch.LongTensor(in_sentence).view(1,-1))
    input = input.cuda() if use_cuda else input
    encoder_output, encoder_hidden = model_sections['encoder'](input, encoder_hidden)
    
    # initialize decoder hidden layer with final encoder hidden layer
    decoder_hidden = encoder_hidden[0].clone().view(1,1,-1)
    
    # initialize loss to 0
    loss = 0 
    
    # pass encoder output through attention + decoder
    for word_index in out_sentence:
        decoder_input = Variable(torch.LongTensor([[int(word_index)]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        input_context = model_sections['attention'](decoder_hidden, encoder_output) 
        decoder_output, decoder_hidden = model_sections['decoder'](input_context, decoder_hidden, decoder_input)
        loss += optimizing_params['loss_function'](decoder_output.view(1,-1), decoder_input.view(-1))
    return loss.data[0] / len(out_sentence)

def train(in_sentences, out_sentences, in_test_sentences, out_test_sentences, model_sections, optimizing_params):
    training_losses = list()
    testing_losses = list()
    current_loss = 0
    testing_loss = 0
    for epoch_num in range(EPOCHS):
        model_sections['encoder'].zero_grad()
        model_sections['decoder'].zero_grad()
        model_sections['attention'].zero_grad()
        timestamp1 = time.time()
        timestamp3 = time.time()
        if DEBUG:
            train_set_length = DEBUG_LENGTH
            test_set_length = int(DEBUG_LENGTH / 10)
        else:
            train_set_length = len(in_sentences)
            test_set_length = len(in_test_sentences)
        for index, (in_sentence, out_sentence) in enumerate(
                zip(in_sentences[:train_set_length], out_sentences[:train_set_length])):
            current_loss += train_step(in_sentence, out_sentence, model_sections, optimizing_params)
            if index % 10 == 0:
                timestamp4 = time.time()
                print('Epoch: {} | STEP: {} | TRAINING Loss: {} | Time: {}'.format(
                    epoch_num + 1, index, current_loss / train_set_length, timestamp4 - timestamp3))
                timestamp3 = time.time()
        for in_test_sentence, out_test_sentence in zip(
                in_test_sentences[:test_set_length], out_test_sentences[:test_set_length]):
            testing_loss += test_step(in_test_sentence, out_test_sentence, model_sections, optimizing_params)
        timestamp2 = time.time()
        print('\nEpoch: {} | TRAINING Loss: {} | TESTING Loss: {} | Time: {}\n'.format(
            epoch_num + 1, current_loss / train_set_length, testing_loss / test_set_length, timestamp2 - timestamp1))
        training_losses.append(current_loss / train_set_length)
        testing_losses.append(testing_loss  / test_set_length)
        current_loss = 0
        testing_loss = 0
    return training_losses, testing_losses

def model_initializer(in_lang_len, out_lang_len):
    # Instantiate model and optimizer
    encoder = EncoderRNN(in_lang_len, HIDDEN_SIZE).cuda() if use_cuda else EncoderRNN(in_lang_len, HIDDEN_SIZE)
    attention = Attention(HIDDEN_SIZE).cuda() if use_cuda else Attention(HIDDEN_SIZE)
    decoder = DecoderRNN(out_lang_len, HIDDEN_SIZE).cuda() if use_cuda else DecoderRNN(out_lang_len, HIDDEN_SIZE)
    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
    attention_optimizer = optim.SGD(attention.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)

    # save model and optimizer params to dict
    model_sections = {'encoder': encoder, 
                      'attention': attention, 
                      'decoder': decoder,
                     }

    optimizing_params = {'loss_function': criterion, 
                         'enc_optimizer': encoder_optimizer,
                         'att_optimizer': attention_optimizer,
                         'dec_optimizer': decoder_optimizer,
                        }
    return model_sections, optimizing_params

def main(language_in, language_out, dataset_path):
    # Load dataset
    processed_dataset = process_train_test_datasets(language_in=language_in, 
                                                    language_out=language_out,
                                                    dataset_path=dataset_path)
    print('Dataset loaded.')
    # Initialize model
    model_sections, optimizing_params = model_initializer(
                                            len(processed_dataset['vocab_{}'.format(language_in)]),
                                            len(processed_dataset['vocab_{}'.format(language_out)]))
    print('Model initialized, starting training...\n')
    # Train model with dataset
    training_losses, testing_losses = train(processed_dataset['train_{}'.format(language_in)], 
                                            processed_dataset['train_{}'.format(language_out)], 
                                            processed_dataset['test_{}'.format(language_in)], 
                                            processed_dataset['test_{}'.format(language_out)], 
                                            model_sections, 
                                            optimizing_params)
    print('Training has ended.')

if __name__ == "__main__":
    if len(sys.argv) == 4:
        langauge_in, language_out, dataset_path = sys.argv[1], sys.argv[2], sys.argv[3]
        main(langauge_in, language_out, dataset_path)
    else:
        print('Command format as follows: python3 trainer.py language_in language_out dataset_path')
        print('Example: python3 trainer.py en vi data')



