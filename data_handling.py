import numpy as np
import matplotlib.pyplot as plt

from CONSTANTS import SOS_token, EOS_token, UNIQUE_WORDS

def load_dataset(language, dataset_type, dataset_path):
    with open('{}/{}.{}.txt'.format(dataset_path, dataset_type, language), 'r', encoding='utf-8') as data:
        sentences = [sentence.rstrip('\n').split(' ') for sentence in data]
    return sentences

def create_vocab(sentences):
    language_dict = dict()
    for sentence in sentences:
        for word in sentence:
            if word in language_dict:
                language_dict[word] += 1
            else:
                language_dict[word] = 1
    word_frequency = [(word, language_dict[word]) for word in language_dict]
    sorted_word_frequency = sorted(word_frequency, key=lambda x: x[1], reverse=True)
    if len(sorted_word_frequency) > UNIQUE_WORDS - 3:
        sorted_word_frequency = sorted_word_frequency[:UNIQUE_WORDS - 3]
    vocab_dict = {'<unk>': 0, '<s>': 1, '</s>': 2}
    counter = 3
    for word_tuple in sorted_word_frequency:
        if word_tuple[0] not in vocab_dict:
            vocab_dict[word_tuple[0]] = counter
            counter += 1
    return vocab_dict

def word_from_dict(word, lan_dict):
    if word in lan_dict:
        return lan_dict[word]
    else:
        return lan_dict['<unk>']

def process_sentences(sentences, vocab, translate_to=False):
	X = list()
	for index, sentence in enumerate(sentences):
		if translate_to:
			index_sentence = [SOS_token] + [word_from_dict(word, vocab) for word in sentence] + [EOS_token]
		else:
			index_sentence = [word_from_dict(word, vocab) for word in sentence] + [EOS_token]
		X.append(index_sentence)
	X = np.array([np.array(Xi) for Xi in X])
	return X

def process_train_test_datasets(language_in='en', language_out='vi', dataset_path='data'):
	processed_dataset = dict()
	for dataset_type in ['train', 'test']:
		for language in [language_in, language_out]:
			sentences = load_dataset(language, dataset_type, dataset_path)
			# Create vocab if not available 
			vocab_key = 'vocab_{}'.format(language)
			if vocab_key not in processed_dataset:
				processed_dataset[vocab_key] = create_vocab(sentences)
			dataset_key = '{}_{}'.format(dataset_type, language)
			vocab = processed_dataset[vocab_key]
			processed_dataset[dataset_key] = process_sentences(
				sentences, vocab, translate_to=(language == language_out))
	return processed_dataset

def showPlot(points, points2):
    plt.plot(points)
    plt.plot(points2)
    plt.show()