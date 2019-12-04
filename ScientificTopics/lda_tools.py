import argparse
import base64
import json
import logging
import os
import re
import subprocess
import warnings
import zlib
from collections import Counter
from functools import lru_cache
from math import ceil

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("Unable to find matplotlib, plotting is disabled")
    plt = None
import numpy
import pyximport

from ScientificTopics.sentence_segmentation import Tokenizer


pyximport.install(
    setup_args={'include_dirs': numpy.get_include()})
from ScientificTopics import lda_infer


class LDAInfer(object):
    """
    LDAInfer is a class that reads LDA models and infer topics for new documents.
    To use LDAInfer, four model files are required:

    1. Punkt parameter file, for segmenting paragraphs into sentences.
    2. SentencePieceModel file, for tokenizing sentences.
    3. Stopwords file.
    4. LDA result directory, for collecting LDA doc-topic/topic-word tables.

    Two methods, tokenize_sentences() and tokenize_paragraph() are used to
    tokenize text into tokens and convert into vocabulary ids.

    Two methods, infer_topic_fast() and infer_topic() are used to infer topics
    for list of token ids representing a document.
    """
    def __init__(self, lda_result_dir,
                 punkt_model, stopwords, spm_model,
                 beta=0.01, alpha=0.1, num_vocab=None):
        """
        Initialize a LDAInfer class to infer topics for new documents.

        :param lda_result_dir: Directory containing tables of a LDA model.
        :param punkt_model: Filename of Punkt model.
        :param stopwords: Filename of stopwords list.
        :param spm_model: Filename of SentencePiece model.
        :param beta: Beta of LDA model.
        :param alpha: Alpha of LDA model.
        :param num_vocab: Number of vocabularies.
        """
        self.lda_result_dir = lda_result_dir
        self.beta = beta
        self.alpha = alpha
        self.num_vocab = num_vocab
        self._ntopics = None

        if self.num_vocab is not None:
            self._betasum = self.beta * self.num_vocab

        self.tokenizer = Tokenizer(punkt_model, spm_model)
        self.spm_tokenizer = self.tokenizer.spm_tokenizer
        with open(stopwords, encoding='utf8') as f:
            self.stopwords = [x.strip().split()[0] for x in f if x.strip()]
        self.stopwords = set(self.spm_tokenizer.PieceToId(x) for x in self.stopwords)

        self.n_t, self.n_tw = self._read_model_parameters()

    @property
    def ntopics(self):
        """
        Return number of topics.
        """
        return self._ntopics

    @lru_cache(maxsize=1000)
    def _get_p(self, word_id):
        p_array = []
        for i in range(self.ntopics):
            p = self.n_tw[i, word_id] * 1. * self.beta / (self.n_t[i] + self.beta * self.num_vocab)
            p_array.append(p)

        p_array = numpy.asarray(p_array)

        summation = numpy.sum(p_array)
        if summation == 0:
            return numpy.ones(shape=(self.ntopics,)) / self.ntopics
        else:
            return p_array / numpy.sum(p_array)

    def _propose_word(self, word_id, size=None):
        return numpy.random.choice(self.ntopics, p=self._get_p(word_id), size=size)

    def _propose_doc(self, total_tokens, size=None):
        """-Int for doc_topics[-i]"""
        rand_numbers = numpy.random.rand(*size) * (total_tokens + self.alpha * self.ntopics)
        proposal = numpy.empty(shape=size, dtype=numpy.int)

        cond = rand_numbers < total_tokens

        take_doc_topics = numpy.where(cond)
        proposal[take_doc_topics] = -(rand_numbers[take_doc_topics]).astype(numpy.int)

        take_random_topics = numpy.where(numpy.logical_not(cond))
        proposal[take_random_topics] = numpy.random.choice(self.ntopics, size=len(take_random_topics[0]))
        return proposal

    def tokenize_sentences(self, text, filter_stopwords=True):
        """
        Tokenize words in a paragraph, and return a list of lists, each list representing
        a sentence. All words are converted to id in vocabulary.

        :param text: Paragraph to tokenize
        :param filter_stopwords: Switch to filter out all stopwords
        :return: a list of lists of token ids.
        """
        token_ids_all = self.tokenizer.tokenize_ids(text)
        valid_tokens = []
        for sentence in token_ids_all:
            sentence = [x for x in sentence if x not in self.stopwords and filter_stopwords]
            valid_tokens.append(sentence)
        return valid_tokens

    def tokenize_paragraph(self, text, filter_stopwords=True):
        """
        Tokenize words in a paragraph, and return a list of token ids.
        See also @tokenize_sentences()

        :param text: Paragraph to tokenize
        :param filter_stopwords: Switch to filter out all stopwords
        :return: a list of token ids.
        """
        token_ids_all = self.tokenize_sentences(text, filter_stopwords)
        # Collect
        token_ids_all = sum(token_ids_all, [])

        return token_ids_all

    def generate_html(self, tokens, doc_topics, monte_carlo_states, doc_topics_states):
        html_template_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'plot.template.html'
        )
        with open(html_template_filename) as f:
            html_template = f.read()

        top_words = [x[1] for x in sorted(self.get_top_words().items(), key=lambda x: x[0])]

        data_base64 = base64.b64encode(zlib.compress(json.dumps({
            'top_words': top_words,
            'states': monte_carlo_states.tolist(),
            'n_tw': doc_topics_states.tolist(),
            'tokens': [(not_stop, self.spm_tokenizer.IdToPiece(x)) for not_stop, x in tokens]
        }).encode())).decode()
        return html_template.replace('{{data_base64}}', data_base64)

    def infer_topic_fast(self, doc, iterations=1000, mh_steps=2, plot_mc_states=False):
        """
        Infer topics for a new document. This method uses Cython implementation so it is
        relatively faster than infer_topic().

        Upon finishing, three objects will be returned:
        1. Topic assignments for each word.
        2. Topic assignments of each iteration of the entire Monte-Carlo process.
        3. Number of topic assignments of each iteration.

        :param doc: List of tokens representing a document.
        :param iterations: Number of Monte-Carlo iterations.
        :param mh_steps: Number of Metropolis-Hastings steps for each iteration.
        :param plot_mc_states: Whether to plot the MC states of all iterations.
        :return:
        """
        doc_topics, monte_carlo_states, doc_topics_states = lda_infer.infer_topic(
            numpy.asarray(doc, dtype=numpy.uint32),
            self.n_t, self.n_tw,
            self.alpha, self.beta, iterations, mh_steps
        )

        if plot_mc_states:
            for i in range(self.ntopics):
                t_arr = monte_carlo_states[i]
                if numpy.sum(t_arr) > iterations:
                    plt.plot(list(range(iterations)), t_arr, linewidth=1, label='Topic %d' % i)
            plt.legend()
            plt.show()

        return doc_topics, monte_carlo_states, doc_topics_states

    def infer_topic(self, doc, iterations=1000, mh_steps=2, plot_mc_states=False):
        """
        Infer topics for a new document. This method uses Cython implementation so it is
        relatively faster than infer_topic().

        Upon finishing, three objects will be returned:
        1. Topic assignments for each word.
        2. Topic assignments of each iteration of the entire Monte-Carlo process.
        3. Number of topic assignments of each iteration.

        :param doc: List of tokens representing a document.
        :param iterations: Number of Monte-Carlo iterations.
        :param mh_steps: Number of Metropolis-Hastings steps for each iteration.
        :param plot_mc_states: Whether to plot the MC states of all iterations.
        :return:
        """
        doc_topics = numpy.random.choice(self.ntopics, size=(len(doc),))
        topic_counter = numpy.zeros((self.ntopics,), dtype=numpy.int)
        for t, c in Counter(doc_topics).items():
            topic_counter[t] += c

        monte_carlo_states = numpy.empty((iterations, len(doc)), dtype=numpy.int)
        doc_topics_states = numpy.empty((iterations, self.ntopics), dtype=numpy.int)

        word_proposals = [self._propose_word(x, size=(iterations, mh_steps)) for x in doc]
        doc_proposals = [self._propose_doc(len(doc), size=(iterations, mh_steps)) for _ in doc]
        random_numbers = numpy.random.rand(iterations, len(doc), mh_steps, 2)

        for i in range(iterations):
            monte_carlo_states[i, :] = doc_topics
            doc_topics_states[i, :] = topic_counter

            for j, word_id in enumerate(doc):
                original_topic = doc_topics[j]

                topic = original_topic
                for k in range(mh_steps):
                    new_topic = word_proposals[j][i, k]

                    if new_topic != topic:
                        # Simplified version of cxx::LightDocSampler::Sample()
                        n_td_alpha = topic_counter[new_topic] + self.alpha
                        n_sd_alpha = topic_counter[topic] + self.alpha

                        if topic == original_topic:
                            n_sd_alpha -= 1
                        if new_topic == original_topic:
                            n_td_alpha -= 1

                        pi = n_td_alpha / n_sd_alpha

                        if random_numbers[i, j, k, 0] < pi:
                            topic = new_topic

                    doc_proposal = doc_proposals[j][i, k]
                    new_topic = doc_topics[-doc_proposal] if doc_proposal < 0 else doc_proposal

                    if new_topic != topic:
                        if new_topic != original_topic:
                            n_tw_beta = self.n_tw[new_topic, word_id] + self.beta
                            n_t_beta_sum = self.n_t[new_topic] + self._betasum
                            pi_t = n_tw_beta / n_t_beta_sum
                        else:
                            n_td_alpha = topic_counter[new_topic] + self.alpha - 1
                            n_tw_beta = self.n_tw[new_topic, word_id] + self.beta
                            n_t_beta_sum = self.n_t[new_topic] + self._betasum
                            proposal_t = topic_counter[new_topic] + self.alpha
                            pi_t = (n_td_alpha * n_tw_beta) / (n_t_beta_sum * proposal_t)

                        if topic != original_topic:
                            n_sw_beta = self.n_tw[topic, word_id] + self.beta
                            n_s_beta_sum = self.n_t[topic] + self._betasum
                            pi_s = n_sw_beta / n_s_beta_sum
                        else:
                            n_sd_alpha = topic_counter[topic] + self.alpha
                            n_sw_beta = self.n_tw[topic, word_id] + self.beta
                            n_s_beta_sum = self.n_t[topic] + self._betasum
                            proposal_s = topic_counter[topic] + self.alpha
                            pi_s = (n_sd_alpha * n_sw_beta) / (n_s_beta_sum * proposal_s)

                        pi = pi_t / pi_s

                        if random_numbers[i, j, k, 1] < pi:
                            topic = new_topic

                if topic != original_topic:
                    topic_counter[original_topic] -= 1
                    topic_counter[topic] += 1

                doc_topics[j] = topic

        if plot_mc_states:
            for i in range(self.ntopics):
                t_arr = monte_carlo_states[i]
                if numpy.sum(t_arr) > iterations:
                    plt.plot(list(range(iterations)), t_arr, linewidth=1, label='Topic %d' % i)
            plt.legend()
            plt.show()

        return doc_topics, monte_carlo_states, doc_topics_states

    def get_top_words(self):
        top_words = {}
        for topic in range(len(self.n_tw)):
            sort_index = self.n_tw[topic].argsort()[:-21:-1]
            top_count = self.n_tw[topic][sort_index]
            string_builder = []
            for word_id, count in zip(sort_index, top_count):
                piece_id = self.spm_tokenizer.IdToPiece(int(word_id)) if self.spm_tokenizer is not None else str(
                    word_id)
                string_builder.append(piece_id)
            top_words[topic] = string_builder
        return top_words

    def find_possible_stopwords(self):
        stopwords = Counter()
        for topic in range(len(self.n_tw)):
            total_counts = sum(self.n_tw[topic])
            sort_index = self.n_tw[topic].argsort()[:-21:-1]
            top_count = self.n_tw[topic][sort_index]
            string_builder = []
            for word_id, count in zip(sort_index, top_count):
                piece_id = self.spm_tokenizer.IdToPiece(int(word_id)) if self.spm_tokenizer is not None else str(
                    word_id)
                stopwords[piece_id] += 1
                string_builder.append('%s (%.4f)' % (piece_id, count / total_counts))
            logging.info('Topic %d: %s', topic, ', '.join(string_builder))

        for word, count in sorted(stopwords.items(), key=lambda x: x[1], reverse=True):
            if count > self.ntopics / 10:
                logging.info('Possible stopword: %s in %d topics', word, count)

    def _read_model_parameters_try_cache(self):
        """
        Read cached LDA model tables n_tw and n_t.
        """
        cache_filename = os.path.join(self.lda_result_dir, 'cached_parameters.npz')

        if not os.path.exists(cache_filename):
            raise FileNotFoundError()

        cache_timestamp = os.path.getmtime(cache_filename)
        model_parameters = [x for x in os.listdir(self.lda_result_dir) if re.match(r'server_\d_table_\d.model', x)]
        total_nodes = max(int(x.split('_')[1]) for x in model_parameters)
        for i in range(total_nodes):
            f_timestamp = os.path.getmtime(os.path.join(self.lda_result_dir, 'server_%d_table_1.model' % i))
            if f_timestamp > cache_timestamp:
                raise FileNotFoundError()
            f_timestamp = os.path.getmtime(os.path.join(self.lda_result_dir, 'server_%d_table_0.model' % i))
            if f_timestamp > cache_timestamp:
                raise FileNotFoundError()

        logging.info('Found cached parameters: %s, loading cache...', cache_filename)
        data = numpy.load(cache_filename)
        n_t = data['n_t']
        n_tw = data['n_tw']
        self._ntopics = n_t.shape[0]

        return n_t, n_tw

    def _read_model_parameters(self):
        """
        Read LDA model tables n_tw and n_t.
        """
        try:
            return self._read_model_parameters_try_cache()
        except FileNotFoundError:
            pass

        model_parameters = [x for x in os.listdir(self.lda_result_dir) if re.match(r'server_\d_table_\d.model', x)]
        total_nodes = max(int(x.split('_')[1]) for x in model_parameters)

        # Alphas
        logging.info('Reading word count by topic...')
        n_t = {}
        for i in range(total_nodes):
            with open(os.path.join(self.lda_result_dir, 'server_%d_table_1.model' % i)) as f:
                line = f.readline()
                if not line:
                    continue
                for topic, topic_count in re.findall(r'(\d+):(\d+)', line):
                    n_t[int(topic)] = int(topic_count)
                logging.info('Loaded topic summary table %s', 'server_%d_table_1.model' % i)

        self._ntopics = max(n_t.keys()) + 1
        logging.info('Found %d topics', self.ntopics)
        n_t = numpy.asarray([x[1] for x in sorted(n_t.items())], dtype=numpy.uint32)
        assert len(n_t) == self.ntopics
        logging.debug('Word count by topics: %r', n_t)

        # Word beta
        logging.info('Reading word count by topic and word id...')
        n_tw = numpy.zeros((self.ntopics, self.num_vocab), dtype=numpy.uint32)
        for i in range(total_nodes):
            with open(os.path.join(self.lda_result_dir, 'server_%d_table_0.model' % i)) as f:
                for line in f:
                    if not line.strip():
                        break

                    matches = re.findall(r'(\d+)(?=\s)|(\d+):(\d+)', line)
                    word_id = int(matches[0][0])
                    for _, topic, topic_count in matches[1:]:
                        n_tw[int(topic), word_id] = int(topic_count)

                logging.info('Loaded word topic table %s', 'server_%d_table_0.model' % i)

        numpy.savez(
            os.path.join(self.lda_result_dir, 'cached_parameters.npz'),
            n_t=n_t,
            n_tw=n_tw
        )

        return n_t, n_tw


def gen_blocks(input_file, nodes, block_size, dump_binary):
    logging.info('Scanning input file...')

    total_docs = 0
    total_tokens = 0
    with open(input_file + '.libsvm') as f:
        for line in f:
            tokens = sum([int(x.split(':')[1]) for x in line.split('\t')[1].split(' ')])
            total_tokens += tokens
            total_docs += 1

    total_block_size = (total_tokens * 2 + total_docs + 2) * 4 / 1024 / 1024
    logging.info('In total %d documents, %d tokens, estimated total block size: %.2f MB',
                 total_docs, total_tokens, total_block_size)

    num_blocks = int(ceil(total_block_size / nodes / block_size))
    tokens_per_block = int(ceil(total_tokens / nodes / num_blocks))

    logging.info('Scheduling %d nodes with %d blocks, each block has %d tokens',
                 nodes, num_blocks, tokens_per_block)

    dump_binary_results = []

    with open(input_file + '.libsvm') as f:
        for node in range(nodes):
            for block in range(num_blocks):
                tokens_this_block = 0
                docs_this_block = 0

                block_input = input_file + '.libsvm.%d.%d' % (node, block)
                with open(block_input, 'w') as f_output:
                    while tokens_this_block < tokens_per_block:
                        try:
                            line = f.readline()
                            if not line.strip():
                                raise EOFError('empty of file reached')
                        except (EOFError, IOError):
                            break

                        tokens = sum([int(x.split(':')[1]) for x in line.split('\t')[1].split(' ')])
                        tokens_this_block += tokens
                        f_output.write(line)
                        docs_this_block += 1

                    logging.info('Node %d, block %d has %d docs and %d tokens',
                                 node, block, docs_this_block, tokens_this_block)

                    if dump_binary is not None:
                        cmd = [
                            dump_binary,
                            block_input, input_file + '.vocab',
                            os.path.realpath(os.path.dirname(block_input)), str(node), str(block)
                        ]
                        logging.info('Executing dump_binary: %r', cmd)
                        dump_binary_results.append(
                            subprocess.Popen(cmd))

    logging.info('Waiting for dump_binary to finish.')
    for i in dump_binary_results:
        i.wait()


def analyze_results(input_dir, spm_model):
    inferer = LDAInfer(input_dir, spm_model=spm_model)
    inferer.find_possible_stopwords()


def infer(input_dir,
          infer_input, infer_output,
          params, spm_model, stopwords,
          alpha, beta, num_vocab, generate_html):
    logging.info('Loading model...')
    inferer = LDAInfer(input_dir, punkt_model=params, spm_model=spm_model, stopwords=stopwords,
                       beta=beta, alpha=alpha, num_vocab=num_vocab)

    top_words = [x[1] for x in sorted(inferer.get_top_words().items(), key=lambda x: x[0])]
    # inferer.find_possible_stopwords()

    with open(infer_input, encoding='utf8') as input_file, open(infer_output, 'w') as output_file:
        logging.info('Starting to infer topics on new documents.')
        for i, line in enumerate(input_file):
            if not line.strip():
                logging.warning('Found an empty line at line %d', i)
                continue

            logging.info('Tokenizing paragraph.')
            token_ids_all = inferer.tokenizer.tokenize_ids(line)
            token_ids_all = sum(token_ids_all, [])
            token_ids_all = [(x not in inferer.stopwords, x) for x in token_ids_all]

            token_ids = [x for not_stop, x in token_ids_all if not_stop]

            logging.info('Inferring paragraph.')
            topics, mc_states, n_tw_states = inferer.infer_topic_fast(token_ids)

            if generate_html:
                with open('plot.template.html') as f:
                    html_template = f.read()
                with open('plot.%d.html' % i, 'w', encoding='utf8') as f:
                    data_base64 = base64.b64encode(zlib.compress(json.dumps({
                        'top_words': top_words,
                        'states': mc_states.tolist(),
                        'n_tw': n_tw_states.tolist(),
                        'tokens': [(not_stop, inferer.spm_tokenizer.IdToPiece(x)) for not_stop, x in token_ids_all]
                    }).encode())).decode()
                    f.write(html_template.replace('{{data_base64}}', data_base64))

            topic_count = Counter(topics)
            builder = []
            for topic in sorted(topic_count):
                builder.append('%d:%d' % (topic, topic_count[topic]))
            output_file.write(' '.join(builder))
            output_file.write('\n')
            logging.info('Inferred %d tokens: %s',
                         len(token_ids), ' '.join('%d:%d' % x for x in Counter(token_ids).items()))


def main():
    parser = argparse.ArgumentParser(description='LDA tools for sentence segmenter and tokenizer.')
    parser.add_argument('--verbose', action='store_true', default=False, help='More debugging logging.')

    subparsers = parser.add_subparsers(help='Action to execute.', dest='action')

    download_paragraph = subparsers.add_parser("gen-blocks", help='Generate blocks for LDA use.')
    download_paragraph.add_argument('--input', action='store', type=str, required=True,
                                    help='Input corpus, for example corpus.lightlda, '
                                         'must has *.lightlda.libsvm and *.lightlda.vocab.')
    download_paragraph.add_argument('--nodes', action='store', type=int, required=True,
                                    help='How many nodes to use.')
    download_paragraph.add_argument('--block-size', action='store', type=int, default=500,
                                    help='The average block size in MB. See lightlda/dump_binary.cpp for block format.')
    download_paragraph.add_argument('--dump-binary', action='store', type=str, default=None,
                                    help='If provided the executive path for dump-binary, I will help you execute it.')

    analyze_arguments = subparsers.add_parser("analyze", help='Analyze LDA result.')
    analyze_arguments.add_argument('--lda-result-dir', action='store', type=str, required=True,
                                   help='LDA result dir')
    analyze_arguments.add_argument('--spm', action='store', type=str, required=True,
                                   help='SPM model file')

    infer_arguments = subparsers.add_parser("infer", help='Infer LDA topics.')
    infer_arguments.add_argument('--lda-result-dir', action='store', type=str, required=True,
                                 help='LDA result dir')
    infer_arguments.add_argument('--input', action='store', type=str, required=True,
                                 help='Paragraphs to infer')
    infer_arguments.add_argument('--output', action='store', type=str, required=True,
                                 help='Inference output')
    infer_arguments.add_argument('--params', action='store', type=str, required=True,
                                 help='Punkt parameters')
    infer_arguments.add_argument('--spm', action='store', type=str, required=True,
                                 help='SPM model file')
    infer_arguments.add_argument('--stopwords', action='store', type=str, required=True,
                                 help='List of stopwords')
    infer_arguments.add_argument('--num-vocabs', action='store', type=int, required=True,
                                 help='Number of vocabulary')
    infer_arguments.add_argument('--alpha', action='store', type=float, default=0.001,
                                 help='Topics prior')
    infer_arguments.add_argument('--beta', action='store', type=float, default=0.01,
                                 help='Words prior')
    infer_arguments.add_argument('--generate-html', action='store_true', default=False,
                                 help='Generate HTML visualizations')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

    if args.action == 'gen-blocks':
        gen_blocks(args.input, args.nodes, args.block_size, args.dump_binary)
    elif args.action == 'analyze':
        analyze_results(args.lda_result_dir, args.spm)
    elif args.action == 'infer':
        infer(args.lda_result_dir,
              args.input, args.output,
              args.params, args.spm, args.stopwords,
              args.alpha, args.beta, args.num_vocabs,
              args.generate_html)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
