import argparse
import json
import logging
import lzma
import multiprocessing
import pickle
import re
from collections import Counter

from tqdm import tqdm
import sentencepiece

from ScientificTopics.efficient_punkt import PunktSentenceTokenizer, PunktTrainer, PunktLanguageVars


class LangVarsForScientificArticles(PunktLanguageVars):
    pass


class Tokenizer(object):
    def __init__(self, punkt_params, spm_params):
        """
        A tokenizer that uses Punkt system and sentence piece.

        References:
        1. Kiss, Tibor, and Jan Strunk. "Unsupervised multilingual
            sentence boundary detection." Computational Linguistics
            32.4 (2006): 485-525.
        2. Kudo, Taku, and John Richardson. "Sentencepiece: A simple
            and language independent subword tokenizer and detokenizer
            for neural text processing." arXiv preprint
            arXiv:1808.06226 (2018).

        :param punkt_params: Path to Punkt parameters.
        :param spm_params: Path to sentence piece model parameters.
        """
        if punkt_params.endswith('.xz'):
            f = lzma.open(punkt_params, 'rb')
        else:
            f = open(punkt_params, 'rb')

        logging.info('Loading punkt parameters...')
        punkt_params = pickle.load(f)
        self.sentence_tokenizer = PunktSentenceTokenizer(punkt_params)
        f.close()

        logging.info('Loading sentence piece model...')
        self.spm_tokenizer = sentencepiece.SentencePieceProcessor()
        self.spm_tokenizer.Load(spm_params)

    def tokenize(self, paragraph):
        """
        Tokenize a paragraph into sentences and tokens.

        :param paragraph: A str object of the text.
        :return: List of list of tokens.
        """
        sentences = self.sentence_tokenizer.tokenize(paragraph)
        tokens = []
        for sentence in sentences:
            tokens.append(self.spm_tokenizer.EncodeAsPieces(sentence))

        return tokens

    def tokenize_ids(self, paragraph):
        """
        Tokenize a paragraph into sentences and tokens.

        :param paragraph: A str object of the text.
        :return: List of list of tokens.
        """
        sentences = self.sentence_tokenizer.tokenize(paragraph)
        tokens = []
        for sentence in sentences:
            tokens.append(self.spm_tokenizer.EncodeAsIds(sentence))

        return tokens


def get_text_file_or_xz(filename):
    if re.match(r'.*?\.xz$', filename):
        return lzma.open(filename, 'rt', encoding='utf8')
    else:
        return open(filename, 'rt', encoding='utf8')


def get_random_paragraphs(n, output):
    from synthesis_project_ceder.database import SynPro

    synpro = SynPro()
    paragraphs_collection = synpro.Paragraphs
    random_paragraphs = paragraphs_collection.aggregate([
        {
            '$sample': {'size': n}
        },
        {
            '$project': {
                'text': '$text'
            }
        }
    ], allowDiskUse=True)

    with open(output, 'w', encoding='utf8') as f:
        for paragraph in tqdm(random_paragraphs, desc='Fetching data', unit='paragraphs', total=n):
            f.write(paragraph['text'])
            f.write('\n')


def learn_from_paragraphs(input_file, output_file):
    def paragraphs_reader():
        read_f = get_text_file_or_xz(input_file)

        for i, line in enumerate(read_f):
            line = line.strip()
            if line:
                yield line

            if i % 1000 == 0:
                logging.debug('Read %d lines.', i)

        read_f.close()

    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.IGNORE_ABBREV_PENALTY = True
    trainer.train(paragraphs_reader(), verbose=True)
    # trainer._params.abbrev_types.update({
    #     'al', 'e.g', 'eg', 'eq', 'e.q', 'etc', 'i.e', 'ie',
    #     'fig',
    #     'a.r', 'ar',
    # })

    logging.info('Training complete, dumping parameters.')
    with open(output_file, 'wb') as f:
        pickle.dump(trainer.get_params(), f)


def segment_html(params, input_file, output_file):
    with open(params, 'rb') as f:
        punkt_params = pickle.load(f)
        tokenizer = PunktSentenceTokenizer(punkt_params)

    input_f = get_text_file_or_xz(input_file)
    with open(output_file, 'w', encoding='utf8') as output_f:
        output_f.write("""<head>
        <style>
        span.break::after {
            content: "<s>";
            background-color: yellow;
            color: red;
            font-weight: bold;
        }
        span.unbreak::after {
            content: "<a>";
            background-color: blue;
            color: green;
            font-weight: bold;
        }
        span.end::after {
            content: "<e>";
            background-color: red;
            color: white;
            font-weight: bold;
        }
        </style>
    </head>
    <body>
    """)
        for paragraph in input_f:
            paragraph = paragraph.strip()
            try:
                decisions = sorted(list(tokenizer.debug_decisions(paragraph)), key=lambda x: x['period_index'])
            except Exception as e:
                logging.exception('Cannot tokenize %r: %s', e, paragraph)
                continue

            output_f.write('<p>\n')
            remaining_paragraph = paragraph
            remaining_start = 0
            while len(remaining_paragraph):
                if len(decisions):
                    skip = decisions[0]['period_index'] + 1 - remaining_start
                    cut_sentence = remaining_paragraph[:skip]
                    remaining_paragraph = remaining_paragraph[skip:]
                    remaining_start += skip

                    if decisions[0]['break_decision']:
                        output_f.write('\t<span class="break">%s</span>\n' % cut_sentence)
                    else:
                        output_f.write('\t<span class="unbreak">%s</span>\n' % cut_sentence)
                    decisions.pop(0)
                else:
                    output_f.write('\t<span class="end">%s</span>\n' % remaining_paragraph)
                    remaining_start += len(remaining_paragraph)
                    remaining_paragraph = ''
            output_f.write('</p>\n')

        output_f.write('</body>')

    input_f.close()


def do_print_params(params):
    with open(params, 'rb') as f:
        punkt_params = pickle.load(f)

    print('Abbreviation types:')
    for abbr in sorted(punkt_params.abbrev_types, key=lambda x: len(x)):
        print(abbr)

    print('Collocations types:')
    for collocation in sorted(punkt_params.collocations):
        print(collocation[0] + ', ' + collocation[1])


def do_test_params(params):
    with open(params, 'rb') as f:
        punkt_params = pickle.load(f)
        tokenizer = PunktSentenceTokenizer(punkt_params)

    sentences_to_test = [
        ('Panić et al. also recognized similar peaks.', 1),
        ('From second order FRF, i.e. E2/E1 spectra, at O2 stoichiometry of 1.75 and 1.5.', 1),
        ('Fig. 6 shows the dependence of the surface hardness.', 1),
        ('Sleath used Eq. (21) to express the velocity distribution.', 1),
        ('Batteries, gas sensors, etc., have been good examples.', 1)
    ]
    for sentence, n_sentence in sentences_to_test:
        try:
            sentences = tokenizer.tokenize(sentence)
            if len(sentences) != n_sentence:
                logging.error('Expecting %d but got %d sentences for: %s. It was tokenized as %r',
                              n_sentence, len(sentences), sentence, sentences)
        except Exception as e:
            logging.exception('Failed to tokenize: %s', sentence)


def segment_text_file(params, input_file, output_file, bulk):
    with open(params, 'rb') as f:
        punkt_params = pickle.load(f)
        tokenizer = PunktSentenceTokenizer(punkt_params)

    input_f = get_text_file_or_xz(input_file)

    with tqdm(desc='Segmenting paragraphs', unit=' tokens',
              miniters=10000, mininterval=0.5) as bar:
        with open(output_file, 'w', encoding='utf8') as output_f:
            for paragraph in input_f:
                paragraph = paragraph.strip()
                try:
                    sentences = tokenizer.tokenize(paragraph)
                except Exception as e:
                    logging.exception('Cannot tokenize %r: %s', e, paragraph)
                    continue

                for sentence in sentences:
                    bar.update(sentence.count(' ') + 1)

                if bulk:
                    for sentence in sentences:
                        output_f.write(sentence)
                        output_f.write('\n')
                else:
                    output_f.write('%s\n' % json.dumps(sentences))

    input_f.close()


def _tokenize(args):
    """Returns is_not_empty, tuple((is_filtered, tokens, token_id_s, bow_s))"""
    line, min_words, produce_sentences = args

    global tokenizer, g_stopwords
    tokens = tokenizer.tokenize_ids(line)

    if not any(len(x) for x in tokens):
        return False, (), None, None

    if not produce_sentences:
        tokens = [sum(tokens, [])]

    results = []
    coverage_data = Counter()
    vocab_tf = Counter()
    for token_ids_org in tokens:
        token_ids = [x for x in token_ids_org if x not in g_stopwords]

        if len(token_ids) < min_words:
            results.append((True, (), None, None))
            continue

        for t in set(token_ids):
            coverage_data[t] += 1

        token_id_s = ' '.join(str(x) for x in token_ids)

        bow = Counter(token_ids)
        bow_string = ['%d:%d' % x for x in bow.items()]
        bow_string = ' '.join(bow_string)

        results.append((False, token_ids, token_id_s, bow_string))

        for token, count in bow.items():
            vocab_tf[token] += count

    return True, results, coverage_data, vocab_tf


def do_tokenize(punkt_params, spm_params,
                input_filename, output_filename,
                produce_sentences=False, coverage_report=None,
                min_words=0, stopwords=None, n_cpus=None):
    global tokenizer, g_stopwords
    tokenizer = Tokenizer(punkt_params, spm_params)
    input_f = get_text_file_or_xz(input_filename)

    coverage_data = Counter()
    vocab_tf = Counter()
    total_docs = 0

    if stopwords is None:
        stopwords = set()
    else:
        with open(stopwords) as f:
            stopwords = [x.strip().split()[0] for x in f if x.strip()]
        stopwords = set(tokenizer.spm_tokenizer.PieceToId(x) for x in stopwords)
        stopwords = set(x for x in stopwords if not tokenizer.spm_tokenizer.IsUnknown(x))

    g_stopwords = stopwords

    output_f = open(output_filename + '.sequence', 'w')
    output_f_libsvm = open(output_filename + '.lightlda.libsvm', 'w')
    output_f_vocab = open(output_filename + '.lightlda.vocab', 'w', encoding='utf8')

    # Write documents
    with multiprocessing.Pool(processes=n_cpus) as pool:
        for i, (is_not_empty, results, _coverage, _vocab_tf) in enumerate(
                pool.imap(
                    _tokenize,
                    ((x, min_words, produce_sentences) for x in input_f),
                    chunksize=1000)):
            if not is_not_empty:
                logging.warning('Found an empty line at line %d', i)
                continue

            for j, (is_filtered, token_ids, token_id_s, bow_string) in enumerate(results):
                if is_filtered:
                    logging.debug('Skipping a output item at line %d', i)
                    continue

                total_docs += 1
                output_f.write(token_id_s)
                output_f.write('\n')
                output_f_libsvm.write('%d,%d\t%s\n' % (i, j, bow_string))

            coverage_data += _coverage
            vocab_tf += _vocab_tf

            if (i + 1) % 10000 == 0:
                logging.info('Processed %d lines', i + 1)

    output_f.close()
    output_f_libsvm.close()

    # Write vocabulary for lightlda
    for token_id, count in sorted(vocab_tf.items()):
        output_f_vocab.write(
            '%d\t%s\t%d\n' %
            (token_id, tokenizer.spm_tokenizer.IdToPiece(token_id), count))
    output_f_vocab.close()

    # Write token coverage report for human inspection.
    if coverage_report is not None:
        with open(coverage_report, 'w', encoding='utf8') as f:
            for token_id, count in sorted(coverage_data.items(),
                                          key=lambda x: x[1], reverse=True):
                f.write('%d\t%s\t%.4f\t%d\n' %
                        (token_id, tokenizer.spm_tokenizer.IdToPiece(token_id),
                         count * 1. / total_docs, count))


def do_tokenize_interact(punkt_params, spm_params):
    tokenizer = Tokenizer(punkt_params, spm_params)
    while True:
        try:
            input_string = input('Paragraph <<< ')
        except (EOFError, KeyboardInterrupt):
            break

        tokens = tokenizer.tokenize(input_string)
        for sentence in tokens:
            sentence = [x.decode('utf8') for x in sentence]
            print(' '.join(sentence))


def main():
    parser = argparse.ArgumentParser(description='Sentence segmenter using Punkt System.')
    parser.add_argument('--verbose', action='store_true', default=False, help='More debugging logging.')

    subparsers = parser.add_subparsers(help='Action to execute.', dest='action')

    download_paragraph = subparsers.add_parser("download", help='Dump SynPro paragraphs.')
    download_paragraph.add_argument('--output', action='store', type=str, required=True,
                                    help='Output filename.')
    download_paragraph.add_argument('--n', action='store', type=int, required=True,
                                    help='How many paragraphs.')

    learn_punk = subparsers.add_parser("learn", help="Learn sentence segmenter using Punkt.")
    learn_punk.add_argument('--input', action='store', type=str, required=True,
                            help='Input list of paragraphs')
    learn_punk.add_argument('--params', action='store', type=str, required=True,
                            help='Output pickle data file for Punkt arguments')

    print_params = subparsers.add_parser("print-params", help="Print parameters learned.")
    print_params.add_argument('--params', action='store', type=str, required=True,
                              help='Punkt parameters')

    test_params = subparsers.add_parser("test-params", help="Test parameters learned.")
    test_params.add_argument('--params', action='store', type=str, required=True,
                             help='Punkt parameters')

    print_html = subparsers.add_parser("html", help="Print tokenizing effects to HTML.")
    print_html.add_argument('--params', action='store', type=str, required=True,
                            help='Punkt parameters')
    print_html.add_argument('--input', action='store', type=str, required=True,
                            help='Input paragraphs file')
    print_html.add_argument('--output', action='store', type=str, required=True,
                            help='Output HTML file')

    segment_text = subparsers.add_parser("segment", help="Segment paragraphs.")
    segment_text.add_argument('--params', action='store', type=str, required=True,
                              help='Punkt parameters')
    segment_text.add_argument('--input', action='store', type=str, required=True,
                              help='Input paragraphs file')
    segment_text.add_argument('--output', action='store', type=str, required=True,
                              help='Output json file')
    segment_text.add_argument('--bulk', action='store_true', default=False,
                              help='Write as a bulk file instead of list of sentences')

    tokenizer = subparsers.add_parser("tokenize", help="Tokenize paragraphs.")
    tokenizer.add_argument('--punkt', action='store', type=str, required=True,
                           help='Punkt parameters')
    tokenizer.add_argument('--spm', action='store', type=str, required=True,
                           help='SPM model file')
    tokenizer.add_argument('--input', action='store', type=str, required=True,
                           help='Input paragraphs file')
    tokenizer.add_argument('--stopwords', action='store', type=str, default=None,
                           help='Stop words list')
    tokenizer.add_argument('--min-words', action='store', type=int, default=1,
                           help='Minimum words per output line')
    tokenizer.add_argument('--output', action='store', type=str, required=True,
                           help='Output token ids file')
    tokenizer.add_argument('--produce-sentences', action='store_true', default=False,
                           help='Produce sentences per line instead of paragraphs')
    tokenizer.add_argument('--coverage-report', action='store', type=str, default=None,
                           help='Token coverage report filename')
    tokenizer.add_argument('--n-cpus', action='store', type=int, default=multiprocessing.cpu_count(),
                           help='Processes to fork')

    tokenizer_interact = subparsers.add_parser("tokenize-interact", help="Tokenize paragraphs via interaction.")
    tokenizer_interact.add_argument('--punkt', action='store', type=str, required=True,
                                    help='Punkt parameters')
    tokenizer_interact.add_argument('--spm', action='store', type=str, required=True,
                                    help='Input paragraphs file')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

    if args.action == 'download':
        get_random_paragraphs(args.n, args.output)
    elif args.action == 'learn':
        learn_from_paragraphs(args.input, args.params)
    elif args.action == 'print-params':
        do_print_params(args.params)
    elif args.action == 'test-params':
        do_test_params(args.params)
    elif args.action == 'html':
        segment_html(args.params, args.input, args.output)
    elif args.action == 'segment':
        segment_text_file(args.params, args.input, args.output, args.bulk)
    elif args.action == 'tokenize':
        do_tokenize(args.punkt, args.spm, args.input, args.output,
                    args.produce_sentences, args.coverage_report,
                    args.min_words, args.stopwords, args.n_cpus)
    elif args.action == 'tokenize-interact':
        do_tokenize_interact(args.punkt, args.spm)
    else:
        parser.print_usage()


if __name__ == '__main__':
    main()
