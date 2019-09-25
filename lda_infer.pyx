# cython: language_level=3
import numpy
cimport numpy
cimport cython

ctypedef numpy.float32_t FLOAT_T
ctypedef numpy.int32_t INT32_T
ctypedef numpy.uint32_t UINT32_T


def get_p(numpy.ndarray[UINT32_T, ndim=2] n_tw,
          numpy.ndarray[UINT32_T, ndim=1] n_t,
          UINT32_T word_id, float beta, UINT32_T num_vocab):
    cdef UINT32_T ntopics = n_tw.shape[0]

    cdef numpy.ndarray[FLOAT_T, ndim=1] p_array = numpy.empty((ntopics,), dtype=numpy.float32)
    for i in range(ntopics):
        p_array[i] = n_tw[i, word_id] * beta / (n_t[i] + beta * num_vocab)

    cdef float summation = numpy.sum(p_array)
    if summation == 0:
        p_array = numpy.ones(shape=(ntopics,), dtype=numpy.float32)
        return p_array / ntopics
    else:
        return p_array / summation


def propose_word(numpy.ndarray[UINT32_T, ndim=2] n_tw,
                 numpy.ndarray[UINT32_T, ndim=1] n_t,
                 float beta, UINT32_T num_vocab,
                 UINT32_T word_id, size=None):
    cdef UINT32_T ntopics = n_tw.shape[0]

    return numpy.random.choice(
        ntopics,
        p=get_p(n_tw, n_t, word_id, beta, num_vocab),
        size=size)


def propose_doc(UINT32_T doc_size, UINT32_T ntopics, float alpha, size=None):
    """-Int for doc_topics[-i]"""
    rand_numbers = numpy.random.rand(*size) * (doc_size + alpha * ntopics)
    proposal = numpy.empty(shape=size, dtype=numpy.int32)

    cond = rand_numbers < doc_size

    take_doc_topics = numpy.where(cond)
    proposal[take_doc_topics] = -(rand_numbers[take_doc_topics]).astype(numpy.int32)

    take_random_topics = numpy.where(numpy.logical_not(cond))
    proposal[take_random_topics] = numpy.random.choice(ntopics, size=len(take_random_topics[0]))
    return proposal


@cython.boundscheck(False)
@cython.wraparound(False)
def infer_topic(
        numpy.ndarray[UINT32_T, ndim=1] doc,
        numpy.ndarray[UINT32_T, ndim=1] n_t, numpy.ndarray[UINT32_T, ndim=2] n_tw,
        float alpha, float beta,
        UINT32_T iterations=1000, UINT32_T mh_steps=2):
    cdef UINT32_T ntopics = n_tw.shape[0]
    cdef UINT32_T num_vocab = n_tw.shape[1]
    cdef UINT32_T doc_size = doc.shape[0]
    cdef float betasum = beta * num_vocab
    cdef UINT32_T i, j, k

    cdef numpy.ndarray[UINT32_T, ndim=1] doc_topics = numpy.random.choice(ntopics, size=(doc_size,)).astype(numpy.uint32)
    cdef numpy.ndarray[UINT32_T, ndim=1] topic_counter = numpy.zeros((ntopics,), dtype=numpy.uint32)

    # Build topic counter
    for i in range(doc_size):
        j = doc_topics[i]
        topic_counter[j] += 1

    cdef numpy.ndarray[UINT32_T, ndim=2] monte_carlo_states = numpy.empty((iterations, doc_size), dtype=numpy.uint32)
    cdef numpy.ndarray[UINT32_T, ndim=2] doc_topics_states = numpy.empty((iterations, ntopics), dtype=numpy.uint32)

    cdef numpy.ndarray[FLOAT_T, ndim=4] random_numbers = numpy.random.rand(iterations, doc_size, mh_steps, 2).astype(numpy.float32)
    cdef numpy.ndarray[UINT32_T, ndim=3] word_proposals = numpy.empty((doc_size, iterations, mh_steps), dtype=numpy.uint32)
    cdef numpy.ndarray[INT32_T, ndim=3] doc_proposals = numpy.empty((doc_size, iterations, mh_steps), dtype=numpy.int32)
    for i in range(doc_size):
        word_proposals[i, :, :] = propose_word(
            n_tw=n_tw, n_t=n_t,
            beta=beta,
            num_vocab=num_vocab,
            word_id=doc[i],
            size=(iterations, mh_steps))
        doc_proposals[i, :, :] = propose_doc(
            doc_size=doc_size,
            ntopics=ntopics,
            alpha=alpha,
            size=(iterations, mh_steps))

    cdef UINT32_T word_id, original_topic, topic, new_topic
    cdef INT32_T doc_proposal
    cdef float n_td_alpha, n_tw_beta, n_t_beta_sum, proposal_t, pi_t
    cdef float n_sd_alpha, n_sw_beta, n_s_beta_sum, proposal_s, pi_s
    cdef float pi

    for i in range(iterations):
        for j in range(doc_size):
            monte_carlo_states[i, j] = doc_topics[j]
        for j in range(ntopics):
            doc_topics_states[i, j] = topic_counter[j]

        for j in range(doc_size):
            word_id = doc[j]
            original_topic = doc_topics[j]

            topic = original_topic
            for k in range(mh_steps):
                new_topic = word_proposals[j, i, k]

                if new_topic != topic:
                    # Simplified version of cxx::LightDocSampler::Sample()
                    n_td_alpha = topic_counter[new_topic] + alpha
                    n_sd_alpha = topic_counter[topic] + alpha

                    if topic == original_topic:
                        n_sd_alpha -= 1
                    if new_topic == original_topic:
                        n_td_alpha -= 1

                    pi = n_td_alpha / n_sd_alpha

                    if random_numbers[i, j, k, 0] < pi:
                        topic = new_topic

                doc_proposal = doc_proposals[j, i, k]
                new_topic = doc_topics[-doc_proposal] if doc_proposal < 0 else doc_proposal

                if new_topic != topic:
                    if new_topic != original_topic:
                        n_tw_beta = n_tw[new_topic, word_id] + beta
                        n_t_beta_sum = n_t[new_topic] + betasum
                        pi_t = n_tw_beta / n_t_beta_sum
                    else:
                        n_td_alpha = topic_counter[new_topic] + alpha - 1
                        n_tw_beta = n_tw[new_topic, word_id] + beta
                        n_t_beta_sum = n_t[new_topic] + betasum
                        proposal_t = topic_counter[new_topic] + alpha
                        pi_t = (n_td_alpha * n_tw_beta) / (n_t_beta_sum * proposal_t)

                    if topic != original_topic:
                        n_sw_beta = n_tw[topic, word_id] + beta
                        n_s_beta_sum = n_t[topic] + betasum
                        pi_s = n_sw_beta / n_s_beta_sum
                    else:
                        n_sd_alpha = topic_counter[topic] + alpha
                        n_sw_beta = n_tw[topic, word_id] + beta
                        n_s_beta_sum = n_t[topic] + betasum
                        proposal_s = topic_counter[topic] + alpha
                        pi_s = (n_sd_alpha * n_sw_beta) / (n_s_beta_sum * proposal_s)

                    pi = pi_t / pi_s

                    if random_numbers[i, j, k, 1] < pi:
                        topic = new_topic

            if topic != original_topic:
                topic_counter[original_topic] -= 1
                topic_counter[topic] += 1

            doc_topics[j] = topic

    return doc_topics, monte_carlo_states, doc_topics_states
