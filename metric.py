import numpy as np
from itertools import product
from collections import Counter
#https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54#:~:text=Precision%20and%20recall%20at%20k%3A%20Definition&text=This%20means%20that%2080%25%20of,are%20relevant%20to%20the%20user.&text=Suppose%20that%20we%20computed%20recall,in%20the%20top%2Dk%20results.
#https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf

# https://www.pinecone.io/learn/offline-evaluation/#Metrics-in-Information-Retrieval
# https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54
# https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Precision
# https://www.educative.io/answers/what-is-the-mean-average-precision-in-information-retrieval
def ap_k(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    Examples
    --------
    actual, predicted = ['a'], ['a', 'a', 'n', 'a', 'j']
    print(ap_k(actual, predicted, 5))   0.92
     """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual[:k]:
            # if p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return 0.0 if num_hits == 0.0 else score / num_hits


def map_k(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two list
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    -------
    example :
    actual, predicted = ['p_a', 'p_b'], [
                                        ['p_a', 'p_b', 'p_c', 'p_d', 'p_e', 'p_f'],
                                        ['p_c', 'p_d', 'p_e', 'p_f', 'p_a', 'p_b'],
                                        ['p_d', 'p_a', 'p_c', 'p_b', 'p_e', 'p_f'],
                                        ]
    print(map_k(actual, predicted, 6))
    actual, predicted = ['a'], [['a', 'b', 'a', 'c', 'a', 'c', 'a', 'c', 'c', 'b'],
                                ['c', 'b', 'a', 'v', 'a', 'c', 'a', 'v', 'v', 'v'],
                                ['a', 'x', 'a', 'c', 'a', 'x', 'x', 'c', 'c', 'v'],
                                ['a', 'x', 'a', 'x', 'x', 'c', 'x', 'c', 'c', 'b']]
    print(map_k(actual, predicted, 10))

    print(map_k([1], [[1,1,0,1,0]], k=5)) 0.916
    """
    return np.mean([ap_k(a, p, k) for a, p in product([actual], predicted)])


# is maybe hit_rate and mMV are similar ???
def hit_rate_k(actual, predicted, k=10):
    """
    example :
    actual, predicted = ['y'], ['y', 'y', 'n', 'y', 'n']
    print(hit_rate_k(actual, predicted, 5))
    """
    # hits = sum(predicted[i] in actual for i in range(k))
    # return hits / float(k)
    hits = any(predicted[i] in actual for i in range(k))
    return 1 if hits else 0


def mMV_k(actual, predicted, k=10):
    """
    # Fast and Scalable Image Search For Histology
    # the mean majority vote == mMV_k
    the majority top-5 accuracy
    Returns:
    list: The class label with the majority vote for each prediction.
    example:
    actual, predicted = ['w'], ['w','jj','ss','eee','w']
    print(mMV_k(actual, predicted, k=5)) 1
    """
    # for pred in predicted:
    top_k_predictions = predicted[:k]
    top_k_predictions_numbers = [int(pred[0]) for pred in top_k_predictions]

    vote_count = Counter(top_k_predictions_numbers)
    majority_vote = vote_count.most_common(1)[0][0]
    #     mmv_results.append(majority_vote)
    # word_counts = Counter(mmv_results)
    # most_common_word = word_counts.most_common(1)[0][1]
    # return most_common_word/len(predicted)
    return 1 if majority_vote in actual else 0


