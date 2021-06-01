import math

import numpy as np
import pandas as pd

percent = 0.3
windowSize = 1
V = []
stop_words = ['.', 'is', 'a', 'of', 'and']

sentences = [
    'John likes NLP',
    'He likes Mary',
    'John likes machine learning',
    'Deep learning is a subfield of machine learning',
    'John wrote a post about NLP and got likes'
]

train_words = ['John', 'likes', 'NLP', 'He', 'Mary', 'machine', 'learning', 'Deep', 'is', 'a', 'subfield', 'of',
               'wrote', 'post', 'about', 'and', 'got']


def get_V():
    for sent in sentences:
        for word in sent.split():
            if word.endswith('.'):
                word = word[:-1]
            if word in stop_words:
                continue
            if word not in V:
                V.append(word)
    print(V, len(V))


def filter_words(sent):
    return [x for x in sent if x not in stop_words]


def co_occurrences_matrix2():
    get_V()
    df = pd.DataFrame(np.zeros(shape=(len(V), len(V))), dtype=int, index=V, columns=V)
    for sent in sentences:
        sent_words = filter_words(sent.split())
        for i, word in enumerate(sent_words):
            if i == 0:
                df[word][sent_words[1]] += 1
            elif i == len(sent_words) - 1:
                df[word][sent_words[i - 1]] += 1
            else:
                df[word][sent_words[i + 1]] += 1
                df[word][sent_words[i - 1]] += 1
    return df


def cosine_similarity(u_tag, x, y):
    return ((x @ u_tag) @ (y @ u_tag)) / (
            np.linalg.norm(x @ u_tag) * np.linalg.norm(y @ u_tag))


def procedure_2_3_4(embedding):
    john_vec = np.zeros(len(V))
    john_vec[V.index('John')] = 1
    he_vec = np.zeros(len(V))
    he_vec[V.index('He')] = 1
    subfield_vec = np.zeros(len(V))
    subfield_vec[V.index('subfield')] = 1
    deep_vec = np.zeros(len(V))
    deep_vec[V.index('Deep')] = 1
    machine_vec = np.zeros(len(V))
    machine_vec[V.index('machine')] = 1

    john_with_he = cosine_similarity(embedding, john_vec, he_vec)
    john_with_subfield = cosine_similarity(embedding, john_vec, subfield_vec)
    deep_with_machine = cosine_similarity(embedding, deep_vec, machine_vec)
    print('john_with_he')
    print(round(john_with_he, 3))
    print('john_with_subfield')
    print(round(john_with_subfield, 3))
    print('deep_with_machine')
    print(round(deep_with_machine, 3))


def procedure_6_7(embedding):
    wrote_vec = np.zeros(len(V))
    wrote_vec[V.index('wrote')] = 1
    post_vec = np.zeros(len(V))
    post_vec[V.index('post')] = 1
    likes_vec = np.zeros(len(V))
    likes_vec[V.index('likes')] = 1

    wrote_with_post = cosine_similarity(embedding, wrote_vec, post_vec)
    like_with_like = cosine_similarity(embedding, likes_vec, likes_vec)

    print('wrote_with_post')
    print(round(wrote_with_post, 3))
    print('like_with_like')
    print(round(like_with_like, 3))


def get_svd_embedding(matrix, num_of_vals):
    """
    Question 4.2 and 4.3
    """
    # write co_occurrences_matrix to excel
    df.to_excel('co_occurrences_matrix.xlsx')

    u, s, vt = np.linalg.svd(matrix)
    u_tag = u[:, :num_of_vals]
    s_tag = np.diag(s[:num_of_vals])
    vt_tag = vt[:num_of_vals, :]

    pd.DataFrame(u).to_excel('u.xlsx')
    pd.DataFrame(np.diag(s)).to_excel('s.xlsx')
    pd.DataFrame(vt).to_excel('v.xlsx')
    pd.DataFrame(u_tag).to_excel('u_tag.xlsx')
    pd.DataFrame(s_tag).to_excel('s_tag.xlsx')
    pd.DataFrame(vt_tag).to_excel('v_tag.xlsx')
    u_tag = u[:, :num_of_vals]
    return u_tag, vt_tag


def get_evd_embedding(matrix, num_of_vals):
    eigen_val, p_matrix = np.linalg.eig(matrix)
    biggest_indexes = np.argpartition(eigen_val, -num_of_vals)[-num_of_vals:]
    res = None

    for j in biggest_indexes:
        if res is None:
            res = np.array(p_matrix[:, j:j + 1])
        else:
            res = np.append(res, p_matrix[:, j:j + 1], axis=1)
    return np.array(res)


def round_mat(X):
    re_X = X
    for i, row in enumerate(re_X):
        for j, entry in enumerate(row):
            if entry < 0.1:
                re_X[i][j] = 0
    return re_X


# create the matrix
df = co_occurrences_matrix2()
biggest = math.floor(percent * len(V))

# get the SVD
svd_embedding = get_svd_embedding(df, biggest)[0]
print("SVD 2,3,4")
procedure_2_3_4(svd_embedding)
evd_embedding = get_evd_embedding(df, biggest)
print("EVD 2,3,4")
procedure_2_3_4(evd_embedding)
print("SVD 6,7")
procedure_6_7(svd_embedding)
print("EVD 6,7")
procedure_6_7(evd_embedding)

"""
    TRYING
"""
print("------////////////////////----------")
print("------////////////////////----------")
print("------////////////////////----------")
print("------////////////////////----------")


def cosine_similarity2(emb, x, y):
    return ((x @ emb[0] @ emb[1]) @ (y @ emb[0] @ emb[1])) / (
            np.linalg.norm(x @ emb[0] @ emb[1]) * np.linalg.norm(y @ emb[0] @ emb[1]))


john_vec = np.zeros(len(V))
john_vec[V.index('John')] = 1
he_vec = np.zeros(len(V))
he_vec[V.index('He')] = 1
subfield_vec = np.zeros(len(V))
subfield_vec[V.index('subfield')] = 1
deep_vec = np.zeros(len(V))
deep_vec[V.index('Deep')] = 1
machine_vec = np.zeros(len(V))
machine_vec[V.index('machine')] = 1

embedding = get_svd_embedding(df, biggest)

john_with_he = cosine_similarity2(embedding, john_vec, he_vec)
john_with_subfield = cosine_similarity2(embedding, john_vec, subfield_vec)
deep_with_machine = cosine_similarity2(embedding, deep_vec, machine_vec)
print('john_with_he')
print(round(john_with_he, 3))
print('john_with_subfield')
print(round(john_with_subfield, 3))
print('deep_with_machine')
print(round(deep_with_machine, 3))

wrote_vec = np.zeros(len(V))
wrote_vec[V.index('wrote')] = 1
post_vec = np.zeros(len(V))
post_vec[V.index('post')] = 1
likes_vec = np.zeros(len(V))
likes_vec[V.index('likes')] = 1

wrote_with_post = cosine_similarity2(embedding, wrote_vec, post_vec)
like_with_like = cosine_similarity2(embedding, likes_vec, likes_vec)

print('wrote_with_post')
print(round(wrote_with_post, 3))
print('like_with_like')
print(round(like_with_like, 3))

# u, s, vh = np.linalg.svd(df)

# Sigma = np.zeros((u.shape[1], vh.shape[0]))
# Sigma[:u.shape[0], :vh.shape[1]] = np.diag(s)
# SAME (round_mat(u @ Sigma @ vh) == round_mat(u @ np.diag(s) @ vh)):

#
# X_tag = round_mat(u[:biggest, :] @ np.diag(s) @ vh[:, :biggest])
# Sigma_tag = np.zeros((biggest, 13))
# Sigma_tag[:biggest, :biggest] = np.diag(s[:biggest])
# X_tag_2 = round_mat(u[:biggest, :] @ np.diag(s) @ vh[:, :biggest])
# print(X_tag)
# print(X_tag_2)

# def co_occurrences_matrix():
#     get_V()
#     df = pd.DataFrame(np.zeros(shape=(len(V), len(V))), dtype=int, index=V, columns=V)
#     # print(df['is']['likes'])
#     print(df)
#     for sent in sentences:
#         sent_words = sent.split()
#         sent_words = filter_words(sent_words)
#
#         for i, word in enumerate(sent_words):
#             # if word in stop_words:
#             #     continue
#             if i == 0:
#                 j = 1
#                 df[sent_words[j]][word] += 1
#                 while j + 1 < len(sent_words) and (sent_words[j] in stop_words):
#                     j += 1
#                     df[sent_words[j]][word] += 1
#             elif i == len(sent_words) - 1:
#                 j = len(sent_words) - 2
#                 df[sent_words[j]][word] += 1
#                 while j - 1 >= 0 and (sent_words[j] in stop_words):
#                     j -= 1
#                     df[sent_words[j]][word] += 1
#             else:
#                 j = i + 1
#                 df[sent_words[j]][word] += 1
#                 while j + 1 < len(sent_words) and sent_words[j] in stop_words:
#                     j += 1
#                     df[sent_words[j]][word] += 1
#                 j = i - 1
#                 df[sent_words[j]][word] += 1
#                 while j - 1 >= 0 and sent_words[j] in stop_words:
#                     j -= 1
#                     df[sent_words[j]][word] += 1
#     return df
