from sklearn.feature_extraction.text import CountVectorizer
import json
from nltk import word_tokenize
from tqdm import tqdm
import numpy as np
import pickle as pk
import torch
import os
import static_representations
from collections import Counter
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

def readData(args, translator):
    with open(f"episode_dataset/{args.theme}/{args.title}/{args.title}_segmented_raw.txt", "r") as raw_seg, open(f"episode_dataset/{args.theme}/{args.title}/{args.title}_segmented.txt", "r") as coref_seg:
        all_raw_segs = []
        raw_dataset = [[]]
        doc_id = 0

        for raw_text, coref_text in zip(raw_seg.readlines(), coref_seg.readlines()):
            if coref_text.strip() == '':
                doc_id += 1
                raw_dataset.append([])
            else:
                all_raw_segs.append(raw_text.strip())
                raw_dataset[doc_id].append(raw_text.strip())

        raw_dataset = [doc for doc in raw_dataset if len(doc) > 0]

        raw_dataset_tok_lower = [[word_tokenize(seg.lower().translate(translator)) for seg in doc] for doc in raw_dataset if len(doc) > 0]

        all_raw_segs = [seg.lower() for doc in raw_dataset for seg in doc]

    return all_raw_segs, raw_dataset, raw_dataset_tok_lower

def getGroundTruth(args, gt_root="groundtruth"):
    all_ev_id = set()
    gt = {}
    with open(os.path.join(gt_root, f'{args.title}_groundtruth.txt'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '': continue
            ev_id, text = line.split('\t')
            gt[text] = ev_id
            all_ev_id.add(ev_id)
    
    return all_ev_id, gt

def salience(word, num_bg, segs, background_dataset, stop_words, min_freq=5, verbal=False):
    if word in stop_words:
        return -1

    count_fg = sum([1 for i in set(segs) if word.lower() in i])
    count_bg = sum([1 for i in set(background_dataset) if word.lower() in i])

    if verbal:
        print(num_bg, "fg:", count_fg, "bg:", count_bg)
    if (count_bg != 0) and (count_fg > min_freq):
        return (1 + np.log(count_fg)**2) * np.log(num_bg/count_bg)
    elif (count_bg == 0) and (count_fg > min_freq):
        return (1 + np.log(count_fg)**2) * np.log(num_bg)
    return -1

def expandTerms(all_terms, all_cos, sim_thresh=0.75):

    in_cluster = [-1 for i in all_terms]
    word_clusters = []
    num_clusters = 0

    for w_id, i in enumerate(all_terms):
        score = max(all_cos[w_id])
        max_id = np.argmax(all_cos[w_id])
        
        if score < sim_thresh:
            word_clusters.append([i])
            num_clusters += 1
        elif (in_cluster[w_id] != -1) and (in_cluster[max_id] != -1) and (in_cluster[w_id] == in_cluster[max_id]):
            pass
        elif (in_cluster[w_id] != -1) and (in_cluster[max_id] != -1) and (in_cluster[w_id] != in_cluster[max_id]):
            merged = word_clusters[in_cluster[w_id]] + word_clusters[in_cluster[max_id]]
            word_clusters[in_cluster[w_id]] = []
            word_clusters[in_cluster[max_id]] = []
            word_clusters.append(merged)
            for w in merged:
                in_cluster[all_terms.index(w)] = num_clusters
            num_clusters += 1
        elif (in_cluster[max_id] != -1) and (in_cluster[w_id] == -1):
            word_clusters[in_cluster[max_id]].append(i)
            in_cluster[w_id] = in_cluster[max_id]
        elif (in_cluster[max_id] == -1) and (in_cluster[w_id] != -1):
            word_clusters[in_cluster[w_id]].append(all_terms[max_id])
            in_cluster[max_id] = in_cluster[w_id]
        else:
            word_clusters.append([i, all_terms[max_id]])
            in_cluster[w_id] = num_clusters
            in_cluster[max_id] = num_clusters
            num_clusters += 1

    return in_cluster, word_clusters

def average_with_harmonic_series(representations):
    weights = [0.0] * len(representations)
    for i in range(len(representations)):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=0)

def co_occurrence(segs, terms, in_cluster, cluster_segment_freq):
    # tokenize and clean
    translator = str.maketrans(punctuation, ' '*len(punctuation)) #map punctuation to space

    tokenized_segs = [word_tokenize(s.translate(translator)) for s in segs]
    freq_matrix = np.zeros((len(terms), len(terms)))
    scaled_matrix = np.zeros((len(terms), len(terms)))
    for k, s in tqdm(enumerate(tokenized_segs), total=len(tokenized_segs)):
        for i, t1 in enumerate(terms):
            if ((t1 in s) or ((in_cluster[i] != -1) and cluster_segment_freq[k, in_cluster[i]])):
                for j, t2 in enumerate(terms[(i+1):]):
                    if ((t2 in s) or ((in_cluster[i+1+j] != -1) and cluster_segment_freq[k, in_cluster[i+1+j]])):
                        freq_matrix[i, i+1+j] += 1
                        freq_matrix[i+1+j, i] += 1
        
    mean_freq = np.mean(freq_matrix, axis=0)

    for i, t1 in enumerate(terms):
        # ensure that the pair frequency is statistically significant
        # ensure that the term is a discriminative matching
        scaled_matrix[i] = np.log(np.divide(freq_matrix[i], np.maximum(mean_freq, mean_freq[i])), where=(mean_freq != 0)) * np.log(len(terms)/(np.maximum(np.sum(freq_matrix > 1, axis=0), np.sum(freq_matrix[i] > 1)) + 1))

    return freq_matrix, scaled_matrix

def segment_joint_salience(terms, all_terms, key_term_dict, co_occurrence_matrix, cosim_matrix, verbal=False):
    all_saliences = [[-1 for s in np.arange(len(terms[i]))] for i in np.arange(len(terms))]
    all_overall = [[] for i in np.arange(len(terms))]
    all_co_occur = [[] for i in np.arange(len(terms))]
    all_cosim = [[] for i in np.arange(len(terms))]

    for doc_id in np.arange(len(terms)):
        for seg_id, seg_t in enumerate(terms[doc_id]):
            # if the seg has no key terms
            if len(seg_t) == 0:
                all_saliences[doc_id][seg_id] = 0
            else:
                # overall salience
                overall = np.array([key_term_dict[i] for i in seg_t])
                all_overall[doc_id].append(np.log(len(terms[doc_id][seg_id])+1) * overall)

                # co-occurrence
                co_occur = np.ones((len(seg_t), len(seg_t)))
                for i, t1 in enumerate(seg_t):
                    for j, t2 in enumerate(seg_t):
                        if (i == j):
                            co_val = 0 # we don't want duplicate terms to impact the inner-co-occurrence
                        else:
                            co_val = co_occurrence_matrix[all_terms.index(t2), all_terms.index(t1)]
                        co_occur[i, j] = co_val
                
                # get a co-occurrence score for each of the key terms in the segment
                if len(seg_t) > 1:
                    co_occur = (co_occur.sum(1)-np.diag(co_occur))/(co_occur.shape[1]-1) # np.mean(co_occur, axis=1)
                else:
                    co_occur = np.log(len(terms[doc_id][seg_id])+1) * co_occur[0]
                all_co_occur[doc_id].append(co_occur)

                # semantic similarity
                cosim = np.zeros((len(seg_t), len(seg_t)))
                for i, t1 in enumerate(seg_t):
                    for j, t2 in enumerate(seg_t):
                        if i == j:
                            cosim[i,j] = 0
                        else:
                            cosim[i, j] = cosim_matrix[all_terms.index(t1), all_terms.index(t2)]
                if len(seg_t) > 1:
                    cosim = (cosim.sum(1)-np.diag(cosim))/(cosim.shape[1]-1) #np.mean(cosim, axis=1)
                else:
                    cosim = cosim[0]
                all_cosim[doc_id].append(cosim)

    # normalize across segments
    flat_overall = np.concatenate([np.concatenate(all_overall[i]).ravel() for i in np.arange(len(terms)) if len(all_overall[i]) > 0]).ravel()
    flat_co_occur = np.concatenate([np.concatenate(all_co_occur[i]).ravel() for i in np.arange(len(terms)) if len(all_co_occur[i]) > 0]).ravel()
    flat_cosim = np.concatenate([np.concatenate(all_cosim[i]).ravel() for i in np.arange(len(terms)) if len(all_cosim[i]) > 0]).ravel()

    for doc_id in np.arange(len(all_overall)):
        for seg_id in np.arange(len(all_overall[doc_id])):
            all_overall[doc_id][seg_id] = (all_overall[doc_id][seg_id] - np.min(flat_overall))/(np.max(flat_overall) - np.min(flat_overall))
            all_co_occur[doc_id][seg_id] = (all_co_occur[doc_id][seg_id] - np.min(flat_co_occur))/(np.max(flat_co_occur) - np.min(flat_co_occur))

            all_cosim[doc_id][seg_id] = (all_cosim[doc_id][seg_id] - np.min(flat_cosim))/(np.max(flat_cosim) - np.min(flat_cosim))
    
    # combine
    for doc_id in np.arange(len(terms)):
        i = 0 # for empty segments, we don't have a corresponding list in overall, co_occur, and cosim
        for seg_id in np.arange(len(terms[doc_id])):
            if all_saliences[doc_id][seg_id] == -1:
                all_saliences[doc_id][seg_id] = ((np.mean(all_overall[doc_id][i]) + np.mean(all_co_occur[doc_id][i]) + np.mean(all_cosim[doc_id][i]))/3)
                i += 1
            else:
                all_saliences[doc_id][seg_id] = 0

    return all_saliences, all_overall, all_co_occur, all_cosim



def episode_segmentation(doc_idx, raw_dataset, seg_terms, all_terms, all_saliences, all_overall, 
                         co_occurrence_matrix, cosim_matrix, 
                         static_word_representations, word_to_index, verbal=False):

    s = [s_val for d in all_saliences for s_val in d]# all_saliences[doc_idx] # np.array([np.mean(all_overall[doc_idx][i]) for i in np.arange(len(raw_dataset[doc_idx]))])
    mean_s = np.mean(s)
    std_s = np.std(s)

    salient_segs = list(filter(lambda x: all_saliences[doc_idx][x] >= (mean_s - 1*std_s), np.arange(len(raw_dataset[doc_idx]))))
    if len(salient_segs) == 0:
        return []
    
    min_salient = min(salient_segs)
    if len(salient_segs) == 1:
        return [[min_salient]]

    episode_segments = [[min_salient]]
    if verbal:
        print("salient:", len(salient_segs), salient_segs)
        print(np.mean(all_overall[doc_idx][0]), raw_dataset[doc_idx][0])
    
    track = np.zeros((len(salient_segs)-1, 5))

    for i in np.arange(min_salient+1, len(salient_segs)):
        prev = salient_segs[i-1]
        curr = salient_segs[i]

        cross_co_occur = []
        cross_sim = []
        for tx in seg_terms[doc_idx][curr]:
            occur_temp = []
            sim_temp = []
            for ty in seg_terms[doc_idx][prev]:
                occur_temp.append(co_occurrence_matrix[all_terms.index(ty), all_terms.index(tx)])
                sim_temp.append(cosim_matrix[all_terms.index(tx), all_terms.index(ty)])
            
            # weigh the term co-occurrences w.r.t the prev terms discriminative ranking (seg_terms is in descending rank order)
            cross_co_occur.append(np.mean(occur_temp))
            # cross_co_occur.append(np.mean(occur_temp))
            
            cross_sim.append(np.mean(sim_temp))
        
        cross_avgsim = cosine_similarity([np.mean([static_word_representations[word_to_index[w]] for w in seg_terms[doc_idx][prev]], axis=0), np.mean([static_word_representations[word_to_index[w]] for w in seg_terms[doc_idx][curr]], axis=0)])[0, 1]
        # track[i-1, :] = np.array([np.mean(all_overall[doc_idx][i]), np.mean(cross_co_occur), np.mean(cross_sim), cross_avgsim])
        track[i-1, :] = np.array([all_saliences[doc_idx][curr], np.mean(all_overall[doc_idx][i]), average_with_harmonic_series(cross_co_occur), np.mean(cross_sim), cross_avgsim])
        # track[i-1, :] = np.array([all_saliences[doc_idx][curr], np.mean(all_overall[doc_idx][i]), np.mean(cross_co_occur), np.mean(cross_sim), cross_avgsim])

    mean_track = np.mean(track, axis=0)
    std_track = np.std(track, axis=0)

    curr_seg_id = 0
    for i in np.arange(min_salient+1, len(salient_segs)):
        co_occurs = track[i-1, 2] >= (mean_track[2] - 1*std_track[2])
        similar = track[i-1, 3] >= (mean_track[3] - 1*std_track[3])
        avg_similar = track[i-1, 4] >= (mean_track[4] - 1*std_track[4])

        # if boundary (co-occur OR cosim)
        if not co_occurs or not (similar and avg_similar):
            if verbal:
                print("\n")
            episode_segments.append([salient_segs[i]])
            curr_seg_id += 1
        else:
            episode_segments[curr_seg_id].append(salient_segs[i])
        if verbal:
            print(salient_segs[i], co_occurs, similar, avg_similar, track[i-1],  raw_dataset[doc_idx][salient_segs[i]])
        
    return episode_segments

def compare_episodes(terms_a, terms_b, co_occurrence_matrix, cosim_matrix, all_terms, static_word_representations, word_to_index):
    cross_co_occur = []
    cross_sim = []
    for tx in terms_a:
        occur_temp = []
        sim_temp = []
        for ty in terms_b:
            occur_temp.append(co_occurrence_matrix[all_terms.index(ty), all_terms.index(tx)])
            sim_temp.append(cosim_matrix[all_terms.index(tx), all_terms.index(ty)])
        cross_co_occur.append(np.mean(occur_temp))
        cross_sim.append(np.mean(sim_temp))
    cross_avgsim = cosine_similarity([np.mean([static_word_representations[word_to_index[w]] for w in terms_b], axis=0), np.mean([static_word_representations[word_to_index[w]] for w in terms_a], axis=0)])[0, 1]
    # track[i-1, :] = np.array([np.mean(all_overall[doc_idx][i]), np.mean(cross_co_occur), np.mean(cross_sim), cross_avgsim])
    return [np.mean(cross_co_occur), np.mean(cross_sim), cross_avgsim]

def clusterEpisodes(merging_episodes, candidate_episodes):

    merge_co_occur_norm = 1/((merging_episodes[0] - merging_episodes[0].min())/(merging_episodes[0].max() - merging_episodes[0].min()) + 1)
    merge_co_occur_norm = (merge_co_occur_norm - merge_co_occur_norm.min())/(merge_co_occur_norm.max() - merge_co_occur_norm.min())


    merge_sim_norm = 1/((merging_episodes[2] - merging_episodes[2].min())/(merging_episodes[2].max() - merging_episodes[2].min()) + 1)
    merge_sim_norm = (merge_sim_norm - merge_sim_norm.min())/(merge_sim_norm.max() - merge_sim_norm.min())

    mean_track = np.mean(merge_co_occur_norm + 2 * merge_sim_norm)
    std_track = np.std(merge_co_occur_norm + 2 * merge_sim_norm)

    clustering = AgglomerativeClustering(n_clusters=None, linkage= 'complete', metric='precomputed', distance_threshold=mean_track - std_track).fit(merge_co_occur_norm + 2* merge_sim_norm)

    episode_clusters = [[] for i in np.arange(len(set(clustering.labels_)))]

    for id, e in enumerate(clustering.labels_):
        episode_clusters[e].append(id)

    cluster_lengths = [np.sum([len(candidate_episodes[i][2]) for i in episode_clusters[i]]) for i in np.arange(len(episode_clusters))]

    return episode_clusters, cluster_lengths

class text_LLM():
    def __init__(self, args):
        api_key = args.api_key
        self.anthropic = Anthropic(api_key=api_key)
    def infer(self,prompt):
        completion = self.anthropic.completions.create(model="claude-2.1",
                                            max_tokens_to_sample=2048,
                                            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
                                            temperature=1)
        return completion.completion

def evaluate(args, episodes, gt):
    # episodes = json.load(open(os.path.join(epi_root, f'{args.ep}.json'), encoding='utf-8'))
    thres = args.eval_top // 2 + 1
    
    tp = set()
    p = 0
    for ep in episodes:
        if len(ep) < args.eval_top:
            continue
        p += 1
        for did, text in ep[:args.eval_top]:
            if text not in gt:
                print('Not in gt:', did, text)
        c = Counter([gt[text] for did, text in ep[:args.eval_top] if text in gt and gt[text] not in {'X', 'M'}])
        for gt_ev, count in c.items():
            if count >= thres:
                tp.add(gt_ev)
                break

    try:
        prec = len(tp) / float(p)
    except ZeroDivisionError:
        prec = 0
    print(f'{args.eval_top}-Precision: {prec}')
    
    try:
        recall = len(tp) / float(args.num)
    except ZeroDivisionError:
        recall = 0
    print(f'{args.eval_top}-Recall: {recall}')
    
    try:
        f1 = 2 * prec * recall / (prec + recall)
    except ZeroDivisionError:
        f1 = 0
    print(f'{args.eval_top}-F1: {f1}')

    return prec, recall, f1

# main method

def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("### STEP 1: Read in dataset & process. ###")

    translator = str.maketrans(punctuation, ' '*len(punctuation)) #map punctuation to space

    all_raw_segs, raw_dataset, raw_dataset_tok_lower = readData(args, translator)
    all_ev_id, gt = getGroundTruth(args)

    with open("episode_dataset/background.txt", "r") as f:
        background_dataset = [d.strip() for d in f.readlines()]

    vectorizer = CountVectorizer(stop_words='english')
    count = vectorizer.fit_transform(all_raw_segs + background_dataset).toarray()
    vocab = vectorizer.vocabulary_
    stop_words = list(vectorizer.get_stop_words())

    static_representations.main(args, all_raw_segs) # [s[1] for s in filtered_segs]

    static_repr_path = os.path.join('intermediate_data', args.theme, args.title, f"static_repr_lm-{args.lm_type}-{args.layer}.pk")
    with open(static_repr_path, "rb") as f:
        vocab = pk.load(f)
        static_word_representations = vocab["static_word_representations"]
        word_to_index = vocab["word_to_index"]
        vocab_words = vocab["vocab_words"]
    
    unfiltered = set([(salience(" " + i, len(background_dataset), all_raw_segs, background_dataset, stop_words), i) for i in tqdm(vocab_words)])



    print("### STEP 2: Identify key terms and expand list. ###")

    thresh = np.mean([i[0] for i in unfiltered if i[0] != -1])
    key_terms = sorted(list(filter(lambda x: x[0] > thresh, unfiltered)), key=lambda x: -x[0])
    all_terms = [w[1] for w in sorted(list(set(key_terms)), key=lambda x:-x[0])]

    # expand the set of key terms based with very similar but less frequent terms (to account for journalist stylistic differences)
    excluded_terms = [w for w in vocab_words if w not in all_terms]
    expand_cos = cosine_similarity([static_word_representations[word_to_index[w]] for w in excluded_terms], [static_word_representations[word_to_index[w]] for w in all_terms])
    expand_max = expand_cos.max(axis=1)
    expand_argmax = expand_cos.argmax(axis=1)
    expand_terms = [(i, w, all_terms[expand_argmax[i]], expand_max[i]) for i, w in enumerate(excluded_terms) if (expand_max[i] >= 0.75) and (w not in stop_words)]
    key_terms += [(i[3] * key_terms[expand_argmax[i[0]]][0], i[1]) for i in expand_terms]
    all_terms = [w[1] for w in sorted(list(set(key_terms)), key=lambda x:-x[0])]

    all_cos = cosine_similarity([static_word_representations[word_to_index[w]] for w in all_terms], [static_word_representations[word_to_index[w]] for w in all_terms])
    np.fill_diagonal(all_cos, -1)

    in_cluster, word_clusters = expandTerms(all_terms, all_cos, sim_thresh=0.85)

    for i in word_clusters:
        if len(i) > 1:
            print(i)
    print("Expanded terms:", len(key_terms)/len(vocab_words), len(expand_terms))



    tokenized_segs = [word_tokenize(s.translate(translator)) for s in all_raw_segs]
    cluster_segment_freq = np.zeros((len(tokenized_segs), len(word_clusters)))
    for seg_id, s in tqdm(enumerate(tokenized_segs), total=len(tokenized_segs)):
        for clus_id, c in enumerate(word_clusters):
            cluster_segment_freq[seg_id][clus_id] = any(w in s for w in c)
    

    print("### STEP 3: Computing discriminative co-occurence for detecting candidate episodes. ###")

    freq_matrix, co_occurrence_raw = co_occurrence(all_raw_segs, all_terms, in_cluster, cluster_segment_freq)

    min_co_occur = np.min(np.nan_to_num(co_occurrence_raw, neginf=np.max(co_occurrence_raw)))
    max_co_occur = np.max(co_occurrence_raw)

    co_occurrence_matrix = np.nan_to_num(co_occurrence_raw, neginf=min_co_occur)
    np.fill_diagonal(co_occurrence_matrix, max_co_occur)
    # do the same for all of the words within a word cluster
    for c_id, c in enumerate(word_clusters):
        if len(c) > 0:
            for t1 in set(word_clusters[c_id]):
                for t2 in set(word_clusters[c_id]):
                    if t1 != t2:
                        co_occurrence_matrix[all_terms.index(t1), all_terms.index(t2)] = max_co_occur
                        co_occurrence_matrix[all_terms.index(t2), all_terms.index(t1)] = max_co_occur
    
    cosim_matrix = cosine_similarity([static_word_representations[word_to_index[w]] for w in all_terms], [static_word_representations[word_to_index[w]] for w in all_terms])
    cosim_matrix = (cosim_matrix - cosim_matrix.min())/(cosim_matrix.max() - cosim_matrix.min())

    min_sal = min(key_terms, key=lambda x : x[0])[0]
    max_sal = max(key_terms, key=lambda x : x[0])[0]
    key_term_dict = {i[1]:(i[0]-min_sal)/(max_sal-min_sal) for i in key_terms}
    term_rank = sorted(all_terms, key = lambda x: np.log(np.max(freq_matrix[all_terms.index(x)])) * np.max(co_occurrence_raw[all_terms.index(x)]), reverse=True)

    seg_terms = []
    # gather all key terms in segment
    # for seg in raw_dataset[idx]
    for doc_idx in np.arange(len(raw_dataset)):
        seg_terms.append([])
        for seg_id in np.arange(len(raw_dataset[doc_idx])):
            counts = Counter(raw_dataset_tok_lower[doc_idx][seg_id])
            seg_terms[doc_idx].append([])
            for w in term_rank:
                freq_w = counts[w]
                if freq_w:
                    seg_terms[doc_idx][seg_id].extend([w])

    all_saliences, all_overall, all_co_occur, all_cosim = segment_joint_salience(seg_terms, all_terms, key_term_dict, co_occurrence_matrix, cosim_matrix)
    

    print("### STEP 4: Partition each article based on consecutive segment co-occurrence and semantic similarity. ###")
    
    contained = []
    episode_segs = []
    flat_episode_segs = []
    num_episodes = 0
    for doc_idx in np.arange(len(raw_dataset)):
        episode_segs.append([])
        res = episode_segmentation(doc_idx, raw_dataset, seg_terms, all_terms, all_saliences, all_overall, co_occurrence_matrix, cosim_matrix, static_word_representations, word_to_index)
        for e_id, r in enumerate(res):
            episode_text = " ".join([raw_dataset[doc_idx][i] for i in r])
            if not (episode_text in contained):
                contained.append(episode_text)
                episode_segs[doc_idx].append((r, episode_text))
                flat_episode_segs.append((doc_idx, e_id, r, num_episodes, " ".join([raw_dataset[doc_idx][i] for i in r])))
                num_episodes += 1

    print(num_episodes, "episodes")

    i = 0
    ranks = []
    for total, overall, co_occur, cosim in zip(all_saliences, all_overall, all_co_occur, all_cosim):
        val = np.mean(np.concatenate(co_occur).ravel()) if len(co_occur) > 0 else 0
        ranks.append(val*np.log(len(episode_segs[i]) + 1))
        i += 1
    
    # true_top = [(id, set([gt[raw_dataset[id][s[0][0]]] for s in d if gt[raw_dataset[id][s[0][0]]] not in ["M", "X"]])) for id, d in enumerate(episode_segs)]
    top_docs = sorted(list(enumerate(ranks)), key=lambda x: -x[1])

    candidate_docs = top_docs[:int(len(top_docs)*args.doc_thresh)]
    candidate_episodes = []
    for d in candidate_docs:
        for e_id, e in enumerate(episode_segs[d[0]]):
            candidate_episodes.append((d[0], e_id, e[0], e[1]))

    merging_episodes = np.zeros((3, len(candidate_episodes), len(candidate_episodes)))

    epi_terms = []
    for a in np.arange(len(candidate_episodes)):
        doc_idx = candidate_episodes[a][0]
        epi_terms.append([])
        for i in candidate_episodes[a][2]:
            for t in seg_terms[doc_idx][i]:
                if not (t in epi_terms[a]):
                    epi_terms[a].append(t)
        # can ignore this sorting
        epi_terms[a] = sorted(epi_terms[a], key = lambda x: np.log(max(freq_matrix[all_terms.index(x)])) * max(co_occurrence_raw[all_terms.index(x)]), reverse=True)
    
                     
    for a in tqdm(np.arange(len(candidate_episodes))):
        for b in np.arange(a + 1, len(candidate_episodes)):
            if a != b:
                merging_episodes[:, a, b] = compare_episodes(epi_terms[a], epi_terms[b], co_occurrence_matrix, cosim_matrix, all_terms, static_word_representations, word_to_index)
                merging_episodes[:, b, a] = merging_episodes[:, a, b].copy()

    


    print("### STEP 5: Cluster the top documents' candidate episodes. ###")

    episode_clusters, cluster_lengths = clusterEpisodes(merging_episodes, candidate_episodes)

    
    final_prompt = ""

    avg_len = []
    final_episodes = []
    id = 0
    for e in episode_clusters:
        if len(set(e)) >= 1:
            avg_len.append(len(set(e)))

    if args.theme == 'natural_disasters':
        lower_thresh = np.mean(cluster_lengths) - np.std(cluster_lengths)
    else:
        lower_thresh = np.mean(cluster_lengths) + np.std(cluster_lengths)

    for e_id, e in enumerate(episode_clusters):
        if (len(set(e)) >= 1) and (cluster_lengths[e_id] >= (lower_thresh)):
            final_prompt += f"Group #{id}:\n"
            final_episodes.append([])
            for s in set(e):
                doc_idx = candidate_episodes[s][0]
                for seg_id in candidate_episodes[s][2]:
                    final_episodes[id].append((all_saliences[doc_idx][seg_id], raw_dataset[doc_idx][seg_id]))
            for i in final_episodes[id][:args.eval_top]:
                final_prompt += i[1] + "\n"
                    
            final_prompt += "\n"
            id += 1
    
    final_prompt += f"You are a key news event analyzer that is aiming to detect episodes (a representative subevent that reflects a critical sequence of actions performed by a subject at a certain and/or location) based on text segments from different news articles. Given the above groups of article segments, predict at least 2 and at most {max(3, id)} potential episodes of the key event. Some groups may fall under the same episode. Output your answer inside the tags <answer></answer> as a JSON object where each item is also a JSON with the key \"title\" with the value containing the [subject, action, object, time, location] of the episode, a key \"keywords\" with the string value being a list of 5-10 associated keywords unique to that specific episode, and a final key \"example_sentences\" with a value being a list of 2-5 extracted sentences from the input segment groups. Feel free to output less than {id} episodes if you feel that any are redundant (could fit under an existing candidate episode). The title, keywords, and example sentences of a predicted episode should not be able to be placed under another different predicted episode."

    text_model = text_LLM(args)
    client = Anthropic()

    sentmodel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device) # for encoding location
    flat_dataset = [s for d in raw_dataset for s in d]
    total_segs = len(flat_dataset)
    seg_embs = []
    start = 0
    for i in tqdm(np.arange(start=0, stop=total_segs, step=args.batch_size)):
        seg_embs.append(sentmodel.encode(flat_dataset[i:min(total_segs, i+args.batch_size)]))
    seg_embs = np.concatenate(seg_embs, axis=0)

    input = final_prompt
    final_results = None

    for i in np.arange(args.trials):
        print(f"Trial {i}")
        max_attempts = 5
        success = 0
        attempts = 0
        llm_episodes = None
        while (not success) and (attempts < 5):
            try:
                episodes_out = text_model.infer(input)
                if "```json" in episodes_out:
                    llm_episodes = json.loads(episodes_out.split("```json")[1].split("```")[0])
                else:
                    llm_episodes = json.loads(episodes_out.split("<answer>\n", maxsplit=1)[1].split("</answer>", maxsplit=1)[0])
                if (type(llm_episodes[0]) == dict) and (len(llm_episodes) >= 2) and ("example_sentences" in llm_episodes[0].keys()):
                    success = True
                else:
                    print(f"candidate llm-based episode failed! attempt #{attempts+1}")
                attempts += 1
            except:
                success = False
                print(f"candidate llm-based episode failed! attempt #{attempts+1}")
                attempts += 1

        if final_results is None:
            print(llm_episodes)
            print("### STEP 6: Classification based on LLM-based episode detection. ###")

        final_episode_clusters = None

        if (llm_episodes is not None) and (type(llm_episodes[0]) == dict) and (len(llm_episodes) >= 2) and ("example_sentences" in llm_episodes[0].keys()):
            cluster_reprs = [average_with_harmonic_series(np.concatenate((sentmodel.encode(" ".join(i["title"])).reshape((-1,args.emb_dim)), sentmodel.encode(i["example_sentences"]).reshape((-1, args.emb_dim))), axis=0)) for i in llm_episodes]

            cluster_sim = cosine_similarity(seg_embs, cluster_reprs)
            toptwo = np.partition(cluster_sim, -2)[:, -2:]
            toptwo = (toptwo[:, 1] - toptwo[:, 0])
            score = toptwo / np.sum(toptwo)
            mapping = cluster_sim.argmax(axis=1)


            final_clusters = [[] for i in llm_episodes]

            for i in np.arange(len(seg_embs)):
                if score[i] >= (np.mean(score) + np.std(score)):
                    final_clusters[mapping[i]].append((score[i], flat_dataset[i]))

            final_episode_clusters = []
            for i in np.arange(len(final_clusters)):
                if len(final_clusters[i]) > 1:
                    final_episode_clusters.append(sorted(final_clusters[i], key= lambda x : x[0], reverse=True))

            prec, recall, f1 = evaluate(args, final_episode_clusters, gt)

            if final_results is None:
                final_results = np.array([prec, recall, f1]).reshape((1,-1))
            else:
                final_results = np.concatenate((final_results, np.array([prec, recall, f1]).reshape((1,-1))), axis=0)
    
    if final_results is None:
        return llm_episodes, final_episode_clusters, None, None

    max_score = np.argmax(np.mean(final_results, axis=1))

    return llm_episodes, final_episode_clusters, np.mean(final_results, axis=0), final_results[max_score]