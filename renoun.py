import spacy
from spacy.tokens import Token

import networkx as nx
import pandas as pd

import re
import matplotlib.pyplot as plt

Rules = [
r"the {att} of (.+), {obj}",
r"the {att} of (.+) is {obj}",
r"{obj}, {att} of (.+)",
r"{obj}, the {att} of (.+)",
r"{obj}, (.+)’s {att}",
r"{obj}, (.+) {att}",
r"(.+) {att} {obj}",
r"(.+)’s {att}, {obj}",
r"(.+) {att}, {obj}"
]

# official language を追加（トリプルの翻訳結果が "official teminology" となってしまっているため
# CEO などの略語はそのまま、それ以外は小文字にする
Attributes = ["CEO", "official language","Prime Minister","prime minister","District Judge"]
#df = pd.read_csv("./data/openie_eval_200221_GG.csv")
#SPO_en = df['S P O_en']
#for triple in SPO_en:
 #   rel = triple.split("|")[1]
  #  if rel == rel.upper():
   #     Attributes.append(rel)
    #else:
     #   Attributes.append(rel.lower())

#Attributes = list(set(Attributes))

NLP = spacy.load("en_core_web_sm")

# TODO
def resolution(word_list):
    ret = []
    for word in word_list:
        ret.append((word, 1))
    return ret

def get_noun_list(doc):
    noun_chunk_list = [chunk.text for chunk in doc.noun_chunks]
    ner_list = [ent.text for ent in doc.ents]
    noun_list = list(set(noun_chunk_list) | set(ner_list))
    trans_dict = {'(':'', ')':'', '[':'', ']':'', '\\': '', '"': ''}
    noun_list = [noun.translate(noun.maketrans(trans_dict)) for noun in noun_list]
    return noun_list

def get_noun_list_over(doc):
    noun_list = [token.text for token in doc if token.pos_ in ["NOUN", "PRON", "PROPN", "X"]]
    noun_chunk_list = [chunk.text for chunk in doc.noun_chunks]
    ner_list = [ent.text for ent in doc.ents]
    return list(set(noun_chunk_list) | set(ner_list) | set(noun_list))

def extract_seed_fact(doc):
    noun_list = get_noun_list(doc)

    matched_atts = []
    for att in Attributes:
        if att in doc.text:
            matched_atts.append(att)
            if att in noun_list:
                noun_list.remove(att)
    if matched_atts == []:
        return []

    noun_ids = resolution(noun_list)
    attribute_ids = resolution(matched_atts)

    AO_list = []
    for att, att_id in attribute_ids:
        for noun, noun_id in noun_ids:
            if att_id == noun_id:
                AO_list.append((att, noun))
    if AO_list == []:
        return []

    SAO_list = []
    for A, O in AO_list:
        for i, rule in enumerate(Rules):
            r_att = rule.format(att=A, obj=O)
            try:
                m = re.match(r_att, doc.text)
            except re.error:
                #print(r_att, A, O)
                continue
            if m != None:
                S = m.groups()[0]
                if (i == 2) or (i == 3):
                    found = False
                    for noun in noun_list:
                        if S.find(noun) == 0:
                            S = noun
                            found = True
                            break
                    if found:
                        SAO_list.append((S, A, O))
                else:
                    if S in noun_list:
                        SAO_list.append((S, A, O))
    return SAO_list

def get_root(s):
    for token in NLP(s):
        if token.dep_ == 'ROOT':
            return token.text

def create_graph(doc):
    G = nx.Graph()
    for token in doc:
        if token != token.head: # except for ROOT
            G.add_edge(token, token.head, head=token.head, relation=token.dep_)
    return G

# 翻訳がうまくいかず文章が分断されてしまい、木構造とならないケースがあるので除く
# 具体的には、分断された連結成分については最後は一つの点となるはずなのでそれを無条件に消してしまう
# 残った側の連結成分にsaoが含まれていたら返して、ダメなら例外として空リストを返す

def get_subgraph(graph, s, a, o):
    leaf_list = []
    for node in graph.nodes:
        if graph.degree(node) == 1:
            leaf_list.append((node, 1))
        elif graph.degree(node) == 0:
            leaf_list.append((node, 0))
    removed = False
    for leaf, degree in leaf_list:
        if degree == 0:
            graph.remove_node(leaf)
            removed = True
        elif degree == 1:
            if (leaf != s) and (leaf != a) and (leaf != o):
                graph.remove_node(leaf)
                removed = True
    if removed:
        return get_subgraph(graph, s, a, o)
    else:
        sao_count = 0
        for node in graph.nodes:
            if (node == s) or (node == a) or (node == o):
                sao_count += 1
        if sao_count == 3:
            return graph
        else:
            return []

def dependency_pattern_generation(doc, seed_fact):
    S, A, O = seed_fact
    S_root, A_root, O_root = get_root(S), get_root(A), get_root(O)
    S_tokens, A_tokens, O_tokens = [], [], []
    for token in doc:
        if token.text == S_root:
            S_tokens.append(token)
        if token.text == A_root:
            A_tokens.append(token)
        if token.text == O_root:
            O_tokens.append(token)

    # doc 中の全通り
    SAO_tokens = [(s, a, o) for s in S_tokens for a in A_tokens for o in O_tokens]

    dep_pattern_set = set()
    for s, a, o in SAO_tokens:
        # remove は graph のメソッドなのでグラフ本体を変更してしまう
        # deepcopy はエラー（Spacy の token object が未対応）
        # そこまでのボトルネックではないと見ているが、毎回同じグラフを作るのは望ましくなさそう
        dep_graph = create_graph(doc)

        dep_subgraph = get_subgraph(dep_graph, s, a, o)
        if dep_subgraph == []:
            continue

        mapping = {}
        for node in list(dep_subgraph.nodes):
            if node == s:
                mapping[node] = ("S", 0)
            elif node == a:
                mapping[node] = ("A", 0)
            elif node == o:
                mapping[node] = ("O", 0)
            else:
                not_found = True
                for v in mapping.values():
                    if v[0] == node.text:
                        mapping[node] = (node.text, v[1]+1)
                        not_found = False
                        break
                if not_found:
                    mapping[node] = (node.text, 0)

        delex_subgraph = nx.relabel_nodes(dep_subgraph, mapping)

        for edge_info in dict(delex_subgraph.edges).values():
            if edge_info['head'] == s:
                edge_info['head'] = "S";
            elif edge_info['head'] == a:
                edge_info['head'] = "A";
            elif edge_info['head'] == o:
                edge_info['head'] = "O";
            else:
                edge_info['head'] = edge_info['head'].text

        dep_pattern_set.add(delex_subgraph)

    return (dep_pattern_set, A)

def find_SO(att_token, node_t, node_p, dep_tree, dep_patt, token_check_dict):
    def get_another_node(node, edge):
        node1, node2 = edge
        if node1 == node:
            return node2
        else:
            return node1
    def word_eq(token, text, att_token):
        noun_pos = ["NOUN", "PRON", "PROPN", "X"]
        if token == att_token:
            if text == "A":
                return True
        elif (text == "S") or (text == "O"):
            if token.pos_ in noun_pos:
                return True
        else:
            if token.text == text:
                return True
        return False

    for edge_t in list(dep_tree.edges(node_t)):
        for edge_p in list(dep_patt.edges(node_p)):
            another_node_t = get_another_node(node_t, edge_t)
            another_node_p = get_another_node(node_p, edge_p)
            e_t = dep_tree[node_t][another_node_t]
            e_p = dep_patt[node_p][another_node_p]
            #print("edge_t:", node_t, another_node_t, e_t)
            #print("edge_p:", node_p, another_node_p, e_p)
            #print()
            if (e_t['relation'] == e_p['relation']) and \
               (word_eq(e_t['head'], e_p['head'], att_token)) and \
               (word_eq(another_node_t, another_node_p[0], att_token)) and \
               (not token_check_dict[another_node_p[0]]):
                token_check_dict[another_node_p[0]] = another_node_t.text
                #print("added:", another_node_t)
                token_check_dict = find_SO(att_token, another_node_t, another_node_p, dep_tree, dep_patt, token_check_dict)

    return token_check_dict

def candidate_generation(doc, att_patt_graph):
    noun_list = get_noun_list(doc)
    SO_dict = {}
    for noun_phrase in noun_list:
        SO_dict.setdefault(get_root(noun_phrase), []).append(noun_phrase)

    A_dict = {}

    for att_p in Attributes:
        if att_p in doc.text:
            for token in doc:
                if (get_root(att_p) == token.text):
                    A_dict[token] = att_p

    if A_dict == {}:
        return []

    dep_tree = create_graph(doc)

    found_sao = []
    for att_token, att_p in A_dict.items():
        init_node_t = att_token
        init_node_p = ("A", 0)
        try:
            matched_patt_list = list(nx.all_neighbors(att_patt_graph, att_token.text))
        except nx.exception.NetworkXError:
            continue

        for i, dep_patt in enumerate(matched_patt_list, 1):
            token_check_dict = {}
            for patt_node, _ in list(dep_patt.nodes):
                token_check_dict[patt_node] = False
            token_check_dict["A"] = att_token.text
            token_check_dict = find_SO(att_token, init_node_t, init_node_p, dep_tree, dep_patt, token_check_dict)

            found = True
            for v in token_check_dict.values():
                if v == False:
                    found = False
            if found:
                try:
                    extracted_triple = (SO_dict[token_check_dict["S"]], A_dict[att_token], SO_dict[token_check_dict["O"]])
                    found_sao.append(extracted_triple)
                except KeyError:
                    continue
                    # print("!!Noun not found!!")
                    # print(token_check_dict)
                    # print(SO_dict)

    return found_sao