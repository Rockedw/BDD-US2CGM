from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from main import *
import spacy


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cal_sim(sentence1, sentence2):
    sentences = [sentence1, sentence2]
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    # Compute cosine-similarity
    cosine_similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
    return cosine_similarity


nlp = spacy.load("en_core_web_sm")


def decapitalize(string):
    return string[:1].lower() + string[1:]


def capitalize(string):
    return string[:1].upper() + string[1:]


def analyze_sentence(sentence):
    doc = nlp(sentence)
    passive = False
    for token in doc:
        print(token.text, token.dep_, token.head.text)
        if 'pass' in token.dep_:
            passive = True
    if not passive:
        sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
        verb_phrase = []
        for token in doc:
            if token.pos_ == "VERB":
                verb_phrase.append(token.text)
            elif token.dep_ == "prt" and token.head.pos_ == "VERB":
                verb_phrase.append(token.text)
        dobj_toks = [tok for tok in doc if (tok.dep_ == "dobj")]
        iobj_toks = [tok for tok in doc if (tok.dep_ == "dative")]
        if sub_toks:
            subject = [tok for tok in sub_toks[0].subtree]
        else:
            subject = []

        if dobj_toks:
            direct_object = [tok for tok in dobj_toks[0].subtree]
        else:
            direct_object = []
        if iobj_toks:
            indirect_object = [tok for tok in iobj_toks[0].subtree]
        else:
            indirect_object = []

        # print("Subject:", subject)
        # print("Verb:", verb_phrase)
        # print("Direct Object:", direct_object)
        # print("Indirect Object:", indirect_object)

        # if verb_phrase:
        #     temp = []
        #     auxs = [tok for tok in doc if tok.dep_ == "aux"]
        #     for v in verb_object:
        #         for aux in auxs:
        #             if v == aux.head:
        #                 temp.append(aux)
        #         temp.append(v)
        #     verb_object = temp
        new_sentence = [subject, verb_phrase, direct_object, indirect_object]
    else:
        dobj_toks = [tok for tok in doc if (tok.dep_ == "nsubjpass")]
        # 第一个字母小写
        if dobj_toks:
            direct_object = [tok for tok in dobj_toks[0].subtree]
        else:
            direct_object = []
        # 输出动词原形
        # verb = [tok for tok in doc if (tok.pos_ == "VERB")][0].lemma_
        verb_phrase = []
        for token in doc:
            if token.pos_ == "VERB":
                verb_phrase.append(token.lemma_)
            elif token.dep_ == "prt" and token.head.pos_ == "VERB":
                verb_phrase.append(token.text)
        sub_toks = [tok for tok in doc if (tok.dep_ == "pobj")]
        # 第一个字母大写
        if sub_toks:
            subject = [tok for tok in sub_toks[0].subtree]
        else:
            subject = []
        # print("Subject:", subject)
        # print("Verb:", verb_phrase)
        # print("Direct Object:", direct_object)
        new_sentence = [subject, verb_phrase, direct_object, []]
    return new_sentence


def extract_predicate(sentence):
    doc = nlp(sentence)
    predicate = []
    for token in doc:
        if "VERB" in token.pos_:
            subtree = [t.text for t in token.subtree]
            predicate.extend(subtree)
            break
    return " ".join(predicate)


sentence = "User can not sign up with invalid data"



def remove_modifiers(sentence: str):
    doc = nlp(sentence)
    new_sentence = []
    for token in doc:
        # print(token.text, token.dep_, token.head.text, token.head.pos_)
        if token.dep_ not in ["amod", "advmod", "npadvmod"]:
            new_sentence.append(token.text)
    return " ".join(new_sentence)


def get_coref(sentence):
    doc = nlp(sentence)
    coref = []
    for token in doc:
        if token.dep_ == "nsubj":
            coref.append(token.text)
    verb_phrase = []
    for token in doc:
        if token.pos_ == "VERB":
            verb_phrase.append(token.text)
        elif token.dep_ == "prt" and token.head.pos_ == "VERB":
            verb_phrase.append(token.text)
    verb = " ".join(verb_phrase)
    return coref, verb


if __name__ == '__main__':

    sentence4 = "Users sign up with invalid data"

    print(analyze_sentence(sentence4))
