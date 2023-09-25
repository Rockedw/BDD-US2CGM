import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def scan_feature(dir_path, save_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.feature'):
                with open(os.path.join(root, file), 'r', encoding='utf8') as f:
                    content = f.read()
                with open(os.path.join(save_path, file, ), 'w', encoding='utf8') as f:
                    f.write(content)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cal_sim(sentence1, sentence2,tokenizer,model):
    sentences = [sentence1, sentence2]
    # Load model from HuggingFace Hub
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


def summarize_goals(goals: list, tokenizer, model):
    text = f"In the system ,There are {len(goals)} behaviors for achieving a goal,"
    for goal in goals:
        text += f" {goal},"
    text += 'What might this goal be? Please answer me with "This goal may be" at the beginning of your answer.'
    response, _ = model.chat(tokenizer, text, history=[])
    res = response.split('This goal may be')[1]
    return res


def test():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
    requirements = """我们需要开发一个在线教育平台，该平台需要支持学生、教师和管理员等多种用户角色。学生可以在平台上浏览课程、观看课程视频、完成课后作业和参加在线考试。教师可以在平台上发布课程、上传课程视频、布置作业和出卷考试。管理员负责管理平台上的用户账号、课程内容和订单信息。
此外，学生还可以与教师和其他学生进行在线交流，讨论课程内容。教师可以查看学生的作业和考试成绩，并对学生进行评价。管理员可以查看平台的运营数据，包括用户数量、课程销售情况等。
该平台还需要支持多种支付方式，包括信用卡、支付宝和微信支付。学生可以使用这些支付方式购买课程。管理员可以查看订单信息，并对订单进行管理。"""
    text = f'有一段需求文本"{requirements}",请对需求文本信息按照用户、用户目标进行分类拆解.'
    response, _ = model.chat(tokenizer, text, history=[])
    print(response)


from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_word_vector(word):
    input_ids = tokenizer.encode(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_state = outputs[0]
    word_vec = last_hidden_state[0, 1, :].numpy()
    return word_vec




if __name__ == '__main__':
    # for file in os.listdir('dataset'):
    #     # 如果 os.path.join('user_story_feature', file) 不存在，创建
    #     if not os.path.exists(os.path.join('user_story_feature', file)):
    #         os.mkdir(os.path.join('user_story_feature', file))
    #
    #     scan_feature(os.path.join('./dataset', file), os.path.join('user_story_feature', file))
    # print(cal_sim('I sort students', 'I sort teachers'))
    print(similarity('healthy', 'unhealthy'))
    print(cal_sim('healthy', 'unhealthy'))

    print(similarity('sick', 'unhealthy'))
    print(cal_sim('sick', 'unhealthy'))

    print(similarity('healthy', 'healthy'))
    print(cal_sim('healthy', 'healthy'))

    print(similarity('healthy', 'not healthy'))
    print(cal_sim('healthy', 'not healthy'))

    print(similarity('healthy', 'happy'))
    print(cal_sim('healthy', 'happy'))

    print(similarity('healthy', 'sad'))
    print(cal_sim('healthy', 'sad'))

    print(similarity('happy', 'unhappy'))
    print(cal_sim('happy', 'unhappy'))

    print(similarity('happy', 'sad'))
    print(cal_sim('happy', 'sad'))
