import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
import timm
import math
from t2iattention import BertCrossEncoder

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class SIAMNER(nn.Module):
    def __init__(self, label_list, args):
        """
        label_list: the list of target labels
        args: argparse
        """
        super(SIAMNER, self).__init__()
        self.args = args
        model_path = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(model_path)  # get the pre-trained BERT model for the text

        self.model_resnet50 = timm.create_model('resnet50', pretrained=True)  # get the pre-trained ResNet model for the image
        self.num_labels = len(label_list)  # the number of target labels
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(1000, self.num_labels)
        self.dropout = nn.Dropout(0.2)

        self.linear_pic = nn.Linear(2048, self.bert.config.hidden_size)  # Used to map ResNet features to BERT hidden size
        self.vismap2text = nn.Linear(2048, self.bert.config.hidden_size)  # Convert image features to text-compatible embeddings
        self.txt2img_attention = BertCrossEncoder(self.bert.config, 1)  # Cross attention module for text and image embeddings
        self.norm_t2i_process = nn.LayerNorm(self.bert.config.hidden_size)

        self.Gate = nn.Linear(self.bert.config.hidden_size * 4, self.bert.config.hidden_size)  # Gate mechanism
        self.Gate_image = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)  # Linear layer for image part of the gate

        self.combine_linear = nn.Linear(self.bert.config.hidden_size * 2, 1000)  # Combine text and image information

    def forward(self, input_ids=None, attention_mask=None,added_input_mask=None, labels=None, images=None, weight=None):

        output_text = self.bert(input_ids, attention_mask)
        hidden_text = self.dropout(output_text['last_hidden_state'])

        feature_OriImg_FineGrained = self.model_resnet50.forward_features(images)
        pic_ori = feature_OriImg_FineGrained.view(-1, 2048, 49).permute(0, 2, 1)
        pic_ori_ = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori_ = torch.transpose(pic_ori, 1, 2)
        pic_ori_ = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori_ = self.linear_pic(pic_ori)

        ori_converted_vis_embed_map = self.vismap2text(pic_ori)

        img_mask = added_input_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        ori_cross_encoder = self.txt2img_attention(hidden_text, ori_converted_vis_embed_map,extended_img_mask)

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_origin = self.linear_q_fine(pic_ori_)

        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)
        pic_ori_final = (pic_original + torch.sum(pic_ori_, dim=1)) * weight[:, 1].reshape(-1, 1)
        pic_ori_final = self.norm_t2i_process(pic_ori_final)
        res_pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.args.max_seq, self.image2token_emb))

        Gate = self.sigmoid(self.Gate(torch.cat((self.Gate_image(ori_cross_encoder[-1]),res_pic_ori)
                                                              ,dim=-1)))

        final_output = torch.cat((hidden_text,Gate),
                                 dim=-1)

        final_output = self.combine_linear(final_output)
        
        emissions = self.fc(torch.relu(final_output))

        # classification
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='sum')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def att(self, query, key, value):
        scale = torch.sqrt(torch.FloatTensor([self.bert.config.hidden_size])).to(self.args.device)
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        scaled_scores = scores / scale
        att_map = self.dropout(F.softmax(scaled_scores, dim=-1))
        return torch.matmul(att_map, value)


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(sum=0.0, std=0.05)


