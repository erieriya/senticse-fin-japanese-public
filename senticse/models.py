import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.models.electra.modeling_electra import ElectraModel, ElectraPreTrainedModel
from transformers.activations import gelu
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from random import randint
from types import SimpleNamespace
from senticse.data import load_sentiment_vocab


def _model_arguments(config, supplied):
    defaults = dict(pooler_type="cls", temp=0.05, do_mlm=False, mlp_only_train=False,
                    mlm_weight=0.15, positive_weight=0.0, negative_weight=0.5,
                    hard_neg_weight=1.0)
    defaults.update(getattr(config, "senticse_args", {}))
    if supplied is not None:
        defaults.update(vars(supplied))
    return SimpleNamespace(**defaults)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], \
            "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden2 = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden2) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden2 = hidden_states[-1]
            pooled_result = ((last_hidden2 + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    # Store hyperparameters, never a private filesystem path, with the checkpoint.
    cls.config.senticse_args = {key: value for key, value in vars(cls.model_args).items()
                               if key not in {"sentiment_vocab_file", "model_name_or_path", "cache_dir",
                                              "config_name", "tokenizer_name", "use_auth_token"}}
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()


# -----------------------------
# helpers for 6-tuple loss
# -----------------------------
def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(float(x), eps))

def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def _cos_logits(anchor: torch.Tensor, cands: torch.Tensor, temp: float) -> torch.Tensor:
    """
    anchor: (B, H)
    cands : (N, H)
    return: (B, N)
    """
    a = _normalize(anchor)
    c = _normalize(cands)
    return torch.matmul(a, c.t()) / temp

def _all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """
    All-gather that keeps gradients for local rank (like SimCSE trick):
    we gather tensors, then replace local slice with original x.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    x_list = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(x_list, x.contiguous())
    x_list[dist.get_rank()] = x
    return torch.cat(x_list, dim=0)

def _block_bias(num_total: int, blocks: list) -> torch.Tensor:
    """
    Create bias vector length num_total for candidate concatenation blocks.
    blocks: [(start, end, logw), ...]
    """
    b = torch.zeros((num_total,), device="cpu")
    for s, e, v in blocks:
        b[s:e] = v
    return b


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    positive_score=None,
    negative_score=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    mlm_outputs = None

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))

    # If using "cls", apply MLP
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # always define loss_fct
    loss_fct = nn.CrossEntropyLoss()

    # =========================================================
    # (A) num_sent == 6: fixed order (pos,pos,neu,neg,neg,neu)
    # with in-batch negatives + α,β,δ weighting (paper style)
    # =========================================================
    if num_sent == 6:
        # indices:
        # 0 pos0, 1 pos1, 2 neu0, 3 neg0, 4 neg1, 5 neu1
        Z = pooler_output  # (B,6,H)

        p  = Z[:, 0, :]
        p_ = Z[:, 1, :]
        c  = Z[:, 2, :]
        c_ = Z[:, 5, :]
        n  = Z[:, 3, :]
        n_ = Z[:, 4, :]

        # gather across GPUs (DDP) so negatives include other processes
        if cls.training and dist.is_available() and dist.is_initialized():
            p_all  = _all_gather_with_grad(p)
            c_all  = _all_gather_with_grad(c)
            n_all  = _all_gather_with_grad(n)
            ppos_all = _all_gather_with_grad(p_)  # for positives block
            cpos_all = _all_gather_with_grad(c_)
            npos_all = _all_gather_with_grad(n_)
        else:
            p_all, c_all, n_all = p, c, n
            ppos_all, cpos_all, npos_all = p_, c_, n_

        # weights (paper α,β,δ)
        pos_neu_weight = getattr(cls.model_args, "pos_neu_weight", 1.0)  # pos-neu
        pos_neg_weight = getattr(cls.model_args, "pos_neg_weight", 2.0)  # pos-neg
        neu_neg_weight = getattr(cls.model_args, "neu_neg_weight", 1.0)  # neu-neg

        log_a = _safe_log(pos_neu_weight)
        log_b = _safe_log(pos_neg_weight)
        log_d = _safe_log(neu_neg_weight)

        temp = cls.model_args.temp

        # global size
        Btot = p_all.size(0)

        # targets for each local sample correspond to same index in gathered positives.
        # In DDP, local batch indices occupy a contiguous block per rank in the gathered tensor.
        # We compute offset = rank * local_batch_size, assuming each rank has same local batch size.
        # If your last batch differs per rank, you should drop_last=True in dataloader.
        if cls.training and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            # local batch size = current batch_size
            offset = rank * batch_size
        else:
            offset = 0

        # -------------------------
        # L_pos: anchor p_i  -> positive p_i+
        # denom: p_i+ + α*C_j + β*N_j
        # candidates: [p+_all, c_all, n_all] : (Btot + Btot + Btot) = 3Btot
        # correct index in first block: offset + i
        # -------------------------
        cand_pos = torch.cat([ppos_all, c_all, n_all], dim=0)  # (3Btot,H)
        logits_pos = _cos_logits(p, cand_pos, temp)  # (B,3Btot)

        # add log-weights to denom blocks (C and N blocks)
        # block ranges: [0:Btot)=p+, [Btot:2Btot)=C, [2Btot:3Btot)=N
        bias_pos = torch.zeros((1, logits_pos.size(1)), device=logits_pos.device)
        bias_pos[:, Btot:2*Btot] += log_a
        bias_pos[:, 2*Btot:3*Btot] += log_b
        logits_pos = logits_pos + bias_pos

        target_pos = (torch.arange(batch_size, device=logits_pos.device) + offset).long()

        # -------------------------
        # L_com: anchor c_i -> positive c_i+
        # denom: c_i+ + α*P_j + δ*N_j
        # candidates: [c+_all, p_all, n_all]
        # -------------------------
        cand_com = torch.cat([cpos_all, p_all, n_all], dim=0)
        logits_com = _cos_logits(c, cand_com, temp)

        bias_com = torch.zeros((1, logits_com.size(1)), device=logits_com.device)
        bias_com[:, Btot:2*Btot] += log_a  # P block
        bias_com[:, 2*Btot:3*Btot] += log_d  # N block
        logits_com = logits_com + bias_com

        target_com = (torch.arange(batch_size, device=logits_com.device) + offset).long()

        # -------------------------
        # L_neg: anchor n_i -> positive n_i+
        # denom: n_i+ + β*P_j + δ*C_j
        # candidates: [n+_all, p_all, c_all]
        # -------------------------
        cand_neg = torch.cat([npos_all, p_all, c_all], dim=0)
        logits_neg = _cos_logits(n, cand_neg, temp)

        bias_neg = torch.zeros((1, logits_neg.size(1)), device=logits_neg.device)
        bias_neg[:, Btot:2*Btot] += log_b  # P block
        bias_neg[:, 2*Btot:3*Btot] += log_d  # C block
        logits_neg = logits_neg + bias_neg

        target_neg = (torch.arange(batch_size, device=logits_neg.device) + offset).long()

        # losses
        loss_pos = loss_fct(logits_pos, target_pos)
        loss_com = loss_fct(logits_com, target_com)
        loss_neg = loss_fct(logits_neg, target_neg)

        # optional outer weights (if you want different importance among the 3 losses)
        w_pos = getattr(cls.model_args, "loss_w_pos", 1.0)
        w_com = getattr(cls.model_args, "loss_w_neu", 1.0)
        w_neg = getattr(cls.model_args, "loss_w_neg", 1.0)
        denom = (w_pos + w_com + w_neg) if (w_pos + w_com + w_neg) > 0 else 1.0
        loss = (w_pos * loss_pos + w_com * loss_com + w_neg * loss_neg) / denom

        # logits output for compatibility (minimal): return diag scores (B,3)
        # (downstreamがlogits shapeに依存しない前提。依存しているならここは要調整)
        diag_pos = logits_pos[:, :Btot][torch.arange(batch_size, device=logits_pos.device), target_pos]
        diag_com = logits_com[:, :Btot][torch.arange(batch_size, device=logits_com.device), target_com]
        diag_neg = logits_neg[:, :Btot][torch.arange(batch_size, device=logits_neg.device), target_neg]
        cos_sim = torch.stack([diag_pos, diag_com, diag_neg], dim=1)  # (B,3)

    # =========================================================
    # (B) original behavior: num_sent != 6 (2/3/4)
    # =========================================================
    else:
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        z3 = pooler_output[:, 2] if num_sent >= 3 else None
        z4 = pooler_output[:, 3] if num_sent == 4 else None

        if dist.is_initialized() and cls.training:
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

            if num_sent >= 3:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            if num_sent == 4:
                z4_list = [torch.zeros_like(z4) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z4_list, tensor=z4.contiguous())
                z4_list[dist.get_rank()] = z4
                z4 = torch.cat(z4_list, 0)

        cos_sim_pos = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        cos_sim_pos = torch.diagonal(cos_sim_pos).unsqueeze(1)

        if num_sent == 4:
            cos_sim_neg = cls.sim(z4.unsqueeze(1), z3.unsqueeze(0))
            cos_sim_neg = torch.diagonal(cos_sim_neg).unsqueeze(1)

        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim_pos = torch.cat([cos_sim_pos, z1_z3_cos], 1)

            if num_sent == 4:
                z3_z1_cos = cls.sim(z4.unsqueeze(1), z2.unsqueeze(0))
                cos_sim_neg = torch.cat([cos_sim_neg, z3_z1_cos], 1)

        labels_ = torch.zeros(cos_sim_pos.size(0)).long().to(cls.device)

        if num_sent == 4:
            weights = torch.tensor(
                [[cls.model_args.positive_weight]
                 + [cls.model_args.negative_weight] * i
                 + [cls.model_args.hard_neg_weight]
                 + [cls.model_args.negative_weight] * (z1_z3_cos.size(-1) - i - 1)
                 for i in range(z1_z3_cos.size(-1))]
            ).to(cls.device)

            cos_sim_pos = cos_sim_pos + weights
            cos_sim_neg = cos_sim_neg + weights

            cos_sim = (cos_sim_pos + cos_sim_neg) / 2
            loss1 = loss_fct(cos_sim_pos, labels_)
            loss2 = loss_fct(cos_sim_neg, labels_)
            loss = loss1 + loss2
        else:
            cos_sim = cos_sim_pos
            loss = loss_fct(cos_sim_pos, labels_)

    # =========================
    # MLM loss (original)
    # =========================
    do_mlm = (
    (mlm_outputs is not None)
    and (mlm_labels is not None)
    and (getattr(cls, "lm_head", None) is not None)
    and (getattr(cls.model_args, "mlm_weight", 0) > 0)
    )
    
    if do_mlm:
        if positive_score is None or negative_score is None:
            raise ValueError("Sentiment MLM training requires sentiment_vocab_file for this tokenizer.")
        mlm_labels = mlm_labels.reshape(-1)
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        predicts = prediction_scores.reshape(-1, cls.config.vocab_size)
        predicted_ids = predicts.argmax(dim=1)
        # Keep lookup and masking on the model device. Indexing NumPy with CUDA
        # tensors fails and a Python loop would synchronize once per token.
        positive = torch.as_tensor(positive_score, device=predicts.device)
        negative = torch.as_tensor(negative_score, device=predicts.device)
        valid = mlm_labels != -100
        target_ids = mlm_labels.clamp_min(0)
        target_positive = (positive[target_ids] > 0) & (negative[target_ids] == 0)
        target_negative = (negative[target_ids] > 0) & (positive[target_ids] == 0)
        predicted_positive = (positive[predicted_ids] > 0) & (negative[predicted_ids] == 0)
        predicted_negative = (negative[predicted_ids] > 0) & (positive[predicted_ids] == 0)
        opposite = valid & ((target_positive & predicted_negative) | (target_negative & predicted_positive))
        sentiword_loss = predicts.sum() * 0
        if opposite.any():
            sentiword_loss = loss_fct(predicts[opposite], mlm_labels[opposite])
        loss = loss + cls.model_args.mlm_weight * sentiword_loss

    # debug (you can remove later)
    # print('loss:', loss)

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def get_output_embeddings(self):
        return self.lm_head.decoder if hasattr(self, "lm_head") else None

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = _model_arguments(config, model_kargs.get("model_args"))
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)
        self.positive_score = None
        self.negative_score = None
        vocab_path = getattr(self.model_args, "sentiment_vocab_file", None)
        if self.model_args.do_mlm and vocab_path:
            sentimlm_info = load_sentiment_vocab(vocab_path, config.vocab_size)
            self.positive_score = sentimlm_info[1]
            self.negative_score = sentimlm_info[2]

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        positive_score=None,
        negative_score=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                positive_score=self.positive_score,
                negative_score=self.negative_score,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def get_output_embeddings(self):
        return self.lm_head.decoder if hasattr(self, "lm_head") else None

    def set_output_embeddings(self, embeddings):
        self.lm_head.decoder = embeddings

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = _model_arguments(config, model_kargs.get("model_args"))
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)
        self.positive_score = None
        self.negative_score = None
        vocab_path = getattr(self.model_args, "sentiment_vocab_file", None)
        if self.model_args.do_mlm and vocab_path:
            sentimlm_info = load_sentiment_vocab(vocab_path, config.vocab_size)
            self.positive_score = sentimlm_info[1]
            self.negative_score = sentimlm_info[2]

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        positive_score=None,
        negative_score=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                positive_score=self.positive_score,
                negative_score=self.negative_score,
            )


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def electra_sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    encoder_outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    sequence_output = encoder_outputs[0]
    roberta_pooler = RobertaPooler(cls.config).to(cls.device)
    pooled_output = roberta_pooler(sequence_output) if roberta_pooler is not None else None
    outputs2 = BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )

    pooler_output = cls.pooler(attention_mask, outputs2)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs2[0], pooler_output) + outputs2[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs2.last_hidden_state,
        hidden_states=outputs2.hidden_states,
    )


class ElectraModelForCL(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = _model_arguments(config, model_kargs.get("model_args"))
        self.electra = ElectraModel(config)
        self.config = config
        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return electra_sentemb_forward(self, self.electra,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.electra,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
