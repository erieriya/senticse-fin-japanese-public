import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    T5Tokenizer,
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling
from senticse.models import RobertaForCL, BertForCL, ElectraModelForCL
from senticse.trainers import CLTrainer
from senticse.data import ContrastiveCollator, load_sentiment_vocab

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.15,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    sentiment_vocab_file: Optional[str] = field(
        default=None, metadata={"help": "External .npy vocabulary for sentiment MLM; not needed without --do_mlm."}
    )

    # oridinal sentiCSE's arguments
    positive_weight: float = field(default=0.0, metadata={"help": "positive weight"})
    inbatch_weight: float = field(default=1.0, metadata={"help": "inbatch_negative weight"})
    negative_weight: float = field(default=0.5, metadata={"help": "neagtive weight"})
    hard_neg_weight: float = field(default=1.0, metadata={"help": "hard_negative weight"})
    diagonal: bool = field(default=False, metadata={"help": "diagonal setting for CSEloss"})
    ppnn: bool = field(default=True, metadata={"help": "if you want use ppnn not hardneg"})

    # proposed models' arguments
    pos_neu_weight: float = field(
        default=1.0,
        metadata={"help": "weight for pos-neu separation"}
    )
    pos_neg_weight: float = field(
        default=2.0,
        metadata={"help": "weight for pos-neg separation"}
    )
    neu_neg_weight: float = field(
        default=1.0,
        metadata={"help": "weight for neu-neg separation"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    embedding_plot_file: Optional[str] = field(default=None, metadata={"help": "External text,label CSV for fixed-sentence PCA snapshots during training."})
    embedding_plot_steps: int = field(default=250)
    embedding_plot_max_samples: int = field(default=300)
    embedding_plot_max_frames: int = field(default=30)

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    sentimlm_probability: float = field(
        default=0.1, 
        metadata={"help": "Ratio of tokens to mask for senti MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.rsplit(".", 1)[-1].lower()
            if extension not in {"csv", "tsv", "json", "jsonl"}:
                raise ValueError("--train_file must be CSV, TSV, JSON or JSONL with six sentence columns.")


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Local experiments should not upload metrics to installed tracking services.
    report_to: str = field(default="none")
    senteval_data_dir: Optional[str] = field(default=None)
    evaluation_strategy: Optional[str] = field(default=None, metadata={"help": "Legacy alias for --eval_strategy."})

    def __post_init__(self):
        if self.evaluation_strategy is not None:
            self.eval_strategy = self.evaluation_strategy
        super().__post_init__()

    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        #model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
        and not training_args.resume_from_checkpoint
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.process_index == 0 else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.world_size > 1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.process_index == 0:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    datasets = None
    if training_args.do_train:
        if data_args.train_file:
            extension = data_args.train_file.rsplit(".", 1)[-1].lower()
            loader = {"tsv": "csv", "jsonl": "json"}.get(extension, extension)
            kwargs = {"delimiter": "\t" if extension == "tsv" else ","} if loader == "csv" else {}
            datasets = load_dataset(loader, data_files={"train": data_args.train_file},
                                    cache_dir=model_args.cache_dir, **kwargs)
        elif data_args.dataset_name:
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name,
                                    cache_dir=model_args.cache_dir)
        else:
            raise ValueError("--do_train requires --train_file or --dataset_name.")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
    }

    # どこから tokenizer をロードするか（tokenizer_name があれば優先）
    tokenizer_name_or_path = model_args.tokenizer_name or model_args.model_name_or_path

    if tokenizer_name_or_path is None:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # ★ rinna/japanese-roberta-base 用の特別扱い
    if "japanese-roberta-base" in tokenizer_name_or_path:
        print("Use T5Tokenizer for rinna/japanese-roberta-base")
        tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        # README 推奨の lower_case 対応
        tokenizer.do_lower_case = True

        # 念のため mask_token が設定されていない場合に備えておく
        if tokenizer.mask_token is None:
            # rinna の RoBERTa は <mask> を使っているのでそれを指定
            tokenizer.mask_token = "<mask>"
    else:
        # それ以外は今まで通り AutoTokenizer に任せる
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)


    if model_args.model_name_or_path:
        if config.model_type == "roberta":
            print("Backbone Model is RoBERTa")
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        elif config.model_type == "bert":
            print("Backbone Model is BERT")
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path, **config_kwargs)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        elif config.model_type == "electra":
            if model_args.do_mlm:
                raise ValueError("Sentiment MLM is supported for BERT and RoBERTa only.")
            print("Backbone Model is Electra")
            model = ElectraModelForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=True if model_args.use_auth_token else None,
                model_args=model_args
                )
        else:
            raise ValueError(f"Unsupported backbone: {config.model_type}")

    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    # Prepare features
    column_names = datasets["train"].column_names if datasets is not None else []
    print("column_names:", column_names)

    # ★ 6文組として使う列（先頭6列を文列として固定で使う）
    if training_args.do_train and len(column_names) < 6:
        raise ValueError(f"Need at least 6 sentence columns, but got {len(column_names)} columns: {column_names}")

    sent_cols = column_names[:6]  # (pos,pos,neu,neg,neg,neu) の順で並んでいる前提


    # 5列目以降に何があってもここでは無視する（label 等は features に使わない)


    sentimlm_info = None
    if model_args.do_mlm:
        sentimlm_info = load_sentiment_vocab(model_args.sentiment_vocab_file, len(tokenizer))

    def get_senti_type(sent_features):
        
        senti_type_list = []
        senti_pos_list = []
        senti_neg_list = []
        for idx in range(len(sent_features['input_ids'])):
            senti_list = []
            pos_list = []
            neg_list = []
            for ids in sent_features['input_ids'][idx]:
                if sentimlm_info[0][ids] == 1:
                    senti_list.append(1)
                else:
                    senti_list.append(0)
                pos_list.append(sentimlm_info[1][ids])
                neg_list.append(sentimlm_info[2][ids])
            senti_type_list.append(senti_list)
            senti_pos_list.append(pos_list)
            senti_neg_list.append(neg_list)
            
        sent_features['senti_type'] = senti_type_list
        sent_features['positive_score'] = senti_pos_list
        sent_features['negative_score'] = senti_neg_list
        return sent_features

    def mask_tokens_map(sent_features):
        # Masking by our new method
        ignore_index=-100
        replace_prob=0.1
        orginal_prob=0.1
        mlm_probability: float = data_args.mlm_probability
        sentimask_prob: float = data_args.sentimlm_probability
        
        mask_token_index = tokenizer.mask_token_id
        special_tok_ids = tokenizer.all_special_ids
        vocab_size= tokenizer.vocab_size
    
        masked_inputs_list = []
        is_mlm_applied_list = []
        labels_list =[]
        for idx in range(len(sent_features['input_ids'])):
            inputs = torch.tensor(sent_features['input_ids'][idx]).clone()
            device = inputs.device
            labels = inputs.clone()

            probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
            senti_probability_matrix = torch.tensor(sent_features['senti_type'][idx]).clone() * sentimask_prob
            special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)

            for sp_id in special_tok_ids:
                special_tokens_mask = special_tokens_mask | (inputs == sp_id)

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            senti_probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            # use only senti_mask
            mlm_mask = torch.bernoulli(senti_probability_matrix).bool()
            labels[~mlm_mask] = ignore_index
            mask_prob = 1 - replace_prob - orginal_prob
            mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
            inputs[mask_token_mask] = mask_token_index

            if replace_prob != 0:
                rep_prob = replace_prob / (replace_prob + orginal_prob)
                replace_token_mask = torch.bernoulli(
                    torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
                random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
                inputs[replace_token_mask] = random_words[replace_token_mask]
            pass
            masked_inputs_list.append(inputs)
            is_mlm_applied_list.append(mlm_mask)
            labels_list.append(labels)
            
        sent_features['mlm_input_ids'] = masked_inputs_list
        sent_features['is_mlm_applied'] = is_mlm_applied_list
        sent_features['mlm_labels'] = labels_list
        return sent_features
      
    def prepare_features(examples):
        total = len(examples[sent_cols[0]])

        # None 対策：6列すべてに適用
        for cname in sent_cols:
            for i in range(total):
                if examples[cname][i] is None:
                    examples[cname][i] = " "

        # 6列分を縦に連結（col0の全行→col1の全行→...）
        sentences = []
        for cname in sent_cols:
            sentences += examples[cname]

        # tokenize
        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # senti info / mlm 用の追加
        if model_args.do_mlm:
            sent_features = get_senti_type(sent_features)
            sent_features = mask_tokens_map(sent_features)
            for key in ("senti_type", "positive_score", "negative_score", "is_mlm_applied"):
                sent_features.pop(key)

        # (total*6) のフラットを (total,6,...) に戻す
        features = {}
        for key in sent_features:
            features[key] = [
                [sent_features[key][i + total * k] for k in range(6)]
                for i in range(total)
            ]

        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    data_collator = ContrastiveCollator(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        model_path=model_args.model_name_or_path,
        senteval_data_dir=training_args.senteval_data_dir,
    )
    if data_args.embedding_plot_file:
        if not training_args.do_train:
            raise ValueError("--embedding_plot_file requires --do_train.")
        from senticse.visualization import EmbeddingPlotCallback
        trainer.add_callback(EmbeddingPlotCallback(
            tokenizer, data_args.embedding_plot_file,
            os.path.join(training_args.output_dir, "embeddings.html"),
            every_steps=data_args.embedding_plot_steps,
            max_samples=data_args.embedding_plot_max_samples,
            max_frames=data_args.embedding_plot_max_frames,
            max_length=data_args.max_seq_length,
            seed=training_args.seed,
            resume_checkpoint=training_args.resume_from_checkpoint,
        ))
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "a") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
