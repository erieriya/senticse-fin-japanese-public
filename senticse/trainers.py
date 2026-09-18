"""SentiCSE evaluation on top of the maintained Hugging Face training loop."""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer


class CLTrainer(Trainer):
    def __init__(self, model_path="model", senteval_data_dir=None, **kwargs):
        # Preserve callers of the old tokenizer= constructor.
        if "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        # SentEval has its own task loader, but Trainer validates eval_dataset at init.
        external_eval = senteval_data_dir is not None and kwargs.get("eval_dataset") is None
        if external_eval:
            kwargs["eval_dataset"] = []
        super().__init__(**kwargs)
        if external_eval:
            self.eval_dataset = None
        self.model_path = model_path
        self.senteval_data_dir = senteval_data_dir

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:
        
        if eval_dataset is not None or self.eval_dataset is not None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if not self.senteval_data_dir or not Path(self.senteval_data_dir).is_dir():
            raise ValueError("SentEval evaluation requires --senteval_data_dir pointing to external task data.")
        # Legacy SentEval is optional; normal training needs no evaluation data.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SentEval"))
        import senteval

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.processing_class.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': self.senteval_data_dir, 'usepytorch': True, 'kfold': 5, 'model_path':self.model_path}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        # params = {'task_path': self.senteval_data_dir, 'usepytorch': True, 'kfold': 10, 'model_path':self.model_path}
        # params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
        #                                 'tenacity': 5, 'epoch_size': 4}
        
        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness',
                    'mr','sst2', 'yelp2', 'imdb',
                    ]
        
        if eval_senteval_transfer or getattr(self.args, "eval_transfer", False):
            tasks = ['STSBenchmark', 'SICKRelatedness',
                     'imdb', 'yelp2',  'mr', 'sst2',
                     'IMDB',  'MR', 'SST2', 'SST5', 'YELP2', 
                    ]
            #tasks = ['STSBenchmark', 'SICKRelatedness',
            #         'imdb', 'yelp2', 'yelp5', 'mr', 'sst2',
            #         'IMDB',  'MR', 'SST2', 'SST5', 'YELP2', 
            #        ]
        
        self.model.eval()
        results = se.eval(tasks)
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        
        metrics = {"eval_stsb_spearman": results['STSBenchmark']['dev']['spearman'][0], 
                    "eval_sickr_spearman": results['SICKRelatedness']['dev']['spearman'][0], 
                    "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2,
                    
                    "eval_sst2_sts": results['sst2']['all']['spearman']['mean'],
                    "eval_imdb_sts": results['imdb']['all']['spearman']['mean'], #
                    "eval_yelp2_sts": results['yelp2']['all']['spearman']['mean'], #
                    "eval_mr_sts": results['mr']['all']['spearman']['mean'], #
                    
                    "sst_uniform_loss" : results['sst2']['all']['uniform_loss'],
                    "imdb_uniform_loss" : results['imdb']['all']['uniform_loss'],
                    "yelp_uniform_loss" : results['yelp2']['all']['uniform_loss'],
                    "mr_uniform_loss" : results['mr']['all']['uniform_loss'],
                    "sst_align_loss" : results['sst2']['all']['align_loss'],
                    "imdb_align_loss" : results['imdb']['all']['align_loss'],
                    "yelp_align_loss" : results['yelp2']['all']['align_loss'],
                    "mr_align_loss" : results['mr']['all']['align_loss'],
                    
                    "eval_imdb_mr_sts" : (results['imdb']['all']['spearman']['mean']+ results['mr']['all']['spearman']['mean']) / 2,
                    "eval_mr_imdb_sts" : (results['imdb']['all']['spearman']['mean']+ results['mr']['all']['spearman']['mean']) / 2,
                    "eval_imdb_sst_sts" : (results['imdb']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                    "eval_sst_imdb_sts" : (results['imdb']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean'])  / 2,
                    "eval_imdb_yelp_sts" : (results['imdb']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                    "eval_yelp_imdb_sts" : (results['imdb']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                    "eval_mr_sst_sts" : (results['mr']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                    "eval_sst_mr_sts" : (results['mr']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                    "eval_mr_yelp_sts" : (results['mr']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                    "eval_yelp_mr_sts" : (results['mr']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                    "eval_sst_yelp_sts" : (results['yelp2']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                    "eval_yelp_sst_sts" : (results['yelp2']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                   } 
        
        if eval_senteval_transfer or getattr(self.args, "eval_transfer", False):
            metrics = {"eval_stsb_spearman": results['STSBenchmark']['dev']['spearman'][0], 
                        "eval_sickr_spearman": results['SICKRelatedness']['dev']['spearman'][0], 
                        "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2,

                        "eval_sst2_sts": results['sst2']['all']['spearman']['mean'],
                        "eval_imdb_sts": results['imdb']['all']['spearman']['mean'], #
                        "eval_yelp2_sts": results['yelp2']['all']['spearman']['mean'], #
                        "eval_mr_sts": results['mr']['all']['spearman']['mean'], #
                        
                        "sst_uniform_loss" : results['sst2']['all']['uniform_loss'],
                        "imdb_uniform_loss" : results['imdb']['all']['uniform_loss'],
                        "yelp_uniform_loss" : results['yelp2']['all']['uniform_loss'],
                        "mr_uniform_loss" : results['mr']['all']['uniform_loss'],
                        "sst_align_loss" : results['sst2']['all']['align_loss'],
                        "imdb_align_loss" : results['imdb']['all']['align_loss'],
                        "yelp_align_loss" : results['yelp2']['all']['align_loss'],
                        "mr_align_loss" : results['mr']['all']['align_loss'],
                        
                        "eval_imdb_mr_sts" : (results['imdb']['all']['spearman']['mean']+ results['mr']['all']['spearman']['mean']) / 2,
                        "eval_mr_imdb_sts" : (results['imdb']['all']['spearman']['mean']+ results['mr']['all']['spearman']['mean']) / 2,
                        "eval_imdb_sst_sts" : (results['imdb']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                        "eval_sst_imdb_sts" : (results['imdb']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean'])  / 2,
                        "eval_imdb_yelp_sts" : (results['imdb']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                        "eval_yelp_imdb_sts" : (results['imdb']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                        "eval_mr_sst_sts" : (results['mr']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                        "eval_sst_mr_sts" : (results['mr']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                        "eval_mr_yelp_sts" : (results['mr']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                        "eval_yelp_mr_sts" : (results['mr']['all']['spearman']['mean']+ results['yelp2']['all']['spearman']['mean']) / 2,
                        "eval_sst_yelp_sts" : (results['yelp2']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,
                        "eval_yelp_sst_sts" : (results['yelp2']['all']['spearman']['mean']+ results['sst2']['all']['spearman']['mean']) / 2,

                        "eval_acc_sst2" : results['SST2']['devacc'], 
                        "eval_acc_imdb": results['IMDB']['acc'], #
                        "eval_acc_yelp2": results['YELP2']['acc'], #
                        "eval_acc_mr": results['MR']['acc'], #
                    } 
            
            avg_transfer = 0

        if metric_key_prefix != "eval":
            metrics = {key.replace("eval_", metric_key_prefix + "_", 1): value for key, value in metrics.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
