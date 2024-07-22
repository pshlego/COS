from ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from ColBERT.colbert import Trainer
import hydra
from omegaconf import DictConfig

@hydra.main(config_path = "conf", config_name = "training_colbert")
def main(cfg: DictConfig):
    experiment_name = cfg.experiment_name
    collection_root_dir_path = cfg.collection_root_dir_path
    with Run().context(RunConfig(nranks = 4, experiment = experiment_name)):
        config = ColBERTConfig(bsize=512, lr=1e-05, warmup=40, doc_maxlen=512, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=128, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=cfg.triples, queries=cfg.queries, collection=cfg.collection, config=config)

        checkpoint_path = trainer.train(checkpoint=cfg.colbert_checkpoint)

        print(f"Saved checkpoint to {checkpoint_path}...")
        
if __name__=='__main__':
    main()