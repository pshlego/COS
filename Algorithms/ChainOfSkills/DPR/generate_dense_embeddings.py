#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple
from tqdm import tqdm
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components, init_hf_cos_biencoder
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
from DPR.run_chain_of_skills_hotpot import set_up_encoder
logger = logging.getLogger()
setup_logger(logger)

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True, expert_id=None
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    for j, batch_start in tqdm(enumerate(range(0, n, bsz)), total=n // bsz):
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_token_tensors = [
            tensorizer.text_to_tensor(
                ctx[1].text, title=ctx[1].title if insert_title else None
            )
            for ctx in batch
        ]

        ctx_ids_batch = move_to_device(
            torch.stack(batch_token_tensors, dim=0), cfg.device
        )
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(
            tensorizer.get_attn_mask(ctx_ids_batch), cfg.device
        )
        with torch.no_grad():
            if expert_id is None:
                outputs = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
            else:
                outputs = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask, expert_id=expert_id)
            if cfg.mean_pool:
                out = mean_pooling(outputs[0], ctx_attn_mask)
            else:
                out = outputs[1]
        out = out.cpu()

        ctx_ids = [r[0] for r in batch]
        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        # TODO: refactor to avoid 'if'
        if extra_info:
            results.extend(
                [
                    (ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i])
                    for i in range(out.size(0))
                ]
            )
        else:
            results.extend(
                [(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))]
            )
        # table chunk마다 key가 있고, 해당 key에 대응하여 table chunk의 embedding vector를 저장한다. title, 과 row by row로 serialize하여 embedding vector를 생성하고, 저장한다.
        if total % 10 == 0:
            logger.info("Encoded passages %d", total)
    return results


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    cfg = setup_cfg_gpu(cfg)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)
    # cfg.encoder.encoder_model_type = 'hf_cos'
    # sequence_length = 512
    # cfg.encoder.sequence_length = sequence_length
    if cfg.encoder.pretrained_file is not None:
        print ('setting pretrained file to None', cfg.encoder.pretrained_file)
        cfg.encoder.pretrained_file = None
    if cfg.model_file:
        print ('Since we are loading from a model file, setting pretrained model cfg to bert', cfg.encoder.pretrained_model_cfg)
        cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )
    print (cfg.encoder.encoder_model_type)
    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("ctx_model.")
    }
    if 'encoder.embeddings.position_ids' in ctx_state:
        if 'encoder.embeddings.position_ids' not in model_to_load.state_dict():
            print ('deleting position ids', ctx_state['encoder.embeddings.position_ids'].shape)
            del ctx_state['encoder.embeddings.position_ids']
    else:
        if 'encoder.embeddings.position_ids' in model_to_load.state_dict():
            ctx_state['encoder.embeddings.position_ids'] = model_to_load.state_dict()['encoder.embeddings.position_ids']
    model_to_load.load_state_dict(ctx_state)

    logger.info("reading data source: %s", cfg.ctx_src)

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]

    gpu_id = cfg.gpu_id
    if gpu_id == -1:
        logger.info('Using DDP mode')
        gpu_passages = shard_passages
    else:
        per_gpu_size = math.ceil(len(shard_passages) / cfg.num_gpus)
        gpu_start = per_gpu_size * gpu_id
        gpu_end = gpu_start + per_gpu_size
        logger.info(
            "Producing encodings for passages range: %d to %d (gpu id %d)",
            gpu_start+start_idx,
            gpu_end+start_idx,
            gpu_id,
        )
        gpu_passages = shard_passages[gpu_start:gpu_end]

    expert_id = None
    if cfg.encoder.use_moe:
        # TODO(chenghao): Fix this.
        if cfg.target_expert != -1:
            expert_id = int(cfg.target_expert)
            logger.info("Setting expert_id=%s", expert_id)
        else:
            logger.info("Default mode, Setting expert_id=1")
            expert_id = 1
    if cfg.mean_pool:
        logger.info("Using mean pooling for sentence embeddings")

    data = gen_ctx_vectors(cfg, gpu_passages, encoder, tensorizer, True, expert_id=expert_id)
    if gpu_id == -1:
        file = cfg.out_file + "_" + str(cfg.shard_id)
    else:
        file = cfg.out_file + "_shard" + str(cfg.shard_id) + "_gpu" + str(gpu_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    main()
