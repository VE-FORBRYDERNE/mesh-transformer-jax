import os
os.environ["WANDB_START_METHOD"] = "thread"  # fixes https://github.com/wandb/wandb/issues/3045

WANDB_KEY = ""

import argparse
import json
import time
import sys
import copy

import numpy as np
from tqdm.auto import tqdm

from mesh_transformer.build_model import build_model
from lm_eval import evaluator, tasks
from tasks.eval_harness import EvalHarnessAdaptor
from tfrecord_loader import TFRecordNewInputs
import multiprocessing

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on.")
    parser.add_argument("--tpu_region", type=str, help="Region of TPU to train on.")
    parser.add_argument("--preemptible", action="store_true")

    parser.add_argument("--config", type=str, default=None, help="Config file location")

    parser.add_argument("--new", action="store_true", help="If set, deletes previous checkpoint, if it exists, and "
                                                           "starts a new training run")

    parser.add_argument("--version", type=int, default=1, help="Choose which model version to use")

    parser.add_argument("--continuing", action="store_true", help="If set, continues training from the latest checkpoint instead of starting a new finetuning run.")
    parser.add_argument("--save_opt_state", action="store_true", help="If set, saves checkpoints as full checkpoints (with optimizer states) instead of slim checkpoints.")
    parser.add_argument("--use_wandb", action="store_true", help="If set, logs training stats to wandb.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # huggingface tokenizers gets very angry if you fork
    multiprocessing.set_start_method("spawn")

    args = parse_args()
    params = json.load(open(args.config))

    wandb_project = params.get("wandb_project", "mesh-transformer-jax")
    wandb_entity = params.get("wandb_entity", "20b-united")
    wandb_name = params.get("name", "tpu-training")
    wandb_config = copy.deepcopy(params)


    if args.use_wandb:
        import wandb
        print('—— wandb_config', wandb_config)
        print('— Using wandb project:', wandb_project, 'entity:', wandb_entity, 'name:', wandb_name)
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name, config=params)

    if args.new:
        print(f"Starting experiment {params['name']} from scratch! "
              f"all data in gs://{params['bucket']}/{params['model_dir']}/ will be deleted")
        input("Hit enter to continue")

    tpu_name = args.tpu
    region = args.tpu_region
    preemptible = args.preemptible
    clean_start = args.new

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    tpu_size = params["tpu_size"]
    cores_per_replica = params["cores_per_replica"]

    if cores_per_replica % 8 != 0:
        raise NotImplementedError("cores_per_replica must be divisible by 8")

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    val_batches = params["val_batches"]
    val_every = params["val_every"]
    ckpt_every = params["ckpt_every"]
    keep_every = params["keep_every"]
    eval_tasks = params["eval_harness_tasks"]
    total_steps = params["total_steps"]

    noise_scale_alpha = params.get("noise_scale_alpha", 0.01)

    pe = params["pe"]

    assert pe in [
        "neox_rotary",
        "fixed",
        "rotary",
        "t5"]

    print('—— BUILDING MODEL', tpu_name, args.version)

    t = build_model(params, tpu_name, region, preemptible, version=args.version)

    print('—— BUILT MODEL', t)

    # break

    #
    try:
        t.save(0, bucket, model_dir, init=True, overwrite=clean_start, save_opt_state=args.save_opt_state)
        step = 0
        opt_step = 0
        train_load_restore = None
    except Exception as e:
        print(f"Save failed with error {e}, trying to load instead...", e)
        print("Loading from:", bucket, model_dir)

        step, aux, opt_step = t.load(bucket, model_dir, finetuning=not args.continuing)
        train_load_restore = aux.get("train_loader", None)

        if train_load_restore is None:
            print("Failed to restore train loader state")

    if not args.continuing:
        train_load_restore = None

    print('—— LOADED FROM CHECKPOINT')

    assert gradient_accumulation_steps % (per_replica_batch * tpu_size // cores_per_replica) == 0
    train_dataset = TFRecordNewInputs(f"data/{params['train_set']}",
                                      batch_size=(
                                          gradient_accumulation_steps // (per_replica_batch * tpu_size // cores_per_replica),
                                          per_replica_batch * tpu_size // cores_per_replica),
                                      sample_size=params['seq'],
                                      restore_state=train_load_restore)

    global_val_batch = int(1 * params.get("val_batch_multiplier", 1))


    print('— restored_idx:', train_dataset.restored_idx)


    val_sets = {}

    for k, v in params['val_set'].items():
        val_sets[k] = TFRecordNewInputs(f"data/{v}",
                                        batch_size=(global_val_batch,),
                                        sample_size=seq)

    # use dynamic seq length unless pe is fixed
    adaptor = EvalHarnessAdaptor(t,
                                 seq,
                                 global_val_batch,
                                 shrink=pe != "fixed",
                                 min_seq=1024 if args.version == 2 else None)  # work around suboptimal pjit layout

    # nosta_adaptor = EvalHarnessAdaptor(t, seq, global_val_batch * 4, shrink=pe != "fixed")

    samples = train_dataset.get_samples()
    obs = samples[:, :, :-1]
    target = samples[:, :, 1:]
    print(f'—— STARTING TRAINING (samples shape: {samples.shape}; obs shape: {obs.shape}; target shape: {target.shape})', flush=True)

    start = time.time()
    t.train(samples)
    step += 1
    opt_step += 1
    print(f"Train fn compiled in {time.time() - start:.06}s")

    start = time.time()
    for val_set in val_sets.values():
        t.eval(val_set.get_samples())
    print(f"Eval fn compiled in {time.time() - start:.06}s")

    eval_task_dict = tasks.get_task_dict(eval_tasks)

    restored_idx = train_dataset.restored_idx if train_dataset.restored_idx > 1 else 1

    pbar = tqdm(
        initial=restored_idx,
        total=total_steps,
        desc="Training progress"
    )

    first_step = True

    G_noise_avg = None
    S_noise_avg = None

    iteration = 1

    sequences_per_step = gradient_accumulation_steps
    tokens_per_step = params['seq'] * sequences_per_step
    print("Sequences per step:", sequences_per_step)
    print("Tokens per step:", tokens_per_step)

    while True:
        iteration += 1

        start = time.time()
        loss, last_loss, grad_norm, grad_norm_micro = t.train(train_dataset.get_samples())
        opt_step += 1
        steps_per_sec = 1 / (time.time() - start)
        tokens_per_sec = tokens_per_step * steps_per_sec

        grad_norm = grad_norm / gradient_accumulation_steps

        stats = {
            'train/loss': loss,
            'train/last_loss': last_loss,
            'train/grad_norm': grad_norm,
            "train/steps_per_sec": steps_per_sec,
            "train/tokens_per_sec": tokens_per_sec,
            "train/learning_rate": float(t.scheduler(opt_step)),
        }

        gbsmall = grad_norm_micro ** 2
        gbbig = grad_norm ** 2
        G_noise = (gradient_accumulation_steps * gbbig - gbsmall) / (
            gradient_accumulation_steps - 1
        )
        S_noise = (gbsmall - gbbig) / (1 - 1 / gradient_accumulation_steps)
        use_step_in_noise_avgs = grad_norm < 2
        if use_step_in_noise_avgs:
            if G_noise_avg is None:
                G_noise_avg = G_noise
            else:
                G_noise_avg = (1 - noise_scale_alpha) * G_noise_avg + noise_scale_alpha * G_noise
            if S_noise_avg is None:
                S_noise_avg = S_noise
            else:
                S_noise_avg = (1 - noise_scale_alpha) * S_noise_avg + noise_scale_alpha * S_noise
            B_simple = S_noise_avg / G_noise_avg
            stats.update(
                {
                    "noise/G_noise_avg": G_noise_avg,
                    "noise/S_noise_avg": S_noise_avg,
                    "noise/grad_noise_scale": B_simple,
                }
            )

        if (args.use_wandb): wandb.log(stats, step + 1)

        if not first_step and ((iteration % ckpt_every == 0 and step) or iteration == total_steps):
            print(f"Saving (step {step + 1})")
            t.save(step + 1, bucket, model_dir,
                   aux={"train_loader": train_dataset.get_state()},
                   init=False,
                   save_opt_state=args.save_opt_state,
                   delete_old=iteration % keep_every != 0)

            if iteration == total_steps:
                pbar.set_postfix({'loss': loss, 'last_loss': last_loss, 'grad_norm': grad_norm})
                pbar.update()
                print("training completed!")
                exit()

        first_step = False

        if iteration % 100 == 0:
            print(f"step {step + 1} done")

        if iteration % val_every == 0:
            for name, val_set in val_sets.items():
                val_loss = []
                for i, _ in tqdm(zip(val_set.sample_once(), range(val_batches)),
                                 desc=f"validation for step {step + 1}, set {name}",
                                 total=val_batches):
                    val_loss.append(t.eval(i))
                val_loss = np.array(val_loss).mean()
                print(f"validation loss for step {step + 1}, set {name}: {val_loss}")

                if (args.use_wandb): wandb.log({f'val/loss_{name}': float(val_loss)}, step + 1)

            results = evaluator.evaluate(adaptor, eval_task_dict, False, 0, None)

            flat_results = {}

            for task_name, task_res in results["results"].items():
                version = results["versions"][task_name]
                for metric_name, metric_res in task_res.items():
                    flat_results[f"{task_name}-v{version}/{metric_name}"] = float(metric_res)
                    # nosta_flat_results[f"{task_name}/{metric_name}"] = float(metric_res)

            dumped = json.dumps(results, indent=2)
            print(f"step {step + 1} val results: {dumped}")

            if (args.use_wandb): wandb.log(flat_results, step)
        step += 1

        pbar.set_postfix({'loss': loss, 'last_loss': last_loss, 'grad_norm': grad_norm})
        pbar.update()
