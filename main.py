import os
import torch
import config as cfg
from data import load_data
from model import SherlockModel
from train import train_model, compute_perplexity
from generate import generate_text, print_step_breakdown
from logger import setup_logger, save_run_summary


def main():

    # -- 0. Setup logging -----------------------------------------------
    logger, run_id = setup_logger()
    logger.info("Starting Sherlock next-word prediction training")

    # -- 1. Load and preprocess data ------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("=" * 55)
    input_vocab, w2i, _, out_w2i, out_i2w, train_seqs, test_seqs = load_data()

    # -- 2. Build model -------------------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 2: Building model")
    logger.info("=" * 55)
    model        = SherlockModel(
                       input_vocab_size=len(input_vocab),
                       output_vocab_size=len(out_w2i)
                   )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {total_params:,}")

    # -- 3. Train -------------------------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 3: Training")
    logger.info("=" * 55)
    history = train_model(model, train_seqs, test_seqs)

    # -- 4. Load best model and evaluate --------------------------------
    logger.info("=" * 55)
    logger.info("STEP 4: Evaluating best model")
    logger.info("=" * 55)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, weights_only=True))

    best_train_acc = max(history["train_acc"])
    best_val_acc   = max(history["val_acc"])
    best_val_loss  = min(history["val_loss"])
    perplexity     = compute_perplexity(best_val_loss)

    logger.info("FINAL METRICS")
    logger.info(f"Train Accuracy : {best_train_acc:.4f}  (target >0.80)  {'PASS' if best_train_acc > 0.80 else 'FAIL'}")
    logger.info(f"Val Accuracy   : {best_val_acc:.4f}  (target >0.75)  {'PASS' if best_val_acc > 0.75 else 'FAIL'}")
    logger.info(f"Perplexity     : {perplexity:.2f}     (target <250)   {'PASS' if perplexity < 250 else 'FAIL'}")

    # -- 5. Generate text examples --------------------------------------
    logger.info("=" * 55)
    logger.info("STEP 5: Generating text examples")
    logger.info("=" * 55)

    seed_phrases = [
        "I saw Holmes",
        "the mystery was",
        "Watson looked at the door",
        "it is a curious case",
        "sherlock holmes stepped into the room"
    ]

    os.makedirs("logs", exist_ok=True)
    output_log_path = f"logs/generated_{run_id}.txt"

    with open(output_log_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("GENERATED TEXT EXAMPLES\n")
        f.write("=" * 65 + "\n")

        for seed in seed_phrases:
            generated, _ = generate_text(
                seed, max_phrase_len=30,
                model=model, w2i=w2i, out_i2w=out_i2w, temperature=0.7
            )
            logger.info(f"SEED   : \"{seed}\"")
            logger.info(f"OUTPUT : {generated}")
            logger.info("-" * 65)

            f.write(f"\nSEED   : \"{seed}\"\n")
            f.write(f"OUTPUT : {generated}\n")
            f.write("-" * 65 + "\n")

        # -- 6. Step by step breakdown ----------------------------------
        f.write("\n" + "=" * 65 + "\n")
        f.write("STEP-BY-STEP BREAKDOWN\n")
        f.write("=" * 65 + "\n")

        breakdown_seed = "I saw Holmes"
        generated, steps = generate_text(
            breakdown_seed, max_phrase_len=15,
            model=model, w2i=w2i, out_i2w=out_i2w, temperature=0.7
        )

        logger.info("=" * 55)
        logger.info("STEP 6: Step-by-step breakdown")
        logger.info("=" * 55)
        logger.info(f"Seed     : \"{breakdown_seed}\"")
        logger.info(f"Generated: {generated}")
        print_step_breakdown(breakdown_seed, steps)

        f.write(f"Seed     : \"{breakdown_seed}\"\n")
        f.write(f"Generated: {generated}\n\n")

        for i, step in enumerate(steps):
            f.write(f"\nStep {i+1:02d} | Chosen: '{step['chosen_word']}'\n")
            f.write(f"  Context: '...{step['input_context']}'\n")
            f.write("  Top 5:\n")
            for rank, candidate in enumerate(step["top5"], 1):
                marker = " <- chosen" if candidate["word"] == step["chosen_word"] else ""
                f.write(f"    {rank}. '{candidate['word']:<15}' "
                        f"{candidate['probability']:.4f}{marker}\n")

    logger.info(f"Generated text saved to: {output_log_path}")

    # -- 7. Save run summary --------------------------------------------
    metrics = {
        "best_train_acc": best_train_acc,
        "best_val_acc"  : best_val_acc,
        "perplexity"    : perplexity,
    }

    config_dict = {
        "SEQ_LEN"         : cfg.SEQ_LEN,
        "INPUT_MIN_FREQ"  : cfg.INPUT_MIN_FREQ,
        "OUTPUT_MIN_FREQ" : cfg.OUTPUT_MIN_FREQ,
        "EMBED_DIM"       : cfg.EMBED_DIM,
        "LSTM_UNITS"      : cfg.LSTM_UNITS,
        "BATCH_SIZE"      : cfg.BATCH_SIZE,
        "LEARNING_RATE"   : cfg.LEARNING_RATE,
        "DROPOUT"         : cfg.DROPOUT,
        "PATIENCE"        : cfg.PATIENCE,
    }

    summary_path = save_run_summary(history, metrics, config_dict, run_id)
    logger.info(f"Run summary saved to: {summary_path}")
    logger.info("All done!")


if __name__ == "__main__":
    main()
