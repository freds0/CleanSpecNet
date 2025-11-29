import yaml
import argparse
from argparse import Namespace
import logging
import copy
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Import CleanSpecNet specific modules
from lightning_modules.cleanspecnet_module import CleanSpecNetLightningModule
from lightning_modules.data_module import CleanSpecNetDataModule

# Configure logging for console output
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_cleanspecnet")

def _safe_instantiate_callbacks(callbacks_config: dict):
    """
    Instantiate callbacks from a config dict in a safe, explicit manner.
    Supported: ModelCheckpoint, EarlyStopping.
    """
    callbacks = []
    for name, cb_cfg in (callbacks_config or {}).items():
        if not isinstance(cb_cfg, dict):
            logger.warning("Callback config for '%s' is not a dict; skipping.", name)
            continue

        cfg = copy.deepcopy(cb_cfg)
        target = cfg.pop("_target_", None)
        if target is None:
            logger.warning("Callback '%s' has no '_target_' field; skipping.", name)
            continue

        if "ModelCheckpoint" in target:
            logger.info("Instantiating ModelCheckpoint for callback '%s'.", name)
            callbacks.append(ModelCheckpoint(**cfg))
        elif "EarlyStopping" in target:
            logger.info("Instantiating EarlyStopping for callback '%s'.", name)
            callbacks.append(EarlyStopping(**cfg))
        else:
            logger.warning("Callback '%s' with target '%s' is not supported explicitly.", name, target)
            # Optional: Dynamic instantiation fallback could go here if needed
    return callbacks

def _create_single_logger(cfg: dict):
    """
    Helper to instantiate a single logger based on its _target_.
    """
    if not cfg:
        return None
    
    config_copy = copy.deepcopy(cfg)
    target = config_copy.pop("_target_", None)

    if not target:
        return None

    if "TensorBoardLogger" in target:
        logger.info("Instantiating TensorBoardLogger.")
        return TensorBoardLogger(**config_copy)
    
    elif "WandbLogger" in target:
        logger.info("Instantiating WandbLogger.")
        return WandbLogger(**config_copy)
    
    else:
        logger.warning(f"Logger target '{target}' not supported. Skipping.")
        return None

def _safe_instantiate_logger(logger_config: dict):
    """
    Instantiate logger(s) from config safely.
    Supports 'choice' logic: 'tensorboard', 'wandb', or 'both'.
    """
    if not logger_config:
        logger.info("No logger configuration provided; proceeding without logger.")
        return None

    choice = logger_config.get("choice", None)

    # Strategy: 'choice' logic
    if choice:
        choice = choice.lower()
        loggers_list = []

        if choice in ["tensorboard", "both"]:
            tb_conf = logger_config.get("tensorboard")
            if tb_conf:
                l = _create_single_logger(tb_conf)
                if l: loggers_list.append(l)
        
        if choice in ["wandb", "both"]:
            wb_conf = logger_config.get("wandb")
            if wb_conf:
                l = _create_single_logger(wb_conf)
                if l: loggers_list.append(l)

        if not loggers_list:
            logger.warning(f"Logger choice was '{choice}' but no valid configuration found.")
            return None
        
        return loggers_list[0] if len(loggers_list) == 1 else loggers_list

    # Fallback: Direct instantiation (legacy structure)
    if "_target_" in logger_config:
        return _create_single_logger(logger_config)

    return None

def train(config: dict):
    """
    Main training function using configuration dictionary.
    """
    # 1. Merge model and data parameters into a single namespace for the LightningModule
    hparams_dict = {**config.get('model', {}), **config.get('data', {})}
    hparams = Namespace(**hparams_dict)

    # 2. Instantiate DataModule
    logger.info("Instantiating DataModule.")
    data_module = CleanSpecNetDataModule(**config['data'])

    # 3. Instantiate LightningModule
    logger.info("Instantiating CleanSpecNetLightningModule.")
    model = CleanSpecNetLightningModule(hparams)

    # 4. Instantiate Callbacks
    callbacks = _safe_instantiate_callbacks(config.get('callbacks', {}))

    # 5. Instantiate Logger
    lightning_logger = _safe_instantiate_logger(config.get('logger', {}))

    # 6. Instantiate Trainer
    logger.info("Creating PyTorch Lightning Trainer.")
    trainer_kwargs = copy.deepcopy(config.get('trainer', {}))
    
    trainer = Trainer(
        logger=lightning_logger,
        callbacks=callbacks,
        **trainer_kwargs
    )

    # 7. Start training
    logger.info("Starting training...")
    trainer.fit(model, datamodule=data_module)
    logger.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CleanSpecNet using a YAML configuration.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)
