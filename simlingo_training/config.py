from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time

from hydra.core.config_store import ConfigStore

@dataclass
class VLMEncoderConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    embed_dim: int = 512
    freeze: bool = False

    _target_: str = "simlingo_training.models.encoder.vlm.VLMEncoderModel"


@dataclass
class LanguageModelConfig:
    variant: str = 'OpenGVLab/InternVL2-1B'
    lora: bool = True
    lora_alpha: int = 64
    lora_r: int = 32
    lora_dropout: float = 0.1

    _target_: str = "simlingo_training.models.language_model.llm.LLM"


@dataclass
class VlaadConfig:
    """VLAAD collision/anomaly signal injection (ported from carla_garage TF++).

    - mode == 'off':        no VLAAD signal (default; identical to upstream SimLingo).
    - mode == 'logit':      feed the 1-d frozen collision logit as one extra <VLAAD> token.
    - mode == 'projected':  feed the 768-d frozen projected feature as one extra <VLAAD> token.

    ``checkpoint_path`` points at the converted VLAAD bundle
    (carla_garage/team_code/vlaad/convert_checkpoint.py); only ``anomaly_head_state_dict``
    is loaded into the frozen SupervisedAnomalyDetector.

    ``trainable_scope`` (frozen VLAAD head is NEVER trained; vlaad_encoder is ALWAYS trained
    when the numeric slot is on — it is randomly initialized):
      - 'full':      upstream regime — LoRA + adaptors + heads + vision + vlaad_encoder
                     (base LLM frozen by PEFT). Matches how SimLingo itself fine-tunes.
      - 'heads':     vlaad_encoder + driving adaptors/heads (route+speed waypoints) + wp_encoder
                     (target-point input adaptor); LLM & vision frozen.
      - 'heads_llm': 'heads' + the LLM's LoRA adapters (vision frozen).
      - 'llm':       vlaad_encoder + the LLM's LoRA adapters only (heads, wp_encoder, vision frozen).
    """
    mode: str = "off"  # off | logit | projected
    # How the signal enters the LLM:
    #   'numeric' — the frozen head's (logit|projected) feature fills a <VLAAD> token slot via the
    #               interleaver (learned vlaad_encoder; full signal, not human-readable).
    #   'text'    — the frozen head's collision logit is verbalized (e.g. "Collision risk: high.")
    #               and appended to the text prompt. No <VLAAD> token / vlaad_encoder / splice.
    #               Runs the tiny frozen detector in the dataloader. 'mode' picks the numeric
    #               feature for the 'numeric' path; the 'text' path always verbalizes the logit.
    injection: str = "numeric"  # numeric | text
    checkpoint_path: Optional[str] = None
    input_dim: int = 768
    hidden_dim: int = 256
    encoder_hidden_size: int = 256
    trainable_scope: str = "full"  # full | heads | heads_llm | llm


@dataclass
class DrivingModelConfig:
    vision_model: Any
    language_model: Any

    lr: float = 5e-2

    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    pct_start: float = 0.05
    speed_wps_mode: str = '2d'
    predict_route_as_wps: bool = True

    vlaad: VlaadConfig = field(default_factory=VlaadConfig)

    _target_: str = "simlingo_training.models.driving.DrivingModel"


@dataclass
class DatasetBaseConfig:
    data_path: str = "/home/katrinrenz/coding/wayve_carla/database/expertv3_2*"
    bucket_path: str = "data/buckets"

    cut_bottom_quarter: bool = False
    use_1d_wps: bool = False

    use_commentary: bool = False
    use_qa: bool = False
    qa_augmentation: bool = True
    commentary_augmentation: bool = True
    use_old_towns: bool = False
    use_only_old_towns: bool = False
    use_town13: bool = False

    skip_first_n_frames: int = 10
    pred_len: int = 11 # including the current time step
    hist_len: int = 1 # including the current time step
    hist_len_commentary: int = 5 # including the current time step
    
    img_augmentation: bool = True
    img_augmentation_prob: float = 0.5
    img_shift_augmentation: bool = True
    img_shift_augmentation_prob: float = 0.5
    
    use_safety_flag: bool = False
    
    num_route_points: int = 20

    route_as: str = 'target_point_command' # target_point_command, target_point, command
    use_lmdrive_commands: bool = True

    # Route discovery glob relative to data_path. Default is SimLingo's native 4-level layout;
    # for a TransFuser++ dataset (<root>/<Scenario>/Town*) set route_glob: '*/Town*'.
    route_glob: str = 'data/simlingo/*/*/*/Town*'
    # Town-name-regex holdout (carla_garage style). When set (e.g. 'Town13'), val = routes whose
    # dir name matches it, train = the rest. Overrides the routes_training/validation substring split.
    holdout_town: Optional[str] = None
    # VLAAD: when != 'off', the dataset loads embeddings/NNNN.pt for the current frame.
    # Kept in sync with model.vlaad.* by train.py.
    vlaad_mode: str = "off"
    vlaad_injection: str = "numeric"  # numeric | text (see VlaadConfig)
    # Used only by the 'text' injection path to run the frozen detector in the dataloader:
    vlaad_checkpoint_path: Optional[str] = None
    vlaad_input_dim: int = 768
    vlaad_hidden_dim: int = 256

@dataclass
class DrivingDatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_driving.Data_Driving"
    
@dataclass
class DreamerDatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_dreamer.Data_Dreamer"
    
@dataclass
class QADatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_qa_comm.Data_Eval"
    
@dataclass
class InstEvalDatasetConfig:
    # base: DatasetBaseConfig = field(default_factory=DatasetBaseConfig)
    _target_: str = "simlingo_training.dataloader.dataset_eval_dreamer.Eval_Dreamer"

@dataclass
class DrivingDataModuleConfig:
    
    base_dataset: DatasetBaseConfig
    
    driving_dataset:Optional[ DrivingDatasetConfig] = field(default_factory=DrivingDatasetConfig)
    dreamer_dataset: Optional[DreamerDatasetConfig] = field(default_factory=DreamerDatasetConfig)
    qa_dataset: Optional[QADatasetConfig] = field(default_factory=QADatasetConfig)
    insteval_dataset: Optional[InstEvalDatasetConfig] = field(default_factory=InstEvalDatasetConfig)

    batch_size: int = 16
    num_workers: int = 10
    
    train_partitions: Optional[Dict[str, float]] = None
    train_partitions_dreamer: Optional[Dict[str, float]] = None
    use_global_img: bool = False
    
    _target_: str = "simlingo_training.dataloader.datamodule.DataModule"


@dataclass
class TrainConfig:
    model: DrivingModelConfig
    data_module: Any

    seed: int = 42
    gpus: int = 8

    resume: bool = False
    resume_path: Optional[str] = None

    debug: bool = False
    overfit: int = 0
    fast_dev_run: int = 0  # >0 runs N train+val batches with no logging/checkpointing (smoke test)
    fp16_loss_scale: float = 32.0 # 0.0 means dynamic loss scaling, only used with deepspeed

    enable_wandb: bool = True
    wandb_project: Optional[str] = "simlingo"
    wandb_group: Optional[str] = None          # groups related runs in the wandb UI (e.g. an ablation)
    wandb_tags: Optional[List[str]] = None      # filterable tags; CLI: 'wandb_tags=[vlaad,full,logit]'
    if debug:
        wandb_name: Optional[str] = f"debug"
        gpus: int = 1
    else:
        # wandb_name: Optional[str] = f"debug"
        name: Optional[str] = 'test'
        wandb_name: Optional[str] = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    
    # max_steps: int = 100_000
    max_epochs: int = 20
    precision: str = "16-mixed"
    strategy: str = "deepspeed_stage_2" # deepspeed_stage_2 ddp
    # val_check_interval: int = 5000
    val_every_n_epochs: int = 1

    checkpoint: Optional[str] = None


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="train_base", node=TrainConfig)
    cs.store(group="data_module", name="driving", node=DrivingDataModuleConfig)
    cs.store(group="data_module/base_dataset", name="dataset", node=DatasetBaseConfig)
    cs.store(group="model", name="driving", node=DrivingModelConfig)
    cs.store(group="model/vision_model", name="vlm", node=VLMEncoderConfig)
    cs.store(group="model/language_model", name="llm", node=LanguageModelConfig)


register_configs()
