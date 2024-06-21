import torch
from whisperplus.pipelines.text2speech import TextToSpeechPipeline
from whisperplus import SpeechToTextPipeline
from transformers import BitsAndBytesConfig, HqqConfig
from TTS.api import TTS

hqq_config = HqqConfig(
    nbits=4,
    group_size=64,
    quant_zero=False,
    quant_scale=False,
    axis=0,
    offload_meta=False,
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
stt_pipeline = SpeechToTextPipeline(
    model_id="distil-whisper/distil-large-v3",
    quant_config=hqq_config,
    flash_attention_2=False,
)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")