LlavaLlamaModel(
  (llm): LlamaForCausalLM(
    (model): LlamaModel(
      (embed_tokens): Embedding(32000, 2560, padding_idx=0)
      (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
          (self_attn): LlamaFlashAttention2(
            (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
            (k_proj): Linear(in_features=2560, out_features=2560, bias=False)
            (v_proj): Linear(in_features=2560, out_features=2560, bias=False)
            (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=2560, out_features=6912, bias=False)
            (up_proj): Linear(in_features=2560, out_features=6912, bias=False)
            (down_proj): Linear(in_features=6912, out_features=2560, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=2560, out_features=32000, bias=False)
  )
  (vision_tower): SiglipVisionTower(
    (vision_tower): SiglipVisionModel(
      (vision_model): SiglipVisionTransformer(
        (embeddings): SiglipVisionEmbeddings(
          (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
          (position_embedding): Embedding(729, 1152)
        )
        (encoder): SiglipEncoder(
          (layers): ModuleList(
            (0-26): 27 x SiglipEncoderLayer(
              (self_attn): SiglipAttention(
                (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
              )
              (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
              (mlp): SiglipMLP(
                (activation_fn): PytorchGELUTanh()
                (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                (fc2): Linear(in_features=4304, out_features=1152, bias=True)
              )
              (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
        (head): SiglipMultiheadAttentionPoolingHead(
          (attention): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=1152, out_features=1152, bias=True)
          )
          (layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (mlp): SiglipMLP(
            (activation_fn): PytorchGELUTanh()
            (fc1): Linear(in_features=1152, out_features=4304, bias=True)
            (fc2): Linear(in_features=4304, out_features=1152, bias=True)
          )
        )
      )
    )
  )
  (mm_projector): MultimodalProjector(
    (layers): Sequential(
      (0): DownSampleBlock()
      (1): LayerNorm((4608,), eps=1e-05, elementwise_affine=True)
      (2): Linear(in_features=4608, out_features=2560, bias=True)
      (3): GELU(approximate='none')
      (4): Linear(in_features=2560, out_features=2560, bias=True)
    )
  )
)