"""
Implementation with LoRA (Low-Rank Adaptation) applied to each prediction head
using the PEFT library.
"""

from transformers import DebertaV2Model, DebertaV2PreTrainedModel, DebertaV2Config
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Trainer

class DebertaV2ForAIDetectionWithLoRA(DebertaV2PreTrainedModel):
    def __init__(self, config, num_ai_models, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__(config)
        # Initialize base DeBERTa model
        self.deberta = DebertaV2Model(config)

        # Apply LoRA to the base model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query_proj", "value_proj", "key_proj"],  # Correct DeBERTa modules
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"  # Important for non-sequence-classification tasks
        )
        self.deberta = get_peft_model(self.deberta, lora_config)

        # Regular dropout and classification heads (no LoRA)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Task 1: Human (0) vs. AI (1) - Binary head
        self.human_ai_head = nn.Linear(config.hidden_size, 1)

        # Task 2: If AI, classify which model - Multiclass head
        self.ai_model_head = nn.Linear(config.hidden_size, num_ai_models)

        self.post_init()

    def freeze_params(self, freeze):
        """
        Function for the possibility of separate training of the "head" and "body" of the BERT-like model.
        """
        if freeze:
            for param in self.deberta.parameters():
                param.requires_grad = False
        if not freeze:
            for param in self.deberta.parameters():
                param.requires_grad = True

    def unfreeze_top_layers(self, n_layers):
        """Unfreeze the top n layers (0 = none, all_layers = full unfreeze)"""
        layers = self.deberta.encoder.layer[-n_layers:]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            human_ai_labels=None,  # Binary labels (0=human, 1=AI)
            ai_model_labels=None,  # Model labels (if AI)
            **kwargs
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)

        # Task 1: Human vs. AI logits
        human_ai_logits = self.human_ai_head(pooled_output)

        # Task 2: AI model logits (only used if AI)
        ai_model_logits = self.ai_model_head(pooled_output)

        loss = None
        if human_ai_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            human_ai_loss = loss_fct(
                human_ai_logits.view(-1),
                human_ai_labels.float().view(-1)
            )

            # Mask AI model loss (only compute for AI-generated texts)
            if ai_model_labels is not None:
                ai_mask = (human_ai_labels == 1)  # Only AI samples
                if ai_mask.any():
                    loss_fct = nn.CrossEntropyLoss()
                    ai_model_loss = loss_fct(
                        ai_model_logits[ai_mask],
                        ai_model_labels[ai_mask]
                    )
                    loss = human_ai_loss + ai_model_loss
                else:
                    loss = human_ai_loss
            else:
                loss = human_ai_loss

        return {
            "human_ai_logits": human_ai_logits,
            "ai_model_logits": ai_model_logits,
            "loss": loss,
        }

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        Useful for verifying LoRA is working correctly.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def print_layer_info(self):
        """
        Prints the number of layers in the model and their trainable status.
        Also shows total number of trainable parameters.
        """
        # Get the number of layers from the encoder
        num_layers = len(self.deberta.encoder.layer)
        print(f"\nModel Architecture Info:")
        print(f"- Total layers in DeBERTa encoder: {num_layers}")
        print(f"- LoRA rank: {self.deberta.peft_config['default'].r}")

        # Count trainable vs frozen layers
        trainable_layers = 0
        for i, layer in enumerate(self.deberta.encoder.layer):
            if any(p.requires_grad for p in layer.parameters()):
                trainable_layers += 1
                status = "Trainable"
            else:
                status = "Frozen"
            print(f"  Layer {i + 1}/{num_layers}: {status}")

        print(f"\n- Trainable layers: {trainable_layers}/{num_layers}")
        print(f"- Trainable heads: Human-AI and {self.ai_model_head.out_features}-way AI classifier")
        self.print_trainable_parameters()