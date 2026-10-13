"""The frozen SmolVLM2 trunk: images and language in, a prefix of vectors out.

This is the "VLM half" of the model. It is loaded once, put in eval mode, and never
trained - everything that learns lives in the action expert that reads its output. Because
nothing feeding it is trainable, its forward pass builds no autograd graph at all, so it
costs about the same during training as during inference.

Only the first `num_layers` text layers are kept, which is what SmolVLA does; the depth of
the language model halves and the last layers' output was not being used for anything.

The three calls below are copied from SmolVLA's own `embed_image` and
`embed_language_tokens`, so the prefix this produces is the one SmolVLA's expert expects.
"""

import torch
from transformers import AutoModelForImageTextToText

TRUNK_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


class SmolVLM2Backbone(torch.nn.Module):
    """Frozen SmolVLM2-500M, exposing only the prefix hidden states."""

    def __init__(self, num_layers=16, dtype=torch.bfloat16):
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(TRUNK_NAME, dtype=dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # keep the first num_layers text layers and drop the rest
        self.model.model.text_model.layers = self.model.model.text_model.layers[:num_layers]

        # the language modelling head is unused: we want hidden states, not token logits
        del self.model.lm_head

        self.hidden_size = self.model.config.text_config.hidden_size

    def train(self, mode=True):
        """Refuse to leave eval mode.

        `policy.train()` recurses into every submodule, which would switch the trunk's
        dropout back on and make its output random from step to step. The trunk is frozen
        and must stay in eval mode, so this override ignores the request entirely.
        """
        return super().train(False)

    @torch.no_grad()
    def forward(self, images, text_ids, text_mask):
        """images: (batch, num_cameras, 3, S, S)，取值 [0, 1]。text_ids, text_mask: (batch, T)。

        返回 (hidden, prefix_mask)。hidden 是 (batch, L, hidden_size)，图像 token 在前、
        文字 token 在后；prefix_mask 是 (batch, L)，真 token 处为 True。mask 从这里返回、
        而不是让调用方重建，因为它的长度取决于图像尺寸和相机数，只有这个模块知道。

        整个前向跑在 no_grad 下：这些参数和输入都不需要梯度，没有计算图可建。
        """
        batch = images.shape[0]  # B
        num_cameras = images.shape[1]  # 2

        flat_images = images.reshape(batch * num_cameras, *images.shape[2:])  # (B*2, 3, S, S)
        # SigLIP 要 (pixel - 0.5) / 0.5；数据集给的已经是 [0, 1]
        pixels = ((flat_images - 0.5) / 0.5).to(self.model.model.vision_model.dtype)

        # 视觉塔：patch 网格 → (B*2, patch 数, 768)
        patches = self.model.model.vision_model(pixel_values=pixels).last_hidden_state
        # connector 把 patch 网格每个方向压到 1/4：2 相机 × 256² 只剩 32 个 token，不是 512
        patches = self.model.model.connector(patches)
        patches = patches.reshape(batch, -1, self.hidden_size)  # (B, 32, 960)

        words = self.model.model.text_model.get_input_embeddings()(text_ids)  # (B, T, 960)

        # SmolVLA 把两个模态都乘上 √hidden_size 再拼接，让它们的起始量级相当
        scale = self.hidden_size ** 0.5  # ≈ 31
        embeddings = torch.cat([patches * scale, words * scale], dim=1)  # (B, 32+T, 960)

        # 图像 token 永远是真的；文字可能被补齐到 batch 里最长的那一条
        image_mask = torch.ones(batch, patches.shape[1], dtype=torch.bool, device=images.device)
        prefix_mask = torch.cat([image_mask, text_mask], dim=1)  # (B, 32+T)

        hidden = self.model.model.text_model(
            inputs_embeds=embeddings,
            attention_mask=prefix_mask.long(),
        ).last_hidden_state  # (B, 32+T, 960)
        return hidden, prefix_mask
