import torch

class TokenProjector(torch.nn.Module):
    """Projection Model
    Takes in input embeddings of shape (BS, L, clip_embeddings_dim) and projects them to (BS, L, cross_attention_dim)
    """

    def __init__(self, clip_embeddings_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        """
        image_embeds: torch.Tensor of shape (BS, L, clip_embeddings_dim)

        returns: torch.Tensor of shape (BS, L, cross_attention_dim)
        """
        # image embeds in shape (BS, L, C)
        embeds = image_embeds
        # BS, C, L = embeds.shape
        projected_tokens = self.proj(embeds)
        projected_tokens = self.norm(projected_tokens)
        return projected_tokens
    
class ImageProjModel(torch.nn.Module):
    """Projection Model from the original IPAdapter"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
class SequenceTextProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        B, L, C = embeds.shape
        clip_extra_context_tokens = self.proj(embeds).reshape(
            B, L, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens