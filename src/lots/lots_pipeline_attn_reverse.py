from typing import List, Optional, Union # Added Optional, Union for type hinting in generate
from PIL import Image
import torch
import os
import json
from transformers import AutoImageProcessor
from lots.pair_former import PairFormer
from ip_adapter.utils import is_torch2_available, get_generator
from lots.token_projection import ImageProjModel, TokenProjector, SequenceTextProjModel
from utils.dinov2_utils import get_pooling_dim, get_feature_dim, extract_features
from lots.global_attn_reverse import GlobalSketchAttention 

if is_torch2_available():
    from lots.cross_attn import AttnProcessor2_0 as AttnProcessor
    from lots.cross_attn import LOTSAttnProcessor2_0 as LOTSAttnProcessor
else:
    from lots.cross_attn import AttnProcessor
    from lots.cross_attn import LOTSAttnProcessor

class LOTSPipeline:

    def __init__(self, sd_pipe, lots_ckpt, device, image_encoder=None, num_global_tokens=77, num_tokens=32, model_type='vits14'):
        self.device = device
        self.image_encoder = image_encoder
        self.lots_ckpt = lots_ckpt
        self.num_global_tokens = num_global_tokens
        self.num_tokens = num_tokens
        self.model_type = model_type
        

        self.pipe = sd_pipe.to(self.device)
        self.add_cross_attn(num_global_tokens=num_global_tokens)

        # Allow two way to import encoder
        self.image_encoder = image_encoder.to(self.device, dtype=torch.float16)
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        # image proj model
        # 1. MODIFICATION: init_proj 需返回 global_attn_module
        self.image_proj_model, self.text_proj_model, self.pair_former, self.global_attn_module = self.init_proj() 
        self.load_cross_attn()

    def add_cross_attn(self, num_global_tokens=77):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = LOTSAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_global_tokens=num_global_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

    def init_proj(self):
        
        base_dim = get_feature_dim(self.model_type)
        embeddings_dim = get_pooling_dim(base_dim, "cls")

        image_proj_model = TokenProjector(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=embeddings_dim,
        ).to(self.device, dtype=torch.float16)

        text_proj_model = SequenceTextProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.pipe.text_encoder.config.projection_dim + self.pipe.text_encoder_2.config.projection_dim,
            clip_extra_context_tokens=4,
        ).to(self.device, dtype=torch.float16)

        # check if config is available from ckpt folder
        config_path = os.path.join(os.path.dirname(self.lots_ckpt), "fusion_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                fusion_config = json.load(f)
            pair_former_model = PairFormer(**fusion_config).to(self.device, dtype=torch.float16)
        else:
            # use default parameters
            pair_former_model = PairFormer(
                in_channels=self.pipe.unet.config.cross_attention_dim,
                inner_dim=self.pipe.unet.config.cross_attention_dim,
                fusion_strategy="deferred",
                num_layers=2,
                num_attention_heads=8,
                dropout=0.0,
                activation_fn="geglu",
                norm_num_groups=32,
                masking_strategy="compression",
                num_cls_tokens=self.num_tokens,
            ).to(self.device, dtype=torch.float16)
        
        # 2. ADDITION: Global Attention Module 初始化
        global_attn_module = GlobalSketchAttention( 
            sketch_embed_dim=base_dim, # DINOv2 feature dim
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            num_attention_heads=8, 
            prepend_cls=True
        ).to(self.device, dtype=torch.float16)

        # 3. MODIFICATION: 返回 global_attn_module
        return image_proj_model, text_proj_model, pair_former_model, global_attn_module

    def load_cross_attn(self):
        state_dict = torch.load(self.lots_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.text_proj_model.load_state_dict(state_dict["text_proj"], strict=True)
        self.pair_former.load_state_dict(state_dict["pair_former"], strict=True)
        
        # 4. ADDITION: 加载 global_attn 权重 (与训练代码对齐)
        if "global_attn" in state_dict:
            self.global_attn_module.load_state_dict(state_dict["global_attn"], strict=True)
        else:
            print("Warning: 'global_attn' key not found in checkpoint. Assuming global attention is a new module or not trained.")

        attn_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        attn_layers.load_state_dict(state_dict["cross_attn"], strict=True)
       
    def generate(
        self,
        pil_images: Union[Image.Image, List[Image.Image]],
        descriptions: Union[str, List[str]],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        scale: float = 1.0,
        num_samples: int = 4,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        resolution: int = 512,
        global_sketch: Optional[Union[Image.Image, List[Image.Image]]] = None, # 5. MODIFICATION: 添加 global_sketch 参数
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 
        num_sketches = len(pil_images)

        if prompt is None:
            prompt = "High quality photo of a model, artistic, 4k"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # 1. Local Sketch and Text Embeds (Projection)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_images)
        text_prompt_embeds, uncond_text_prompt_embeds = self.get_text_embeds(descriptions)

        # 2. Local PairFormer Fusion
        mask = [[True for _ in range(num_sketches)]] # extra dimension for batching
        pair_embeds = self.pair_former(image_embeds=image_prompt_embeds, text_embeds=text_prompt_embeds, image_masks=mask, text_masks=mask)
        uncond_pair_embeds = self.pair_former(image_embeds=uncond_image_prompt_embeds, text_embeds=uncond_text_prompt_embeds, image_masks=mask, text_masks=mask)

        # 3. Process Global Sketch Embeds (Raw DINOv2)
        global_sketch_embeds, uncond_global_sketch_embeds = self._get_global_sketch_embeds(global_sketch)

        # 4. Global Attention Fusion (Local Pair Embeds + Global Sketch Embeds)
        
        # Mask for pair_embeds before global fusion (all True in inference)
        pair_cross_attn_mask = torch.ones((pair_embeds.shape[0], pair_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # Conditioned Fusion
        fused_local_and_global_tokens, _ = self.global_attn_module(
            global_sketch_embeds=global_sketch_embeds,
            pair_embeds=pair_embeds,
            pair_mask=pair_cross_attn_mask,
        )

        # Unconditioned Fusion (for CFG)
        uncond_fused_local_and_global_tokens, _ = self.global_attn_module(
            global_sketch_embeds=uncond_global_sketch_embeds,
            pair_embeds=uncond_pair_embeds,
            pair_mask=pair_cross_attn_mask, 
        )

        # 5. Global Text Encoding and Final Concatenation
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            # 6. MODIFICATION: 拼接 Global Text + Fused Local/Global Tokens
            prompt_embeds = torch.cat([prompt_embeds, fused_local_and_global_tokens], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_fused_local_and_global_tokens], dim=1)

        # 7. ADDITION: 准备 UNet 的 Cross Attention Mask (Global Text Mask + Fused Tokens Mask)
        # Global text sequence length is self.num_global_tokens (77), always attended to (True)
        #text_mask = torch.ones((final_cross_attn_mask.shape[0], self.num_global_tokens), dtype=torch.bool, device=self.device)
        
        #final_unet_mask = torch.cat([text_mask, final_cross_attn_mask], dim=1)
        #final_uncond_unet_mask = torch.cat([text_mask, uncond_final_cross_attn_mask], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            height=resolution,
            width=resolution,
            #encoder_attention_mask=final_unet_mask, # 8. MODIFICATION: 传入 mask
            #negative_encoder_attention_mask=final_uncond_unet_mask, # 9. MODIFICATION: 传入 mask
            **kwargs,
        ).images

        return images
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_images: Union[Image.Image, List[Image.Image]]):
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]

        sketches = [self.image_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16) for pil_image in pil_images]
        sketches = torch.cat(sketches, dim=0)
        outputs = self.image_encoder(sketches)
    
        image_embeds = outputs.last_hidden_state.unsqueeze(0) # add batch dimension

        image_prompt_embeds = self.image_proj_model(image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    @torch.inference_mode()
    def get_text_embeds(self, descriptions: Union[str, List[str]]):
        if descriptions is not None:
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            descriptions_ids = [self.pipe.tokenizer(description, return_tensors="pt", padding="max_length", truncation=True, max_length=self.pipe.tokenizer.model_max_length).input_ids.to(self.device) 
                                for description in descriptions]
            text_embeds = [self.pipe.text_encoder(description_ids)['pooler_output'] for description_ids in descriptions_ids]
            descriptions_ids_2 = [self.pipe.tokenizer_2(description, return_tensors="pt", padding="max_length", truncation=True, max_length=self.pipe.tokenizer_2.model_max_length).input_ids.to(self.device)
                                 for description in descriptions]
            text_embeds_2 = [self.pipe.text_encoder_2(description_ids_2)['text_embeds'] for description_ids_2 in descriptions_ids_2]
            text_embeds = torch.cat(text_embeds, dim=0)
            text_embeds_2 = torch.cat(text_embeds_2, dim=0)
            text_embeds = torch.cat([text_embeds, text_embeds_2], dim=1).unsqueeze(0) # add batch dimension

        text_prompt_embeds = self.text_proj_model(text_embeds)
        uncond_text_prompt_embeds = self.text_proj_model(torch.zeros_like(text_embeds))
        return text_prompt_embeds, uncond_text_prompt_embeds
    
    # 10. ADDITION: Helper function to process global sketch into raw DINOv2 features
    @torch.inference_mode()
    def _get_global_sketch_embeds(self, global_sketch: Optional[Union[Image.Image, List[Image.Image]]]):
        
        if global_sketch is None:
            # Fallback to zero tensor of expected shape if global sketch is missing
            L_patches = 257 # DINOv2 ViT/14 patches + CLS token
            D_embed = get_feature_dim(self.model_type)
            zero_embeds = torch.zeros(1, L_patches, D_embed, device=self.device, dtype=torch.float16)
            return zero_embeds, zero_embeds

        if isinstance(global_sketch, Image.Image):
            pil_sketches_list = [global_sketch]
        elif isinstance(global_sketch, list):
            pil_sketches_list = global_sketch
        else:
            raise ValueError("`global_sketch` must be a PIL Image or a list of PIL Images.")
        
        # Process the image(s) to get raw DINOv2 features
        sketches_pixels = [self.image_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16) 
                            for pil_image in pil_sketches_list]
        sketches_pixels = torch.cat(sketches_pixels, dim=0) # [N_sketches, 3, H, W]
        
        # Outputs.last_hidden_state is [N_sketches, L_patches, D_embed]
        outputs = self.image_encoder(sketches_pixels)
        raw_features = outputs.last_hidden_state
        
        # For inference, assume a single final global sketch feature [1, L_patches, D_embed]
        # If multiple are provided, we take the first one or need an aggregation rule (taking first for now)
        global_sketch_embeds = raw_features[0].unsqueeze(0) 
        
        # Unconditioned version (zeros)
        uncond_global_sketch_embeds = torch.zeros_like(global_sketch_embeds)
        
        return global_sketch_embeds, uncond_global_sketch_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LOTSAttnProcessor):
                attn_processor.scale = scale
