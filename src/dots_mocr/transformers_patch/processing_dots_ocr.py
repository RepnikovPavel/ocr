from transformers import Qwen2_5_VLProcessor

class DotsVLProcessor(Qwen2_5_VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, 
                 video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)
        self.image_token = "<|imgpad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_token_id = 151665
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.video_token_id = 151656
