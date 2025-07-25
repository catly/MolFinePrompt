import clip
import torch


class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype
        self.device = "cuda:0"
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

    def tokenize(self, text , operate = None):
        if operate is None:
            return torch.cat([clip.tokenize(tok, truncate=True, operate = operate) for tok in text])
        else:
            result_sum = torch.empty(0, dtype=torch.int32)
            pos_sum = torch.empty(0, dtype=torch.int32)
            for tok in text:
                result, pos = clip.tokenize(tok, truncate=True, operate=operate)
                result_sum = torch.cat((result_sum,result), dim=0)
                pos_sum = torch.cat((pos_sum,pos), dim=0)

            return result_sum, pos_sum

    def encode_text(self, text):

        token_ids = self.tokenize(text)
        text_features = self.forward(token_ids)

        return text_features

    def forward(self, token_ids):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """

        token_ids = token_ids.to(self.device)
        text_features = self.token_embedding(token_ids)
        text_features = text_features.type(self.dtype)
        text_features = text_features.type(self.dtype)

        x = (
                text_features + self.positional_embedding.type(self.dtype)
        )
        x = x.permute(1, 0, 2)
        x = self.transformer(x.to(torch.float16))
        x = x.permute(1, 0, 2)
        x = self.ln_final(x.to(self.dtype))
        tf = (
            x[
                torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
            ]
            # @ self.text_projection
        )
        return tf
