cfg = {
    "rna": {
        "encoder": {'in_dim': 128, 'hidden_dim': 100, 'out_dim': 100, 'num_heads': 8,
                    'attn_drop': 0.0, 'add_drop': 0.2, 'num_layers': 4, 'final_embed': 256},
        "decoder": {'input_size': 128, 'hidden_size': 256, 'depth': 4, 'num_heads': 8, 'classes': 6, 'mlp_ratio': 2,'cond_size': 256,
                    'dit_type': 'dit'},
        "diffusion": {'betas': [1.0e-4, 0.02], 'n_T': 1000, 'drop_prob':0,"schedule":"cosine"}
    },
    "atac": {
        "encoder": {'in_dim': 128, 'hidden_dim': 100, 'out_dim': 100, 'num_heads': 8,
                    'attn_drop': 0.0, 'add_drop': 0.2, 'num_layers': 4, 'final_embed': 256},
        "decoder": {'input_size': 128, 'hidden_size': 256, 'depth': 5, 'num_heads': 8, 'classes': 6, 'mlp_ratio': 2,'cond_size': 256,
                    'dit_type': 'dit'},
        "diffusion": {'betas': [1.0e-4, 0.02], 'n_T': 1000, 'drop_prob':0,"schedule":"cosine"}
    },
    "adt": {
        "encoder": {'in_dim': 128, 'hidden_dim': 100, 'out_dim': 100, 'num_heads': 8,
                    'attn_drop': 0.0, 'add_drop': 0.2, 'num_layers': 4, 'final_embed': 256},
        "decoder": {'input_size': 128, 'hidden_size': 256, 'depth': 5, 'num_heads': 8, 'classes': 6, 'mlp_ratio': 2,'cond_size': 256,
                    'dit_type': 'dit'},
        "diffusion": {'betas': [1.0e-4, 0.02], 'n_T': 1000, 'drop_prob':0,"schedule":"cosine"}
    },
    "spatial": {
        "encoder": {'in_dim': 128, 'hidden_dim': 100, 'out_dim': 100, 'num_heads': 8,
                    'attn_drop': 0.0, 'add_drop': 0.2, 'num_layers': 4, 'final_embed': 256},
        "decoder": {'input_size': 128, 'hidden_size': 256, 'depth': 5, 'num_heads': 8, 'classes': 6, 'mlp_ratio': 2,'cond_size': 256,
                    'dit_type': 'dit'},
        "diffusion": {'betas': [1.0e-4, 0.02], 'n_T': 1000, 'drop_prob':0,"schedule":"cosine"}
    },
    "morph": {
        "encoder": {'in_dim': 128, 'hidden_dim': 100, 'out_dim': 100, 'num_heads': 8,
                    'attn_drop': 0.0, 'add_drop': 0.2, 'num_layers': 4, 'final_embed': 256},
        "decoder": {'input_size': 128, 'hidden_size': 256, 'depth': 5, 'num_heads': 8, 'classes': 6, 'mlp_ratio': 2,'cond_size': 256,
                    'dit_type': 'dit'},
        "diffusion": {'betas': [1.0e-4, 0.02], 'n_T': 1000, 'drop_prob':0,"schedule":"cosine"}
    },
    "ephy": {
        "encoder": {'in_dim': 68, 'hidden_dim': 100, 'out_dim': 100, 'num_heads': 8,
                    'attn_drop': 0.0, 'add_drop': 0.2, 'num_layers': 4, 'final_embed': 256},
        "decoder": {'input_size': 68, 'hidden_size': 256, 'depth': 5, 'num_heads': 8, 'classes': 6, 'mlp_ratio': 2,'cond_size': 256,
                    'dit_type': 'dit'},
        "diffusion": {'betas': [1.0e-4, 0.02], 'n_T': 1000, 'drop_prob':0,"schedule":"cosine"}
    },
   "mrna": {
        "encoder": {'in_dim': 100, 'hidden_dim': 100, 'out_dim': 100, 'num_heads': 8,
                    'attn_drop': 0.0, 'add_drop': 0.2, 'num_layers': 4, 'final_embed': 256},
        "decoder": {'input_size': 100, 'hidden_size': 256, 'depth': 5, 'num_heads': 8, 'classes': 6, 'mlp_ratio': 2,'cond_size': 256,
                    'dit_type': 'dit'},
        "diffusion": {'betas': [1.0e-4, 0.02], 'n_T': 1000, 'drop_prob':0,"schedule":"cosine"}
    }
}
