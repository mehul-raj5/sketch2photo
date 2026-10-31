# ours models package. No eager imports: scripts set HF_HOME / TORCH_HOME (utils.config.apply_env)
# before anything imports diffusers or transformers, and the x̂-only path must not import them at all.
