"""SD 1.5 (Realistic Vision 5.1) + 19-channel dual-path ControlNet over DECP's frozen photo codebook — v2.

Frozen: UNet, VAE (sd-vae-ft-mse), CLIP text encoder (all fp32 weights, eval), CVQ-VAE encoder + codebook
(eval-locked, fp32 tokens). Trainable: ControlNet (scribble init, conv_in widened with zero columns, CN-L6)
and SketchAdapter (GroupNorm on L2-normalised tokens, CN-L2).
The pipeline is a plain object, not an nn.Module, so no recursive .train() can reach the codebook (NEW-codebook-1).

Training step (P5–P7, SD-L2, CN-L1, CN-L5):
  t from `mixed` (half uniform, half Beta(1, 3)), shuffled across the batch; per-sample text dropout and
  condition / path dropout; one residual scale × U[0.8, 1.2] per step; ε-MSE × min-SNR-γ (γ = 5);
  latent L1 + Scharr on x̂₀ vs z₀ at every t × min(1, SNR_t);
  VAE decode of at most `aux.max_decode` randomly chosen rows with t ≤ `aux.max_t`; LPIPS-VGG, Scharr, GMSD on
  256² copies × min(1, SNR_t); IR-SE50 ID on aligned crops for t ≤ `aux.id.max_t`; FSIM surrogate and MS-SSIM
  behind flags (P12). All random draws except the diffusion noise and the VAE posterior sample come from one
  CPU generator whose state the loop checkpoints; those two use the checkpointed global CUDA RNG.
"""
import os

import torch
import torch.nn.functional as F

from models.conditioning import (assemble_conditioning, build_controlnet, draw_condition_dropout,
                                 draw_residual_scale, scale_residuals, soft_edges, DROP_ADAPTER, DROP_ALL,
                                 DROP_EDGES)
from models.cvqvae_encoder import build_frozen_cvqvae
from models.sketch_adapter import SketchAdapter
from models.structure_losses import resize_antialiased
from utils.precision import autocast_ctx, autocast_disabled

SCHARR_X = ((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0))


def sample_timesteps(batch_size, num_train_timesteps, mode, shuffle, generator):
    """(B,) int64 on CPU. `low` = Beta(1, 3) by inverse CDF (seedable), `mixed` = half uniform + half low."""
    total = int(num_train_timesteps)

    def low(n):
        u = torch.rand(n, generator=generator, dtype=torch.float64)
        x = 1.0 - (1.0 - u) ** (1.0 / 3.0)
        return (x * total).long().clamp(0, total - 1)

    if mode == 'uniform':
        t = torch.randint(0, total, (batch_size,), generator=generator)
    elif mode == 'low':
        t = low(batch_size)
    elif mode == 'mixed':
        half = batch_size // 2
        t = torch.cat([torch.randint(0, total, (half,), generator=generator), low(batch_size - half)])
        if shuffle:  # NEW-pipeline-7: no position-correlated t inside the batch
            t = t[torch.randperm(batch_size, generator=generator)]
    else:
        raise ValueError('unknown timestep sampler %r' % mode)
    return t.long()


def mse_weights(snr, timesteps, mode, gamma, num_train_timesteps):
    if mode == 'min_snr':  # Hang et al. 2023, ε-prediction form
        return torch.clamp(snr, max=gamma) / snr
    if mode == 'legacy':   # v1 ramp, ablation only (NEW-sdunet-2)
        return 1.0 + 2.0 * (1.0 - timesteps.float() / num_train_timesteps)
    return torch.ones_like(snr)


def _scharr_latent(x):
    c = x.shape[1]
    kx = torch.tensor(SCHARR_X, device=x.device, dtype=x.dtype) / 16.0
    ky = kx.t().contiguous()
    return (F.conv2d(x, kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1), padding=1, groups=c),
            F.conv2d(x, ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1), padding=1, groups=c))


def _laplacian_abs(images):
    """Mean |Laplacian| of the grey channel in 0-255 grey levels — the speckle guard of F46."""
    from models.structure_losses import to_gray01
    grey = to_gray01(images.float()) * 255.0
    lap = (4.0 * grey[:, :, 1:-1, 1:-1] - grey[:, :, :-2, 1:-1] - grey[:, :, 2:, 1:-1]
           - grey[:, :, 1:-1, :-2] - grey[:, :, 1:-1, 2:])
    return float(lap.abs().mean())


def latent_l1_scharr(x0_pred, z0):
    """Per-sample L1 + Scharr-component L1 between latent x̂₀ and z₀ (SD-L2d)."""
    dims = (1, 2, 3)
    px, py = _scharr_latent(x0_pred)
    tx, ty = _scharr_latent(z0)
    return ((x0_pred - z0).abs().mean(dim=dims) + (px - tx).abs().mean(dim=dims)
            + (py - ty).abs().mean(dim=dims))


class SketchToPhotoPipeline(object):
    def __init__(self, cfg, device, with_decoder=False):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        self.cfg = cfg
        self.device = torch.device(device)
        m = cfg['model']
        self.vae = AutoencoderKL.from_pretrained(m['vae_id'], torch_dtype=torch.float32)
        self.unet = UNet2DConditionModel.from_pretrained(m['sd_model_id'], subfolder='unet', torch_dtype=torch.float32)
        self.text_encoder = CLIPTextModel.from_pretrained(m['sd_model_id'], subfolder='text_encoder',
                                                          torch_dtype=torch.float32)
        self.tokenizer = CLIPTokenizer.from_pretrained(m['sd_model_id'], subfolder='tokenizer')
        self.noise_scheduler = DDPMScheduler.from_pretrained(m['sd_model_id'], subfolder='scheduler')
        prediction_type = getattr(self.noise_scheduler.config, 'prediction_type', 'epsilon')
        if prediction_type != 'epsilon':  # NEW-sdunet-6: loss target, x̂₀ and CFG assume ε
            raise RuntimeError('%s scheduler prediction_type=%r, v2 assumes epsilon' % (m['sd_model_id'], prediction_type))
        for module in (self.vae, self.unet, self.text_encoder):
            module.requires_grad_(False)
            module.eval()
        self.vae_scale = float(self.vae.config.scaling_factor)
        self.vae_decoder_sha256 = None
        if m.get('vae_decoder'):  # fine-tuned decoder + post_quant_conv (tools/finetune_vae_decoder.py); encoder stays stock
            from utils.config import sha256_file
            path = m['vae_decoder']
            if not os.path.isabs(path):
                path = os.path.join(cfg['_resolved']['project_root'], path)
            state = torch.load(path, map_location='cpu')
            self.vae.decoder.load_state_dict(state['decoder'])
            self.vae.post_quant_conv.load_state_dict(state['post_quant_conv'])
            self.vae_decoder_sha256 = sha256_file(path)
        self.controlnet = build_controlnet(m['controlnet_id'], cfg['adapter']['out_channels'])
        self.adapter = SketchAdapter(in_channels=cfg['codebook']['embed_dim'], out_channels=cfg['adapter']['out_channels'],
                                     norm_groups=cfg['adapter']['norm_groups'],
                                     l2_normalise_tokens=cfg['adapter']['l2_normalise_tokens'])
        self.cvq = build_frozen_cvqvae(cfg, with_decoder=with_decoder)
        for module in (self.vae, self.unet, self.text_encoder, self.controlnet, self.adapter, self.cvq):
            module.to(self.device)
        if m['gradient_checkpointing']:  # SD-L8 VRAM lever
            self.controlnet.enable_gradient_checkpointing()
            self.unet.enable_gradient_checkpointing()
            self.unet.train()  # diffusers checkpoints only in training mode; the SD 1.5 UNet has no dropout / BatchNorm
        self.tone_stats = None
        if cfg['tone']['enabled']:
            from utils.tone import load_tone_stats
            self.tone_stats = load_tone_stats(cfg['_resolved']['tone_stats_dir'], cfg['tone']['stats_split'])
        self.num_train_timesteps = int(self.noise_scheduler.config.num_train_timesteps)
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=torch.float32)
        self.structure_losses = None
        self.id_loss = None
        # Per-step multipliers on the LPIPS and identity terms (v1 two-phase schedule, aux.boost). The trainer sets
        # them from the step before every training_step; 1.0 reproduces the unboosted objective exactly.
        self.aux_multipliers = {'lpips': 1.0, 'id': 1.0}
        self.info = {
            'prediction_type': prediction_type,
            'vae_scaling_factor': self.vae_scale,
            'vae_decoder': m.get('vae_decoder'), 'vae_decoder_sha256': self.vae_decoder_sha256,
            'scheduler_config': dict(self.noise_scheduler.config),
            'controlnet_conditioning_channels': int(self.controlnet.config.conditioning_channels),
            'controlnet_parameters': int(sum(p.numel() for p in self.controlnet.parameters())),
            'adapter_parameters': int(self.adapter.count_parameters()),
            'codebook': self.cvq.describe(),
        }

    # ------------------------------------------------------------------ modes, modules
    def attach_losses(self, structure_losses=None, id_loss=None):
        self.structure_losses = structure_losses.to(self.device) if structure_losses is not None else None
        self.id_loss = id_loss.to(self.device) if id_loss is not None else None

    def trainable_modules(self):
        return {'controlnet': self.controlnet, 'adapter': self.adapter}

    def train_mode(self):
        self.controlnet.train()
        self.adapter.train()
        self.cvq.eval()  # a no-op by construction (train() is overridden) — kept explicit

    def eval_mode(self):
        self.controlnet.eval()
        self.adapter.eval()
        self.cvq.eval()

    # ------------------------------------------------------------------ building blocks
    @torch.no_grad()
    def encode_prompt(self, texts):
        """list of str → (N, 77, 768) float32 CLIP last hidden states, computed once and cached by callers."""
        tokens = self.tokenizer(list(texts), padding='max_length', max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
        with autocast_disabled(self.device):
            emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return emb.float()

    def conditioning(self, sketch, sketch_raw, drop_mode=None, paths=None):
        """(B, 3, 256, 256) + (B, 3, 512, 512) in [-1, 1] → c_spatial (B, 19, 512, 512) float32."""
        c = self.cfg['conditioning']
        paths = paths or c['paths']
        edges = soft_edges(sketch_raw, c['edge_amplification'], c['edge_border_fix'])
        if paths == 'edges_only':
            b, _, h, w = edges.shape
            adapter_out = edges.new_zeros(b, self.adapter.out_channels, h, w)
        else:
            z_q = self.cvq.tokens(sketch)       # float32, autocast off (CB-L1)
            adapter_out = self.adapter(z_q)     # under the caller's autocast
        return assemble_conditioning(edges, adapter_out, drop_mode, paths)

    def controlnet_residuals(self, z_t, t, text, c_spatial, scale):
        down, mid = self.controlnet(z_t, t, encoder_hidden_states=text, controlnet_cond=c_spatial, return_dict=False)
        return scale_residuals(down, mid, scale)

    def unet_eps(self, z_t, t, text, down=None, mid=None):
        return self.unet(z_t, t, encoder_hidden_states=text, down_block_additional_residuals=down,
                         mid_block_additional_residual=mid, return_dict=False)[0]

    def predict_x0(self, z_t, eps, t):
        a = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return (z_t.float() - (1.0 - a).sqrt() * eps.float()) / a.sqrt()

    @torch.no_grad()
    def encode_latents_mean(self, images, precision):
        with autocast_ctx(self.device, precision):
            z = self.vae.encode(images.to(self.device)).latent_dist.mean
        return z.float() * self.vae_scale

    @torch.no_grad()
    def anchor_latents(self, sketch, size, precision):
        """Latent of the tone-normalised codebook estimate x̂: frozen decoder → tone → bicubic → VAE mean.

        The single definition of the x̂ anchor, shared by `generate()` (SDEdit start) and the anchored training
        rows, so the model trains on exactly the anchor it meets at inference.
        """
        xhat = self.tone_normalise(self.cvq.estimate(sketch))
        xhat = F.interpolate(xhat, size=size, mode='bicubic', align_corners=False)
        return self.encode_latents_mean(xhat.clamp(-1.0, 1.0), precision)

    # ------------------------------------------------------------------ training
    def training_step(self, batch, text_cond, text_uncond, generator):
        """Returns (loss, scalars: dict of floats, per_sample: dict of (B,) tensors, per_row: {term: (t, loss)})."""
        cfg = self.cfg
        dev = self.device
        precision = cfg['run']['precision']
        aux = cfg['aux']
        cond = cfg['conditioning']
        photo = batch['photo'].to(dev, non_blocking=True).float()
        sketch = batch['sketch'].to(dev, non_blocking=True).float()
        sketch_raw = batch['sketch_raw'].to(dev, non_blocking=True).float()
        bsz = photo.shape[0]

        # random draws in a fixed order from the checkpointed CPU generator
        t_cpu = sample_timesteps(bsz, self.num_train_timesteps, cfg['timesteps']['sampler'],
                                 cfg['timesteps']['shuffle'], generator)
        text_drop = torch.rand(bsz, generator=generator, dtype=torch.float64) < cfg['prompt']['dropout_per_sample']
        drop_mode = draw_condition_dropout(bsz, cond['dropout_all'], cond['dropout_adapter'], cond['dropout_edges'],
                                           generator)
        scale = draw_residual_scale(cond['residual_scale'], cond['scale_jitter'], generator)
        decode_order = torch.randperm(bsz, generator=generator)
        an = cfg['anchor']
        # Drawn only when enabled, so runs without the anchor keep bit-identical random streams (and resume).
        anchor_draw = torch.rand(bsz, generator=generator, dtype=torch.float64) if an['enabled'] else None
        t = t_cpu.to(dev)

        with torch.no_grad(), autocast_ctx(dev, precision):
            z0 = self.vae.encode(photo).latent_dist.sample() * self.vae_scale
        z0 = z0.float()
        noise = torch.randn_like(z0)
        z_t = self.noise_scheduler.add_noise(z0, noise, t)
        eps_target = noise

        # Anchored rows (plan phase 3): the noisy input starts from the x̂ anchor, as at inference, while the target
        # still points at the real photo. With z_t = sqrt(a) z_xhat + sqrt(1-a) eps, the eps that makes
        # predict_x0(z_t, eps*, t) == z0 is eps* = eps + sqrt(a / (1 - a)) (z_xhat - z0). Regressing x̂ itself
        # instead would re-learn the pale anchor tone (F48).
        n_anchored = 0
        if anchor_draw is not None:
            chosen = (t_cpu >= an['t_min']) & (t_cpu <= an['t_max']) & (anchor_draw < an['prob'])
            rows_a = torch.nonzero(chosen, as_tuple=False).view(-1).to(dev)
            n_anchored = int(rows_a.numel())
            if n_anchored > 0:
                z_xhat = self.anchor_latents(sketch[rows_a], tuple(sketch_raw.shape[-2:]),
                                             cfg['inference']['precision'])
                a_rows = self.alphas_cumprod[t[rows_a]].view(-1, 1, 1, 1)
                z_t = z_t.clone()
                z_t[rows_a] = self.noise_scheduler.add_noise(z_xhat, noise[rows_a], t[rows_a])
                eps_target = noise.clone()
                eps_target[rows_a] = noise[rows_a] + (a_rows / (1.0 - a_rows)).sqrt() * (z_xhat - z0[rows_a])
        text = torch.where(text_drop.to(dev).view(bsz, 1, 1), text_uncond.expand(bsz, -1, -1),
                           text_cond.expand(bsz, -1, -1))

        with autocast_ctx(dev, precision):
            c_spatial = self.conditioning(sketch, sketch_raw, drop_mode)
            down, mid = self.controlnet_residuals(z_t, t, text, c_spatial, scale)
            eps = self.unet_eps(z_t, t, text, down, mid)
        eps = eps.float()

        alpha = self.alphas_cumprod[t]
        snr = alpha / (1.0 - alpha)
        w_mse = mse_weights(snr, t, cfg['timesteps']['mse_weighting'], cfg['timesteps']['gamma'],
                            self.num_train_timesteps)
        per_mse = F.mse_loss(eps, eps_target, reduction='none').mean(dim=(1, 2, 3))
        loss_mse = (w_mse * per_mse).mean()
        total = loss_mse
        w_aux = snr.clamp(max=1.0) if aux['snr_weight'] else torch.ones_like(snr)

        scalars = {
            'loss_mse': float(loss_mse.detach()),
            'residual_scale': float(scale),
            't_mean': float(t_cpu.float().mean()),
            'text_dropped': float(text_drop.double().mean()),
            'cond_dropped_all': float((drop_mode == DROP_ALL).double().mean()),
            'cond_dropped_adapter': float((drop_mode == DROP_ADAPTER).double().mean()),
            'cond_dropped_edges': float((drop_mode == DROP_EDGES).double().mean()),
        }
        if an['enabled']:
            scalars['anchored_frac'] = n_anchored / float(bsz)
        if aux['boost']['start_step'] > 0:
            scalars['aux_boost_lpips'] = float(self.aux_multipliers['lpips'])
            scalars['aux_boost_id'] = float(self.aux_multipliers['id'])
        per_sample = {'t': t.detach(), 'mse': per_mse.detach()}
        per_row = {}

        x0 = self.predict_x0(z_t, eps, t)
        if aux['latent_l1_scharr'] > 0:
            per_lat = latent_l1_scharr(x0, z0)
            loss_lat = (w_aux * per_lat).mean()
            total = total + aux['latent_l1_scharr'] * loss_lat
            scalars['loss_latent'] = float(loss_lat.detach())
            per_sample['latent'] = per_lat.detach()

        masked = t <= aux['max_t']
        n_masked = int(masked.sum().item())
        scalars['aux_samples'] = n_masked
        use_pixel = self.structure_losses is not None and len(self.structure_losses.active) > 0
        use_id = self.id_loss is not None and aux['id']['weight'] > 0
        decoded = 0
        if (use_pixel or use_id) and aux['max_decode'] > 0 and n_masked > 0:
            order = decode_order.to(dev)
            rows = order[masked[order]][:aux['max_decode']]   # random subset of the masked rows (P5)
            decoded = int(rows.numel())
            with autocast_ctx(dev, precision):
                img = self.vae.decode(x0[rows] / self.vae_scale).sample
            img = img.float().clamp(-1.0, 1.0)
            pred = resize_antialiased(img, aux['loss_size'])
            with torch.no_grad():
                target = resize_antialiased(photo[rows], aux['loss_size'])
                # Speckle guard (F46): high-frequency energy of the prediction vs the real photo, in grey levels.
                # Real photos sit near 9; the run that invented texture reached 44.
                scalars['lap_abs'] = _laplacian_abs(pred)
                scalars['lap_abs_target'] = _laplacian_abs(target)
            t_rows = t[rows]
            w_rows = w_aux[rows]
            if use_pixel:
                for name, (values, eligible) in self.structure_losses(pred, target, t_rows).items():
                    n_e = int(eligible.sum().item())
                    scalars['n_' + name] = n_e
                    if n_e == 0:
                        continue
                    term = (w_rows * values * eligible.float()).sum() / n_e
                    total = total + self.structure_losses.weights[name] * self.aux_multipliers.get(name, 1.0) * term
                    scalars['loss_' + name] = float(term.detach())
                    per_row[name] = (t_rows[eligible].detach(), values[eligible].detach())
            if use_id:
                if 'id_theta' not in batch:
                    raise RuntimeError('aux.id.weight > 0 but the batch has no id_theta (landmarks not loaded)')
                eligible = t_rows <= aux['id']['max_t']
                n_e = int(eligible.sum().item())
                scalars['n_id'] = n_e
                if n_e > 0:
                    idx = torch.nonzero(eligible, as_tuple=False).view(-1)
                    rows_cpu = rows[idx].cpu()
                    theta = batch['id_theta'][rows_cpu].to(dev)
                    values = self.id_loss(pred[idx], target[idx], theta)
                    term = (w_rows[idx] * values).sum() / n_e
                    total = total + aux['id']['weight'] * self.aux_multipliers['id'] * term
                    scalars['loss_id'] = float(term.detach())
                    scalars['id_fallback_crops'] = int((~batch['id_landmarks_found'][rows_cpu].bool()).sum().item())
                    per_row['id'] = (t_rows[idx].detach(), values.detach())
        scalars['aux_decoded'] = decoded
        scalars['loss_total'] = float(total.detach())
        return total, scalars, per_sample, per_row

    # ------------------------------------------------------------------ inference (P4, SD-L1/L3/L4/L7, CN-L1/L8)
    def build_inference_scheduler(self):
        """Sampler from `inference.sampler` on the model's own noise schedule; trailing spacing by default (SD-L1)."""
        from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
                               UniPCMultistepScheduler)
        inf = self.cfg['inference']
        base = self.noise_scheduler.config
        spacing = inf['timestep_spacing']
        name = inf['sampler']
        if name == 'ddim':
            sched = DDIMScheduler.from_config(base, timestep_spacing=spacing, clip_sample=False)
            if sched.config.clip_sample:  # NEW-sdunet-9
                raise RuntimeError('DDIM clip_sample must be False')
        elif name == 'dpmpp_2m_karras':
            sched = DPMSolverMultistepScheduler.from_config(base, algorithm_type='dpmsolver++', solver_order=2,
                                                            use_karras_sigmas=True, timestep_spacing=spacing)
        elif name == 'unipc':
            sched = UniPCMultistepScheduler.from_config(base, timestep_spacing=spacing)
        elif name == 'euler_a':
            sched = EulerAncestralDiscreteScheduler.from_config(base, timestep_spacing=spacing)
        else:
            raise ValueError('unknown sampler %r' % name)
        if getattr(sched.config, 'prediction_type', 'epsilon') != 'epsilon':
            raise RuntimeError('inference scheduler prediction_type must be epsilon')
        return sched

    @torch.no_grad()
    def xhat_images(self, sketch):
        """Frozen-decoder estimate x̂ (B, 3, 256, 256) in [-1, 1] (CB §8e), tone-normalised when enabled."""
        return self.tone_normalise(self.cvq.estimate(sketch.to(self.device)))

    def tone_normalise(self, xhat):
        """Map x̂ onto the training photos' tone (phase 0). No-op unless tone.enabled."""
        if self.tone_stats is None:
            return xhat
        from utils.precision import autocast_disabled
        from utils.tone import normalise_tone
        t = self.cfg['tone']
        with autocast_disabled(self.device):
            return normalise_tone(xhat.float(), self.tone_stats, t['method'], t['strength'], t['target'])

    @torch.no_grad()
    def generate(self, sketch, sketch_raw, text_cond, text_neg, generators):
        """Sketch-only generation → (B, 3, 512, 512) float32 in [-1, 1].
        text_cond / text_neg: (1, 77, 768) cached embeddings; generators: one torch.Generator per image."""
        import inspect
        cfg = self.cfg
        inf = cfg['inference']
        dev = self.device
        precision = inf['precision']
        sketch = sketch.to(dev).float()
        sketch_raw = sketch_raw.to(dev).float()
        bsz = sketch.shape[0]
        if len(generators) != bsz:
            raise ValueError('need one generator per image')
        scale = cfg['conditioning']['residual_scale']
        cfg_scale = inf['guidance_scale']
        do_cfg = cfg_scale > 1.0
        sched = self.build_inference_scheduler()
        sched.set_timesteps(inf['steps'], device=dev)
        timesteps = sched.timesteps
        latent_shape = (1, 4, sketch_raw.shape[-2] // 8, sketch_raw.shape[-1] // 8)

        with autocast_ctx(dev, precision):
            c_spatial = self.conditioning(sketch, sketch_raw)
        cond = text_cond.expand(bsz, -1, -1)
        uncond = text_neg.expand(bsz, -1, -1)

        noise = torch.cat([torch.randn(latent_shape, generator=g, device=dev, dtype=torch.float32) for g in generators])
        if inf['start'] == 'noise':
            latents = noise * sched.init_noise_sigma
        else:  # SDEdit start from the frozen-decoder estimate (SD-L3, CB-L2)
            n = len(timesteps) // sched.order if sched.order > 1 else len(timesteps)
            init_steps = min(int(round(inf['steps'] * inf['strength'])), inf['steps'])
            t_start = max(inf['steps'] - init_steps, 0)
            timesteps = sched.timesteps[t_start * sched.order:]
            if len(timesteps) == 0:
                raise ValueError('inference.strength %.3f leaves no denoising steps' % inf['strength'])
            if hasattr(sched, 'set_begin_index'):
                sched.set_begin_index(t_start * sched.order)
            if inf['start'] == 'sketch':   # ablation: SDEdit from the sketch image itself, no codebook
                z = self.encode_latents_mean(sketch_raw.clamp(-1.0, 1.0), precision)
            else:
                z = self.anchor_latents(sketch, tuple(sketch_raw.shape[-2:]), precision)
            latents = sched.add_noise(z, noise, timesteps[:1].repeat(bsz))
            del n
        lf = inf.get('lowfreq_guide') or {'enabled': False}
        if lf['enabled']:
            if inf['start'] == 'noise':
                z = self.anchor_latents(sketch, tuple(sketch_raw.shape[-2:]), precision)

            def lowpass(x):
                h, w = x.shape[-2:]
                small = F.interpolate(x, size=(max(1, h // lf['factor']), max(1, w // lf['factor'])), mode='bilinear',
                                      align_corners=False, antialias=True)
                return F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)

        step_params = inspect.signature(sched.step).parameters
        step_kwargs = {}
        if 'generator' in step_params:
            step_kwargs['generator'] = generators
        if 'eta' in step_params and inf['sampler'] == 'ddim':
            step_kwargs['eta'] = inf['eta']
        n_steps = len(timesteps)
        for i, t in enumerate(timesteps):
            t_batch = t.reshape(1).expand(bsz).to(dev)
            latent_in = sched.scale_model_input(latents, t)
            control_on = (i + 1) / float(n_steps) <= inf['control_guidance_end'] + 1e-9
            with autocast_ctx(dev, precision):
                if do_cfg:
                    lat2 = torch.cat([latent_in, latent_in])
                    t2 = torch.cat([t_batch, t_batch])
                    txt2 = torch.cat([uncond, cond])
                    down = mid = None
                    if control_on and inf['guess_mode']:  # CN-L8: residuals only in the conditional branch
                        d_c, m_c = self.controlnet_residuals(latent_in, t_batch, cond, c_spatial, scale)
                        down = [torch.cat([torch.zeros_like(d), d]) for d in d_c]
                        mid = torch.cat([torch.zeros_like(m_c), m_c])
                    elif control_on:
                        down, mid = self.controlnet_residuals(lat2, t2, txt2, torch.cat([c_spatial, c_spatial]), scale)
                    eps_u, eps_c = self.unet_eps(lat2, t2, txt2, down, mid).float().chunk(2)
                    eps = eps_u + cfg_scale * (eps_c - eps_u)
                else:  # CFG 1.0: no unconditional branch (SD-L4)
                    down = mid = None
                    if control_on:
                        down, mid = self.controlnet_residuals(latent_in, t_batch, cond, c_spatial, scale)
                    eps = self.unet_eps(latent_in, t_batch, cond, down, mid).float()
            latents = sched.step(eps, t, latents, **step_kwargs).prev_sample.float()
            if lf['enabled'] and i + 1 < n_steps and (i + 1) / float(n_steps) <= lf['until'] + 1e-9:
                # ILVR: y_{t-1} = x_{t-1} - w * (phi(x_{t-1}) - phi(q(z | t-1))); the reference is x̂'s latent noised with the
                # same noise as the start, in the scheduler's own parametrisation (add_noise matches step's latent space)
                ref = sched.add_noise(z, noise, timesteps[i + 1:i + 2].repeat(bsz)).float()
                latents = latents - lf['weight'] * (lowpass(latents) - lowpass(ref))
        with autocast_ctx(dev, precision):
            images = self.vae.decode(latents / self.vae_scale).sample
        return images.float().clamp(-1.0, 1.0)
