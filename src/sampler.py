import torch
from tqdm import tqdm
import numpy as np

def _model_sample_diffusion(model, device, dataloader, total_samples, time, n_T):
    noise = []
    i = 0
    for _, x_cond in dataloader:
        x_cond = x_cond.float().to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        mask= torch.zeros(x_cond.shape[0]).to(device)
        n = model(total_samples[i:i+len(x_cond)], t/n_T, x_cond, mask)
        noise.append(n)
        i = i+len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_diffusion(model,
                dataloader,
                noise_scheduler,
                gt = None,
                num_step=1000,
                device=torch.device('cuda'),
                sample_shape=(7060, 2000),
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                is_classifier_guidance=False,
                omega=0.1):
    model.eval()
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]
    gt = torch.tensor(gt).to(device)
    n_T= num_step
    
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            model_output = _model_sample_diffusion(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_samples=x_t,
                                        time=time,
                                        n_T= n_T)

            if is_classifier_guidance:
                model_output_uncondi = _model_sample_diffusion(model,
                                                           device=device,
                                                           dataloader=dataloader,
                                                           total_samples=x_t,
                                                           time=time,
                                                           n_T=n_T)
                model_output = model_output_uncondi + omega * (model_output - model_output_uncondi)

        x_t, _ = noise_scheduler.step(model_output,
                                         torch.from_numpy(np.array(time)).long().to(device),
                                         x_t,
                                         model_pred_type=model_pred_type)

        if time == 0 and model_pred_type == 'x_start':
            sample = model_output

    return x_t
