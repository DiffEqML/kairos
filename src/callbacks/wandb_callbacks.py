import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize as NormalizeCmap

import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import PIL

from src.utils.numerics import idct2d


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder
            # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(str(git_dir_path))  # noqa: W503
                    # ignore files ignored by git
                    and (  # noqa: W503
                        subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 1
                    )
                ):
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(targets, preds, average=None)
            r = recall_score(targets, preds, average=None)
            p = precision_score(targets, preds, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()



class LogReconstructions(Callback):
    """Logs a validation batch and their reconstructions
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            step = experiment.step

            # get a validation batch from the validation dat loader
            val_imgs, val_targets = next(iter(trainer.datamodule.val_dataloader()))

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            val_targets = val_targets.to(device=pl_module.device)

            val_recs = pl_module(val_imgs)

            for channel in range(val_recs.shape[1]):
                inp_imgs = [wandb.Image(val_imgs[: self.num_samples, channel], 
                            caption=f"Input, channel: {channel}")
                ]
                targ_imgs = [wandb.Image(val_targets[: self.num_samples, channel], 
                            caption=f"Target, channel: {channel}")
                ]
                rec_imgs = [wandb.Image(val_recs[: self.num_samples, channel], 
                            caption=f"Recs, channel: {channel}")
                ]

                experiment.log(
                    {
                        f"Images/{experiment.name}_channel": 
                            [*inp_imgs, *targ_imgs, *rec_imgs]                       
                    }, 
                )

class LogReconstructionMetrics(Callback):
    """Logs a validation batch and their reconstructions
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, forward_transform_on_preds=None):
        super().__init__()
        self.ready = True
        self.forward_transform_on_preds = forward_transform_on_preds

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            step = experiment.step

            # get a validation batch from the validation dat loader
            val_imgs, val_targets = next(iter(trainer.datamodule.val_dataloader()))

            # keep only a single image to avoid OOM
            val_imgs = val_imgs[:1]

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            # val_targets = val_targets.to(device=pl_module.device)

            val_recs = pl_module(val_imgs).detach().cpu()
            
            if self.forward_transform_on_preds == "dct":
                val_recs = idct2d(val_recs, norm="ortho")

            val_imgs = val_imgs.detach().cpu()
            val_targets = val_targets.detach().cpu()

            vx_in, vy_in, mask = val_imgs[0][0,:,:], val_imgs[0][1,:,:], val_imgs[0][2,:,:]
            vx_r, vy_r, p_r = val_recs[0][0,:,:], val_recs[0][1,:,:], val_recs[0][2,:,:]
            vx_t, vy_t, p_t = val_targets[0][0,:,:], val_targets[0][1,:,:], val_targets[0][2,:,:]

            fig, axs = plt.subplots(3, 4, figsize=(12, 9))

            # COL 1: Input velocities and mask
            axs[0,0].imshow(vx_in, cmap='magma')
            axs[1,0].imshow(vy_in, cmap='magma')
            axs[2,0].imshow(mask, cmap='magma')
            axs[0,0].set_title("v_x (input)") ; axs[1,0].set_title("v_y (input)") ; axs[2,0].set_title("Mask (input)")

            # COL 2: Target velocities and pressure
            axs[0,1].imshow(vx_t, cmap='magma')
            axs[1,1].imshow(vy_t, cmap='magma')
            axs[2,1].imshow(p_t, cmap='magma')
            axs[0,1].set_title("v_x (target)") ; axs[1,1].set_title("v_y (target)") ; axs[2,1].set_title("Pressure (target)")

            # COL 3: Reconstructed velocities and pressure
            axs[0,2].imshow(vx_r, cmap='magma')
            axs[1,2].imshow(vy_r, cmap='magma')
            axs[2,2].imshow(p_r, cmap='magma')
            axs[0,2].set_title("v_x (recon)") ; axs[1,2].set_title("v_y (recon)") ; axs[2,2].set_title("Pressure (recon)")

            # COL 4: Error
            err_vx = axs[0,3].imshow(abs(vx_t - vx_r), cmap='magma')
            err_vy = axs[1,3].imshow(abs(vy_t - vy_r), cmap='magma')
            err_vp = axs[2,3].imshow(abs(p_t - p_r), cmap='magma')
            axs[0,3].set_title("v_x (error)") ; axs[1,3].set_title("v_y (error)") ; axs[2,3].set_title("Pressure (error)")
            div_vx = make_axes_locatable(axs[0,3]) ; cax_vx = div_vx.append_axes("right", size="5%", pad=0.05)
            div_vy = make_axes_locatable(axs[1,3]) ; cax_vy = div_vy.append_axes("right", size="5%", pad=0.05)
            div_p = make_axes_locatable(axs[2,3]) ; cax_p = div_p.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(err_vx, cax=cax_vx)
            plt.colorbar(err_vy, cax=cax_vy)
            plt.colorbar(err_vp, cax=cax_p)

            plt.tight_layout()
        
            experiment.log(
                {
                    f"rec_metric/{experiment.name}": 
                        wandb.Image(fig)          
                }, 
            )


class LogReconstructionsScalarFlow(Callback):
    """Logs a validation batch and their reconstructions
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY

       # TODO: add features for rollout, different context steps
    """

    def __init__(self, num_samples: int = 4, channels: int = 1, is_normalized: bool = True):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.channels = channels
        self.is_normalized = is_normalized

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

        # # Debug logging: uncomment the following to log the first step
        self.on_validation_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            step = experiment.step

            # get a validation batch from the validation dat loader
            val_imgs, val_targets = next(iter(trainer.datamodule.val_dataloader()))

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            val_targets = val_targets.to(device=pl_module.device)
            _, val_recs, _ = pl_module.step((val_imgs, val_targets), is_train=False) # use step for automatic rollout
            val_recs = val_recs.detach().cpu()
            val_imgs = val_imgs.detach().cpu()
            val_targets = val_targets.detach().cpu()

            # Detect if we are doing a rollout
            # NOTE: may need to change for more channels
            if val_targets.ndim == 3: rollout = False
            else: rollout = True

            if not rollout:
                one_step_reconstruction = self._plot_reconstructions(val_imgs[:, -1], val_targets, val_recs)
                one_step_residuals = self._plot_residuals(val_imgs[:, -1], val_targets, val_recs)
            else: 
                one_step_reconstruction = self._plot_reconstructions(val_imgs[:, -1], val_targets[:, 0], val_recs[:, 0])
                one_step_residuals = self._plot_residuals(val_imgs[:, -1], val_targets[:, 0], val_recs[:, 0])
                # Slide window to last elements
                last_step_reconstruction = self._plot_reconstructions(val_imgs[:, -1], val_targets[:, -1], val_recs[:, -1])
                last_step_residuals = self._plot_residuals(val_imgs[:, -1], val_targets[:, -1], val_recs[:, -1])
                error_propagation = self._plot_error_propagation(val_recs, val_targets)
    
            logs = {
                    f"Predictions/{experiment.name}_one_step": 
                        wandb.Image(one_step_reconstruction),
                    f"Residuals/{experiment.name}_one_step": 
                        wandb.Image(one_step_residuals)                              
            }
            

            # Add extra logs if we have a rollout ( | operation merges dicts) (only Py3.9+)
            # better to use legacy method for compatibility across environments
            if rollout: 
                logs.update({ 
                    f"Predictions/{experiment.name}_last_step": 
                        wandb.Image(last_step_reconstruction),
                    f"Residuals/{experiment.name}_last_step": 
                        wandb.Image(last_step_residuals),
                    f"Errors/{experiment.name}_mae_propagation": 
                        wandb.Image(error_propagation)       
                })

            experiment.log(
                logs, 
            )
            
    def _plot_reconstructions(self, val_imgs, val_targets, val_recs):
            fig, axs = plt.subplots(self.num_samples, 3, figsize=(12, 6*self.num_samples))
            if axs.ndim == 1: axs = axs[None] # expand dimension
        
            cmaps = ['inferno', 'inferno', 'inferno']
            titles = ['Input', 'Target', 'Prediction']

            if self.is_normalized: vmin = -0.1; vmax = 0.7
            else:  vmin = -0.4; vmax = 3
            norm = NormalizeCmap(vmin=vmin, vmax=vmax) # we use from -0.1 to 0 to detect if the model is learning "negative" densities
            
            # Transpose for iterating over first dim
            for sample_idx, axs_ in enumerate(axs): #
                
                img0 = val_imgs[sample_idx]
                img1 = val_targets[sample_idx]
                img2 = val_recs[sample_idx]

                if img0.squeeze().shape[0] > 1.2*img0.squeeze().shape[1]: aspect_plot = 1
                else: aspect_plot = 2.333

                imgs = [img0, img1, img2]

                for i, (img, ax, cmap, title) in enumerate(zip(imgs, axs_, cmaps, titles)):
                    img_ = ax.imshow(img.squeeze(), origin='lower', interpolation='nearest', cmap=cmap, norm=norm)
                    ax.set_title(title)
                    ax.set_aspect(aspect_plot)
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.tight_layout()
            return fig

    def _plot_residuals(self, val_imgs, val_targets, val_recs):
            fig, axs = plt.subplots(self.num_samples, 3, figsize=(12, 6*self.num_samples))
            if axs.ndim == 1: axs = axs[None] # expand dimension

            cmaps = ['seismic', 'seismic', 'seismic']
            titles = ['Target Residual', 'Predicted Residual', f'Difference $\hat y - y$']

            if self.is_normalized: vmin = -0.6; vmax = 0.6
            else:  vmin = -3; vmax = 3
            norm = NormalizeCmap(vmin=vmin, vmax=vmax)
            
            for sample_idx, axs_ in enumerate(axs): #            
                img = val_imgs[sample_idx].squeeze()
                target = val_targets[sample_idx].squeeze()
                rec = val_recs[sample_idx].squeeze()

                target_res = target - img
                rec_res = rec - img

                if img.squeeze().shape[0] > 1.2*img.squeeze().shape[1]: aspect_plot = 1
                else: aspect_plot = 2.333

                imgs = [target_res, rec_res, rec_res - target_res]

                for i, (img, ax, cmap, title) in enumerate(zip(imgs, axs_, cmaps, titles)):
                    img_ = ax.imshow(img, origin='lower', interpolation='nearest', cmap=cmap, norm=norm)
                    div = make_axes_locatable(ax) ; cax = div.append_axes("right", size="2%", pad=-1.45)
                    plt.colorbar(img_, cax=cax) 
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect(aspect_plot)

            fig.tight_layout()    
            return fig

    def _plot_error_propagation(self, y_hat, y):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        steps = torch.arange(1, y.shape[1]+1, 1)
        means = (abs(y_hat - y)).mean(dim=[0,2,3])
        stds = (abs(y_hat - y)).std(dim=[0,2,3])

        ax.plot(steps, means)
        ax.fill_between(steps, means-stds, means+stds, alpha=0.1)
        ax.set_ylim(0)

        ax.set_xlabel('Step number')
        ax.set_ylabel('MAE')
        ax.set_title('Error propagation over time steps')
        ax.yaxis.grid(True)
        fig.tight_layout()    
        return fig
