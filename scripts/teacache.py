import torch
import numpy as np
from torch import Tensor
import gradio as gr
from modules import scripts
from modules.ui_components import InputAccordion
from backend.nn.flux import IntegratedFluxTransformer2DModel, timestep_embedding


class TeaCache(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enable_teacache = False
        self.rel_l1_thresh = 0.4
        self.steps = 25
        self.last_input_shape = None  # Record the last input dimensions

    def title(self):
        return "TeaCache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title(), elem_id="extensions-teacache") as teacache_enabled:
            with gr.Row():
                enable_teacache_checkbox = gr.Checkbox(
                    label="Enable TeaCache",
                    value=self.enable_teacache,
                    tooltip="Enable TeaCache to speed up inference by caching intermediate results."
                )
            with gr.Row(visible=False) as teacache_settings:  # Hide settings by default
                rel_l1_thresh_slider = gr.Slider(
                    label="Relative L1 Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=self.rel_l1_thresh,
                    tooltip="Threshold for caching intermediate results. Lower values cache more aggressively."
                )
                steps_slider = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=self.steps,
                    tooltip="Number of steps to cache intermediate results."
                )
            with gr.Row():
                clear_cache_button = gr.Button("Clear Residual Cache", variant="secondary")
            with gr.Row():
                gr.Markdown("**Note**: Clear residual cache when changing image size or disabling TeaCache.")

        # Define a function to toggle the visibility of TeaCache settings
        def toggle_teacache_settings(enable_teacache):
            return {teacache_settings: gr.update(visible=enable_teacache)}

        # Bind the checkbox change event to toggle settings visibility
        enable_teacache_checkbox.change(
            fn=toggle_teacache_settings,
            inputs=enable_teacache_checkbox,
            outputs=teacache_settings
        )

        # Define the function to clear residual cache
        def clear_residual_cache_ui():
            self.clear_residual_cache()
            return "Residual cache cleared."

        # Bind the button click event
        clear_cache_button.click(
            fn=clear_residual_cache_ui,
            outputs=None
        )

        # Return UI components
        return [teacache_enabled, enable_teacache_checkbox, rel_l1_thresh_slider, steps_slider]

    def clear_residual_cache(self):
        """Clear residual cache and free GPU memory."""
        if hasattr(IntegratedFluxTransformer2DModel, "previous_residual"):
            setattr(IntegratedFluxTransformer2DModel, "previous_residual", None)
        if hasattr(IntegratedFluxTransformer2DModel, "previous_modulated_input"):
            setattr(IntegratedFluxTransformer2DModel, "previous_modulated_input", None)
        if hasattr(IntegratedFluxTransformer2DModel, "cnt"):
            setattr(IntegratedFluxTransformer2DModel, "cnt", 0)
        if hasattr(IntegratedFluxTransformer2DModel, "accumulated_rel_l1_distance"):
            setattr(IntegratedFluxTransformer2DModel, "accumulated_rel_l1_distance", 0)
        # Free GPU memory
        torch.cuda.empty_cache()
        print("Residual cache cleared and GPU memory freed.")

    def process(self, p, teacache_enabled, enable_teacache_checkbox, rel_l1_thresh_slider, steps_slider):
        # Get the current input dimensions
        current_input_shape = (p.width, p.height)
        if self.last_input_shape is not None and current_input_shape != self.last_input_shape:
            # If input dimensions change, clear residual cache
            self.clear_residual_cache()
        self.last_input_shape = current_input_shape

        # Debug information only when TeaCache is enabled
        if teacache_enabled:
            print("TeaCache enabled:", teacache_enabled)
            print("Enable TeaCache checkbox:", enable_teacache_checkbox)
            print("Relative L1 Threshold:", rel_l1_thresh_slider)
            print("Steps:", steps_slider)

        # If TeaCache is enabled, add parameters to generation parameters
        if teacache_enabled:
            p.extra_generation_params.update({
                "enable_teacache": enable_teacache_checkbox,
                "rel_l1_thresh": rel_l1_thresh_slider,
                "steps": steps_slider,
            })

            # Dynamically modify class attributes of IntegratedFluxTransformer2DModel
            setattr(IntegratedFluxTransformer2DModel, "enable_teacache", enable_teacache_checkbox)
            setattr(IntegratedFluxTransformer2DModel, "cnt", 0)
            setattr(IntegratedFluxTransformer2DModel, "rel_l1_thresh", rel_l1_thresh_slider)
            setattr(IntegratedFluxTransformer2DModel, "steps", steps_slider)
            setattr(IntegratedFluxTransformer2DModel, "accumulated_rel_l1_distance", 0)
            setattr(IntegratedFluxTransformer2DModel, "previous_modulated_input", None)
            setattr(IntegratedFluxTransformer2DModel, "previous_residual", None)

            # Replace the original inner_forward method
            if hasattr(IntegratedFluxTransformer2DModel, "inner_forward"):
                # Save the original inner_forward method
                setattr(IntegratedFluxTransformer2DModel, "original_inner_forward", IntegratedFluxTransformer2DModel.inner_forward)
                # Replace with the new inner_forward method
                IntegratedFluxTransformer2DModel.inner_forward = patched_inner_forward

            # Replace the original forward method
            if not hasattr(IntegratedFluxTransformer2DModel, "original_forward"):
                IntegratedFluxTransformer2DModel.original_forward = IntegratedFluxTransformer2DModel.forward
                IntegratedFluxTransformer2DModel.forward = patched_forward
        else:
            # If TeaCache is disabled, restore the original inner_forward method
            if hasattr(IntegratedFluxTransformer2DModel, "original_inner_forward"):
                IntegratedFluxTransformer2DModel.inner_forward = IntegratedFluxTransformer2DModel.original_inner_forward
                # Clear residual cache and reset TeaCache attributes
                self.clear_residual_cache()
                setattr(IntegratedFluxTransformer2DModel, "enable_teacache", False)
                setattr(IntegratedFluxTransformer2DModel, "cnt", 0)
                setattr(IntegratedFluxTransformer2DModel, "rel_l1_thresh", 0.4)
                setattr(IntegratedFluxTransformer2DModel, "steps", 25)
                setattr(IntegratedFluxTransformer2DModel, "accumulated_rel_l1_distance", 0)
                setattr(IntegratedFluxTransformer2DModel, "previous_modulated_input", None)
                setattr(IntegratedFluxTransformer2DModel, "previous_residual", None)
                print("TeaCache fully disabled and cache cleared.")

            # Restore the original forward method
            if hasattr(IntegratedFluxTransformer2DModel, "original_forward"):
                IntegratedFluxTransformer2DModel.forward = IntegratedFluxTransformer2DModel.original_forward


def patched_inner_forward(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
    # Print "TeaCache is enabled!" only once per generation
    if self.enable_teacache:
        if not hasattr(self, "_teacache_enabled_printed"):  # Check if the message has been printed
            print("TeaCache is enabled!")
            self._teacache_enabled_printed = True  # Set flag to avoid repeated printing

    # If TeaCache is not enabled, call the original method
    if not hasattr(self, "enable_teacache") or not self.enable_teacache:
        return self.original_inner_forward(img, img_ids, txt, txt_ids, timesteps, y, guidance)

    # Get parameters from UI
    rel_l1_thresh = getattr(self, "rel_l1_thresh", 0.4)
    steps = getattr(self, "steps", 25)

    # TeaCache logic
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # Image and text embedding
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))

    # If guidance_embed is enabled, add guidance information
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    # Merge image and text IDs
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    # TeaCache logic
    inp = img.clone()
    vec_ = vec.clone()
    modulated_inp = img.clone()  # Use img.clone() directly as modulated_inp

    # Check if previous_modulated_input exists and has the correct shape
    if hasattr(self, "previous_modulated_input") and self.previous_modulated_input is not None:
        if self.previous_modulated_input.shape != modulated_inp.shape:
            # Clear cache if shapes don't match
            self.previous_modulated_input = None
            self.previous_residual = None
            print("  Cleared cache due to shape mismatch.")

    if self.cnt == 0 or self.cnt == steps - 1:
        should_calc = True
        self.accumulated_rel_l1_distance = 0
    else:
        if hasattr(self, "previous_modulated_input") and self.previous_modulated_input is not None:
            coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item()
            )
            if self.accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        else:
            should_calc = True

    self.previous_modulated_input = modulated_inp
    self.cnt += 1
    if self.cnt == steps:
        self.cnt = 0

    if not should_calc:
        if hasattr(self, "previous_residual") and self.previous_residual is not None:
            img += self.previous_residual
    else:
        ori_img = img.clone()
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]
        self.previous_residual = img - ori_img

    # Final output
    img = self.final_layer(img, vec)
    return img


def patched_forward(self, x, timestep, context, y, guidance=None, **kwargs):
    # Set TeaCache parameters if provided and enabled
    if hasattr(self, "enable_teacache") and kwargs.get("enable_teacache", False):
        self.enable_teacache = kwargs.get("enable_teacache", self.enable_teacache)
        self.rel_l1_thresh = kwargs.get("rel_l1_thresh", self.rel_l1_thresh)
        self.steps = kwargs.get("steps", self.steps)

    # Call the original forward method
    return self.original_forward(x, timestep, context, y, guidance, **kwargs)