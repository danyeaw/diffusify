import asyncio
import functools
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from transformers import CLIPImageProcessor

MODEL_ID = "Lykon/dreamshaper-xl-1-0"


class DiffusionModel:
    """Model class handling the AI image generation pipeline."""

    def __init__(self):
        self.pipeline = None
        self.safety_checker = None
        self.feature_extractor = None
        self.model_loaded = False
        self.total_steps = 0
        self._progress_callback = None
        self._main_loop = None

    def set_progress_callback(self, callback: Callable[[int], None] | None):
        """Set a callback function for progress updates.

        Args:
            callback: Function that receives the progress percentage
        """
        self._progress_callback = callback
        # Store the main event loop when setting up the callback
        self._main_loop = asyncio.get_event_loop()

    def _progress_callback_wrapper(self, pipe, step_index, timestep, callback_kwargs):
        """Wrapper for the pipeline's callback on step end."""
        # StableDiffusionXL pipeline uses 0-based indexing for steps
        current_step = step_index + 1
        percentage = int((current_step / self.total_steps) * 100)

        # Call the external progress callback if set
        if self._progress_callback and self._main_loop:
            # Schedule the callback to run in the main event loop
            self._main_loop.call_soon_threadsafe(
                functools.partial(self._progress_callback, percentage)
            )

        return callback_kwargs

    async def load_pipeline(
        self, use_attention_slicing: bool = True, use_karras: bool = False
    ) -> tuple[bool, str | None]:
        """Initialize the diffusion pipeline asynchronously."""
        try:
            # Define work to be done in executor
            def _load_pipeline_sync():
                # Load safety checker and feature extractor
                self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker"
                )
                self.feature_extractor = CLIPImageProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )

                # Move safety checker to appropriate device
                if torch.cuda.is_available():
                    self.safety_checker = self.safety_checker.to("cuda")
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.safety_checker = self.safety_checker.to("mps")

                # Load SDXL pipeline
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16
                    if torch.cuda.is_available()
                    else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                    use_safetensors=True,
                )

                # Enable attention slicing if requested
                if use_attention_slicing:
                    pipe.enable_attention_slicing()

                # Apply Karras scheduler if requested
                if use_karras:
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config,
                        use_karras_sigmas=True,
                        algorithm_type="sde-dpmsolver++",
                    )

                # Move to appropriate device
                if torch.cuda.is_available():
                    return pipe.to("cuda"), None
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    return pipe.to("mps"), None  # Apple Silicon
                return pipe, None

            # Run the pipeline loading in a thread executor
            loop = asyncio.get_event_loop()
            pipe_result, error = await loop.run_in_executor(None, _load_pipeline_sync)

            if error:
                return False, str(error)

            self.pipeline = pipe_result
            self.model_loaded = True
            return True, None
        except Exception as e:
            return False, str(e)

    def apply_safety_checker(self, image: Image) -> tuple[Image, bool]:
        """Apply safety checker to the generated image."""
        if not self.safety_checker or not self.feature_extractor:
            return image, False

        # Make sure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Extract image features
        safety_checker_input = self.feature_extractor([image], return_tensors="pt")

        # Move to the same device as the safety checker
        if torch.cuda.is_available():
            safety_checker_input = safety_checker_input.to("cuda")
            device = "cuda"
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            safety_checker_input = safety_checker_input.to("mps")
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        # Convert image to numpy array
        image_np = np.array(image).astype(np.float32) / 255.0

        # Safety checker expects these dimensions: [batch_size, channels, height, width]
        image_np = np.transpose(image_np, (2, 0, 1))  # [C, H, W]
        image_np = np.expand_dims(image_np, axis=0)  # [1, C, H, W]

        # Convert to torch tensor
        image_tensor = torch.from_numpy(image_np).to(device=device, dtype=dtype)

        # Run safety checker
        result, has_nsfw_concept = self.safety_checker(
            images=image_tensor, clip_input=safety_checker_input.pixel_values.to(dtype)
        )

        # Convert result back to PIL Image
        if has_nsfw_concept[0]:
            # If NSFW content detected, create a black image of the same size
            filtered_image = Image.new("RGB", image.size, (0, 0, 0))
            return filtered_image, True
        else:
            # No NSFW content, return original
            return image, False

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
    ) -> tuple[str | None, int | None, bool, str | None]:
        """Generate an image from text prompt."""
        if not self.model_loaded:
            return None, None, False, "Model not loaded"

        # Store the total steps for progress calculation
        self.total_steps = steps

        try:
            # Report initial progress
            if self._progress_callback:
                self._progress_callback(0)

            # Define the generation work to be done in the executor
            def _generate_image_sync():
                try:
                    # Generate a seed for reproducibility
                    seed = torch.randint(0, 2147483647, (1,)).item()
                    generator = torch.Generator().manual_seed(seed)

                    if self.pipeline:
                        # Disable the built-in progress bar
                        self.pipeline.set_progress_bar_config(disable=True)

                        # Generate the image using the callback method
                        pipeline_output = self.pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=steps,
                            generator=generator,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            callback_on_step_end=self._progress_callback_wrapper,
                            callback_steps=1,  # Call after every step
                        )

                        output_image = pipeline_output.images[0]

                        # Apply safety checker
                        output_image, has_nsfw_content = self.apply_safety_checker(
                            output_image
                        )

                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as temp_file:
                        output_image.save(temp_file.name)
                        return temp_file.name, seed, has_nsfw_content, None
                except Exception as e:
                    return None, None, False, str(e)

            # Run the generation in a thread executor
            loop = asyncio.get_event_loop()
            output_path, seed, has_nsfw_content, error = await loop.run_in_executor(
                None, _generate_image_sync
            )

            if error:
                return None, None, False, error

            return output_path, seed, has_nsfw_content, None
        except Exception as e:
            return None, None, False, str(e)


def _save_file_sync(source_path: Path, save_path: Path) -> str | None:
    try:
        with open(source_path, "rb") as src_file:
            with open(save_path, "wb") as dst_file:
                dst_file.write(src_file.read())
        return None
    except Exception as e:
        return str(e)


async def save_image(source_path: Path, save_path: Path) -> str | None:
    """Save the image to a file."""
    try:
        # Run the file saving in a thread executor
        loop = asyncio.get_event_loop()
        error: str | None = await loop.run_in_executor(
            None, _save_file_sync, source_path, save_path
        )
        return error
    except Exception as e:
        return str(e)
