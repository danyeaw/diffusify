import time
from collections.abc import Callable
from pathlib import Path

from diffusify.model import DiffusionModel, save_image


class DiffusifyViewModel:
    """ViewModel that coordinates between the View and Model."""

    def __init__(self):
        self.model = DiffusionModel()

        # Default values
        self.width = 512
        self.height = 512
        self.output_image_path = None

        # Callbacks for updating the View
        self.on_progress_update = None
        self.on_status_update = None
        self.on_image_generated = None
        self.on_operation_complete = None

    def set_callbacks(
        self,
        on_progress_update: Callable[[int], None] | None = None,
        on_status_update: Callable[[str], None] | None = None,
        on_image_generated: Callable[[str], None] | None = None,
        on_operation_complete: Callable[[bool], None] | None = None,
    ):
        """Set callbacks for updating the View.

        Args:
            on_progress_update: Called when progress changes
            on_status_update: Called when status message changes
            on_image_generated: Called when image is generated
            on_operation_complete: Called when operation completes
        """
        self.on_progress_update = on_progress_update
        self.on_status_update = on_status_update
        self.on_image_generated = on_image_generated
        self.on_operation_complete = on_operation_complete

        # Set model callback
        self.model.set_progress_callback(self.handle_progress_update)

    def handle_progress_update(self, percentage: int):
        """Handle progress updates from the model."""
        if self.on_progress_update:
            self.on_progress_update(percentage)

    def update_status(self, message: str):
        """Update status message."""
        if self.on_status_update:
            self.on_status_update(message)

    def validate_generation_parameters(
        self, prompt: str, negative_prompt: str, steps: int, guidance_scale: float
    ):
        """Validate and normalize generation parameters.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            steps: Number of inference steps
            guidance_scale: Guidance scale parameter

        Returns:
            Tuple of (valid_prompt, valid_neg_prompt, valid_steps, valid_guidance)
        """
        # Validate prompt
        valid_prompt = prompt.strip()
        if not valid_prompt:
            valid_prompt = "A beautiful image"
            self.update_status("Warning: Empty prompt, using default.")

        # Validate steps
        valid_steps = max(1, steps)
        if valid_steps != steps:
            self.update_status(f"Warning: Steps must be positive, using {valid_steps}.")

        # Validate guidance scale
        valid_guidance_scale = min(max(1.0, guidance_scale), 15.0)
        if valid_guidance_scale != guidance_scale:
            self.update_status(
                f"Warning: Adjusted guidance scale to {valid_guidance_scale}."
            )

        return valid_prompt, negative_prompt, valid_steps, valid_guidance_scale

    def update_image_size(self, width: int, height: int):
        """Update image dimensions.

        Args:
            width: Image width in pixels (must be positive)
            height: Image height in pixels (must be positive)
        """
        # Validate dimensions
        if width <= 0:
            width = 512
            self.update_status("Warning: Width must be positive, using default (512).")

        if height <= 0:
            height = 512
            self.update_status("Warning: Height must be positive, using default (512).")

        self.width = width
        self.height = height

    async def load_model(self, use_attention_slicing: bool, use_karras: bool) -> bool:
        """Load the diffusion model.

        Args:
            use_attention_slicing: Whether to enable attention slicing
            use_karras: Whether to use Karras scheduler

        Returns:
            Success flag
        """
        self.update_status("Loading model... Please wait.")

        # Signal indeterminate progress during model loading
        if self.on_progress_update:
            self.on_progress_update(-1)  # Special value for indeterminate progress

        # Perform the actual model loading
        success, error = await self.model.load_pipeline(
            use_attention_slicing=use_attention_slicing, use_karras=use_karras
        )

        # Ensure success is a proper boolean
        success_result = bool(success)

        if success_result:
            # Create status message
            status_message_parts = []
            if use_attention_slicing:
                status_message_parts.append("with attention slicing")
            # Safety checker is always on
            status_message_parts.append("with safety checker")

            status_suffix = ""
            if status_message_parts:
                status_suffix = " " + ", ".join(status_message_parts)

            self.update_status(f"Model loaded successfully{status_suffix}.")
        else:
            self.update_status(f"Error loading model: {error}")

        # Signal completion of the loading process
        if self.on_progress_update:
            self.on_progress_update(100)

        if self.on_operation_complete:
            self.on_operation_complete(success_result)

        return success_result

    async def generate_image(
        self, prompt: str, negative_prompt: str, steps: int, guidance_scale: float
    ) -> bool:
        """Generate an image from the given parameters.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            steps: Number of inference steps (must be positive)
            guidance_scale: Guidance scale (recommended range: 1.0-15.0)

        Returns:
            Success flag
        """
        # Update status
        self.update_status("Preparing to generate image...")

        # Load model if needed
        if not self.model.model_loaded:
            success = await self.load_model(True, False)
            if not success:
                return False

        # Validate and normalize parameters
        prompt, negative_prompt, steps, guidance_scale = (
            self.validate_generation_parameters(
                prompt, negative_prompt, steps, guidance_scale
            )
        )

        self.update_status("Generating the image...")
        start_time = time.time()

        # Generate the image - progress will be reported by model callbacks
        output_path, seed, has_nsfw_content, error = await self.model.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=self.width,
            height=self.height,
        )

        end_time = time.time()
        generation_time = round(end_time - start_time, 1)

        if error:
            self.update_status(f"Error generating image: {error}")
            if self.on_operation_complete:
                self.on_operation_complete(False)
            return False

        # Store the generated image path
        self.output_image_path = output_path

        # Update status to show completion
        status_text = f"Image generated in {generation_time}s! (Seed: {seed})"
        if has_nsfw_content:
            status_text += " - NSFW content detected and filtered."

        self.update_status(status_text)

        # Notify view about the new image
        if self.on_image_generated:
            self.on_image_generated(output_path)

        # Signal completion
        if self.on_progress_update:
            self.on_progress_update(100)

        if self.on_operation_complete:
            self.on_operation_complete(True)

        return True

    async def save_image(self, save_path: str) -> bool:
        """Save the generated image to a file.

        Args:
            save_path: Path where to save the image

        Returns:
            Success flag
        """
        # Validate prerequisites
        if not self.output_image_path:
            self.update_status("No image to save.")
            if self.on_operation_complete:
                self.on_operation_complete(False)
            return False

        if not save_path:
            self.update_status("Error: Save path cannot be empty.")
            if self.on_operation_complete:
                self.on_operation_complete(False)
            return False

        # Update status without progress tracking
        self.update_status("Saving image...")

        # Perform the save operation without progress tracking
        error = await save_image(self.output_image_path, save_path)

        # Handle result
        if error:
            self.update_status(f"Error saving image: {error}")
            if self.on_operation_complete:
                self.on_operation_complete(False)
            return False

        # Report success
        file_name = Path(save_path).name
        self.update_status(f"Image saved to {file_name}")

        if self.on_operation_complete:
            self.on_operation_complete(True)
        return True
