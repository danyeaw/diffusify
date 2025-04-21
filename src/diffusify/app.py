"""Diffusify - A simple app to generate images from text using diffusion models."""

import asyncio
import tempfile
from pathlib import Path

import toga
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

MODEL_ID = "Lykon/dreamshaper-xl-1-0"


class DiffusifyApp(toga.App):
    def startup(self):
        """Initialize the application."""
        # Main window setup
        self.main_window = toga.MainWindow(title=self.formal_name, size=(800, 600))
        main_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        content_box = toga.SplitContainer(style=Pack(flex=1))

        # Left column - controls
        self.control_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Prompt input
        self.prompt_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        self.prompt_label = toga.Label("Prompt:", style=Pack(margin_bottom=5))
        self.prompt_box.add(self.prompt_label)
        self.prompt_input = toga.MultilineTextInput(
            placeholder="Enter prompt here...",
            value="a professional photograph of a mountain landscape, high quality",
            style=Pack(margin=5, height=100),
        )
        self.prompt_box.add(self.prompt_input)
        self.control_box.add(self.prompt_box)

        # Negative prompt input
        self.neg_prompt_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        self.neg_prompt_label = toga.Label("Negative Prompt:", style=Pack(margin_bottom=5))
        self.neg_prompt_box.add(self.neg_prompt_label)
        self.neg_prompt_input = toga.MultilineTextInput(
            placeholder="Enter negative prompt here...",
            value="low quality, blurry, distorted, ugly, bad anatomy",
            style=Pack(margin=5, height=60),
        )
        self.neg_prompt_box.add(self.neg_prompt_input)
        self.control_box.add(self.neg_prompt_box)

        # Image size selection
        self.size_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        self.size_label = toga.Label("Image Size:", style=Pack(margin_bottom=5))
        self.size_box.add(self.size_label)

        # Size options box (horizontal)
        size_options_box = toga.Box(style=Pack(direction=ROW))
        self.size_selection = toga.Selection(
            items=["512×512", "768×768", "1024×1024", "768×512", "512×768"],
            style=Pack(margin=5)
        )
        self.size_selection.on_select = self.size_changed
        size_options_box.add(self.size_selection)
        self.size_box.add(size_options_box)
        self.control_box.add(self.size_box)

        # Steps slider
        self.steps_box = toga.Box(style=Pack(direction=ROW, margin_bottom=10))
        self.steps_label = toga.Label("Steps:", style=Pack(margin_right=5, width=80))
        self.steps_box.add(self.steps_label)
        self.steps_slider = toga.Slider(
            min=15,
            max=50,
            value=30,
            style=Pack(margin=5, flex=1),
        )
        self.steps_slider.on_change = self.steps_changed
        self.steps_box.add(self.steps_slider)
        self.steps_value_label = toga.Label("30", style=Pack(width=40))
        self.steps_box.add(self.steps_value_label)
        self.control_box.add(self.steps_box)

        # Guidance scale slider
        self.guidance_box = toga.Box(style=Pack(direction=ROW, margin_bottom=10))
        self.guidance_label = toga.Label("Guidance:", style=Pack(margin_right=5, width=80))
        self.guidance_box.add(self.guidance_label)
        self.guidance_slider = toga.Slider(
            min=1.0,
            max=15.0,
            value=7.5,
            style=Pack(margin=5, flex=1),
        )
        self.guidance_slider.on_change = self.guidance_changed
        self.guidance_box.add(self.guidance_slider)
        self.guidance_value_label = toga.Label("7.5", style=Pack(width=40))
        self.guidance_box.add(self.guidance_value_label)
        self.control_box.add(self.guidance_box)

        # Attention slicing switch
        self.attention_slicing_switch = toga.Switch(
            "Enable attention slicing",
            value=True,
            style=Pack(margin=5)
        )
        self.control_box.add(self.attention_slicing_switch)

        # Karras scheduler option
        self.karras_switch = toga.Switch(
            "Use Karras scheduler",
            style=Pack(margin=5)
        )
        self.control_box.add(self.karras_switch)

        # Generate button
        self.generate_button = toga.Button(
            "Generate Image",
            style=Pack(margin=10)
        )
        self.generate_button.on_press = self.generate_image
        self.control_box.add(self.generate_button)

        # Right column - image display
        self.display_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Output image display
        self.output_image_label = toga.Label(
            "Generated Image", style=Pack(margin_bottom=5, text_align="center")
        )
        self.display_box.add(self.output_image_label)
        self.output_image_view = toga.ImageView(style=Pack(width=512, height=512))
        self.display_box.add(self.output_image_view)

        # Save button for output
        self.save_button = toga.Button(
            "Save Image",
            style=Pack(margin_top=10)
        )
        self.save_button.on_press = self.save_output_image
        self.save_button.enabled = False
        self.display_box.add(self.save_button)

        # Add content to the SplitContainer
        content_box.content = [self.control_box, self.display_box]

        # Status label at the bottom
        self.status_label = toga.Label(
            "Ready to generate images.",
            style=Pack(margin=10, text_align="center"),
        )

        # Assemble final layout
        main_box.add(content_box)
        main_box.add(self.status_label)
        self.main_window.content = main_box

        # Initialize other variables
        self.output_image = None
        self.pipeline = None
        self.model_loaded = False
        self.width = 512
        self.height = 512

        # Show the main window
        self.main_window.show()

    def steps_changed(self, widget):
        """Update the steps value label when the slider changes."""
        value = int(self.steps_slider.value)
        self.steps_value_label.text = str(value)

    def guidance_changed(self, widget):
        """Update the guidance scale value label when the slider changes."""
        value = round(self.guidance_slider.value, 1)
        self.guidance_value_label.text = str(value)

    def size_changed(self, widget):
        """Update width and height values when size selection changes."""
        size_text = self.size_selection.value
        width, height = map(int, size_text.split("×"))
        self.width = width
        self.height = height
        # Update image view size
        self.output_image_view.style.update(width=min(512, width), height=min(512, height))

    async def load_pipeline(self):
        """Initialize the diffusion pipeline asynchronously."""
        self.status_label.text = "Loading model... Please wait."

        try:
            # Define the work to be done in the executor
            def _load_pipeline_sync():
                # Use the text-to-image pipeline instead of img2img
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                    use_safetensors=True,
                )

                # Enable attention slicing if the option is selected
                if self.attention_slicing_switch.value:
                    pipe.enable_attention_slicing()

                # Apply Karras scheduler if requested
                if self.karras_switch.value:
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config, use_karras_sigmas=True
                    )

                # Move to appropriate device
                if torch.cuda.is_available():
                    return pipe.to("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return pipe.to("mps")  # Apple Silicon
                return pipe

            # Run the pipeline loading in a thread executor
            self.pipeline = await asyncio.get_event_loop().run_in_executor(None, _load_pipeline_sync)
            self.model_loaded = True

            # Update status text in the main thread after the pipeline is loaded
            slicing_status = "with attention slicing" if self.attention_slicing_switch.value else "without attention slicing"
            self.status_label.text = f"Model loaded successfully {slicing_status}."

            return True
        except Exception as e:
            self.status_label.text = f"Error loading model: {str(e)}"
            return False

    async def generate_image(self, widget):
        """Generate an image from text prompt."""
        # Disable the button during processing
        self.generate_button.enabled = False
        self.save_button.enabled = False

        # Load model if needed
        if not self.model_loaded:
            model_loaded = await self.load_pipeline()
            if not model_loaded:
                self.generate_button.enabled = True
                return

        self.status_label.text = "Generating image..."

        # Get parameters
        prompt = self.prompt_input.value
        negative_prompt = self.neg_prompt_input.value
        steps = int(self.steps_slider.value)
        guidance_scale = float(self.guidance_slider.value)

        try:
            # Define the generation work to be done in the executor
            def _generate_image_sync():
                # Generate a seed for reproducibility
                seed = torch.randint(0, 2147483647, (1,)).item()
                generator = torch.Generator().manual_seed(seed)

                # Generate the image
                pipeline_output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    width=self.width,
                    height=self.height,
                )

                output_image = pipeline_output.images[0]

                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    output_image.save(temp_file.name)
                    return temp_file.name, seed

            # Run the generation in a thread executor
            output_path, seed = await asyncio.get_event_loop().run_in_executor(None, _generate_image_sync)

            # Update UI with the result
            self.output_image = toga.Image(output_path)
            self.output_image_view.image = self.output_image
            self.save_button.enabled = True
            self.status_label.text = f"Image generated successfully! (Seed: {seed})"
        except Exception as e:
            self.status_label.text = f"Error generating image: {str(e)}"
        finally:
            # Always re-enable the button
            self.generate_button.enabled = True

    async def save_output_image(self, widget):
        """Save the generated image to a file."""
        if not self.output_image:
            return

        # Open a save file dialog
        save_dialog = toga.SaveFileDialog(
            title="Save Generated Image",
            filename="diffusify_output.png",
            file_types=["png"],
        )

        save_path = await self.main_window.dialog(save_dialog)

        if save_path:
            try:
                # Define the file saving function
                def _save_file_sync():
                    with open(self.output_image.path, "rb") as src_file:
                        with open(save_path, "wb") as dst_file:
                            dst_file.write(src_file.read())

                # Run the file saving in a thread executor
                await asyncio.get_event_loop().run_in_executor(None, _save_file_sync)
                self.status_label.text = f"Image saved to {Path(save_path).name}"
            except Exception as e:
                self.status_label.text = f"Error saving image: {str(e)}"


def main():
    """Entry point for the application."""
    return DiffusifyApp("Diffusify", "com.example.diffusify")


if __name__ == "__main__":
    app = main()
    app.main_loop()
