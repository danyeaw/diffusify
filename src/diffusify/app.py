#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diffusify - A simple app to transform images using diffusion models
"""

import asyncio
import tempfile
from pathlib import Path

import toga
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,  # Use specific pipeline instead of Auto
    DPMSolverMultistepScheduler,
)
from PIL import Image
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

MODEL_ID = "Lykon/dreamshaper-xl-1-0"


class DiffusifyApp(toga.App):
    def startup(self):
        """Initialize the application."""
        # Main window setup and UI components (unchanged)
        self.main_window = toga.MainWindow(title=self.formal_name, size=(800, 600))
        main_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        content_box = toga.SplitContainer(style=Pack(flex=1))

        # Left column - controls
        self.control_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Image selection
        self.image_selection_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        self.image_label = toga.Label("Input Image:", style=Pack(margin_bottom=5))
        self.image_selection_box.add(self.image_label)
        self.select_file_button = toga.Button(
            "Select Image", on_press=self.select_image, style=Pack(margin=5)
        )
        self.image_selection_box.add(self.select_file_button)
        self.image_path_label = toga.Label("No image selected", style=Pack(margin=5))
        self.image_selection_box.add(self.image_path_label)
        self.control_box.add(self.image_selection_box)

        # Prompt input
        self.prompt_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        self.prompt_label = toga.Label("Prompt:", style=Pack(margin_bottom=5))
        self.prompt_box.add(self.prompt_label)
        self.prompt_input = toga.MultilineTextInput(
            placeholder="Enter prompt here...",
            value="a professional photograph, high quality",
            style=Pack(margin=5, height=60),
        )
        self.prompt_box.add(self.prompt_input)
        self.control_box.add(self.prompt_box)

        # Strength slider
        self.strength_box = toga.Box(style=Pack(direction=ROW, margin_bottom=10))
        self.strength_label = toga.Label("Strength:", style=Pack(margin_right=5, width=80))
        self.strength_box.add(self.strength_label)
        self.strength_slider = toga.Slider(
            min=0.1,
            max=1.0,
            value=0.5,
            on_change=self.strength_changed,
            style=Pack(margin=5, flex=1),
        )
        self.strength_box.add(self.strength_slider)
        self.strength_value_label = toga.Label("0.5", style=Pack(width=40))
        self.strength_box.add(self.strength_value_label)
        self.control_box.add(self.strength_box)

        # Steps slider
        self.steps_box = toga.Box(style=Pack(direction=ROW, margin_bottom=10))
        self.steps_label = toga.Label("Steps:", style=Pack(margin_right=5, width=80))
        self.steps_box.add(self.steps_label)
        self.steps_slider = toga.Slider(
            min=15,
            max=40,
            value=25,
            on_change=self.steps_changed,
            style=Pack(margin=5, flex=1),
        )
        self.steps_box.add(self.steps_slider)
        self.steps_value_label = toga.Label("25", style=Pack(width=40))
        self.steps_box.add(self.steps_value_label)
        self.control_box.add(self.steps_box)

        # Karras scheduler option
        self.karras_switch = toga.Switch("Use Karras scheduler", style=Pack(margin=5))
        self.control_box.add(self.karras_switch)

        # Process button
        self.process_button = toga.Button(
            "Convert Image", on_press=self.process_image, style=Pack(margin=10)
        )
        self.process_button.enabled = False
        self.control_box.add(self.process_button)

        # Right column - image display
        self.display_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Input image display
        self.input_image_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        self.input_image_label = toga.Label(
            "Input Image", style=Pack(margin_bottom=5, text_align="center")
        )
        self.input_image_box.add(self.input_image_label)
        self.input_image_view = toga.ImageView(style=Pack(width=300, height=300))
        self.input_image_box.add(self.input_image_view)
        self.display_box.add(self.input_image_box)

        # Output image display
        self.output_image_box = toga.Box(style=Pack(direction=COLUMN, margin_top=10))
        self.output_image_label = toga.Label(
            "Output Image", style=Pack(margin_bottom=5, text_align="center")
        )
        self.output_image_box.add(self.output_image_label)
        self.output_image_view = toga.ImageView(style=Pack(width=300, height=300))
        self.output_image_box.add(self.output_image_view)

        # Save button for output
        self.save_button = toga.Button(
            "Save Output Image", on_press=self.save_output_image, style=Pack(margin=5)
        )
        self.save_button.enabled = False
        self.output_image_box.add(self.save_button)
        self.display_box.add(self.output_image_box)

        # Add content to the SplitContainer
        content_box.content = [self.control_box, self.display_box]

        # Status label at the bottom
        self.status_label = toga.Label(
            "Ready. Select an image to begin.", style=Pack(margin=10, text_align="center")
        )

        # Assemble final layout
        main_box.add(content_box)
        main_box.add(self.status_label)
        self.main_window.content = main_box

        # Initialize other variables
        self.input_image_path = None
        self.output_image = None
        self.pipeline = None
        self.model_loaded = False

        # Show the main window
        self.main_window.show()

    def strength_changed(self, widget):
        """Update the strength value label when the slider changes."""
        value = round(self.strength_slider.value, 2)
        self.strength_value_label.text = str(value)

    def steps_changed(self, widget):
        """Update the steps value label when the slider changes."""
        value = int(self.steps_slider.value)
        self.steps_value_label.text = str(value)

    async def select_image(self, widget):
        """Open a file dialog to select an input image."""
        try:
            file_dialog = toga.OpenFileDialog(
                title="Select Image File",
                file_types=["png", "jpg", "jpeg", "webp"]
            )

            self.input_image_path = await self.main_window.dialog(file_dialog)

            if self.input_image_path:
                # Update the label with the file name
                self.image_path_label.text = Path(self.input_image_path).name

                # Display the input image
                self.input_image_view.image = toga.Image(self.input_image_path)

                # Reset the output image
                self.output_image_view.image = None
                self.output_image = None
                self.save_button.enabled = False

                # Enable the process button
                self.process_button.enabled = True
                self.status_label.text = "Image selected. Ready to convert."

        except Exception as e:
            self.status_label.text = f"Error selecting image: {str(e)}"

    async def load_pipeline(self):
        """Initialize the diffusion pipeline asynchronously."""
        self.status_label.text = "Loading model... Please wait."

        try:
            def _load_pipeline():
                # Use the specific pipeline class instead of Auto
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    variant="fp16" if torch.cuda.is_available() else None,
                    use_safetensors=True,
                    safety_checker=True,
                )

                # Apply Karras scheduler if requested
                if self.karras_switch.value:
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config,
                        use_karras_sigmas=True
                    )

                # Move to appropriate device
                if torch.cuda.is_available():
                    return pipe.to("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return pipe.to("mps")  # Apple Silicon
                return pipe

            # Load the pipeline in a background task
            self.pipeline = await asyncio.get_event_loop().run_in_executor(None, _load_pipeline)
            self.model_loaded = True
            self.status_label.text = "Model loaded successfully."
            return True

        except Exception as e:
            self.status_label.text = f"Error loading model: {str(e)}"
            return False

    async def process_image(self, widget):
        """Process the selected image with SDXL Turbo."""
        if not self.input_image_path:
            self.status_label.text = "Please select an input image first."
            return

        # Disable the button during processing
        self.process_button.enabled = False

        try:
            # Load model if needed
            if not self.model_loaded:
                model_loaded = await self.load_pipeline()
                if not model_loaded:
                    self.process_button.enabled = True
                    return

            self.status_label.text = "Processing image..."

            # Get parameters
            prompt = self.prompt_input.value
            strength = self.strength_slider.value
            steps = int(self.steps_slider.value)

            negative_prompt="nudity, nude, breast, nsfw, explicit, foul language, pornography, mature, topless"

            # Ensure steps * strength is at least 1
            if steps * strength < 1:
                steps = max(steps, int(1.0 / strength) + 1)

            # Process the image with a background task
            def _process_image():
                # Load the input image
                input_image = Image.open(self.input_image_path).convert("RGB")

                # Ensure dimensions are multiples of 8 (SDXL requirement)
                width = (input_image.width // 8) * 8
                height = (input_image.height // 8) * 8
                if width == 0 or height == 0:
                    width, height = 512, 512  # Fallback to safe dimensions

                input_image = input_image.resize((width, height))

                # Generate a seed for reproducibility
                seed = torch.randint(0, 2147483647, (1,)).item()
                generator = torch.Generator().manual_seed(seed)

                pipeline_output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=strength,
                    num_inference_steps=steps,
                    generator=generator,
                    guidance_scale=2.0,
                )

                output_image = pipeline_output.images[0]

                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    output_image.save(temp_file.name)
                    return temp_file.name

            try:
                output_path = await asyncio.get_event_loop().run_in_executor(None, _process_image)

                # Update UI with the result
                self.output_image = toga.Image(output_path)
                self.output_image_view.image = self.output_image
                self.save_button.enabled = True
                self.status_label.text = "Image processed successfully!"
            except Exception as e:
                self.status_label.text = f"Error during image processing: {str(e)}"

        except Exception as e:
            self.status_label.text = f"Error processing image: {str(e)}"
        finally:
            # Always re-enable the button
            self.process_button.enabled = True

    async def save_output_image(self, widget):
        """Save the processed output image to a file."""
        if not self.output_image:
            return

        try:
            # Open a save file dialog
            save_dialog = toga.SaveFileDialog(
                title="Save Output Image",
                filename="sdxl_output.png",
                file_types=["png"],
            )

            save_path = await self.main_window.dialog(save_dialog)

            if save_path:
                # Define a function to copy the file
                def _save_file():
                    with open(self.output_image.path, "rb") as src_file:
                        with open(save_path, "wb") as dst_file:
                            dst_file.write(src_file.read())

                # Execute the save in the background
                await asyncio.get_event_loop().run_in_executor(None, _save_file)

                self.status_label.text = f"Image saved to {Path(save_path).name}"

        except Exception as e:
            self.status_label.text = f"Error saving image: {str(e)}"


def main():
    """Entry point for the application."""
    return DiffusifyApp("Diffusify", "com.example.diffusify")


if __name__ == "__main__":
    app = main()
    app.main_loop()
