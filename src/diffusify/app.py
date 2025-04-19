"""
Transform images using diffusion models
"""

"""
SDXL Turbo Image Converter
A BeeWare/Toga app for converting images using SDXL Turbo with memory optimizations.
"""

import os
import tempfile
import threading
import asyncio
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image
from huggingface_hub import snapshot_download


class Diffusify(toga.App):
    def startup(self):
        """
        Initialize the application
        """
        # Main window
        self.main_window = toga.MainWindow(title=self.formal_name, size=(800, 800))

        # Main box with vertical layout
        main_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Split main area into two columns
        content_box = toga.SplitContainer(style=Pack(flex=1))

        # Left column - controls
        self.control_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Image selection
        self.image_selection_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.image_label = toga.Label('Input Image:', style=Pack(margin=(0, 0, 5, 0)))
        self.image_selection_box.add(self.image_label)

        self.select_file_button = toga.Button(
            'Select Image',
            on_press=self.select_image,
            style=Pack(margin=5)
        )
        self.image_selection_box.add(self.select_file_button)

        self.image_path_label = toga.Label('No image selected', style=Pack(margin=5))
        self.image_selection_box.add(self.image_path_label)

        self.control_box.add(self.image_selection_box)

        # Prompt input
        self.prompt_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.prompt_label = toga.Label('Prompt:', style=Pack(margin=(0, 0, 5, 0)))
        self.prompt_box.add(self.prompt_label)
        self.prompt_input = toga.MultilineTextInput(
            placeholder='Enter prompt here...',
            value='a professional photograph, high quality',
            style=Pack(margin=5, height=60)
        )
        self.prompt_box.add(self.prompt_input)

        # Negative prompt input
        self.neg_prompt_label = toga.Label('Negative Prompt:', style=Pack(margin=(5, 0, 5, 0)))
        self.prompt_box.add(self.neg_prompt_label)
        self.neg_prompt_input = toga.MultilineTextInput(
            placeholder='Enter negative prompt...',
            value='blurry, low quality, distorted',
            style=Pack(margin=5, height=60)
        )
        self.prompt_box.add(self.neg_prompt_input)

        self.control_box.add(self.prompt_box)

        # Strength slider
        self.strength_box = toga.Box(style=Pack(direction=ROW, margin=(5, 0)))
        self.strength_label = toga.Label('Strength:', style=Pack(margin=(0, 5, 0, 0), width=80))
        self.strength_box.add(self.strength_label)
        self.strength_slider = toga.Slider(
            min=0.0,
            max=1.0,
            value=0.75,
            on_change=self.strength_changed,
            style=Pack(margin=5, flex=1)
        )
        self.strength_box.add(self.strength_slider)
        self.strength_value_label = toga.Label('0.75', style=Pack(margin=(0, 0, 0, 5), width=40))
        self.strength_box.add(self.strength_value_label)
        self.control_box.add(self.strength_box)

        # Steps slider
        self.steps_box = toga.Box(style=Pack(direction=ROW, margin=(5, 0)))
        self.steps_label = toga.Label('Steps:', style=Pack(margin=(0, 5, 0, 0), width=80))
        self.steps_box.add(self.steps_label)
        self.steps_slider = toga.Slider(
            min=1,
            max = 4,
            value=1,
            tick_count=10,
            on_change=self.steps_changed,
            style=Pack(margin=5, flex=1)
        )
        self.steps_box.add(self.steps_slider)
        self.steps_value_label = toga.Label('1', style=Pack(margin=(0, 0, 0, 5), width=40))
        self.steps_box.add(self.steps_value_label)
        self.control_box.add(self.steps_box)

        # Memory optimization settings
        self.optimization_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.optimization_label = toga.Label('Memory Optimizations:', style=Pack(margin=(5, 0, 5, 0)))
        self.optimization_box.add(self.optimization_label)

        self.vae_tiling = toga.Switch('Enable VAE Tiling', style=Pack(margin=5))
        self.optimization_box.add(self.vae_tiling)

        self.vae_slicing = toga.Switch('Enable VAE Slicing', style=Pack(margin=5))
        self.optimization_box.add(self.vae_slicing)

        self.sequential_offload = toga.Switch('Sequential CPU Offload', style=Pack(margin=5))
        self.optimization_box.add(self.sequential_offload)

        self.model_offload = toga.Switch('Model CPU Offload', style=Pack(margin=5))
        self.optimization_box.add(self.model_offload)

        self.control_box.add(self.optimization_box)

        # Process button
        self.process_button = toga.Button(
            'Convert Image',
            on_press=self.process_image,
            style=Pack(margin=10)
        )
        self.process_button.enabled = False
        self.control_box.add(self.process_button)

        # Right column - image display
        self.display_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Create a box for the input image
        self.input_image_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.input_image_label = toga.Label('Input Image', style=Pack(margin=(0, 0, 5, 0), text_align='center'))
        self.input_image_box.add(self.input_image_label)
        self.input_image_view = toga.ImageView(style=Pack(width=300, height=300))
        self.input_image_box.add(self.input_image_view)
        self.display_box.add(self.input_image_box)

        # Create a box for the output image
        self.output_image_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.output_image_label = toga.Label('Output Image', style=Pack(margin=(10, 0, 5, 0), text_align='center'))
        self.output_image_box.add(self.output_image_label)
        self.output_image_view = toga.ImageView(style=Pack(width=300, height=300))
        self.output_image_box.add(self.output_image_view)

        # Add save button for output
        self.save_button = toga.Button(
            'Save Output Image',
            on_press=self.save_output_image,
            style=Pack(margin=5)
        )
        self.save_button.enabled = False
        self.output_image_box.add(self.save_button)

        self.display_box.add(self.output_image_box)

        # Status label at the bottom
        self.status_label = toga.Label(
            'Ready.',
            style=Pack(margin=10, text_align='center')
        )

        # Add content to the SplitContainer
        content_box.content = [self.control_box, self.display_box]

        # Final layout assembly
        main_box.add(content_box)
        main_box.add(self.status_label)

        self.main_window.content = main_box

        # Initialize other variables
        self.input_image_path = None
        self.output_image = None
        self.pipeline = None
        self.model_loaded = False
        self.main_window.show()

        self.status_label.text = "Startup complete. Please select an image."

        self.on_startup = self.check_model()

    async def check_model(self, sender=None):
        """Check if the model is downloaded already"""
        model_path = os.path.join(self.get_app_folder(), "models", "sdxl-turbo")

        if not os.path.exists(model_path):
            # Show download dialog
            should_download = await self.confirm_dialog("Model Not Found",
                                                        "SDXL Turbo model not found. Would you like to download it now?")
            if should_download:
                self.download_model()
            else:
                self.status_label.text = "Model not downloaded. Some features may not work."
        else:
            self.status_label.text = "Model found. Ready to convert images."

    def confirm_dialog(self, title, message):
        """Show a confirmation dialog and return True if confirmed"""
        future = asyncio.Future()

        def handler(dialog, result):
            future.set_result(result)

        dialog = toga.OptionDialog(
            self.main_window,
            title,
            message,
            options=[
                ('Yes', True),
                ('No', False)
            ],
            on_result=handler
        )
        dialog.show()

        return future

    def get_app_folder(self):
        """Get the application's data folder"""
        app_data_folder = os.path.join(os.path.expanduser("~"), ".sdxl_converter")

        # Create the folder if it doesn't exist
        if not os.path.exists(app_data_folder):
            os.makedirs(app_data_folder)

        # Create models subfolder
        models_folder = os.path.join(app_data_folder, "models")
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        return app_data_folder

    def download_model(self):
        """Download the SDXL Turbo model from Hugging Face"""
        self.status_label.text = "Downloading SDXL Turbo model... this may take a while."

        def _download():
            try:
                # Create path
                model_path = os.path.join(self.get_app_folder(), "models", "sdxl-turbo")

                # Download the model using HuggingFace Hub
                snapshot_download(
                    repo_id="stabilityai/sdxl-turbo",
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )

                # Update UI on main thread
                def update_ui():
                    self.status_label.text = "Model downloaded successfully!"

                self.add_background_task(update_ui)

            except Exception as e:
                # Update UI on main thread with error
                def update_error():
                    self.status_label.text = f"Error downloading model: {str(e)}"

                self.add_background_task(update_error)

        # Run in a background thread
        thread = threading.Thread(target=_download, daemon=True)
        thread.start()

    def strength_changed(self, widget):
        """Update the strength value label when the slider changes"""
        value = round(self.strength_slider.value, 2)
        self.strength_value_label.text = str(value)

    def steps_changed(self, widget):
        """Update the steps value label when the slider changes"""
        value = int(self.steps_slider.value)
        self.steps_value_label.text = str(value)

    async def select_image(self, widget):
        """Open a file dialog to select an input image"""
        try:
            file_dialog = toga.OpenFileDialog(
                title="Select Image File",
                file_types=["png", "jpg", "jpeg", "webp"]
            )
            self.input_image_path = await self.main_window.dialog(file_dialog)

            if self.input_image_path:
                # Update the label with the file name
                self.image_path_label.text = os.path.basename(self.input_image_path)

                # Display the input image
                self.input_image_view.image = toga.Image(self.input_image_path)

                # Enable the process button
                self.process_button.enabled = True

                # Reset the output image
                self.output_image_view.image = None
                self.output_image = None
                self.save_button.enabled = False

                # Update status
                self.status_label.text = "Image selected. Ready to convert."
        except Exception as e:
            self.status_label.text = f"Error selecting image: {str(e)}"

    def init_pipeline(self):
        """Initialize the diffusers pipeline with optimizations"""
        try:
            # Check if the model is saved locally
            model_path = os.path.join(self.get_app_folder(), "models", "sdxl-turbo")

            if os.path.exists(model_path):
                model_id = model_path
            else:
                model_id = "stabilityai/sdxl-turbo"

            self.status_label.text = "Loading pipeline... Please wait."

            # Load the pipeline
            pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            )

            # Apply optimizations based on selected options
            if self.vae_tiling.value:
                pipe.enable_vae_tiling()

            if self.vae_slicing.value:
                pipe.enable_vae_slicing()

            if self.sequential_offload.value:
                pipe.enable_sequential_cpu_offload()
            elif self.model_offload.value:
                pipe.enable_model_cpu_offload()
            else:
                # Move to GPU if no offloading is selected
                if torch.cuda.is_available():
                    pipe = pipe.to("cuda")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    pipe = pipe.to("mps")  # For Apple Silicon GPUs

            self.pipeline = pipe
            self.model_loaded = True

            return True
        except Exception as e:
            self.status_label.text = f"Error loading model: {str(e)}"
            return False

    def process_image(self, widget):
        """Process the selected image with SDXL Turbo"""
        if not self.input_image_path:
            self.status_label.text = "Please select an input image first."
            return

        # Disable the button during processing
        self.process_button.enabled = False
        self.status_label.text = "Initializing pipeline..."

        def _process():
            try:
                # Initialize pipeline if not already done
                if not self.model_loaded:
                    if not self.init_pipeline():
                        # If initialization failed, re-enable the button and return
                        def update_ui_error():
                            self.process_button.enabled = True
                        self.add_background_task(update_ui_error)
                        return

                # Load the input image
                input_image = Image.open(self.input_image_path).convert("RGB")

                # Get the parameters from the UI
                prompt = self.prompt_input.value
                negative_prompt = self.neg_prompt_input.value
                strength = self.strength_slider.value
                steps = int(self.steps_slider.value)

                # Update status
                def update_status_processing():
                    self.status_label.text = "Processing image... This may take a while."
                self.add_background_task(update_status_processing)

                # Run the pipeline
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=0.0,  # SDXL Turbo works best with guidance_scale=0.0
                ).images[0]

                # Save the result to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    output.save(temp_file.name)
                    output_path = temp_file.name

                # Update UI on main thread
                def update_ui():
                    # Display the output image
                    self.output_image = toga.Image(output_path)
                    self.output_image_view.image = self.output_image

                    # Enable the save button
                    self.save_button.enabled = True

                    # Re-enable the process button
                    self.process_button.enabled = True

                    # Update status
                    self.status_label.text = "Image processed successfully!"

                self.add_background_task(update_ui)

            except Exception as e:
                # Update UI on main thread with error
                def update_error():
                    self.status_label.text = f"Error processing image: {str(e)}"
                    self.process_button.enabled = True

                self.add_background_task(update_error)

        # Run in a background thread
        thread = threading.Thread(target=_process, daemon=True)
        thread.start()

    async def save_output_image(self, widget):
        """Save the processed output image to a file"""
        if not self.output_image:
            return

        try:
            # Open a save file dialog
            save_dialog = toga.SaveFileDialog(
                title="Save Output Image",
                filename="sdxl_output.png",
                file_types=["png"]
            )
            save_path = await self.main_window.dialog(save_dialog)
            if save_path:
                # Copy the image to the selected location
                output_path = self.output_image.path

                with open(output_path, "rb") as src_file:
                    with open(save_path, "wb") as dst_file:
                        dst_file.write(src_file.read())

                self.status_label.text = f"Image saved to {os.path.basename(save_path)}"

        except Exception as e:
            self.status_label.text = f"Error saving image: {str(e)}"


def main():
    return Diffusify()
