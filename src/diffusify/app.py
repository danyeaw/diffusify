"""
Transform images using diffusion models
"""

import asyncio
import tempfile

import toga
import torch
from diffusers import AutoPipelineForImage2Image
from huggingface_hub import snapshot_download
from PIL import Image
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


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
        self.image_label = toga.Label("Input Image:", style=Pack(margin=(0, 0, 5, 0)))
        self.image_selection_box.add(self.image_label)

        self.select_file_button = toga.Button(
            "Select Image", on_press=self.select_image, style=Pack(margin=5)
        )
        self.image_selection_box.add(self.select_file_button)

        self.image_path_label = toga.Label("No image selected", style=Pack(margin=5))
        self.image_selection_box.add(self.image_path_label)

        self.control_box.add(self.image_selection_box)

        self.model_status_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.model_status_box.style.background_color = "#f0f0f0"  # Light gray background

        # Status heading
        self.model_status_label = toga.Label(
            "Model Status:", style=Pack(margin=(0, 0, 5, 0), font_weight="bold")
        )
        self.model_status_box.add(self.model_status_label)

        # Progress bar
        self.model_progress = toga.ProgressBar(max=100, value=0, style=Pack(margin=5))
        self.model_status_box.add(self.model_progress)

        # Status text
        self.model_download_status = toga.Label(
            "Checking...", style=Pack(margin=5, text_align="center")
        )
        self.model_status_box.add(self.model_download_status)

        # Model download button
        self.download_button = toga.Button(
            "Download Model", on_press=self.download_model, style=Pack(margin=5)
        )
        self.model_status_box.add(self.download_button)

        # Add model status box at the top of the control panel
        self.control_box.add(self.model_status_box)

        # Prompt input
        self.prompt_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.prompt_label = toga.Label("Prompt:", style=Pack(margin=(0, 0, 5, 0)))
        self.prompt_box.add(self.prompt_label)
        self.prompt_input = toga.MultilineTextInput(
            placeholder="Enter prompt here...",
            value="a professional photograph, high quality",
            style=Pack(margin=5, height=60),
        )
        self.prompt_box.add(self.prompt_input)

        # Negative prompt input
        self.neg_prompt_label = toga.Label(
            "Negative Prompt:", style=Pack(margin=(5, 0, 5, 0))
        )
        self.prompt_box.add(self.neg_prompt_label)
        self.neg_prompt_input = toga.MultilineTextInput(
            placeholder="Enter negative prompt...",
            value="blurry, low quality, distorted",
            style=Pack(margin=5, height=60),
        )
        self.prompt_box.add(self.neg_prompt_input)

        self.control_box.add(self.prompt_box)

        # Strength slider
        self.strength_box = toga.Box(style=Pack(direction=ROW, margin=(5, 0)))
        self.strength_label = toga.Label(
            "Strength:", style=Pack(margin=(0, 5, 0, 0), width=80)
        )
        self.strength_box.add(self.strength_label)
        self.strength_slider = toga.Slider(
            min=0.0,
            max=1.0,
            value=0.75,
            on_change=self.strength_changed,
            style=Pack(margin=5, flex=1),
        )
        self.strength_box.add(self.strength_slider)
        self.strength_value_label = toga.Label(
            "0.75", style=Pack(margin=(0, 0, 0, 5), width=40)
        )
        self.strength_box.add(self.strength_value_label)
        self.control_box.add(self.strength_box)

        # Steps slider
        self.steps_box = toga.Box(style=Pack(direction=ROW, margin=(5, 0)))
        self.steps_label = toga.Label(
            "Steps:", style=Pack(margin=(0, 5, 0, 0), width=80)
        )
        self.steps_box.add(self.steps_label)
        self.steps_slider = toga.Slider(
            min=1,
            max=4,
            value=1,
            tick_count=10,
            on_change=self.steps_changed,
            style=Pack(margin=5, flex=1),
        )
        self.steps_box.add(self.steps_slider)
        self.steps_value_label = toga.Label(
            "1", style=Pack(margin=(0, 0, 0, 5), width=40)
        )
        self.steps_box.add(self.steps_value_label)
        self.control_box.add(self.steps_box)

        # Memory optimization settings
        self.optimization_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.optimization_label = toga.Label(
            "Memory Optimizations:", style=Pack(margin=(5, 0, 5, 0))
        )
        self.optimization_box.add(self.optimization_label)

        self.vae_tiling = toga.Switch("Enable VAE Tiling", style=Pack(margin=5))
        self.optimization_box.add(self.vae_tiling)

        self.vae_slicing = toga.Switch("Enable VAE Slicing", style=Pack(margin=5))
        self.optimization_box.add(self.vae_slicing)

        self.sequential_offload = toga.Switch(
            "Sequential CPU Offload", style=Pack(margin=5)
        )
        self.optimization_box.add(self.sequential_offload)

        self.model_offload = toga.Switch("Model CPU Offload", style=Pack(margin=5))
        self.optimization_box.add(self.model_offload)

        self.control_box.add(self.optimization_box)

        # Process button
        self.process_button = toga.Button(
            "Convert Image", on_press=self.process_image, style=Pack(margin=10)
        )
        self.process_button.enabled = False
        self.control_box.add(self.process_button)

        # Right column - image display
        self.display_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Create a box for the input image
        self.input_image_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.input_image_label = toga.Label(
            "Input Image", style=Pack(margin=(0, 0, 5, 0), text_align="center")
        )
        self.input_image_box.add(self.input_image_label)
        self.input_image_view = toga.ImageView(style=Pack(width=300, height=300))
        self.input_image_box.add(self.input_image_view)
        self.display_box.add(self.input_image_box)

        # Create a box for the output image
        self.output_image_box = toga.Box(style=Pack(direction=COLUMN, margin=5))
        self.output_image_label = toga.Label(
            "Output Image", style=Pack(margin=(10, 0, 5, 0), text_align="center")
        )
        self.output_image_box.add(self.output_image_label)
        self.output_image_view = toga.ImageView(style=Pack(width=300, height=300))
        self.output_image_box.add(self.output_image_view)

        # Add save button for output
        self.save_button = toga.Button(
            "Save Output Image", on_press=self.save_output_image, style=Pack(margin=5)
        )
        self.save_button.enabled = False
        self.output_image_box.add(self.save_button)

        self.display_box.add(self.output_image_box)

        # Status label at the bottom
        self.status_label = toga.Label(
            "Ready.", style=Pack(margin=10, text_align="center")
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

        self.on_startup = self.check_model
        self.asyncio_loop = asyncio.get_event_loop()

        self.check_model_sync()

    def check_model_sync(self):
        """Check if the model is downloaded (blocking version)"""
        model_path = self.paths.app / "resources" / "models"

        # Update UI
        self.model_download_status.text = "Checking..."

        if not model_path.is_dir() or not model_path.iterdir():
            # Model not found or directory empty
            self.model_download_status.text = "Model not downloaded"
            self.model_progress.value = 0
            self.download_button.enabled = True
            return False
        else:
            # Model found
            self.model_download_status.text = "Model downloaded"
            self.model_progress.value = 100
            self.download_button.enabled = False
            return True

    def download_model(self, widget=None):
        """Download the SDXL Turbo model from Hugging Face (blocking version)"""
        # Update UI
        self.status_label.text = "Downloading SDXL Turbo model... UI will freeze until complete."
        self.download_button.enabled = False
        self.model_download_status.text = "Downloading..."
        self.model_progress.value = 10  # Just to show something is happening

        try:
            # Force UI update before blocking
            self.main_window.content.refresh()

            # Create path
            model_path = self.paths.app / "resources" / "models"

            # Download the model - this will block the UI
            snapshot_download(
                repo_id="stabilityai/sdxl-turbo",
                local_dir=model_path,
            )

            # Update UI after download completes
            self.model_download_status.text = "Model downloaded successfully!"
            self.model_progress.value = 100
            self.download_button.enabled = False
            self.status_label.text = "Model downloaded successfully!"

            # Enable process button if an image is selected
            if self.input_image_path:
                self.process_button.enabled = True

            return True
        except Exception as e:
            # Update UI with error
            self.model_download_status.text = f"Error: {str(e)}"
            self.status_label.text = f"Error downloading model: {str(e)}"
            self.download_button.enabled = True
            return False

    async def check_model(self, sender=None):
        """Check if the model is downloaded already"""
        model_path = self.paths.app / "resources" / "models"

        if not model_path.is_dir() or not model_path.iterdir():
            # Model not found or directory empty
            self.model_download_status.text = "Model not downloaded"
            self.model_progress.value = 0
            self.download_button.enabled = True

            # Show download dialog if appropriate
            should_download = await self.confirm_dialog(
                "Model Not Found",
                "SDXL Turbo model not found. Would you like to download it now?",
            )
            if should_download:
                self.download_model()
            else:
                self.status_label.text = "Model not downloaded. Some features may not work."
        else:
            # Model found
            self.model_download_status.text = "Model downloaded"
            self.model_progress.value = 100
            self.download_button.enabled = False
            self.status_label.text = "Model found. Ready to convert images."



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
                title="Select Image File", file_types=["png", "jpg", "jpeg", "webp"]
            )
            self.input_image_path = await self.main_window.dialog(file_dialog)

            if self.input_image_path:
                # Update the label with the file name
                self.image_path_label.text = self.input_image_path.name

                # Display the input image
                self.input_image_view.image = toga.Image(self.input_image_path)

                # Reset the output image
                self.output_image_view.image = None
                self.output_image = None
                self.save_button.enabled = False

                # Check if model is available before enabling processing
                model_path = self.paths.app / "resources" / "models"
                model_available = model_path.is_dir() and len(model_path.iterdir()) > 0

                # Only enable the button if the model is available
                self.process_button.enabled = model_available

                # Update status message based on model availability
                if model_available:
                    self.status_label.text = "Image selected. Ready to convert."
                else:
                    self.status_label.text = "Image selected, but model not available. Please download the model first."

        except Exception as e:
            self.status_label.text = f"Error selecting image: {str(e)}"

    def init_pipeline(self):
        """Initialize the diffusers pipeline with optimizations"""
        try:
            # Check if the model is saved locally
            model_path = self.paths.app / "resources" / "models"

            if model_path:
                model_id = model_path
            else:
                model_id = "stabilityai/sdxl-turbo"

            # Don't update UI here, save the status for later
            status_message = "Loading pipeline... Please wait."

            # Function to update UI on main thread
            def update_status():
                self.status_label.text = status_message

            # Dispatch to main thread
            asyncio.run_coroutine_threadsafe(update_status(), asyncio.get_event_loop())

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
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    pipe = pipe.to("mps")  # For Apple Silicon GPUs

            self.pipeline = pipe
            self.model_loaded = True

            return True
        except Exception as e:
            # Don't update UI here, return False and handle in calling function
            self.pipeline_error = str(e)
            return False

    def process_image(self, widget):
        """Process the selected image with SDXL Turbo (blocking version)"""
        if not self.input_image_path:
            self.status_label.text = "Please select an input image first."
            return

        # Disable the button during processing
        self.process_button.enabled = False

        try:
            # Step 1: Load the model if needed
            if not self.model_loaded:
                self.status_label.text = "Loading model... This may take a while."

                # Check if the model is saved locally
                model_path = self.paths.app / "resources" / "models"

                if model_path.is_dir():
                    model_id = model_path
                else:
                    model_id = "stabilityai/sdxl-turbo"

                # Load the pipeline - will block UI but that's ok for now
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )

                # Apply optimizations
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
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        pipe = pipe.to("mps")  # For Apple Silicon GPUs

                self.pipeline = pipe
                self.model_loaded = True

            # Step 2: Process the image
            self.status_label.text = "Processing image... UI will freeze until complete."

            # Load the input image
            input_image = Image.open(self.input_image_path).convert("RGB")

            # Get parameters
            prompt = self.prompt_input.value
            negative_prompt = self.neg_prompt_input.value
            strength = self.strength_slider.value
            steps = int(self.steps_slider.value)

            # Run the pipeline - UI will freeze here
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=0.0,
            ).images[0]

            # Step 3: Save the output
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                output.save(temp_file.name)
                output_path = temp_file.name

            # Step 4: Update UI
            self.output_image = toga.Image(output_path)
            self.output_image_view.image = self.output_image
            self.save_button.enabled = True
            self.status_label.text = "Image processed successfully!"

        except Exception as e:
            self.status_label.text = f"Error processing image: {str(e)}"
        finally:
            # Always re-enable the button
            self.process_button.enabled = True

    async def save_output_image(self, widget):
        """Save the processed output image to a file"""
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
                # Copy the image to the selected location
                output_path = self.output_image.path

                with open(output_path, "rb") as src_file:
                    with open(save_path, "wb") as dst_file:
                        dst_file.write(src_file.read())

                self.status_label.text = f"Image saved to {save_path.name}"

        except Exception as e:
            self.status_label.text = f"Error saving image: {str(e)}"


def main():
    return Diffusify()
