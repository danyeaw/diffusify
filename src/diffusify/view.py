import asyncio

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from diffusify.view_model import DiffusifyViewModel


class DiffusifyView(toga.Box):
    """View class that handles UI elements and layout."""

    def __init__(self, viewmodel: DiffusifyViewModel):
        super().__init__(style=Pack(direction=COLUMN, margin=10))
        self.viewmodel = viewmodel

        # Set up callbacks
        self.viewmodel.set_callbacks(
            on_progress_update=self.update_progress,
            on_status_update=self.update_status,
            on_image_generated=self.set_output_image,
            on_operation_complete=self.on_operation_complete,
        )

        # Thread-safe progress tracking
        self._progress_visible = False

        # Create UI components
        self._create_content()

    def _create_content(self):
        """Create the main UI layout."""
        content_box = toga.SplitContainer(style=Pack(flex=1))

        # Create left and right columns
        self.control_box = self._create_control_box()
        self.display_box = self._create_display_box()

        # Add content to the SplitContainer
        content_box.content = [self.control_box, self.display_box]

        # Create status bar
        self.status_box = self._create_status_box()

        # Assemble final layout
        self.add(content_box)
        self.add(self.status_box)

    def _create_control_box(self) -> toga.Box:
        """Create the control panel UI (left side)."""
        control_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Prompt input
        prompt_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        prompt_label = toga.Label("Prompt:", style=Pack(margin_bottom=5))
        prompt_box.add(prompt_label)
        self.prompt_input = toga.MultilineTextInput(
            placeholder="Enter prompt here...",
            value="a professional photograph of a mountain landscape, high quality",
            style=Pack(margin=5, height=100),
        )
        prompt_box.add(self.prompt_input)
        control_box.add(prompt_box)

        # Negative prompt input
        neg_prompt_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        neg_prompt_label = toga.Label("Negative Prompt:", style=Pack(margin_bottom=5))
        neg_prompt_box.add(neg_prompt_label)
        self.neg_prompt_input = toga.MultilineTextInput(
            placeholder="Enter negative prompt here...",
            value="low quality, blurry, distorted, ugly, bad anatomy",
            style=Pack(margin=5, height=60),
        )
        neg_prompt_box.add(self.neg_prompt_input)
        control_box.add(neg_prompt_box)

        # Image size selection
        size_box = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        size_label = toga.Label("Image Size:", style=Pack(margin_bottom=5))
        size_box.add(size_label)

        # Size options box (horizontal)
        size_options_box = toga.Box(style=Pack(direction=ROW))
        self.size_selection = toga.Selection(
            items=["512×512", "768×768", "1024×1024", "768×512", "512×768"],
            style=Pack(margin=5),
        )
        self.size_selection.on_select = self.size_changed
        size_options_box.add(self.size_selection)
        size_box.add(size_options_box)
        control_box.add(size_box)

        # Steps slider
        steps_box = toga.Box(style=Pack(direction=ROW, margin_bottom=10))
        steps_label = toga.Label("Steps:", style=Pack(margin_right=5, width=80))
        steps_box.add(steps_label)
        self.steps_slider = toga.Slider(
            min=15,
            max=50,
            value=30,
            style=Pack(margin=5, flex=1),
        )
        self.steps_slider.on_change = self.steps_changed
        steps_box.add(self.steps_slider)
        self.steps_value_label = toga.Label("30", style=Pack(width=40))
        steps_box.add(self.steps_value_label)
        control_box.add(steps_box)

        # Guidance scale slider
        guidance_box = toga.Box(style=Pack(direction=ROW, margin_bottom=10))
        guidance_label = toga.Label("Guidance:", style=Pack(margin_right=5, width=80))
        guidance_box.add(guidance_label)
        self.guidance_slider = toga.Slider(
            min=1.0,
            max=15.0,
            value=7.5,
            style=Pack(margin=5, flex=1),
        )
        self.guidance_slider.on_change = self.guidance_changed
        guidance_box.add(self.guidance_slider)
        self.guidance_value_label = toga.Label("7.5", style=Pack(width=40))
        guidance_box.add(self.guidance_value_label)
        control_box.add(guidance_box)

        # Attention slicing switch
        self.attention_slicing_switch = toga.Switch(
            "Enable attention slicing", value=True, style=Pack(margin=5)
        )
        control_box.add(self.attention_slicing_switch)

        # Karras scheduler option
        self.karras_switch = toga.Switch("Use Karras scheduler", style=Pack(margin=5))
        control_box.add(self.karras_switch)

        # Generate button
        self.generate_button = toga.Button("Generate Image", style=Pack(margin=10))
        self.generate_button.on_press = self.generate_image
        control_box.add(self.generate_button)

        return control_box

    def _create_display_box(self) -> toga.Box:
        """Create the image display UI (right side)."""
        display_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Output image display
        self.output_image_label = toga.Label(
            "Generated Image", style=Pack(margin_bottom=5, text_align="center")
        )
        display_box.add(self.output_image_label)
        self.output_image_view = toga.ImageView(style=Pack(width=512, height=512))
        display_box.add(self.output_image_view)

        # Save button for output
        self.save_button = toga.Button("Save Image", style=Pack(margin_top=10))
        self.save_button.on_press = self.save_output_image
        self.save_button.enabled = False
        display_box.add(self.save_button)

        return display_box

    def _create_status_box(self) -> toga.Box:
        """Create the status bar UI (bottom)."""
        status_box = toga.Box(style=Pack(direction=COLUMN, margin=10))

        # Status label
        self.status_label = toga.Label(
            "Ready to generate images.",
            style=Pack(margin_bottom=5, text_align="center"),
        )
        status_box.add(self.status_label)

        # Progress bar
        self.progress_bar = toga.ProgressBar(max=100, value=0, style=Pack(margin=5))
        status_box.add(self.progress_bar)
        # Initially hide the progress bar
        self.progress_bar.style.update(visibility="hidden")

        return status_box

    # === Event handlers ===
    def steps_changed(self, widget):
        """Handle steps slider value change."""
        value = int(self.steps_slider.value)
        self.steps_value_label.text = str(value)

    def guidance_changed(self, widget):
        """Handle guidance scale slider value change."""
        value = round(self.guidance_slider.value, 1)
        self.guidance_value_label.text = str(value)

    def size_changed(self, widget):
        """Handle image size selection change."""
        size_text = self.size_selection.value
        width, height = map(int, size_text.split("×"))

        # Update ViewModel
        self.viewmodel.update_image_size(width, height)

        # Update image view size
        self.output_image_view.style.update(
            width=min(512, width), height=min(512, height)
        )

    async def generate_image(self, widget):
        """Handle generate button press."""
        # Disable the button during processing
        self.generate_button.enabled = False
        self.save_button.enabled = False

        # Get parameters from the UI
        prompt = self.prompt_input.value
        negative_prompt = self.neg_prompt_input.value
        steps = int(self.steps_slider.value)
        guidance_scale = float(self.guidance_slider.value)

        # Generate the image using the ViewModel
        await self.viewmodel.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
        )

    async def save_output_image(self, widget):
        """Handle save button press."""
        # Open a save file dialog
        save_dialog = toga.SaveFileDialog(
            title="Save Generated Image",
            filename="diffusify_output.png",
            file_types=["png"],
        )

        save_path = await widget.window.dialog(save_dialog)

        if save_path:
            # Save the image using the ViewModel
            await self.viewmodel.save_image(save_path)

    # === Callback handlers ===
    def update_progress(self, value):
        """Update the progress bar."""
        if value == 0:
            self.show_progress(True)

        self.progress_bar.value = value

        if value >= 100:
            # Schedule hiding the progress bar after a delay
            async def hide_progress_after_delay():
                try:
                    await asyncio.sleep(1.5)
                    self.show_progress(False)
                except Exception as e:
                    print(f"Error hiding progress bar: {e}")

            loop = asyncio.get_event_loop()
            loop.create_task(hide_progress_after_delay())

    def update_status(self, message):
        """Update the status label."""
        self.status_label.text = message

    def set_output_image(self, image_path):
        """Set the output image."""
        self.output_image = toga.Image(image_path)
        self.output_image_view.image = self.output_image

    def on_operation_complete(self, success):
        """Handle operation completion."""
        self.generate_button.enabled = True
        if success and hasattr(self, "output_image"):
            self.save_button.enabled = True

    def show_progress(self, show=True):
        """Show or hide the progress bar."""
        if show and not self._progress_visible:
            self.progress_bar.style.update(visibility="visible")
            self._progress_visible = True
        elif not show and self._progress_visible:
            self.progress_bar.style.update(visibility="hidden")
            self._progress_visible = False
