import asyncio
import tempfile
from pathlib import Path

import pytest

from diffusify.view import DiffusifyView

# --- Basic fixtures ---


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# --- Custom test implementation of DiffusifyViewModel ---


class _TestViewModel:
    """Simplified test implementation of DiffusifyViewModel."""

    def __init__(self, test_image_path=None):
        self.width = 512
        self.height = 512
        self.test_image_path = test_image_path
        self.params = None
        self.save_path = None

        # Callbacks
        self.on_progress_update = None
        self.on_status_update = None
        self.on_image_generated = None
        self.on_operation_complete = None

    def set_callbacks(self, **callbacks):
        """Set callbacks for testing."""
        self.on_progress_update = callbacks.get("on_progress_update")
        self.on_status_update = callbacks.get("on_status_update")
        self.on_image_generated = callbacks.get("on_image_generated")
        self.on_operation_complete = callbacks.get("on_operation_complete")

    def update_image_size(self, width, height):
        """Update image dimensions."""
        self.width = width
        self.height = height

    async def generate_image(
        self, prompt, negative_prompt, steps, guidance_scale, **kwargs
    ):
        """Test implementation of generate_image."""
        # Store parameters
        self.params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }

        # Simulate the generation process with callbacks
        if self.on_progress_update:
            self.on_progress_update(0)
            await asyncio.sleep(0.1)
            self.on_progress_update(50)
            await asyncio.sleep(0.1)
            self.on_progress_update(100)

        if self.on_status_update:
            self.on_status_update("Generating image...")
            await asyncio.sleep(0.1)
            self.on_status_update("Image generated successfully!")

        # Skip calling on_image_generated to avoid toga.Image issues

        if self.on_operation_complete:
            self.on_operation_complete(True)

    async def save_image(self, save_path):
        """Test implementation of save_image."""
        self.save_path = save_path
        if self.on_status_update:
            self.on_status_update(f"Image saved to {save_path}")
        return True


# --- Custom view for testing ---


class _TestDiffusifyView(DiffusifyView):
    """Modified DiffusifyView for testing that avoids UI component issues."""

    def set_output_image(self, image_path):
        """Override to avoid using actual toga.Image."""
        # Store the path but don't create an actual Image
        self._image_path = image_path

        # Skip setting the image_view.image property

        # Enable the save button directly
        self.save_button.enabled = True

    async def save_output_image(self, widget):
        """Override to avoid dialog issues."""
        # For testing, just call the viewmodel save_image with a test path
        if hasattr(self, "_test_save_path"):
            await self.viewmodel.save_image(self._test_save_path)
        else:
            # Default test path
            await self.viewmodel.save_image(Path("test_save.png"))


# --- Fixtures for testing ---


@pytest.fixture
def test_image_path(temp_dir):
    """Create a test image file path."""
    test_image_path = temp_dir / "test_output.png"
    test_image_path.touch()
    return test_image_path


@pytest.fixture
def viewmodel(test_image_path):
    """Provide a test viewmodel for integration testing."""
    return _TestViewModel(test_image_path)


@pytest.fixture
def view(viewmodel):
    """Provide a test view for integration testing."""
    return _TestDiffusifyView(viewmodel)


# --- Test view initialization ---


async def test_view_has_required_containers(view):
    """Test that the view initializes with all required containers."""
    assert hasattr(view, "control_box")
    assert hasattr(view, "display_box")
    assert hasattr(view, "status_box")


async def test_view_has_input_components(view):
    """Test that the view initializes with all input components."""
    assert hasattr(view, "prompt_input")
    assert hasattr(view, "neg_prompt_input")
    assert hasattr(view, "size_selection")
    assert hasattr(view, "steps_slider")
    assert hasattr(view, "guidance_slider")
    assert hasattr(view, "attention_slicing_switch")
    assert hasattr(view, "karras_switch")


async def test_view_has_output_components(view):
    """Test that the view initializes with all output components."""
    assert hasattr(view, "output_image_view")
    assert hasattr(view, "save_button")
    assert hasattr(view, "generate_button")
    assert hasattr(view, "progress_bar")
    assert hasattr(view, "status_label")


async def test_view_default_values(view):
    """Test that the view has correct default values."""
    assert (
        view.prompt_input.value
        == "a professional photograph of a mountain landscape, high quality"
    )
    assert (
        view.neg_prompt_input.value
        == "low quality, blurry, distorted, ugly, bad anatomy"
    )
    assert view.karras_switch.value is True
    assert view.save_button.enabled is False


# --- Test UI interactions ---


async def test_steps_slider_updates_label(view):
    """Test that steps slider updates the label correctly."""
    view.steps_slider.value = 25
    view.steps_changed(None)
    assert view.steps_value_label.text == "25"


async def test_guidance_slider_updates_label(view):
    """Test that guidance slider updates the label correctly."""
    view.guidance_slider.value = 5.5
    view.guidance_changed(None)
    assert view.guidance_value_label.text == "5.5"


async def test_size_selection_updates_viewmodel(view, viewmodel):
    """Test that size selection updates the view model correctly."""
    view.size_selection.value = "768×768"
    view.size_changed(None)

    assert viewmodel.width == 768
    assert viewmodel.height == 768


async def test_size_selection_updates_image_view(view):
    """Test that size selection updates the image view dimensions."""
    # Test changing to 768×768
    view.size_selection.value = "768×768"
    view.size_changed(None)

    # Check that image view was updated (limited to max 512)
    assert view.output_image_view.style.width == 512
    assert view.output_image_view.style.height == 512

    # Test changing to 768×512
    view.size_selection.value = "768×512"
    view.size_changed(None)

    # Check image view update
    assert view.output_image_view.style.width == 512
    assert view.output_image_view.style.height == 512


# --- Test progress and status handling ---


async def test_progress_visibility_toggle(view):
    """Test that progress bar visibility toggles correctly."""
    # Progress should be initially hidden
    assert view._progress_visible is False
    assert view.progress_bar.style.visibility == "hidden"

    # Show progress
    view.show_progress(True)
    assert view._progress_visible is True
    assert view.progress_bar.style.visibility == "visible"

    # Hide progress
    view.show_progress(False)
    assert view._progress_visible is False
    assert view.progress_bar.style.visibility == "hidden"


async def test_update_status_changes_label(view):
    """Test that status updates are reflected in the UI."""
    test_message = "Testing status update"
    view.update_status(test_message)
    assert view.status_label.text == test_message


async def test_update_progress_updates_bar(view):
    """Test that progress updates are reflected in the UI."""
    # First call update_progress with 0 to ensure progress bar is shown
    view.update_progress(0)
    # Then update to 50%
    view.update_progress(50)
    assert view.progress_bar.value == 50
    assert view._progress_visible is True


async def test_update_progress_completion_sets_max(view):
    """Test that progress completion sets the progress bar to max."""
    # Complete progress (will schedule hiding after delay)
    view.update_progress(100)
    assert view.progress_bar.value == 100


# --- Test operation completion ---


async def test_operation_complete_enables_generate_button(view):
    """Test that operation complete enables the generate button."""
    # Disable generate button
    view.generate_button.enabled = False

    # Call operation complete
    view.on_operation_complete(False)
    assert view.generate_button.enabled is True


async def test_operation_complete_without_success_keeps_save_disabled(view):
    """Test that operation complete without success keeps save button disabled."""
    # Disable buttons
    view.generate_button.enabled = False
    view.save_button.enabled = False

    # Call operation complete with failure
    view.on_operation_complete(False)

    assert view.generate_button.enabled is True
    assert view.save_button.enabled is False


# --- End-to-end tests ---


async def test_generate_image_workflow(view, viewmodel):
    """Test the complete image generation workflow."""
    # Set custom values in the UI
    view.prompt_input.value = "test prompt"
    view.neg_prompt_input.value = "test negative prompt"
    view.steps_slider.value = 20
    view.steps_changed(None)
    view.guidance_slider.value = 8.0
    view.guidance_changed(None)

    # Start the generation
    await view.generate_image(None)

    # Check parameters were passed correctly
    assert viewmodel.params["prompt"] == "test prompt"
    assert viewmodel.params["negative_prompt"] == "test negative prompt"
    assert viewmodel.params["steps"] == 20
    assert viewmodel.params["guidance_scale"] == 8.0


async def test_save_image_workflow(view, viewmodel, temp_dir):
    """Test the image save workflow."""
    # Setup test save path
    test_save_path = temp_dir / "saved_image.png"
    view._test_save_path = test_save_path

    # Enable save button
    view.save_button.enabled = True

    # Trigger save operation
    await view.save_output_image(view.save_button)

    # Check the save path was passed correctly
    assert viewmodel.save_path == test_save_path


async def test_ui_control_interactions(view):
    """Test interactions with UI controls."""
    # Test switches
    view.attention_slicing_switch.value = True
    assert view.attention_slicing_switch.value is True

    view.karras_switch.value = False
    assert view.karras_switch.value is False

    # Test text inputs
    test_prompt = "test text input"
    view.prompt_input.value = test_prompt
    assert view.prompt_input.value == test_prompt


async def test_size_selection_all_options(view, viewmodel):
    """Test all size selection options."""
    size_options = ["512×512", "768×768", "1024×1024", "768×512", "512×768"]

    for size_option in size_options:
        view.size_selection.value = size_option
        view.size_changed(None)

        # Extract expected width and height
        width, height = map(int, size_option.split("×"))

        # Check that viewmodel was updated correctly
        assert viewmodel.width == width
        assert viewmodel.height == height
