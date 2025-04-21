from pathlib import Path

import pytest
from PIL import Image

from diffusify.view_model import DiffusifyViewModel


@pytest.fixture(scope="module")
async def shared_viewmodel():
    """Load model once for all tests in this module."""
    vm = DiffusifyViewModel()

    # Set up test callback trackers
    vm.progress_updates = []
    vm.status_messages = []
    vm.generated_images = []
    vm.operation_results = []

    # Set callbacks to track calls
    vm.set_callbacks(
        on_progress_update=lambda p: vm.progress_updates.append(p),
        on_status_update=lambda s: vm.status_messages.append(s),
        on_image_generated=lambda i: vm.generated_images.append(i),
        on_operation_complete=lambda r: vm.operation_results.append(r),
    )

    # Set minimal dimensions for all tests
    vm.update_image_size(width=64, height=64)

    # Load model once with optimal configuration
    success = await vm.load_model(use_attention_slicing=True, use_karras=True)
    assert success is True

    yield vm

    # Clean up all generated images at the end of all tests
    for img_path in vm.generated_images:
        if Path(img_path).exists():
            Path(img_path).unlink()

    if vm.output_image_path and Path(vm.output_image_path).exists():
        Path(vm.output_image_path).unlink()


@pytest.fixture
async def reset_viewmodel(shared_viewmodel):
    """Reset the viewmodel state before each test."""
    shared_viewmodel.progress_updates.clear()
    shared_viewmodel.status_messages.clear()
    shared_viewmodel.operation_results.clear()

    yield shared_viewmodel


async def test_model_loading_with_karras(shared_viewmodel):
    """Test that the model loads successfully with Karras scheduler."""
    # This test doesn't need to load the model again since the fixture has done it

    # Just verify the model was loaded with Karras scheduler
    assert len(shared_viewmodel.progress_updates) > 0
    assert shared_viewmodel.progress_updates[-1] == 100
    assert any(
        "Model loaded successfully" in msg for msg in shared_viewmodel.status_messages
    )
    assert len(shared_viewmodel.operation_results) > 0
    assert shared_viewmodel.operation_results[-1] is True


async def test_image_generation(reset_viewmodel):
    """Test that images can be generated successfully."""
    vm = reset_viewmodel

    success = await vm.generate_image(
        prompt="A simple test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    assert success is True
    assert vm.output_image_path is not None
    assert Path(vm.output_image_path).exists()
    assert len(vm.progress_updates) > 0
    assert vm.progress_updates[-1] == 100
    assert any("Image generated" in msg for msg in vm.status_messages)
    assert any("Seed: " in msg for msg in vm.status_messages)
    assert len(vm.generated_images) >= 1
    assert vm.generated_images[-1] == vm.output_image_path
    assert len(vm.operation_results) >= 1
    assert vm.operation_results[-1] is True


async def test_custom_image_dimensions(reset_viewmodel):
    """Test that custom image dimensions are used correctly."""
    vm = reset_viewmodel

    # Set slightly different dimensions for this test
    custom_width = 96
    custom_height = 64
    vm.update_image_size(width=custom_width, height=custom_height)

    # Generate image with custom dimensions
    success = await vm.generate_image(
        prompt="A dimension test",
        negative_prompt="",
        steps=2,  # Minimal steps
        guidance_scale=7.5,
    )

    assert success is True

    # Verify image has correct dimensions
    with Image.open(vm.output_image_path) as img:
        width, height = img.size

    assert width == custom_width
    assert height == custom_height

    # Reset dimensions for other tests
    vm.update_image_size(width=64, height=64)


async def test_save_image(reset_viewmodel):
    """Test that generated images can be saved to disk."""
    vm = reset_viewmodel

    # First generate a minimal test image
    success = await vm.generate_image(
        prompt="Save test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    assert success is True

    # Clear callback trackers for this specific test
    vm.progress_updates.clear()
    vm.status_messages.clear()
    vm.operation_results.clear()

    # Save the image
    save_path = "test_saved_image.png"
    try:
        success = await vm.save_image(save_path)

        assert success is True
        assert Path(save_path).exists()
        assert len(vm.progress_updates) > 0
        assert vm.progress_updates[-1] == 100
        assert any(
            f"Image saved to {Path(save_path).name}" in msg
            for msg in vm.status_messages
        )
        assert len(vm.operation_results) == 1
        assert vm.operation_results[0] is True
    finally:
        # Clean up the saved file
        if Path(save_path).exists():
            Path(save_path).unlink()


async def test_save_with_no_image(reset_viewmodel):
    """Test error handling when saving with no image available."""
    # Create a new viewmodel to ensure no image is generated
    new_vm = DiffusifyViewModel()

    # Set up callback tracking
    status_messages = []
    new_vm.set_callbacks(on_status_update=lambda s: status_messages.append(s))

    # Try to save
    save_path = Path("test_nonexistent.png")
    success = await new_vm.save_image(str(save_path))

    assert success is False
    assert any("No image to save" in msg for msg in status_messages)
    assert not save_path.exists()


async def test_multiple_image_generations(reset_viewmodel):
    """Test that multiple images can be generated in sequence."""
    vm = reset_viewmodel

    # Generate first image
    success1 = await vm.generate_image(
        prompt="First test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    assert success1 is True
    first_image_path = vm.output_image_path
    assert Path(first_image_path).exists()

    # Generate second image
    success2 = await vm.generate_image(
        prompt="Second test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    assert success2 is True
    second_image_path = vm.output_image_path
    assert Path(second_image_path).exists()

    # Images should be different
    assert first_image_path != second_image_path
    assert len(vm.generated_images) >= 2
    assert vm.generated_images[-2] != vm.generated_images[-1]


async def test_negative_prompt_handling(reset_viewmodel):
    """Test that negative prompts are properly handled."""
    vm = reset_viewmodel

    success = await vm.generate_image(
        prompt="Test negative prompt",
        negative_prompt="blurry, distorted",
        steps=2,
        guidance_scale=7.5,
    )

    assert success is True
    assert vm.output_image_path is not None
    assert Path(vm.output_image_path).exists()
    assert any("Image generated" in msg for msg in vm.status_messages)


async def test_guidance_scale_parameter(reset_viewmodel):
    """Test that different guidance scale values are properly handled."""
    vm = reset_viewmodel

    success = await vm.generate_image(
        prompt="Guidance scale test",
        negative_prompt="",
        steps=2,
        guidance_scale=3.0,  # Lower guidance scale for speed
    )

    assert success is True
    assert vm.output_image_path is not None
    assert Path(vm.output_image_path).exists()
    assert any("Image generated" in msg for msg in vm.status_messages)
