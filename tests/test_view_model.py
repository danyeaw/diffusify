from pathlib import Path

import pytest
from PIL import Image

from diffusify.view_model import DiffusifyViewModel

# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


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


@pytest.fixture
async def fresh_viewmodel():
    """Create a fresh viewmodel without loading the model."""
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

    yield vm

    # Clean up any generated images
    for img_path in vm.generated_images:
        if Path(img_path).exists():
            Path(img_path).unlink()

    if vm.output_image_path and Path(vm.output_image_path).exists():
        Path(vm.output_image_path).unlink()


# -----------------------------------------------------------------------------
# Model Loading Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_loading_status_updates(fresh_viewmodel):
    """Test that model loading updates status correctly."""
    vm = fresh_viewmodel

    success = await vm.load_model(use_attention_slicing=True, use_karras=True)

    assert success is True
    assert any("Model loaded successfully" in msg for msg in vm.status_messages)
    assert len(vm.operation_results) > 0
    assert vm.operation_results[-1] is True


@pytest.mark.asyncio
async def test_model_loading_progress_callbacks(fresh_viewmodel):
    """Test that model loading triggers progress callbacks."""
    vm = fresh_viewmodel

    await vm.load_model(use_attention_slicing=True, use_karras=True)

    assert len(vm.progress_updates) > 0
    assert vm.progress_updates[-1] == 100  # Should reach 100% at the end


@pytest.mark.asyncio
async def test_model_loading_with_karras(shared_viewmodel):
    """Test that the model loads successfully with Karras scheduler."""
    # Verify the model was loaded with Karras scheduler (already done in fixture)
    assert any(
        "Model loaded successfully" in msg for msg in shared_viewmodel.status_messages
    )


# -----------------------------------------------------------------------------
# Image Generation Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_image_generation(reset_viewmodel):
    """Test that basic image generation succeeds."""
    vm = reset_viewmodel

    success = await vm.generate_image(
        prompt="A simple test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    assert success is True
    assert vm.output_image_path is not None
    assert Path(vm.output_image_path).exists()
    assert any("Image generated" in msg for msg in vm.status_messages)


@pytest.mark.asyncio
async def test_image_generation_callbacks(reset_viewmodel):
    """Test that image generation triggers appropriate callbacks."""
    vm = reset_viewmodel

    await vm.generate_image(
        prompt="Callback test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    # Check progress callbacks
    assert len(vm.progress_updates) > 0
    assert vm.progress_updates[-1] == 100  # Should reach 100% at the end

    # Check image generated callback
    assert len(vm.generated_images) >= 1
    assert vm.generated_images[-1] == vm.output_image_path

    # Check operation complete callback
    assert len(vm.operation_results) >= 1
    assert vm.operation_results[-1] is True


@pytest.mark.asyncio
async def test_seed_in_status_message(reset_viewmodel):
    """Test that the seed is included in status messages."""
    vm = reset_viewmodel

    await vm.generate_image(
        prompt="Seed test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    assert any("Seed: " in msg for msg in vm.status_messages)


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_guidance_scale_parameter(reset_viewmodel):
    """Test that different guidance scale values are properly handled."""
    vm = reset_viewmodel

    success = await vm.generate_image(
        prompt="Guidance scale test",
        negative_prompt="",
        steps=2,
        guidance_scale=3.0,  # Lower guidance scale
    )

    assert success is True
    assert vm.output_image_path is not None
    assert Path(vm.output_image_path).exists()


# -----------------------------------------------------------------------------
# Multiple Generation Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_image_generations(reset_viewmodel):
    """Test that consecutive images can be generated successfully."""
    vm = reset_viewmodel

    # Generate first image
    success1 = await vm.generate_image(
        prompt="First test image", negative_prompt="", steps=2, guidance_scale=7.5
    )
    assert success1 is True
    first_image_path = vm.output_image_path

    # Generate second image
    success2 = await vm.generate_image(
        prompt="Second test image", negative_prompt="", steps=2, guidance_scale=7.5
    )
    assert success2 is True
    second_image_path = vm.output_image_path

    # Both should exist
    assert Path(first_image_path).exists()
    assert Path(second_image_path).exists()


@pytest.mark.asyncio
async def test_different_images_generated(reset_viewmodel):
    """Test that different images are generated in sequence."""
    vm = reset_viewmodel

    # Generate first image
    await vm.generate_image(
        prompt="First test image", negative_prompt="", steps=2, guidance_scale=7.5
    )
    first_image_path = vm.output_image_path

    # Generate second image
    await vm.generate_image(
        prompt="Second test image", negative_prompt="", steps=2, guidance_scale=7.5
    )
    second_image_path = vm.output_image_path

    # Images should be different
    assert first_image_path != second_image_path
    assert len(vm.generated_images) >= 2
    assert vm.generated_images[-2] != vm.generated_images[-1]


# -----------------------------------------------------------------------------
# Image Dimension Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_image_size(reset_viewmodel):
    """Test that image size can be updated."""
    vm = reset_viewmodel

    # Set new dimensions
    custom_width = 96
    custom_height = 64
    vm.update_image_size(width=custom_width, height=custom_height)

    # Verify dimensions were updated in the viewmodel
    # This assumes these attributes are accessible, adjust if needed
    assert vm.width == custom_width
    assert vm.height == custom_height


@pytest.mark.asyncio
async def test_custom_image_dimensions_applied(reset_viewmodel):
    """Test that custom image dimensions are applied to generated images."""
    vm = reset_viewmodel

    # Set custom dimensions
    custom_width = 96
    custom_height = 64
    vm.update_image_size(width=custom_width, height=custom_height)

    # Generate image with custom dimensions
    await vm.generate_image(
        prompt="A dimension test",
        negative_prompt="",
        steps=2,
        guidance_scale=7.5,
    )

    # Verify image has correct dimensions
    with Image.open(vm.output_image_path) as img:
        width, height = img.size

    assert width == custom_width
    assert height == custom_height


@pytest.mark.asyncio
async def test_different_aspect_ratios(reset_viewmodel):
    """Test that various aspect ratios are handled correctly."""
    vm = reset_viewmodel

    # Test a few different aspect ratios
    dimension_pairs = [
        (64, 64),  # 1:1
        (96, 64),  # 3:2
        (64, 96),  # 2:3
        (128, 64),  # 2:1
    ]

    for width, height in dimension_pairs:
        vm.update_image_size(width=width, height=height)

        await vm.generate_image(
            prompt=f"Aspect ratio test {width}x{height}",
            negative_prompt="",
            steps=2,
            guidance_scale=7.5,
        )

        # Verify dimensions
        with Image.open(vm.output_image_path) as img:
            actual_width, actual_height = img.size

        assert actual_width == width
        assert actual_height == height


# -----------------------------------------------------------------------------
# Image Saving Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_save_image(reset_viewmodel):
    """Test that a generated image can be saved to disk."""
    vm = reset_viewmodel

    # First generate a test image
    await vm.generate_image(
        prompt="Save test image", negative_prompt="", steps=2, guidance_scale=7.5
    )

    # Clear callback trackers
    vm.progress_updates.clear()
    vm.status_messages.clear()
    vm.operation_results.clear()

    # Save the image
    save_path = "test_saved_image.png"
    try:
        success = await vm.save_image(save_path)
        assert success is True
        assert Path(save_path).exists()
    finally:
        # Clean up the saved file
        if Path(save_path).exists():
            Path(save_path).unlink()


@pytest.mark.asyncio
async def test_save_image_callbacks(reset_viewmodel):
    """Test that saving an image triggers appropriate callbacks."""
    vm = reset_viewmodel

    # First generate a test image
    await vm.generate_image(
        prompt="Save callback test", negative_prompt="", steps=2, guidance_scale=7.5
    )

    # Clear callback trackers
    vm.progress_updates.clear()
    vm.status_messages.clear()
    vm.operation_results.clear()

    # Save the image
    save_path = "test_saved_image.png"
    try:
        await vm.save_image(save_path)

        # Check progress callbacks
        assert len(vm.progress_updates) > 0
        assert vm.progress_updates[-1] == 100

        # Check status message
        assert any(
            f"Image saved to {Path(save_path).name}" in msg
            for msg in vm.status_messages
        )

        # Check operation complete callback
        assert len(vm.operation_results) == 1
        assert vm.operation_results[0] is True
    finally:
        # Clean up the saved file
        if Path(save_path).exists():
            Path(save_path).unlink()


@pytest.mark.asyncio
async def test_save_with_no_image(fresh_viewmodel):
    """Test error handling when saving with no image available."""
    vm = fresh_viewmodel

    # Try to save when no image has been generated
    save_path = Path("test_nonexistent.png")
    success = await vm.save_image(str(save_path))

    assert success is False
    assert any("No image to save" in msg for msg in vm.status_messages)
    assert not save_path.exists()


@pytest.mark.asyncio
async def test_save_to_invalid_path(reset_viewmodel):
    """Test error handling when saving to an invalid path."""
    vm = reset_viewmodel

    # First generate a test image
    await vm.generate_image(
        prompt="Invalid path test", negative_prompt="", steps=2, guidance_scale=7.5
    )

    # Try to save to an invalid location (e.g., a non-existent directory)
    save_path = "/nonexistent/directory/image.png"
    success = await vm.save_image(save_path)

    assert success is False
    assert any("Error saving image" in msg for msg in vm.status_messages)


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_with_invalid_parameters(reset_viewmodel):
    """Test handling of invalid generation parameters."""
    vm = reset_viewmodel

    # Try with invalid steps (negative)
    success = await vm.generate_image(
        prompt="Invalid parameter test",
        negative_prompt="",
        steps=-1,  # Invalid negative steps
        guidance_scale=7.5,
    )

    assert success is True, "Generation should succeed with corrected parameters"
    assert vm.output_image_path is not None, "Should have a valid output path"
    assert Path(vm.output_image_path).exists(), "Generated image should exist"
    assert any(
        "Warning: Steps must be positive" in msg for msg in vm.status_messages
    ), "Should display warning about correcting steps"


@pytest.mark.asyncio
async def test_generate_with_invalid_guidance_scale(reset_viewmodel):
    """Test handling of invalid guidance scale parameter."""
    vm = reset_viewmodel

    # Try with invalid guidance scale (negative)
    success = await vm.generate_image(
        prompt="Invalid guidance scale test",
        negative_prompt="",
        steps=2,
        guidance_scale=-2.0,  # Invalid negative guidance scale
    )

    # Should correct the parameter and show a warning
    assert success is True, "Generation should succeed with corrected parameters"
    assert any(
        "Warning: Adjusted guidance scale" in msg for msg in vm.status_messages
    ), "Should display warning about correcting guidance scale"


@pytest.mark.asyncio
async def test_generate_with_empty_prompt(reset_viewmodel):
    """Test handling of empty prompt."""
    vm = reset_viewmodel

    # Try with empty prompt
    success = await vm.generate_image(
        prompt="",  # Empty prompt
        negative_prompt="",
        steps=2,
        guidance_scale=7.5,
    )

    # Should use default prompt and show a warning
    assert success is True, "Generation should succeed with default prompt"
    assert any("Warning: Empty prompt" in msg for msg in vm.status_messages), (
        "Should display warning about empty prompt"
    )


@pytest.mark.asyncio
async def test_update_image_size_with_invalid_dimensions(reset_viewmodel):
    """Test handling of invalid image dimensions."""
    vm = reset_viewmodel

    # Clear status messages
    vm.status_messages.clear()

    # Try with invalid dimensions
    vm.update_image_size(width=-100, height=0)

    # Should correct dimensions and show warnings
    assert vm.width == 512, "Width should be set to default"
    assert vm.height == 512, "Height should be set to default"
    assert any(
        "Warning: Width must be positive" in msg for msg in vm.status_messages
    ), "Should warn about invalid width"
    assert any(
        "Warning: Height must be positive" in msg for msg in vm.status_messages
    ), "Should warn about invalid height"


@pytest.mark.asyncio
async def test_save_with_empty_path(reset_viewmodel):
    """Test error handling when saving with empty path."""
    vm = reset_viewmodel

    # First generate a valid image
    await vm.generate_image(
        prompt="Empty path test",
        negative_prompt="",
        steps=2,
        guidance_scale=7.5,
    )

    # Clear status messages
    vm.status_messages.clear()

    # Try to save with empty path
    success = await vm.save_image("")

    assert success is False, "Save should fail with empty path"
    assert any(
        "Error: Save path cannot be empty" in msg for msg in vm.status_messages
    ), "Should display error about empty path"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_workflow(reset_viewmodel):
    """Test the entire workflow from generation to saving."""
    vm = reset_viewmodel

    # 1. Generate an image
    generate_success = await vm.generate_image(
        prompt="Complete workflow test",
        negative_prompt="blurry, distorted",
        steps=2,
        guidance_scale=7.5,
    )
    assert generate_success is True
    assert vm.output_image_path is not None
    assert Path(vm.output_image_path).exists()

    # 2. Save the generated image
    save_path = "test_workflow_image.png"
    try:
        save_success = await vm.save_image(save_path)
        assert save_success is True
        assert Path(save_path).exists()

        # 3. Verify saved image is valid
        with Image.open(save_path) as img:
            assert img.mode == "RGB"  # Or whatever format is expected
    finally:
        # Clean up
        if Path(save_path).exists():
            Path(save_path).unlink()
