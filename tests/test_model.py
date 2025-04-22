import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from diffusify.model import DiffusionModel, save_image

# Constants for the tests
TEST_PROMPT = "a beautiful sunset over mountains"
TEST_NEG_PROMPT = "ugly, blurry, distorted"
TEST_SIZE = 128
TEST_STEPS = 5

# -----------------------------------------------------------------------------
# Model Loading Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_loading():
    """Test loading the real diffusion model."""
    # Create the model
    model = DiffusionModel()

    # Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)

    # Verify loading succeeded
    assert success is True, f"Failed to load model: {error}"
    assert model.model_loaded is True, "Model not marked as loaded"
    assert model.pipeline is not None, "Pipeline not created"


@pytest.mark.asyncio
async def test_safety_components_loaded():
    """Test that safety components are loaded correctly."""
    # Create the model
    model = DiffusionModel()

    # Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

    # Check if safety checker was loaded
    assert model.safety_checker is not None, "Safety checker not loaded"
    assert model.feature_extractor is not None, "Feature extractor not loaded"


@pytest.mark.asyncio
async def test_reload_behavior():
    """Test loading, unloading, and reloading the model."""
    # Create a fresh model instance
    model = DiffusionModel()

    # 1. Load the model first time
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

    # 2. "Unload" the model by setting attributes to None
    model.pipeline = None
    model.model_loaded = False

    # 3. Load the model again
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to reload model: {error}"
    assert model.model_loaded is True, "Model not marked as loaded after reload"
    assert model.pipeline is not None, "Pipeline not created after reload"


# -----------------------------------------------------------------------------
# Image Generation Tests
# -----------------------------------------------------------------------------


@pytest.fixture
async def loaded_model():
    """Fixture to provide a loaded model for tests."""
    model = DiffusionModel()
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"
    return model


@pytest.mark.asyncio
async def test_basic_image_generation(loaded_model):
    """Test basic image generation functionality."""
    model = loaded_model

    # Generate a small image with few steps to save time
    img_path, seed, has_nsfw, error = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        # Verify generation succeeded
        assert error is None, f"Generation failed: {error}"
        assert img_path is not None, "No image path returned"
        assert seed is not None, "No seed returned"

        # Convert to Path object
        img_path_obj = Path(img_path)
        assert img_path_obj.exists(), "Generated image file doesn't exist"
    finally:
        # Clean up the generated file
        if img_path:
            img_path_obj = Path(img_path)
            if img_path_obj.exists():
                img_path_obj.unlink()


@pytest.mark.asyncio
async def test_generated_image_properties(loaded_model):
    """Test properties of the generated image."""
    model = loaded_model

    # Generate an image
    img_path, _, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        # Verify the image is valid
        img_path_obj = Path(img_path)
        with Image.open(img_path_obj) as img:
            assert img.size == (TEST_SIZE, TEST_SIZE), (
                f"Wrong image size, got {img.size}"
            )
            assert img.mode == "RGB", f"Wrong image mode, got {img.mode}"
    finally:
        # Clean up the generated file
        if img_path:
            img_path_obj = Path(img_path)
            if img_path_obj.exists():
                img_path_obj.unlink()


@pytest.mark.asyncio
async def test_generation_performance(loaded_model):
    """Test the performance of image generation."""
    model = loaded_model

    # Generate a small image with few steps to save time
    start_time = time.time()
    img_path, _, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )
    generation_time = time.time() - start_time

    # Log performance data
    print(f"Image generation took {generation_time:.2f}seconds with {TEST_STEPS} steps")

    try:
        # Simple assertion to ensure generation isn't abnormally slow
        # This is somewhat arbitrary - adjust based on expected performance
        assert generation_time < 60, f"Generation took too long: {generation_time:.2f}s"
    finally:
        # Clean up the generated file
        if img_path:
            img_path_obj = Path(img_path)
            if img_path_obj.exists():
                img_path_obj.unlink()


@pytest.mark.asyncio
async def test_first_load_image_generation():
    """Test that image generation works on first model load."""
    # Create a fresh model instance
    model = DiffusionModel()
    assert not model.model_loaded, "Model should start unloaded"

    # 1. Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

    # 2. Generate an image - first attempt after loading
    img_path, _, _, error = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        # Check if image generation succeeded
        assert error is None, f"First generation failed with error: {error}"
        assert img_path is not None, "No image path returned on first attempt"

        path_obj = Path(img_path)
        assert path_obj.exists(), "Generated image file doesn't exist"
    finally:
        # Clean up the image
        if img_path:
            path_obj = Path(img_path)
            if path_obj.exists():
                path_obj.unlink()


@pytest.mark.asyncio
async def test_consecutive_generations(loaded_model):
    """Test that consecutive image generations work properly."""
    model = loaded_model

    # Generate first image
    img1_path, _, _, error1 = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )
    assert error1 is None, f"First generation failed: {error1}"

    # Generate second image
    img2_path, _, _, error2 = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )
    assert error2 is None, f"Second generation failed: {error2}"

    try:
        # Both should succeed
        assert Path(img1_path).exists(), "First image doesn't exist"
        assert Path(img2_path).exists(), "Second image doesn't exist"
    finally:
        # Clean up generated files
        for path in [img1_path, img2_path]:
            if path:
                path_obj = Path(path)
                if path_obj.exists():
                    path_obj.unlink()


@pytest.mark.asyncio
async def test_generation_after_reload():
    """Test image generation after model reload."""
    # Create a fresh model instance
    model = DiffusionModel()

    # Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

    # "Unload" the model
    model.pipeline = None
    model.model_loaded = False

    # Reload the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to reload model: {error}"

    # Generate image after reload
    img_path, _, _, error = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        assert error is None, f"Generation after reload failed: {error}"
        assert img_path is not None, "No image path returned after reload"
        assert Path(img_path).exists(), "Image doesn't exist after reload"
    finally:
        if img_path:
            path_obj = Path(img_path)
            if path_obj.exists():
                path_obj.unlink()


# -----------------------------------------------------------------------------
# Random Seed Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_random_seed_generation(loaded_model):
    """Test that different calls produce different images (random seeds)."""
    model = loaded_model

    # Generate first image
    img1_path, seed1, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    # Generate second image with same parameters
    img2_path, seed2, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        # Verify seeds are different
        assert seed1 != seed2, "Seeds should be different for different calls"
    finally:
        # Clean up generated files
        for path in [img1_path, img2_path]:
            if path:
                path_obj = Path(path)
                if path_obj.exists():
                    path_obj.unlink()


@pytest.mark.asyncio
async def test_different_seed_produces_different_images(loaded_model):
    """Test that different seeds produce visually different images."""
    model = loaded_model

    # Generate first image
    img1_path, seed1, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    # Generate second image with same parameters
    img2_path, seed2, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        # Convert to Path objects
        img1_path_obj = Path(img1_path)
        img2_path_obj = Path(img2_path)

        # Compare the images - they should be different
        img1 = Image.open(img1_path_obj)
        img2 = Image.open(img2_path_obj)

        # Convert to numpy arrays for comparison
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Images should be different - check if any pixel differs
        assert not np.array_equal(arr1, arr2), (
            "Images should differ with different seeds"
        )
    finally:
        # Clean up generated files
        for path in [img1_path, img2_path]:
            if path:
                path_obj = Path(path)
                if path_obj.exists():
                    path_obj.unlink()


# -----------------------------------------------------------------------------
# Safety Checker Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_checker(loaded_model):
    """Test the safety checker with a simple safe image."""
    model = loaded_model

    # Create a test image to check
    test_image = Image.new("RGB", (TEST_SIZE, TEST_SIZE), color="white")

    # Apply safety checker
    filtered_image, has_nsfw = model.apply_safety_checker(test_image)

    # Plain white image should be safe
    assert has_nsfw is False, "Plain image incorrectly flagged as NSFW"

    # Filtered image should be the same as original for safe content
    assert filtered_image.size == test_image.size, (
        "Filtered image size doesn't match original"
    )


# -----------------------------------------------------------------------------
# Progress Callback Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_callback_registration(loaded_model):
    """Test setting a progress callback."""
    model = loaded_model

    # Track progress
    progress_values = []

    def test_callback(p):
        progress_values.append(p)

    # Set callback
    model.set_progress_callback(test_callback)

    # Generate an image to trigger callback
    img_path, _, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        # Verify progress tracking
        assert len(progress_values) > 0, "No progress updates received"
    finally:
        if img_path:
            path_obj = Path(img_path)
            if path_obj.exists():
                path_obj.unlink()


@pytest.mark.asyncio
async def test_progress_callback_values(loaded_model):
    """Test that progress callback receives expected values."""
    model = loaded_model

    # Track progress
    progress_values = []
    model.set_progress_callback(lambda p: progress_values.append(p))

    # Generate an image
    img_path, _, _, _ = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    try:
        # Progress should start from a low value
        assert min(progress_values) < 50, (
            f"Progress starts too high: {min(progress_values)}"
        )

        # Progress should eventually reach a high value
        assert max(progress_values) >= 80, (
            f"Progress doesn't reach completion: {max(progress_values)}"
        )

        # Progress should increase monotonically
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1], "Progress decreased"
    finally:
        if img_path:
            path_obj = Path(img_path)
            if path_obj.exists():
                path_obj.unlink()


# -----------------------------------------------------------------------------
# File Saving Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_saving():
    """Test saving files with real file operations."""
    # Create a test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as src_file:
        src_path = Path(src_file.name)
        try:
            # Create and save a test image
            test_img = Image.new("RGB", (64, 64), color="red")
            test_img.save(src_path)

            # Test saving to another valid location
            with tempfile.TemporaryDirectory() as tmp_dir:
                save_path = Path(tmp_dir) / "saved_image.png"
                error = await save_image(src_path, save_path)

                # Verify saving succeeded
                assert error is None, f"Save operation failed: {error}"
                assert save_path.exists(), "Saved file doesn't exist"
                assert save_path.stat().st_size > 0, "Saved file is empty"
        finally:
            if src_path.exists():
                src_path.unlink()


@pytest.mark.asyncio
async def test_file_content_preserved():
    """Test that file content is preserved when saving."""
    # Create a test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as src_file:
        src_path = Path(src_file.name)
        try:
            # Create and save a test image
            test_img = Image.new("RGB", (64, 64), color="red")
            test_img.save(src_path)

            # Test saving to another valid location
            with tempfile.TemporaryDirectory() as tmp_dir:
                save_path = Path(tmp_dir) / "saved_image.png"
                await save_image(src_path, save_path)

                # Verify the content is the same
                orig_img = Image.open(src_path)
                saved_img = Image.open(save_path)
                assert list(orig_img.getdata()) == list(saved_img.getdata()), (
                    "Image content changed"
                )
        finally:
            if src_path.exists():
                src_path.unlink()
