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
TEST_SIZE = 128  # Small size for faster tests
TEST_STEPS = 5  # Few steps for faster tests


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

    # Check if safety checker was loaded
    assert model.safety_checker is not None, "Safety checker not loaded"
    assert model.feature_extractor is not None, "Feature extractor not loaded"


@pytest.mark.asyncio
async def test_image_generation():
    """Test generating an image with the real model."""
    # Create the model
    model = DiffusionModel()

    # Track progress
    progress_points = []
    model.set_progress_callback(lambda p: progress_points.append(p))

    # Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

    # Generate a small image with few steps to save time
    start_time = time.time()
    img_path, seed, has_nsfw, error = await model.generate_image(
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
        # Verify generation succeeded
        assert error is None, f"Generation failed: {error}"
        assert img_path is not None, "No image path returned"

        # Convert to Path object
        img_path_obj = Path(img_path)
        assert img_path_obj.exists(), "Generated image file doesn't exist"
        assert seed is not None, "No seed returned"

        # Verify progress was tracked
        assert len(progress_points) > 0, "No progress updates received"
        assert max(progress_points) >= 80, "Progress didn't reach completion"

        # Verify the image is valid
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
async def test_safety_checker():
    """Test the safety checker with real model."""
    # Create the model
    model = DiffusionModel()

    # Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

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

                # Verify the content is the same
                orig_img = Image.open(src_path)
                saved_img = Image.open(save_path)
                assert list(orig_img.getdata()) == list(saved_img.getdata()), (
                    "Image content changed"
                )
        finally:
            if src_path.exists():
                src_path.unlink()


@pytest.mark.asyncio
async def test_random_seed_generation():
    """Test that different calls produce different images (random seeds)."""
    # Create the model
    model = DiffusionModel()

    # Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

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


@pytest.mark.asyncio
async def test_progress_callback():
    """Test that progress callback works with real generation."""
    # Create the model
    model = DiffusionModel()

    # Track progress
    progress_values = []
    model.set_progress_callback(lambda p: progress_values.append(p))

    # Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)
    assert success is True, f"Failed to load model: {error}"

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
        # Verify progress tracking
        assert len(progress_values) > 0, "No progress updates received"

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
        # Clean up the generated file
        if img_path:
            path_obj = Path(img_path)
            if path_obj.exists():
                path_obj.unlink()


async def test_first_load_image_generation():
    """Test that image generation works on first model load."""

    # Create a fresh model instance
    model = DiffusionModel()
    assert not model.model_loaded, "Model should start unloaded"

    # Track progress
    progress_values = []
    model.set_progress_callback(lambda p: progress_values.append(p))

    # 1. Load the model
    success, error = await model.load_pipeline(use_attention_slicing=True)

    # Verify loading succeeded
    assert success is True, f"Failed to load model: {error}"
    assert model.model_loaded is True, "Model not marked as loaded"
    assert model.pipeline is not None, "Pipeline not created"

    # 2. Generate an image - first attempt after loading
    img_path, seed, has_nsfw, error = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    # Cleanup and handling for the first image
    try:
        # Check if image generation succeeded
        assert error is None, f"First generation failed with error: {error}"
        assert img_path is not None, "No image path returned on first attempt"

        path_obj = Path(img_path)
        assert path_obj.exists(), "Generated image file doesn't exist"

        # Verify the image
        with Image.open(path_obj) as img:
            assert img.size == (TEST_SIZE, TEST_SIZE), f"Wrong image size: {img.size}"
            assert img.mode == "RGB", f"Wrong image mode: {img.mode}"

        # Check progress tracking
        assert len(progress_values) > 0, "No progress updates received"

    finally:
        # Clean up the first image
        if img_path:
            path_obj = Path(img_path)
            if path_obj.exists():
                path_obj.unlink()

    # 3. Reset progress tracking
    progress_values.clear()

    # 4. Generate a second image with the same model
    img_path2, seed2, has_nsfw2, error2 = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    # Cleanup and handling for the second image
    try:
        # Compare second attempt results
        assert error2 is None, f"Second generation failed with error: {error2}"
        assert img_path2 is not None, "No image path returned on second attempt"

        path_obj2 = Path(img_path2)
        assert path_obj2.exists(), "Second generated image doesn't exist"

        # Verify that both generations produced valid images
        assert seed != seed2, "Both generations used the same seed"

        # Check progress tracking for second generation
        assert len(progress_values) > 0, "No progress updates on second attempt"

    finally:
        # Clean up the second image
        if img_path2:
            path_obj2 = Path(img_path2)
            if path_obj2.exists():
                path_obj2.unlink()


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

    # 4. Generate an image after reload
    img_path, seed, has_nsfw, error = await model.generate_image(
        prompt=TEST_PROMPT,
        negative_prompt=TEST_NEG_PROMPT,
        steps=TEST_STEPS,
        guidance_scale=7.5,
        width=TEST_SIZE,
        height=TEST_SIZE,
    )

    # Cleanup
    try:
        assert error is None, f"Generation after reload failed: {error}"
        assert img_path is not None, "No image path returned after reload"

        path_obj = Path(img_path)
        assert path_obj.exists(), "Image doesn't exist after reload"

    finally:
        if img_path:
            path_obj = Path(img_path)
            if path_obj.exists():
                path_obj.unlink()
