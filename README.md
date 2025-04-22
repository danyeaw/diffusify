# Diffusify

Diffusify is a cross-platform desktop application for AI image generation using
the Stable Diffusion XL model. Create stunning images from text prompts with an
intuitive interface.

![Screenshot of the main window with a generated scenic picture of mountains
and a lake](docs/images/screenshot-main-window-mountains.png)

## Features

- **AI-Powered Image Generation**: Generate images from text descriptions using
Stable Diffusion XL
- **User-Friendly Interface**: Simple and intuitive controls for all experience levels
- **Customizable Parameters**:
  - Text prompts with negative prompt support
  - Multiple image size options (512×512, 768×768, 1024×1024, etc.)
  - Adjustable diffusion steps and guidance scale
- **Advanced Options**:
  - Attention slicing for reduced memory usage
  - Karras scheduler for improved quality
- **Safety Filtering**: Automatic detection and blocking of NSFW content
- **Image Export**: Save your creations to your device

## Installation

### Prerequisites

- Python 3.12+
- GPU (recommended for faster generation)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/danyeaw/diffusify.git
cd diffusify
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:

```bash
python -m diffusify
```

## Usage

- Enter your prompt: Describe the image you want to generate
- Set negative prompt: Describe elements you want to avoid
- Adjust settings:
  - Image size: Select from preset dimensions
  - Steps: Higher values (30-50) for more detail but slower generation
  - Guidance: Higher values (7-9) for closer adherence to prompt
- Generate: Click the "Generate Image" button
- Save: When satisfied with the result, click "Save Image"

## Project Architecture

Diffusify follows the Model-View-ViewModel (MVVM) architecture pattern:
Model (model.py)

- Handles the AI image generation pipeline
- Manages the Stable Diffusion XL model
- Processes image generation requests

ViewModel (view_model.py)

- Coordinates between View and Model
- Handles parameter validation
- Manages state and callbacks

View (view.py)

- Creates and manages the user interface
- Handles user interactions
- Displays progress and results

Technical Implementation

- Asynchronous Processing: Non-blocking UI during model loading and image generation
- Progress Tracking: Real-time feedback during operations
- Thread-Safe Callbacks: Properly handles updates from worker threads
- Error Handling: Graceful management of errors during generation and saving

## License

[MIT License](LICENSE)

## Acknowledgments

- [Stable Diffusion XL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for the base model.
This model is shared using the openrail++ Responsible AI License
- [Dreamshaper SDXL-1-0](https://huggingface.co/Lykon/dreamshaper-xl-1-0) for the fine-tuned model
- [BeeWare](https://beeware.org) for the Toga cross-platform UI framework
- [Hugging Face](https://huggingface.co) libraries for model download and pipeline creation
