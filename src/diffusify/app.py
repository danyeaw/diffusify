"""Diffusify - A simple app to generate images from text using diffusion models.
Refactored into MVVM architecture."""

import toga

from diffusify.view import DiffusifyView, DiffusifyViewModel


class DiffusifyApp(toga.App):
    """Main application class."""

    def startup(self):
        """Initialize the application."""
        self.main_window = toga.MainWindow(title=self.formal_name, size=(800, 650))

        self.viewmodel = DiffusifyViewModel()
        self.view = DiffusifyView(self.viewmodel)

        self.main_window.content = self.view
        self.main_window.show()


def main():
    """Entry point for the application."""
    return DiffusifyApp("Diffusify", "com.example.diffusify")


if __name__ == "__main__":
    app = main()
    app.main_loop()
