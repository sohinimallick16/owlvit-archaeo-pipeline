from PIL import Image, ImageOps, ImageEnhance, ImageFilter

def to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert the image to grayscale.
    """
    return image.convert("L").convert("RGB")

class PreprocessUtils:
    """
    A collection of preprocessing routines for satellite imagery.
    """

    @staticmethod
    def histogram_equalization(image: Image.Image) -> Image.Image:
        """
        Perform histogram equalization to improve global contrast.
        """
        return ImageOps.equalize(image)

    @staticmethod
    def autocontrast(image: Image.Image, cutoff: float = 0) -> Image.Image:
        """
        Automatically adjust contrast by cutting off extremes.

        Args:
            cutoff (float): Percentage (0-100) of pixels to cut off at
                black and white ends.
        """
        return ImageOps.autocontrast(image, cutoff=cutoff)

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Enhance contrast using PIL's ImageEnhance.

        Args:
            factor (float): 1.0 yields original image, >1.0 increases contrast.
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def unsharp_mask(
        image: Image.Image,
        radius: float = 2,
        percent: int = 150,
        threshold: int = 3
    ) -> Image.Image:
        """
        Apply an unsharp mask to sharpen edges.

        Args:
            radius (float): Radius of the Gaussian blur.
            percent (int): Percent of the difference to add back.
            threshold (int): Threshold for contrast difference.
        """
        return image.filter(ImageFilter.UnsharpMask(radius=radius,
                                                   percent=percent,
                                                   threshold=threshold))

    @staticmethod
    def preprocess_pipeline(
        image: Image.Image,
        equalize: bool = False,
        autocontrast: bool = True,
        contrast_factor: float = 1.5,
        unsharp: bool = False
    ) -> Image.Image:
        """
        Run a configurable sequence of preprocessing steps.

        Args:
            equalize (bool): Apply histogram equalization.
            autocontrast (bool): Apply autocontrast.
            contrast_factor (float): Factor for contrast enhancement.
            unsharp (bool): Apply unsharp mask.

        Returns:
            Preprocessed PIL.Image.Image
        """
        img = image
        if equalize:
            img = PreprocessUtils.histogram_equalization(img)
        if autocontrast:
            img = PreprocessUtils.autocontrast(img)
        if contrast_factor and contrast_factor != 1.0:
            img = PreprocessUtils.enhance_contrast(img, factor=contrast_factor)
        if unsharp:
            img = PreprocessUtils.unsharp_mask(img)
        return img
