"""Module 3: Handwriting recognition (OCR).

Every engine has the same method:  read_lines(list_of_line_images) -> list of text.
Heavy libraries are imported only when an engine is created, so the rest of
the project can be tested without them.
"""


class TrOCREngine:
    """Microsoft TrOCR (Transformer OCR), handwritten model.

    The model reads ONE text line at a time, which is why the page is cut
    into lines first (see preprocessing.segment_lines).
    """

    def __init__(self, model_name="microsoft/trocr-base-handwritten", batch_size=4):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.batch_size = batch_size
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def read_lines(self, line_images):
        from PIL import Image

        texts = []
        for i in range(0, len(line_images), self.batch_size):
            batch = [Image.fromarray(img).convert("RGB") for img in line_images[i:i + self.batch_size]]
            pixels = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            with self.torch.no_grad():
                ids = self.model.generate(pixels, max_new_tokens=64)
            texts.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return [t.strip() for t in texts]


class TesseractEngine:
    """Tesseract OCR. Good for printed or very neat block letters only."""

    def __init__(self):
        import pytesseract

        self.pytesseract = pytesseract

    def read_lines(self, line_images):
        from PIL import Image

        texts = []
        for img in line_images:
            text = self.pytesseract.image_to_string(Image.fromarray(img), config="--psm 7")
            texts.append(text.strip())
        return texts


def create_engine(name, trocr_model="microsoft/trocr-base-handwritten"):
    name = (name or "").lower()
    if name == "trocr":
        return TrOCREngine(trocr_model)
    if name == "tesseract":
        return TesseractEngine()
    raise ValueError(f"Unknown OCR engine '{name}'. Use 'trocr' or 'tesseract'.")
