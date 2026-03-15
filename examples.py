"""Example: How to use the PDF pipeline programmatically."""

import logging
from pathlib import Path

from qmrkg import PDFPipeline, PDFConverter, OCRProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def example_full_pipeline():
    """Example 1: Use the complete pipeline."""
    print("=" * 50)
    print("Example 1: Full Pipeline (PDF -> PNG -> Text)")
    print("=" * 50)

    pipeline = PDFPipeline(
        pdf_dir=Path("data/pdf"),
        image_dir=Path("data/png"),
        text_dir=Path("data/markdown"),
        dpi=200,
        ocr_lang="ch",  # Chinese + English
        use_gpu=False,
    )

    # Process all PDFs
    results = pipeline.process_all(save_images=True)

    # Or process single PDF
    # image_paths, text_path = pipeline.process_pdf(Path("data/pdf/example.pdf"))

    return results


def example_step_by_step():
    """Example 2: Step by step conversion."""
    print("=" * 50)
    print("Example 2: Step by Step")
    print("=" * 50)

    pdf_path = Path("data/pdf/example.pdf")
    if not pdf_path.exists():
        print(f"Create a test PDF at {pdf_path} first")
        return

    # Step 1: PDF -> PNG
    print("\nStep 1: Converting PDF to images...")
    converter = PDFConverter(dpi=200, output_dir=Path("data/png"))
    image_paths = converter.convert(pdf_path)
    print(f"Generated {len(image_paths)} images")

    # Step 2: PNG -> Text
    print("\nStep 2: OCR processing...")
    ocr = OCRProcessor(lang="ch")
    text_path = ocr.process_and_save(image_paths, Path("data/markdown/output.txt"))
    print(f"Text saved to: {text_path}")

    # Or get text directly
    # text = ocr.extract_text(image_paths[0])
    # print(f"Extracted text length: {len(text)} chars")


def example_custom_processing():
    """Example 3: Custom processing with confidence scores."""
    print("=" * 50)
    print("Example 3: Custom Processing with Confidence")
    print("=" * 50)

    ocr = OCRProcessor(lang="ch")

    image_path = Path("data/png/test.png")
    if image_path.exists():
        text, confidence = ocr.extract_text(image_path, return_confidence=True)
        print(f"Confidence: {confidence:.2%}")
        print(f"Text preview: {text[:200]}...")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    # Run examples
    # example_step_by_step()
    # example_custom_processing()

    print("Choose an example to run:")
    print("1. Full pipeline: example_full_pipeline()")
    print("2. Step by step: example_step_by_step()")
    print("3. Custom: example_custom_processing()")
    print("\nOr use CLI: python main.py --help")
