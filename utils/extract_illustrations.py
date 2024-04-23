# Extracts illustrations from a given manual page.
# Dependencies:
#     pip install --upgrade opencv-python numpy pdf2image pytesseract
# Usage:
#     python extract_illustrations.py <page-number> [output-directory] [path-to-pdf]
# Examples:
#     python extract_illustrations.py 1 /tmp ~/Downloads/manual.pdf
#     python extract_illustrations.py 1 /tmp # using path to the manual via MR2_DOCS_MANUAL_PATH environment variable
#     python extract_illustrations.py 1      # output to current directory

import cv2

import helper_functions as util

manualPath, pageNumber, outputDirectory = util.parse_arguments()
manualPage = util.pdf_to_image(manualPath, pageNumber)
illustrations = util.get_illustrations(manualPage)

for illustration in illustrations:
    image = util.remove_border(illustration, 8)
    filename = util.get_filename(image)
    # util.show_image(image)
    image_alpha = util.white_to_alpha(image)
    cv2.imwrite(outputDirectory + "/" + filename + ".webp", image_alpha, [cv2.IMWRITE_WEBP_QUALITY, 100])
