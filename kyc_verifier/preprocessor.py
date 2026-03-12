import os
import cv2
import numpy as np
import tempfile

class ImagePreprocessor:
    """Preprocess license images: auto-crop, deskew, enhance for OCR"""

    @staticmethod
    def order_points(pts):
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        """Apply perspective transform to get a top-down view of the card"""
        rect = ImagePreprocessor.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def auto_crop_card(image):
        """
        Detect and crop the license card from a photo.
        Uses contour detection to find the largest rectangular shape.
        Falls back to the original image if no card is found.
        """
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        # Dilate to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("  [WARN] No contours found, using original image")
            return orig

        # Sort by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                # Found a quadrilateral — likely the card
                area = cv2.contourArea(approx)
                img_area = image.shape[0] * image.shape[1]

                # Card should be at least 10% of the image
                if area > img_area * 0.1:
                    print("  ✓ Card detected, applying perspective transform")
                    pts = approx.reshape(4, 2).astype("float32")
                    return ImagePreprocessor.four_point_transform(orig, pts)

        print("  [WARN] No card rectangle found, using original image")
        return orig

    @staticmethod
    def deskew(image):
        """Correct small rotations by detecting text line angle"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold to get text regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = -(angle - 90)
        else:
            angle = -angle

        # Only correct small angles (< 15 degrees)
        if abs(angle) > 15 or abs(angle) < 0.5:
            return image

        print(f"  [OK] Deskewing by {angle:.1f} degrees")
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def enhance_for_ocr(image):
        """Enhance image contrast for better OCR (light touch)"""
        # Resize if too small
        h, w = image.shape[:2]
        if w < 800:
            scale = 800 / w
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        # Light denoise
        denoised = cv2.bilateralFilter(image, 5, 50, 50)

        # Gentle contrast boost using CLAHE on L channel
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    @staticmethod
    def preprocess(image_path: str) -> str:
        """
        Full preprocessing pipeline.
        Returns path to the preprocessed temp image file.
        """
        print("\nPREPROCESSING:")
        print("-" * 70)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        # Step 1: Auto-crop the card
        cropped = ImagePreprocessor.auto_crop_card(img)

        # Step 2: Deskew
        deskewed = ImagePreprocessor.deskew(cropped)

        # Step 3: Enhance for OCR
        enhanced = ImagePreprocessor.enhance_for_ocr(deskewed)

        # Save to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        cv2.imwrite(temp_path, enhanced)
        print(f"  [OK] Preprocessed image saved ({enhanced.shape[1]}x{enhanced.shape[0]})")

        return temp_path
