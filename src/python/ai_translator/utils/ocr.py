// src/utils/ocr.js
import Tesseract from "tesseract.js";
import sharp from "sharp";
import fs from "fs";
import path from "path";

export async function runOCR(imagePath) {
  try {
    const processedPath = imagePath.replace(/(\.\w+)$/, "_proc$1");

    // 1. Rotate and enhance
    await sharp(imagePath)
      .rotate()             // auto-rotate if metadata exists
      .grayscale()          // convert to grayscale
      .normalize()          // improve contrast
      .toFile(processedPath);

    // 2. OCR
    const { data } = await Tesseract.recognize(processedPath, "eng", {
      logger: (m) => console.log("OCR:", m),
    });

    // Optional: delete processed image to save space
    fs.unlinkSync(processedPath);

    return data.text;
  } catch (error) {
    console.error("❌ OCR error:", error.message);
    return "";
  }
}
