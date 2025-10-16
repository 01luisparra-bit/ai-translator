// src/utils/fileHandler.js
import pdf from "pdf-poppler";
import fs from "fs";
import path from "path";
import { runOCR } from "./ocr.js";

export async function processPDF(pdfPath) {
  try {
    const tempDir = path.join("./temp_images");
    if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir);

    const fileName = path.basename(pdfPath, path.extname(pdfPath));
    const opts = {
      format: "png",
      out_dir: tempDir,
      out_prefix: fileName,
      scale: 1024,
    };

    console.log(`📄 Converting PDF to images for OCR: ${pdfPath}...`);
    await pdf.convert(pdfPath, opts);

    // Process all generated images
    const images = fs
      .readdirSync(tempDir)
      .filter((f) => f.startsWith(fileName) && f.endsWith(".png"));

    let fullText = "";
    for (const img of images) {
      const imgPath = path.join(tempDir, img);
      const text = await runOCR(imgPath);
      if (!text) console.warn(`❌ Failed OCR for image: ${imgPath}`);
      fullText += text + "\n";
    }

    return fullText.trim();
  } catch (error) {
    console.error("❌ PDF processing error:", error.message);
    return "";
  }
}
