import fs from 'fs';
import path from 'path';
import { PDFDocument } from 'pdf-lib';
import { createWorker } from 'tesseract.js';
import franc from 'franc';
import { default as csvWriterPkg } from 'csv-writer';
const { createArrayCsvWriter } = csvWriterPkg;

const pdfDir = './pdfs';
const outputCsv = './extracted_terms.csv';

async function extractTextFromPdf(filePath) {
  try {
    const pdfBytes = fs.readFileSync(filePath);
    const pdfDoc = await PDFDocument.load(pdfBytes, { ignoreEncryption: true });
    const pages = pdfDoc.getPages();
    let fullText = '';

    for (const page of pages) {
      const textContent = await page.getTextContent?.(); // pdf-lib text extraction is limited
      if (textContent) fullText += textContent.items.map(i => i.str).join(' ') + '\n';
    }

    // Fallback to OCR if no text
    if (!fullText.trim()) {
      const worker = await createWorker(); // logger removed for Node 22+ compatibility
      await worker.load();
      await worker.loadLanguage('eng');
      await worker.initialize('eng');
      const { data } = await worker.recognize(filePath);
      fullText = data.text;
      await worker.terminate();
    }

    return fullText;
  } catch (err) {
    console.warn(`⚠️ Could not process ${filePath}:`, err.message);
    return '';
  }
}

async function main() {
  const files = fs.readdirSync(pdfDir).filter(f => f.toLowerCase().endsWith('.pdf'));
  const csvWriter = createArrayCsvWriter({
    path: outputCsv,
    header: ['File', 'Language', 'Text'],
  });

  const rows = [];

  for (const file of files) {
    console.log(`📘 Processing ${file}`);
    const filePath = path.join(pdfDir, file);
    const text = await extractTextFromPdf(filePath);
    if (text) {
      const lang = franc(text) !== 'und' ? franc(text) : 'unknown';
      rows.push([file, lang, text]);
    }
  }

  if (rows.length) {
    await csvWriter.writeRecords(rows);
    console.log(`✅ Extraction complete. CSV saved to ${outputCsv}`);
  } else {
    console.log('⚠️ No text extracted.');
  }
}

main();
