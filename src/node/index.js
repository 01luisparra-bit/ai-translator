// ============================================================
// index.js — Node CLI Translator using OpenAI
// Location: src/node/index.js
// ============================================================

import OpenAI from "openai";
import readline from "readline";
import path from "path";
import fs from "fs/promises";
import { fileURLToPath } from "url";
import "dotenv/config"; // Auto-loads .env from project root

// ---------- Path setup ----------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..", "..");
const DATA = path.join(ROOT, "data");
const INPUT = path.join(DATA, "input");
const OUTPUT = path.join(DATA, "output");
const TEMP = path.join(DATA, "temp");

// ---------- OpenAI client ----------
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ---------- CLI setup ----------
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// ---------- Helper: perform translation ----------
async function translateText(inputText, targetLang) {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a translation assistant. Translate everything into ${targetLang}.`,
      },
      {
        role: "user",
        content: inputText,
      },
    ],
  });

  const translated = completion.choices?.[0]?.message?.content || "";
  return translated.trim();
}

// ---------- Main flow ----------
async function main() {
  try {
    // Ensure folders exist
    await fs.mkdir(OUTPUT, { recursive: true });
    await fs.mkdir(INPUT, { recursive: true });
    await fs.mkdir(TEMP, { recursive: true });

    rl.question("Enter a sentence to translate: ", async (inputText) => {
      rl.question("Enter the target language (e.g. es, fr, de): ", async (targetLang) => {
        try {
          console.log("\nTranslating...");
          const translated = await translateText(inputText, targetLang);

          console.log(`\n🗣️ Translation (${targetLang}):\n${translated}`);

          const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
          const outFile = `manual_${targetLang}_${timestamp}.txt`;
          const outPath = path.join(OUTPUT, outFile);

          await fs.writeFile(outPath, translated, "utf-8");
          console.log(`\n✅ Saved to: ${outPath}`);
        } catch (error) {
          console.error("❌ Error:", error.message || error);
        } finally {
          rl.close();
        }
      });
    });
  } catch (err) {
    console.error("Fatal startup error:", err);
    process.exit(1);
  }
}

// ---------- Entry ----------
main();
