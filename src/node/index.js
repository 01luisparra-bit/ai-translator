import OpenAI from "openai";
import readline from "readline";
import path from "path";
import { fileURLToPath } from "url";
import "dotenv/config"; // loads .env automatically

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

// ---------- Helper ----------
async function translateText(inputText, targetLang) {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are a translation assistant. Translate everything into ${targetLang}.`,
      },
      { role: "user", content: inputText },
    ],
  });

  const translated = completion.choices?.[0]?.message?.content || "";
  return translated.trim();
}

// ---------- Main ----------
async function main() {
  rl.question("Enter a sentence to translate: ", async (inputText) => {
    rl.question("Enter the target language (e.g. es, fr, de): ", async (targetLang) => {
      try {
        console.log("Translating...");
        const translated = await translateText(inputText, targetLang);
        console.log(`\n🗣️ Translation (${targetLang}):\n${translated}`);

        // Optional: save to /data/output
        const outPath = path.join(OUTPUT, `manual_${targetLang}.txt`);
        await Bun.write(outPath, translated);
        console.log(`\n✅ Saved to: ${outPath}`);
      } catch (error) {
        console.error("❌ Error:", error.message || error);
      } finally {
        rl.close();
      }
    });
  });
}

main().catch((err) => console.error("Fatal:", err));
