// src/utils/translator.js
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function translateText(text, targetLang = "es") {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: `Translate the following text to ${targetLang}` },
        { role: "user", content: text },
      ],
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error("❌ Translation error:", error.message);
    return "";
  }
}
