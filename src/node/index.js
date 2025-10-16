import OpenAI from "openai";
import readline from "readline";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function main() {
  rl.question("Enter a sentence to translate: ", async (inputText) => {
    rl.question("Enter the target language: ", async (targetLang) => {
      try {
        console.log("Translating...");
        const completion = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [
            {
              role: "system",
              content: `You are a translation assistant. Translate everything into ${targetLang}.`
            },
            {
              role: "user",
              content: inputText
            }
          ]
        });

        const translated = completion.choices[0].message.content;
        console.log(`\n🗣️ Translation (${targetLang}):`);
        console.log(translated);
      } catch (error) {
        console.error("Error:", error);
      } finally {
        rl.close();
      }
    });
  });
}

main();
