from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "outputs/final_merged_model"
HISTORY_FILE = "chat_history.txt"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Model loaded. Start chatting!\n")

# Initialize history
chat_history = [
    {"role": "system", "content": "You are a helpful Indian legal advisor. Provide clear answers with references to Indian Constitution, IPC, and Laws when possible."}
]

while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Append user input
    chat_history.append({"role": "user", "content": user_input})

    # Build prompt
    prompt = ""
    for msg in chat_history:
        role = msg["role"].capitalize()
        prompt += f"[{role}]: {msg['content']}\n"
    prompt += "[Assistant]:"  # ensure model knows to continue as assistant

    # Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    new_tokens = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    decoded = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

    # ✂️ Extract only the assistant's response
    if "[Assistant]:" in decoded:
        response = decoded.split("[Assistant]:")[-1].strip()
    else:
        response = decoded

    # Append to history
    chat_history.append({"role": "assistant", "content": response})

    # Show clean response
    print(f"🤖 Bot: {response}\n")

    # Save to file
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"👤 You: {user_input}\n")
        f.write(f"🤖 Bot: {response}\n\n")

