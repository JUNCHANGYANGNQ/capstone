document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");
  const university = document.getElementById("university-select").value;
  const question = input.value.trim();
  if (!question) return;

  appendMessage("user", question);
  input.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ university, question }),
    });
    const data = await response.json();
    appendMessage("bot", data.answer || "❌ 无法获取回答");
  } catch (err) {
    appendMessage("bot", "❌ 出现错误，请稍后再试");
  }
}

function appendMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;
  msg.innerText = text;
  document.getElementById("chat-box").appendChild(msg);
}


document.getElementById("upload-btn").addEventListener("click", async () => {
  const university = document.getElementById("university-select").value;
  const fileInput = document.getElementById("file-upload");
  const file = fileInput.files[0];

  if (!file || !university) {
    alert("Please select both a file and a university.");
    return;
  }

  const formData = new FormData();
  formData.append("university", university);
  formData.append("file", file);

  try {
    const res = await fetch("/embed", {
      method: "POST",
      body: formData,
    });

    const result = await res.json();
    if (res.ok) {
      alert(`✅ ${result.university} embedded successfully!`);
    } else {
      alert("❌ Error: " + result.error);
    }
  } catch (err) {
    alert("🚨 Failed to upload: " + err.message);
  }
});

